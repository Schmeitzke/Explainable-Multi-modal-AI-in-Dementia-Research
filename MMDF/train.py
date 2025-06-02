import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import logging
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import config
from data_loader import get_dataloaders
from mmdf_model import MMDF
from components.self_adaptive_loss import SelfAdaptiveLoss

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Training mode: {config.training_mode}")

    try:
        train_loader, val_loader = get_dataloaders()
    except ValueError as e:
        print(f"Error initializing data loaders: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        return
        
    print(f"Training samples: {len(train_loader.dataset)}, Validation samples: {len(val_loader.dataset)}")
    if len(train_loader.dataset) == 0:
        print("Exiting: Training dataset is empty.")
        return

    print("Calculating class weights...")
    all_train_labels = []
    if hasattr(train_loader.dataset, 'labels'):
        all_train_labels = train_loader.dataset.labels.cpu().numpy()
    else:
        print("Warning: Direct label access failed, iterating loader once for class weights. This might be slow.")
        for _, _, _, batch_labels in train_loader:
            all_train_labels.extend(batch_labels.cpu().numpy())
    
    class_weights = None
    if len(all_train_labels) > 0:
        try:
            weights = compute_class_weight(
                class_weight='balanced', 
                classes=np.arange(config.num_classes),
                y=all_train_labels
            )
            class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
            class_counts = np.bincount(all_train_labels, minlength=config.num_classes)
            print(f"Original class counts: {class_counts}")
            print(f"Using sklearn 'balanced' class weights: {class_weights}")
        except ValueError as e:
            print(f"Error calculating sklearn class weights: {e}. Falling back to normalized inverse frequency.")
            class_counts = np.bincount(all_train_labels, minlength=config.num_classes)
            if np.any(class_counts == 0):
                print(f"Warning: Some classes have zero samples in the training set: {class_counts}. Weights for these will be 0 or problematic.")
            
            weights = 1. / class_counts
            weights[class_counts == 0] = 0
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights) * config.num_classes
            else:
                print("Warning: All class weights became zero after inverse. Setting to uniform weights.")
                weights = np.ones(config.num_classes, dtype=float) / config.num_classes * config.num_classes
            class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
            print(f"Fallback class counts: {class_counts}")
            print(f"Fallback normalized inverse frequency weights: {class_weights}")

    else:
        print("Warning: Could not calculate class weights, no labels found.")

    model = MMDF().to(device)
    try:
        patch_embed_module = model.image_net.patch_embed
        img_h, img_w = patch_embed_module.img_h_w_tuple
        patch_size = patch_embed_module.p_val
        num_patches_h = img_h // patch_size
        num_patches_w = img_w // patch_size
        patch_grid_size = (num_patches_h, num_patches_w)
        logging.info(f"Retrieved patch grid size for attention guidance: {patch_grid_size}")
    except AttributeError as e:
        logging.error(f"Could not retrieve patch embedding info from model: {e}. Attention guidance will be skipped.")
        patch_grid_size = None

    criterion = SelfAdaptiveLoss()
    if class_weights is not None:
        criterion_aux_img = nn.CrossEntropyLoss(weight=class_weights)
        criterion_aux_clin = nn.CrossEntropyLoss(weight=class_weights)
        print("Initialized AUX CrossEntropyLoss criteria WITH class weights for SelfAdaptiveLoss run.")
    else:
        criterion_aux_img = nn.CrossEntropyLoss()
        criterion_aux_clin = nn.CrossEntropyLoss()
        print("Warning: class_weights is None. Initialized AUX CrossEntropyLoss criteria WITHOUT class weights for SelfAdaptiveLoss run.")

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate,
                           weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

    best_val_loss = float('inf')

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs} [Train]")
        
        for clin, mri, mask, labels in train_pbar:
            clin_input, mri_input, mask_input, labels = clin.to(device), mri.to(device), mask.to(device), labels.to(device)
            optimizer.zero_grad()

            current_clin_input = clin_input if config.training_mode in ["full", "clinical_only"] else None
            current_mri_input = mri_input if config.training_mode in ["full", "image_only"] else None
            current_mask_input = mask_input if current_mri_input is not None and config.guidance_loss_weight > 0 else None

            main_outputs, image_only_outputs, clinical_only_outputs, last_attn_weights = model(current_clin_input, current_mri_input)

            total_loss = torch.tensor(0.0).to(device)
            main_loss = torch.tensor(0.0).to(device)
            img_aux_loss = torch.tensor(0.0).to(device)
            clin_aux_loss = torch.tensor(0.0).to(device)
            guidance_loss = torch.tensor(0.0).to(device)

            if config.training_mode == "full":
                if main_outputs is not None:
                    main_loss = criterion(main_outputs, labels, epoch, class_weights=class_weights)
                else:
                    logging.warning("Main outputs are None in 'full' training mode. Check data and model.forward logic.")

                if image_only_outputs is not None:
                    img_aux_loss = criterion_aux_img(image_only_outputs, labels)
                
                if clinical_only_outputs is not None and hasattr(config, 'clinical_aux_loss_weight') and config.clinical_aux_loss_weight > 0:
                    clin_aux_loss = criterion_aux_clin(clinical_only_outputs, labels)

                total_loss = main_loss + config.aux_loss_weight * img_aux_loss

            elif config.training_mode == "image_only":
                if image_only_outputs is not None:
                    img_aux_loss = criterion(image_only_outputs, labels, epoch, class_weights=class_weights)
                    total_loss = img_aux_loss
                else:
                    logging.error("Image only outputs are None in 'image_only' training mode. This should not happen.")
                    continue 

            elif config.training_mode == "clinical_only":
                if clinical_only_outputs is not None:
                    clin_aux_loss = criterion(clinical_only_outputs, labels, epoch, class_weights=class_weights)
                    total_loss = clin_aux_loss
                else:
                    logging.error("Clinical only outputs are None in 'clinical_only' training mode. This should not happen.")
                    continue 
            
            if config.training_mode in ["full", "image_only"] and config.guidance_loss_weight > 0 and \
               last_attn_weights is not None and patch_grid_size is not None and current_mask_input is not None:
                
                if last_attn_weights.ndim == 3:
                    B, seq_len_actual, _ = last_attn_weights.shape 
                    N = seq_len_actual - 1 

                    attn = last_attn_weights 
                    cls_attn_to_patches = attn[:, 0, 1:] 

                    expected_num_patches = patch_grid_size[0] * patch_grid_size[1]
                    if N == expected_num_patches:
                        attn_map_2d = cls_attn_to_patches.view(B, patch_grid_size[0], patch_grid_size[1])

                        processed_mask = current_mask_input 
                        if processed_mask.dim() == 3: processed_mask = processed_mask.unsqueeze(1)
                        resized_mask = F.interpolate(processed_mask.float(), size=patch_grid_size, mode='nearest').squeeze(1)
                        
                        guidance_loss = torch.mean(attn_map_2d * (1.0 - resized_mask))
                    else:
                        logging.warning(f"Patch count mismatch: Attn derived N ({N}) vs Grid product ({expected_num_patches}). Skipping guidance loss.")
                
                elif last_attn_weights.ndim == 4:
                    logging.warning("last_attn_weights was 4D. Reverting to averaging heads.")
                    B, H, N_plus_1, _ = last_attn_weights.shape
                    N = N_plus_1 - 1
                    attn = last_attn_weights.mean(dim=1) 
                    cls_attn_to_patches = attn[:, 0, 1:]
                    expected_num_patches = patch_grid_size[0] * patch_grid_size[1]
                    if N == expected_num_patches:
                        attn_map_2d = cls_attn_to_patches.view(B, patch_grid_size[0], patch_grid_size[1])
                        processed_mask = current_mask_input
                        if processed_mask.dim() == 3: processed_mask = processed_mask.unsqueeze(1)
                        resized_mask = F.interpolate(processed_mask.float(), size=patch_grid_size, mode='nearest').squeeze(1)
                        guidance_loss = torch.mean(attn_map_2d * (1.0 - resized_mask))
                    else:
                        logging.warning(f"Patch count mismatch (4D path): Attn derived N ({N}) vs Grid product ({expected_num_patches}). Skipping guidance loss.")
                else:
                    logging.warning(f"last_attn_weights has unexpected dimension: {last_attn_weights.ndim}. Skipping guidance loss.")
            elif config.training_mode in ["full", "image_only"] and config.guidance_loss_weight > 0:
                 logging.debug("Skipping guidance loss: One or more conditions not met (attn_weights, patch_grid_size, mask, or weight is zero).")
            
            if config.training_mode in ["full", "image_only"]: 
                 total_loss += config.guidance_loss_weight * guidance_loss

            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item() * labels.size(0)
            train_pbar.set_postfix({'total_loss': f"{total_loss.item():.4f}"})
            
        avg_train_loss = running_loss / len(train_loader.dataset)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{config.epochs} [Val]")

        with torch.no_grad():
            for clin, mri, mask, labels in val_pbar:
                clin_input, mri_input, mask_input, labels = clin.to(device), mri.to(device), mask.to(device), labels.to(device)
                
                current_clin_input = clin_input if config.training_mode in ["full", "clinical_only"] else None
                current_mri_input = mri_input if config.training_mode in ["full", "image_only"] else None

                main_outputs, image_only_outputs, clinical_only_outputs, _ = model(current_clin_input, current_mri_input)
                
                current_val_loss = torch.tensor(0.0).to(device)
                target_outputs_for_acc = None

                if config.training_mode == "full":
                    if main_outputs is not None:
                        main_val_loss = criterion(main_outputs, labels, epoch, class_weights=class_weights)
                        current_val_loss += main_val_loss
                        target_outputs_for_acc = main_outputs
                    if image_only_outputs is not None: 
                        current_val_loss += config.aux_loss_weight * criterion_aux_img(image_only_outputs, labels)
                    
                    if clinical_only_outputs is not None and hasattr(config, 'clinical_aux_loss_weight') and config.clinical_aux_loss_weight > 0: 
                        current_val_loss += config.clinical_aux_loss_weight * criterion_aux_clin(clinical_only_outputs, labels)

                elif config.training_mode == "image_only":
                    if image_only_outputs is not None:
                        current_val_loss = criterion(image_only_outputs, labels, epoch, class_weights=class_weights)
                        target_outputs_for_acc = image_only_outputs
                    else:
                        logging.warning("Validation: Image_only_outputs is None in image_only mode.")
                
                elif config.training_mode == "clinical_only":
                    if clinical_only_outputs is not None:
                        current_val_loss = criterion(clinical_only_outputs, labels, epoch, class_weights=class_weights)
                        target_outputs_for_acc = clinical_only_outputs
                    else:
                        logging.warning("Validation: clinical_only_outputs is None in clinical_only mode.")

                if target_outputs_for_acc is None and config.training_mode != "full":
                    logging.warning(f"Validation: Target outputs for accuracy are None in {config.training_mode} mode.")
                elif target_outputs_for_acc is None and config.training_mode == "full" and main_outputs is None:
                    logging.warning("Validation: Main outputs are None in full mode, cannot compute primary accuracy.")
                    if image_only_outputs is not None: target_outputs_for_acc = image_only_outputs

                val_loss += current_val_loss.item() * labels.size(0)
                
                if target_outputs_for_acc is not None:
                    preds = target_outputs_for_acc.argmax(dim=1)
                    correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)
                val_pbar.set_postfix({'val_loss': f"{current_val_loss.item():.4f}"})

        avg_val_loss = val_loss / len(val_loader.dataset) if len(val_loader.dataset) > 0 else 0
        val_accuracy = correct_val / total_val if total_val > 0 else 0

        print(f"Epoch {epoch}/{config.epochs} completed. "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_accuracy:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            try:
                torch.save(model.state_dict(), '')
                print(f"Epoch {epoch}: New best model saved with Val Loss: {avg_val_loss:.4f}")
            except Exception as e:
                print(f"Error saving best model at epoch {epoch}: {e}")

    try:
        torch.save(model.state_dict(), '')
        print(f"Training complete. Final epoch model saved to mmdf_final_epoch_model.pth")
    except Exception as e:
        print(f"Error saving final epoch model: {e}")

if __name__ == '__main__':
    main()
