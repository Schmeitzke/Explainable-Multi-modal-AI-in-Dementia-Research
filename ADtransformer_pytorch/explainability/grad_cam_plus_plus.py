import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.transform import resize

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class GradCAMPlusPlus:
    """
    Helper class to compute Grad-CAM++ for ADTransformer model.
    Manages hooks to capture activations and gradients from PatchCNN branches.
    """
    def __init__(self, model: nn.Module, target_branch_idx: int = 0):
        self.model = model
        self.target_branch_idx = target_branch_idx
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        """Registers forward hook to capture activations from a PatchCNN branch."""
        def _forward_hook_fn(module, input, output):
            self.activations = output
            if output.requires_grad:
                output.retain_grad()

        hook_registered = False
        
        if (hasattr(self.model, 'image_emb') and 
            self.model.image_emb is not None and 
            hasattr(self.model.image_emb, 'branches') and 
            self.model.image_emb.branches is not None):
            
            if self.target_branch_idx < len(self.model.image_emb.branches):
                target_layer = self.model.image_emb.branches[self.target_branch_idx]
                for module in target_layer:
                    if isinstance(module, nn.Conv3d):
                        handle_fwd = module.register_forward_hook(_forward_hook_fn)
                        self.hook_handles.append(handle_fwd)
                        hook_registered = True
                        logger.info(f"Registered GradCAM++ hook on branch {self.target_branch_idx}")
                        break
        
        if not hook_registered and hasattr(self.model, 'image_emb') and self.model.image_emb is not None:
            if hasattr(self.model.image_emb, 'shared_cnn'):
                conv_layers = []
                for i, module in enumerate(self.model.image_emb.shared_cnn):
                    if isinstance(module, nn.Conv3d):
                        conv_layers.append((i, module))
                
                if conv_layers:
                    idx, target_conv = conv_layers[-1]
                    handle_fwd = target_conv.register_forward_hook(_forward_hook_fn)
                    self.hook_handles.append(handle_fwd)
                    hook_registered = True
                    logger.info(f"Registered GradCAM++ hook on PatchCNN Conv3d layer at index {idx}: {target_conv}")
        
        if not hook_registered and hasattr(self.model, 'image_emb') and self.model.image_emb is not None:
            conv_layers = []
            def find_conv3d_layers(module, layers_list, path=""):
                for name, child in module.named_children():
                    current_path = f"{path}.{name}" if path else name
                    if isinstance(child, nn.Conv3d):
                        layers_list.append((current_path, child))
                    else:
                        find_conv3d_layers(child, layers_list, current_path)
            
            find_conv3d_layers(self.model.image_emb, conv_layers)
            if conv_layers:
                path, target_conv = conv_layers[-1]
                handle_fwd = target_conv.register_forward_hook(_forward_hook_fn)
                self.hook_handles.append(handle_fwd)
                hook_registered = True
                logger.info(f"Registered GradCAM++ hook on Conv3d layer at path {path}: {target_conv}")
        
        if not hook_registered:
            conv_layers = []
            def find_conv3d_in_model(module, layers_list, path=""):
                for name, child in module.named_children():
                    current_path = f"{path}.{name}" if path else name
                    if isinstance(child, nn.Conv3d) and 'image' in current_path.lower():
                        layers_list.append((current_path, child))
                    else:
                        find_conv3d_in_model(child, layers_list, current_path)
            
            find_conv3d_in_model(self.model, conv_layers)
            if conv_layers:
                path, target_conv = conv_layers[-1]
                handle_fwd = target_conv.register_forward_hook(_forward_hook_fn)
                self.hook_handles.append(handle_fwd)
                hook_registered = True
                logger.info(f"Registered GradCAM++ hook on Conv3d layer from model search at {path}: {target_conv}")
        
        if not hook_registered:
            logger.error("Could not find any suitable Conv3d layers for GradCAM++ in the model")
            logger.error("Model structure inspection:")
            self._log_model_structure()

    def _log_model_structure(self):
        """Log the model structure to help debug hook registration issues."""
        def log_module_structure(module, prefix="", depth=0, max_depth=3):
            if depth > max_depth:
                return
            for name, child in module.named_children():
                child_type = type(child).__name__
                logger.error(f"{prefix}{name}: {child_type}")
                if depth < max_depth:
                    log_module_structure(child, prefix + "  ", depth + 1, max_depth)
        
        logger.error("Model structure:")
        log_module_structure(self.model)

    def _get_cam_plus_plus(self, non_image_data_single: torch.Tensor, image_data_single: torch.Tensor, target_class_idx: int = None) -> tuple:
        """
        Computes the Grad-CAM++ heatmap for a single sample.
        For PatchCNN architecture, this reconstructs spatial heatmaps from patch-based activations.
        """
        self.model.zero_grad()
        self.activations = None

        logits = self.model(non_image=non_image_data_single, image=image_data_single)

        if self.activations is None:
            logger.error("Activations not captured by forward hook. Check target layer and model graph.")
            pred_class = torch.argmax(logits, dim=1).item()
            confidence = torch.max(torch.softmax(logits, dim=1)).item()
            return None, pred_class, confidence

        if target_class_idx is None:
            target_class_idx = torch.argmax(logits, dim=1).item()

        score = logits[0, target_class_idx]
        score.backward()

        if self.activations.grad is None:
            logger.error("Gradients for activations not computed. Check retain_grad() and backward pass integrity.")
            confidence = torch.max(torch.softmax(logits, dim=1)).item()
            return None, target_class_idx, confidence

        activations = self.activations.detach()
        grads = self.activations.grad
        
        if len(activations.shape) == 5:
            logger.debug(f"Processing spatial activations: {activations.shape}")
            cam = self._compute_patch_based_cam_plus_plus(activations, grads, image_data_single)
        elif len(activations.shape) == 2:
            logger.debug(f"Processing pooled activations: {activations.shape}")
            cam = self._compute_pooled_cam_plus_plus(activations, grads, image_data_single)
        else:
            logger.warning(f"Unexpected activation shape: {activations.shape}")
            confidence = torch.max(torch.softmax(logits, dim=1)).item()
            return None, target_class_idx, confidence

        confidence = torch.max(torch.softmax(logits, dim=1)).item()
        return cam, target_class_idx, confidence

    def _compute_patch_based_cam_plus_plus(self, activations, grads, image_data_single):
        """
        Compute Grad-CAM++ from spatial convolutional activations and reconstruct to original image size.
        """
        batch_patches, _, _, _, _ = activations.shape
        
        patch_cams = []
        for i in range(batch_patches):
            patch_grads = grads[i:i+1]
            patch_activations = activations[i:i+1]
            
            grads_power_2 = patch_grads**2
            grads_power_3 = grads_power_2 * patch_grads

            sum_A_k = torch.sum(patch_activations, dim=(2, 3, 4), keepdim=True)

            alpha_numerator = grads_power_2
            alpha_denominator = 2 * grads_power_2 + sum_A_k * grads_power_3 + 1e-7
            alpha = alpha_numerator / alpha_denominator

            relu_grads = F.relu(patch_grads)
            weights_k = torch.sum(alpha * relu_grads, dim=(2, 3, 4))

            sum_relu_grads_per_channel = torch.sum(relu_grads, dim=(2, 3, 4), keepdim=False)
            weights_k = weights_k / (sum_relu_grads_per_channel + 1e-7)
            weights_k = weights_k.squeeze(0)

            cam = torch.sum(weights_k.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * patch_activations.squeeze(0), dim=0)
            cam = F.relu(cam)

            if cam.numel() > 0 and cam.max() > 1e-6:
                cam = cam / cam.max()
            else:
                cam = torch.zeros_like(cam)
            
            patch_cams.append(cam)
        
        reconstructed_cam = self._reconstruct_cam_from_patches(patch_cams, image_data_single)
        return reconstructed_cam.cpu().numpy()

    def _compute_pooled_cam_plus_plus(self, activations, grads, image_data_single):
        """
        Compute simplified CAM++ from pooled activations (less spatial information available).
        """
        grads_power_2 = grads**2
        alpha = grads_power_2 / (2 * grads_power_2 + torch.sum(activations, dim=1, keepdim=True) * grads**3 + 1e-7)
        patch_importances = torch.sum(alpha * F.relu(grads) * activations, dim=1)
        
        if patch_importances.max() > 1e-6:
            patch_importances = patch_importances / patch_importances.max()
        
        reconstructed_cam = self._reconstruct_importance_from_patches(patch_importances, image_data_single)
        return reconstructed_cam.cpu().numpy()

    def _reconstruct_cam_from_patches(self, patch_cams, image_data_single):
        """
        Reconstruct the full image CAM from individual patch CAMs.
        """
        B, C, H, W, D = image_data_single.shape
        
        if hasattr(self.model, 'image_emb') and hasattr(self.model.image_emb, 'patch_size'):
            patch_size = self.model.image_emb.patch_size
            ph, pw, pd = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size, patch_size)
        else:
            num_patches = len(patch_cams)
            estimated_patches_per_dim = int(round(num_patches ** (1/3)))
            ph = pw = pd = H // estimated_patches_per_dim
            logger.warning(f"Estimated patch size: ({ph}, {pw}, {pd})")
        
        nH, nW, nD = H // ph, W // pw, D // pd
        
        reconstructed = torch.zeros((H, W, D), device=image_data_single.device)
        patch_counts = torch.zeros((H, W, D), device=image_data_single.device)
        
        for patch_idx, patch_cam in enumerate(patch_cams):
            patch_h = patch_idx // (nW * nD)
            patch_w = (patch_idx % (nW * nD)) // nD
            patch_d = patch_idx % nD
            
            if patch_h >= nH or patch_w >= nW or patch_d >= nD:
                continue
            
            h_start, h_end = patch_h * ph, (patch_h + 1) * ph
            w_start, w_end = patch_w * pw, (patch_w + 1) * pw
            d_start, d_end = patch_d * pd, (patch_d + 1) * pd
            
            patch_cam_resized = torch.nn.functional.interpolate(
                patch_cam.unsqueeze(0).unsqueeze(0),
                size=(ph, pw, pd),
                mode='trilinear',
                align_corners=False
            ).squeeze(0).squeeze(0)
            
            reconstructed[h_start:h_end, w_start:w_end, d_start:d_end] += patch_cam_resized
            patch_counts[h_start:h_end, w_start:w_end, d_start:d_end] += 1
        
        mask = patch_counts > 0
        reconstructed[mask] = reconstructed[mask] / patch_counts[mask]
        
        return reconstructed

    def _reconstruct_importance_from_patches(self, patch_importances, image_data_single):
        """
        Reconstruct spatial heatmap from patch-level importance scores.
        """
        B, C, H, W, D = image_data_single.shape
        
        if hasattr(self.model, 'image_emb') and hasattr(self.model.image_emb, 'patch_size'):
            patch_size = self.model.image_emb.patch_size
            ph, pw, pd = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size, patch_size)
        else:
            num_patches = len(patch_importances)
            estimated_patches_per_dim = int(round(num_patches ** (1/3)))
            ph = pw = pd = H // estimated_patches_per_dim
        
        nH, nW, nD = H // ph, W // pw, D // pd
        
        reconstructed = torch.zeros((H, W, D), device=image_data_single.device)
        
        for patch_idx, importance in enumerate(patch_importances):
            patch_h = patch_idx // (nW * nD)
            patch_w = (patch_idx % (nW * nD)) // nD
            patch_d = patch_idx % nD
            
            if patch_h >= nH or patch_w >= nW or patch_d >= nD:
                continue
            
            h_start, h_end = patch_h * ph, (patch_h + 1) * ph
            w_start, w_end = patch_w * pw, (patch_w + 1) * pw
            d_start, d_end = patch_d * pd, (patch_d + 1) * pd
            
            reconstructed[h_start:h_end, w_start:w_end, d_start:d_end] = importance
        
        return reconstructed

    def remove_hooks(self):
        """Removes all registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        logger.debug("GradCAM++ hooks removed.")


class GradCAMPlusPlusGenerator:
    """
    Generates Grad-CAM++ visualizations for the ADTransformer model.
    """
    def __init__(self, model: nn.Module, device: torch.device, target_branch_idx: int = 0, class_names: list = None):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.target_branch_idx = target_branch_idx
        self.class_names = class_names if class_names else ['CN', 'MCI', 'AD']

        self.grad_cam_plus_plus_instance = GradCAMPlusPlus(self.model, self.target_branch_idx)
        
        self.hooks_available = len(self.grad_cam_plus_plus_instance.hook_handles) > 0
        if not self.hooks_available:
            logger.warning("No GradCAM++ hooks were registered. GradCAM++ visualization will be skipped.")

        self.stats = {
            'total_samples': 0,
            'correct_predictions': 0,
            'failed_samples': 0
        }

    def generate_heatmap_for_sample(self, non_image_data: torch.Tensor, image_data: torch.Tensor, target_class: int = None) -> tuple:
        """
        Generates a single Grad-CAM++ heatmap for a given sample.
        """
        non_image_data_single = non_image_data.to(self.device)
        image_data_single = image_data.to(self.device)
        self.model.to(self.device)

        heatmap_np, predicted_class_idx, confidence = self.grad_cam_plus_plus_instance._get_cam_plus_plus(
            non_image_data_single, image_data_single, target_class
        )
        return heatmap_np, predicted_class_idx, confidence

    def _plot_grad_cam_plus_plus_visualization(self, original_image_3d: np.ndarray, heatmap_3d_np: np.ndarray,
                                     output_dir: str, sample_idx: int,
                                     pred_class_idx: int, true_label_idx: int, confidence: float):
        """
        Plots and saves the Grad-CAM++ visualization for a single sample.
        """
        if len(original_image_3d.shape) != 4:
            logger.warning(f"Expected 4D image data (C, H, W, D), got {original_image_3d.shape}. Skipping Grad-CAM++ plot for sample {sample_idx}.")
            return

        if heatmap_3d_np is None or heatmap_3d_np.size == 0:
            logger.warning(f"Invalid or empty Grad-CAM++ heatmap for sample {sample_idx}. Skipping plot.")
            return

        c, h, w, d = original_image_3d.shape
        img_h, img_w, img_d = h, w, d

        img_channel = original_image_3d[0]

        try:
            heatmap_resized = resize(heatmap_3d_np, (img_h, img_w, img_d), order=1, preserve_range=True, anti_aliasing=True)
        except Exception as e:
            logger.error(f"Error resizing Grad-CAM++ heatmap for sample {sample_idx}: {e}. Heatmap shape: {heatmap_3d_np.shape}, Target: {(img_h, img_w, img_d)}")
            return
        
        mid_h_orig = img_h // 2
        mid_w_orig = img_w // 2
        mid_d_orig = img_d // 2
        
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        
        pred_name = self.class_names[pred_class_idx] if pred_class_idx < len(self.class_names) else f"Cls {pred_class_idx}"
        true_name = self.class_names[true_label_idx] if true_label_idx < len(self.class_names) else f"Cls {true_label_idx}"
        fig.suptitle(f"Grad-CAM++ Sample {sample_idx} - Pred: {pred_name} ({confidence:.2f}), True: {true_name}", fontsize=14)

        view_titles = ['Sagittal View', 'Coronal View', 'Axial View']
        
        sagittal_slice = img_channel[:, mid_w_orig, :]
        sagittal_heatmap = heatmap_resized[:, mid_w_orig, :]
        
        axes[0, 0].imshow(sagittal_slice, cmap='gray')
        axes[0, 0].set_title(f"{view_titles[0]}")
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(sagittal_heatmap, cmap='jet', vmin=0, vmax=1)
        axes[0, 1].set_title("Grad-CAM++")
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(sagittal_slice, cmap='gray')
        axes[0, 2].imshow(sagittal_heatmap, cmap='jet', alpha=0.5, vmin=0, vmax=1)
        axes[0, 2].set_title("Overlay")
        axes[0, 2].axis('off')

        coronal_slice = img_channel[mid_h_orig, :, :]
        coronal_heatmap = heatmap_resized[mid_h_orig, :, :]
        
        axes[1, 0].imshow(coronal_slice, cmap='gray')
        axes[1, 0].set_title(f"{view_titles[1]}")
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(coronal_heatmap, cmap='jet', vmin=0, vmax=1)
        axes[1, 1].set_title("Grad-CAM++")
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(coronal_slice, cmap='gray')
        axes[1, 2].imshow(coronal_heatmap, cmap='jet', alpha=0.5, vmin=0, vmax=1)
        axes[1, 2].set_title("Overlay")
        axes[1, 2].axis('off')

        axial_slice = img_channel[:, :, mid_d_orig]
        axial_heatmap = heatmap_resized[:, :, mid_d_orig]
        
        axes[2, 0].imshow(axial_slice, cmap='gray')
        axes[2, 0].set_title(f"{view_titles[2]}")
        axes[2, 0].axis('off')
        
        axes[2, 1].imshow(axial_heatmap, cmap='jet', vmin=0, vmax=1)
        axes[2, 1].set_title("Grad-CAM++")
        axes[2, 1].axis('off')
        
        axes[2, 2].imshow(axial_slice, cmap='gray')
        axes[2, 2].imshow(axial_heatmap, cmap='jet', alpha=0.5, vmin=0, vmax=1)
        axes[2, 2].set_title("Overlay")
        axes[2, 2].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plot_path = os.path.join(output_dir, f"grad_cam_plus_plus_sample_{sample_idx}.png")
        try:
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        except Exception as e:
            logger.error(f"Failed to save Grad-CAM++ plot for sample {sample_idx} to {plot_path}: {e}")
        plt.close(fig)

    def process_batch(self, non_image_data_batch: torch.Tensor, image_data_batch: torch.Tensor, labels_batch: torch.Tensor, 
                        output_dir: str, global_sample_offset: int, samples_to_process_in_this_batch: int) -> int:
        """
        Processes a single batch of data for Grad-CAM++, generates visualizations, and updates stats.
        """
        if not self.hooks_available:
            logger.warning("No GradCAM++ hooks available. Skipping batch processing.")
            self.stats['failed_samples'] += min(len(labels_batch), samples_to_process_in_this_batch)
            return min(len(labels_batch), samples_to_process_in_this_batch)
        
        if not self.grad_cam_plus_plus_instance.hook_handles:
            logger.debug("Re-registering GradCAM++ hooks before processing batch.")
            self.grad_cam_plus_plus_instance._register_hooks()
            self.hooks_available = len(self.grad_cam_plus_plus_instance.hook_handles) > 0
            if not self.hooks_available:
                logger.warning("Failed to re-register GradCAM++ hooks. Skipping batch processing.")
                self.stats['failed_samples'] += min(len(labels_batch), samples_to_process_in_this_batch)
                return min(len(labels_batch), samples_to_process_in_this_batch)

        batch_size = non_image_data_batch.size(0)
        num_processed_this_call = 0

        for i in range(min(batch_size, samples_to_process_in_this_batch)):
            current_global_sample_idx = global_sample_offset + i

            current_non_image_sample = non_image_data_batch[i:i+1]
            current_image_sample = image_data_batch[i:i+1] if image_data_batch is not None else None
            true_label = labels_batch[i].item()

            try:
                if current_image_sample is not None:
                    heatmap_np, pred_class_idx, confidence = self.generate_heatmap_for_sample(
                        current_non_image_sample, current_image_sample
                    )
                    
                    if heatmap_np is not None:
                        self.stats['total_samples'] += 1
                        if pred_class_idx == true_label: 
                            self.stats['correct_predictions'] += 1
                        
                        original_image_for_plot = current_image_sample.squeeze(0).cpu().numpy()
                        self._plot_grad_cam_plus_plus_visualization(original_image_for_plot, heatmap_np,
                                                                  output_dir, current_global_sample_idx,
                                                                  pred_class_idx, true_label, confidence)
                    else:
                        logger.warning(f"Grad-CAM++ heatmap generation failed for global_sample_idx {current_global_sample_idx}.")
                        self.stats['failed_samples'] += 1
                else:
                    logger.warning(f"No image data for sample {current_global_sample_idx}, skipping Grad-CAM++.")
                    self.stats['failed_samples'] += 1
                    
            except Exception as e:
                logger.error(f"Error generating Grad-CAM++ for global_sample_idx {current_global_sample_idx}: {e}", exc_info=True)
                self.stats['failed_samples'] += 1
            
            num_processed_this_call += 1
        
        return num_processed_this_call

    def get_final_stats(self) -> dict:
        """Calculates and returns the final statistics."""
        if self.stats['total_samples'] > 0:
            self.stats['accuracy'] = (self.stats['correct_predictions'] / self.stats['total_samples']) * 100
        else:
            self.stats['accuracy'] = 0.0
        
        logger.info(f"Grad-CAM++ final stats: Processed {self.stats['total_samples']} successfully, {self.stats['failed_samples']} failed.")
        if self.stats['total_samples'] > 0:
            logger.info(f"Grad-CAM++ accuracy on processed samples: {self.stats['accuracy']:.2f}%")
        return self.stats

    def cleanup(self):
        """Removes hooks."""
        logger.debug("Cleaning up GradCAMPlusPlusGenerator: removing hooks.")
        self.grad_cam_plus_plus_instance.remove_hooks() 