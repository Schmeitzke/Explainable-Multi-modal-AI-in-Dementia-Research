import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import os
import logging

from model_components.build_model import build_adni_transformer
from prepare_adni_dataset import prepare_adni_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_epoch(model, dataloader, criterion, optimizer, device, use_images):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        non_image = batch['non_image'].to(device)
        labels = batch['label'].to(device)

        image = None
        if use_images:
            image = batch.get('image')
            if image is not None:
                image = image.to(device)

        optimizer.zero_grad()
        outputs = model(non_image=non_image, image=image)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        progress_bar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

    epoch_loss = total_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device, use_images):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            non_image = batch['non_image'].to(device)
            labels = batch['label'].to(device)

            image = None
            if use_images:
                image = batch.get('image')
                if image is not None:
                    image = image.to(device)

            outputs = model(non_image=non_image, image=image)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = total_loss / len(dataloader)
    val_acc = accuracy_score(all_labels, all_preds)

    return val_loss, val_acc, all_preds, all_labels

def train(config_path, train_csv_path, test_csv_path, output_dir):
    """
    Train the AD-Transformer model using data loaded dynamically from CSV and image directories.
    
    Parameters:
    - config_path: Path to the configuration YAML file
    - train_csv_path: Path to the training data CSV file
    - test_csv_path: Path to the test data CSV file
    - output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info("Starting AD-Transformer training...")
    logging.info(f"Config: {config_path}")
    logging.info(f"Train CSV: {train_csv_path}")
    logging.info(f"Test CSV: {test_csv_path}")
    logging.info(f"Output: {output_dir}")

    dataset = prepare_adni_dataset(config_path, train_csv_path, test_csv_path)
    
    if dataset is None:
        logging.error("Failed to prepare dataset. Aborting training.")
        return None
    
    train_loader = dataset['train_loader']
    val_loader = dataset['val_loader']
    feature_columns = dataset['feature_columns']
    config = dataset['config']
    class_weights = dataset['class_weights']
    
    if train_loader is None:
        logging.error("Train loader is None. Aborting training.")
        return None
    
    if val_loader is None:
        logging.warning("Validation loader is None. Training will proceed without validation.")

    with open(os.path.join(output_dir, "feature_columns.txt"), "w") as f:
        for col in feature_columns:
            f.write(f"{col}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs = config.get('training', {}).get('epochs', 20)
    learning_rate = config.get('training', {}).get('learning_rate', 0.001)
    use_images = config.get('image', {}).get('use_images', False)
    
    logging.info(f"Training for {num_epochs} epochs with learning rate {learning_rate}")
    logging.info(f"Using images: {use_images}")

    model = build_adni_transformer(config_path)
    model = model.to(device)

    criterion = None
    use_weight_config = config.get('training', {}).get('use_class_weights', False)

    if use_weight_config and class_weights is not None:
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        if use_weight_config and class_weights is None:
            logging.info("Class weights requested but not available/computable. Using unweighted loss.")
        else:
            logging.info("Not using class weights.")
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    best_val_acc = 0

    logging.info("Starting training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, use_images)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        if val_loader is not None:
            val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device, use_images)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
                print(f"New best model saved with validation accuracy: {val_acc:.4f}")
        else:
            val_losses.append(0.0)
            val_accs.append(0.0)
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print("No validation data available")
            
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
            print("Model saved (no validation for comparison)")

    logging.info("Training completed!")

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curves')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    logging.info(f"Training curves saved to {os.path.join(output_dir, 'training_curves.png')}")

    best_model_path = os.path.join(output_dir, "best_model.pth")
    if os.path.exists(best_model_path):
         logging.info(f"Best model saved at: {best_model_path}")
         return best_model_path
    else:
         logging.warning("Best model file not found after training.")
         return None
