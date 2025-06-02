import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

def get_confidences_predictions_labels(model, dataloader, device, model_modality_mode: str = "full"):
    """
    Gathers confidences, predictions, and true labels for all samples in the dataloader.
    Args:
        model_modality_mode: Mode of the model ("full", "image_only", "clinical_only").
    """
    model.eval()
    all_confidences = []
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for clin_data, mri_data, _, labels in tqdm(dataloader, desc="Calculating Predictions"):
            clin_data_input = clin_data.to(device) if model_modality_mode in ["full", "clinical_only"] else None
            mri_data_input = mri_data.to(device) if model_modality_mode in ["full", "image_only"] else None
            labels = labels.to(device)
            
            main_logits_output, image_only_logits_output, clinical_only_logits_output, _ = model(clin_data_input, mri_data_input)

            target_logits = None
            if model_modality_mode == "image_only":
                if image_only_logits_output is None: 
                    raise ValueError("ConfidenceDiagram: Image-only mode, but image_only_logits are None.")
                target_logits = image_only_logits_output
            elif model_modality_mode == "clinical_only":
                if clinical_only_logits_output is None: 
                    raise ValueError("ConfidenceDiagram: Clinical-only mode, but clinical_only_logits are None.")
                target_logits = clinical_only_logits_output
            else:
                if main_logits_output is None: 
                    raise ValueError("ConfidenceDiagram: Full mode, but main_logits are None. Check data or ensure both modalities are provided if mode is full.")
                target_logits = main_logits_output

            probabilities = F.softmax(target_logits, dim=1)
            
            confidences, predictions = torch.max(probabilities, dim=1)
            
            all_confidences.extend(confidences.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return np.array(all_confidences), np.array(all_predictions), np.array(all_labels)

def plot_confusion_matrix_and_stats(predictions, labels, output_dir, class_names=None):
    """
    Computes, plots, and saves a confusion matrix and classification report.

    Args:
        predictions: Array of predicted class indices.
        labels: Array of true class indices.
        output_dir: Directory to save the plot and report.
        class_names: Optional list of class names for labeling.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_unique_classes = sorted(np.unique(np.concatenate((predictions, labels))))
    num_classes = len(all_unique_classes)
    
    if class_names is None or len(class_names) != num_classes:
        if class_names is not None and len(class_names) != num_classes:
            print(f"Warning: Provided {len(class_names)} class names but found {num_classes} unique classes. Using generic names.")
        class_names = [f"Class {i}" for i in all_unique_classes]
    
    if len(class_names) != num_classes:
        print(f"Warning: Class names length mismatch. Expected {num_classes}, got {len(class_names)}. Using generic names.")
        class_names = [f"Class {i}" for i in all_unique_classes]

    report_str = classification_report(labels, predictions, target_names=class_names, zero_division=0)
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write("Classification Report:\n")
        f.write(report_str)
    print(f"Classification report saved to {report_path}")

    cm = confusion_matrix(labels, predictions)

    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names))))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names[:cm.shape[1]],
           yticklabels=class_names[:cm.shape[0]],
           title='Confusion Matrix',
           ylabel='True Label',
           xlabel='Predicted Label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    cm_plot_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix plot saved to {cm_plot_path}") 