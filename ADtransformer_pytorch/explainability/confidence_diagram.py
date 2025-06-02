import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import logging

logger = logging.getLogger(__name__)

def get_confidences_predictions_labels(model, dataloader, device):
    """
    Gathers confidences, predictions, and true labels for all samples in the dataloader.
    Adapted for ADTransformer which has a single logits output.
    """
    model.eval()
    all_confidences = []
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating Predictions"):
            if isinstance(batch, dict):
                non_image_data = batch['non_image']
                image_data = batch.get('image')
                labels = batch['label']
            elif isinstance(batch, (list, tuple)) and len(batch) >= 3:
                non_image_data, image_data, labels = batch[0], batch[1], batch[2]
            else:
                logger.error(f"Unexpected batch format: {type(batch)} with length {len(batch) if hasattr(batch, '__len__') else 'unknown'}")
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    non_image_data = batch[0]
                    labels = batch[-1]
                    image_data = batch[1] if len(batch) >= 3 else None
                    logger.warning("Using fallback batch parsing - results may be unreliable")
                else:
                    logger.error("Cannot process this batch format. Skipping batch.")
                    continue
            
            non_image_data = non_image_data.to(device)
            image_data = image_data.to(device) if image_data is not None else None
            labels = labels.to(device)
            
            try:
                logits = model(non_image=non_image_data, image=image_data)
                
                if logits is None:
                    logger.error("Model returned None logits. Skipping batch.")
                    continue
                
                probabilities = F.softmax(logits, dim=1)
                confidences, predictions = torch.max(probabilities, dim=1)
                
                all_confidences.extend(confidences.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
            except Exception as e:
                logger.error(f"Error during forward pass: {e}. Skipping batch.")
                continue
            
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

    num_unique_labels = len(np.unique(labels))
    if class_names is None or len(class_names) != num_unique_labels:
        actual_num_classes = len(np.unique(np.concatenate((predictions, labels))))
        class_names = [f"Class {i}" for i in range(actual_num_classes)]
        print(f"Warning: class_names mismatch or not provided. Using generic names: {class_names}")

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

def plot_confidence_histogram(confidences, predictions, labels, output_dir, class_names=None):
    """
    Plots a histogram of prediction confidences.
    
    Args:
        confidences: Array of prediction confidences.
        predictions: Array of predicted class indices.
        labels: Array of true class indices.
        output_dir: Directory to save the plot.
        class_names: Optional list of class names for labeling.
    """
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(np.unique(labels)))]
    
    correct_mask = predictions == labels
    correct_confidences = confidences[correct_mask]
    incorrect_confidences = confidences[~correct_mask]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.hist(confidences, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(np.mean(confidences), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(confidences):.3f}')
    ax1.set_xlabel('Prediction Confidence')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Prediction Confidences')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.hist(correct_confidences, bins=30, alpha=0.7, color='green', 
             label=f'Correct ({len(correct_confidences)} samples)', edgecolor='black')
    ax2.hist(incorrect_confidences, bins=30, alpha=0.7, color='red', 
             label=f'Incorrect ({len(incorrect_confidences)} samples)', edgecolor='black')
    ax2.axvline(np.mean(correct_confidences), color='darkgreen', linestyle='--', linewidth=2, 
                label=f'Correct Mean: {np.mean(correct_confidences):.3f}')
    if len(incorrect_confidences) > 0:
        ax2.axvline(np.mean(incorrect_confidences), color='darkred', linestyle='--', linewidth=2, 
                    label=f'Incorrect Mean: {np.mean(incorrect_confidences):.3f}')
    ax2.set_xlabel('Prediction Confidence')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Confidence Distribution: Correct vs Incorrect Predictions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    confidence_plot_path = os.path.join(output_dir, "confidence_histogram.png")
    plt.savefig(confidence_plot_path, dpi=150)
    plt.close(fig)
    print(f"Confidence histogram saved to {confidence_plot_path}")

def plot_confidence_per_class(confidences, predictions, labels, output_dir, class_names=None):
    """
    Plots confidence distributions per class.
    
    Args:
        confidences: Array of prediction confidences.
        predictions: Array of predicted class indices.
        labels: Array of true class indices.
        output_dir: Directory to save the plot.
        class_names: Optional list of class names for labeling.
    """
    unique_classes = np.unique(labels)
    if class_names is None:
        class_names = [f"Class {i}" for i in unique_classes]
    
    fig, axes = plt.subplots(1, len(unique_classes), figsize=(5 * len(unique_classes), 6))
    if len(unique_classes) == 1:
        axes = [axes]
    
    for i, class_idx in enumerate(unique_classes):
        class_mask = predictions == class_idx
        class_confidences = confidences[class_mask]
        class_true_labels = labels[class_mask]
        
        correct_mask = class_true_labels == class_idx
        correct_confidences = class_confidences[correct_mask]
        incorrect_confidences = class_confidences[~correct_mask]
        
        axes[i].hist(correct_confidences, bins=20, alpha=0.7, color='green', 
                     label=f'Correct ({len(correct_confidences)})', edgecolor='black')
        if len(incorrect_confidences) > 0:
            axes[i].hist(incorrect_confidences, bins=20, alpha=0.7, color='red', 
                         label=f'Incorrect ({len(incorrect_confidences)})', edgecolor='black')
        
        axes[i].set_xlabel('Prediction Confidence')
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f'Predictions as {class_names[i]}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    per_class_plot_path = os.path.join(output_dir, "confidence_per_class.png")
    plt.savefig(per_class_plot_path, dpi=150)
    plt.close(fig)
    print(f"Per-class confidence plot saved to {per_class_plot_path}")

def plot_calibration_curve(confidences, predictions, labels, output_dir, n_bins=10):
    """
    Plots a calibration curve (reliability diagram).
    
    Args:
        confidences: Array of prediction confidences.
        predictions: Array of predicted class indices.
        labels: Array of true class indices.
        output_dir: Directory to save the plot.
        n_bins: Number of bins for calibration.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            count_in_bin = in_bin.sum()
            
            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(avg_confidence_in_bin)
            bin_counts.append(count_in_bin)
        else:
            bin_accuracies.append(0)
            bin_confidences.append(0)
            bin_counts.append(0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax1.plot(bin_confidences, bin_accuracies, 'o-', color='red', linewidth=2, markersize=8, label='Model')
    
    for i, (conf, acc, count) in enumerate(zip(bin_confidences, bin_accuracies, bin_counts)):
        if count > 0:
            ax1.text(conf, acc, f'{count}', ha='center', va='bottom', fontsize=8)
    
    ax1.set_xlabel('Mean Predicted Confidence')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Calibration Plot (Reliability Diagram)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    ax2.hist(confidences, bins=bin_boundaries, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Prediction Confidence')
    ax2.set_ylabel('Count')
    ax2.set_title('Confidence Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    calibration_plot_path = os.path.join(output_dir, "calibration_curve.png")
    plt.savefig(calibration_plot_path, dpi=150)
    plt.close(fig)
    print(f"Calibration curve saved to {calibration_plot_path}")
    
    ece = 0
    total_samples = len(confidences)
    for i, (acc, conf, count) in enumerate(zip(bin_accuracies, bin_confidences, bin_counts)):
        if count > 0:
            ece += (count / total_samples) * abs(acc - conf)
    
    return ece 