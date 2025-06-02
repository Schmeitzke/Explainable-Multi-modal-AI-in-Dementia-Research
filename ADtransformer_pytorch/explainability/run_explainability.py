import numpy as np
import torch
import torch.nn as nn
import os
import sys
import logging
import yaml
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model_components.build_model import build_adni_transformer
from prepare_adni_dataset import prepare_adni_dataset 
from explainability.saliency_maps import SaliencyMapGenerator
from explainability.confidence_diagram import (
    get_confidences_predictions_labels, 
    plot_confusion_matrix_and_stats,
    plot_confidence_histogram,
    plot_confidence_per_class,
    plot_calibration_curve
)
from explainability.grad_cam import GradCAMGenerator
from explainability.grad_cam_plus_plus import GradCAMPlusPlusGenerator
from explainability.gmar import GMARGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_dummy_dataloader(config, test_csv_path, batch_size=1, num_samples=20):
    """
    Creates a dummy dataloader for testing explainability methods.
    This is a simplified version - you may need to adapt based on your actual data loading.
    """
    df = pd.read_csv(test_csv_path)
    print(f"CSV shape: {df.shape}")
    print(f"CSV columns: {df.columns.tolist()}")
    
    if 'DX' in df.columns:
        label_column = 'DX'
    elif 'label' in df.columns:
        label_column = 'label'
    else:
        label_column = df.columns[-1]
    
    clinical_feature_columns = df.columns[2:2+config['non_image']['num_features']].tolist()
    clinical_data = df[clinical_feature_columns].values[:num_samples]
    
    labels = df[label_column].values[:num_samples]
    if isinstance(labels[0], str):
        unique_labels = sorted(df[label_column].unique())
        label_map = {label: i for i, label in enumerate(unique_labels)}
        labels = [label_map[label] for label in labels]
    
    if config['image']['use_images']:
        image_shape = tuple(config['image']['shape'])
        in_channels = config['image']['in_channels']
        image_data = torch.randn(num_samples, in_channels, *image_shape)
    else:
        image_data = None
    
    clinical_data = torch.FloatTensor(clinical_data)
    labels = torch.LongTensor(labels)
    
    class SimpleDataset:
        def __init__(self, clinical_data, image_data, labels, clinical_feature_columns):
            self.clinical_data = clinical_data
            self.image_data = image_data
            self.labels = labels
            self.feature_columns = clinical_feature_columns
        
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            clinical = self.clinical_data[idx]
            image = self.image_data[idx] if self.image_data is not None else None
            label = self.labels[idx]
            return clinical, image, label
    
    dataset = SimpleDataset(clinical_data, image_data, labels, clinical_feature_columns)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return dataloader, clinical_feature_columns

def run_explainability_eval(model_weights_path, test_csv_path, config_path, output_dir, num_explain_samples=20):
    """
    Run explainability evaluation for ADTransformer model.
    
    Args:
        model_weights_path: Path to trained model weights
        test_csv_path: Path to test CSV file
        config_path: Path to model configuration file
        output_dir: Directory to save explainability results
        num_explain_samples: Number of samples to process for explanations
    """
    logging.info("Starting ADTransformer Model Explainability Evaluation...")
    logging.info(f"Model Weights: {model_weights_path}")
    logging.info(f"Test CSV: {test_csv_path}")
    logging.info(f"Config: {config_path}")
    logging.info(f"Output Directory: {output_dir}")
    logging.info(f"Number of samples for explainability: {num_explain_samples}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    with open(config_path, 'r') as f:
        initial_config = yaml.safe_load(f)
    
    if initial_config['classifier']['output_dim'] == 3:
        initial_config['data']['class_names'] = ['CN', 'MCI', 'AD']
        logging.info("Updated class_names for 3-class classification: ['CN', 'MCI', 'AD']")
    
    train_csv_path_from_config = initial_config['data']['train_csv_path']

    os.makedirs(output_dir, exist_ok=True)
    
    logging.info("Preparing dataset for explainability using prepare_adni_dataset...")
    dataset_prepared = prepare_adni_dataset(config_path, train_csv_path_from_config, test_csv_path)

    if dataset_prepared is None:
        logging.error("Failed to prepare dataset for explainability. Aborting.")
        return

    test_loader = dataset_prepared['val_loader']
    clinical_feature_names = dataset_prepared['feature_columns']
    config = dataset_prepared['config']
    
    if config['classifier']['output_dim'] == 3:
        config['data']['class_names'] = ['CN', 'MCI', 'AD']
        logging.info("Ensured config class_names match 3-class classification")

    if test_loader is None:
        logging.error("Test loader (val_loader from prepare_adni_dataset) is None. Aborting explainability.")
        return
    logging.info(f"Successfully prepared dataset. Test loader has {len(test_loader.dataset)} samples.")

    logging.info("Checking label distribution in dataset...")
    all_labels = []
    for batch_idx, batch_data in enumerate(test_loader):
        labels_batch = batch_data['label']
        all_labels.extend(labels_batch.tolist())
        if batch_idx >= 10:
            break
    
    unique_labels = sorted(set(all_labels))
    logging.info(f"Unique labels found in dataset: {unique_labels}")
    logging.info(f"Expected class names: {config['data']['class_names']}")
    logging.info(f"Number of class names: {len(config['data']['class_names'])}")
    
    max_label = len(config['data']['class_names']) - 1
    invalid_labels = [label for label in unique_labels if label < 0 or label > max_label]
    if invalid_labels:
        logging.error(f"Found invalid labels {invalid_labels} outside range [0, {max_label}]")
        logging.error("Please check your data preprocessing and class mapping")
        return

    try:
        model = build_adni_transformer(config_path)
        model.load_state_dict(torch.load(model_weights_path, map_location=device, weights_only=True), strict=False)
        model.to(device)
        model.eval()
        logging.info(f"Successfully loaded model from {model_weights_path}")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return

    class_names = config['data']['class_names']
    
    saliency_output_dir = os.path.join(output_dir, "saliency_maps")
    os.makedirs(saliency_output_dir, exist_ok=True)
    saliency_generator = SaliencyMapGenerator(
        model, device, 
        feature_columns=clinical_feature_names, 
        class_names=class_names
    )

    grad_cam_generator = None
    if config['image']['use_images']:
        grad_cam_output_dir = os.path.join(output_dir, "grad_cam_maps")
        os.makedirs(grad_cam_output_dir, exist_ok=True)
        try:
            grad_cam_generator = GradCAMGenerator(
                model, device, 
                target_branch_idx=0,
                class_names=class_names
            )
            logging.info("GradCAMGenerator initialized")
        except Exception as e:
            logging.error(f"Could not init GradCAMGenerator: {e}")

    grad_cam_pp_generator = None
    if config['image']['use_images']:
        grad_cam_pp_output_dir = os.path.join(output_dir, "grad_cam_plus_plus_maps")
        os.makedirs(grad_cam_pp_output_dir, exist_ok=True)
        try:
            grad_cam_pp_generator = GradCAMPlusPlusGenerator(
                model, device,
                target_branch_idx=0,
                class_names=class_names
            )
            logging.info("GradCAMPlusPlusGenerator initialized")
        except Exception as e:
            logging.error(f"Could not init GradCAMPlusPlusGenerator: {e}")

    gmar_generator = None
    try:
        transformer_layers = None
        
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'layers'):
            transformer_layers = model.transformer.layers
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'encoder') and hasattr(model.transformer.encoder, 'layers'):
            transformer_layers = model.transformer.encoder.layers
        elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
            transformer_layers = model.encoder.layers
        elif hasattr(model, 'layers'):
            transformer_layers = model.layers
        
        if transformer_layers and len(transformer_layers) > 0:
            gmar_output_dir = os.path.join(output_dir, "gmar_maps")
            os.makedirs(gmar_output_dir, exist_ok=True)
            gmar_generator = GMARGenerator(
                model, device, 
                transformer_layers, 
                class_names=class_names
            )
            logging.info(f"GMARGenerator initialized for {len(transformer_layers)} transformer layers")
        else:
            logging.warning("Could not initialize GMARGenerator: Transformer layers not found or empty")
            logging.warning("Tried paths: model.transformer.layers, model.transformer.encoder.layers, model.encoder.layers, model.layers")
    except Exception as e:
        logging.error(f"Could not init GMARGenerator: {e}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")

    logging.info(f"\nStarting generation of explainability maps for up to {num_explain_samples} samples...")
    
    processed_samples_count = 0

    with tqdm(total=num_explain_samples, desc="Processing Samples for XAI") as pbar:
        for batch_data in test_loader:
            if processed_samples_count >= num_explain_samples:
                break
            
            non_image_data_batch = batch_data['non_image'].to(device)
            labels_batch = batch_data['label'].to(device)
            image_data_batch = batch_data.get('image')
            if image_data_batch is not None:
                image_data_batch = image_data_batch.to(device)

            samples_remaining_to_explain = num_explain_samples - processed_samples_count
            samples_to_process_this_batch = min(labels_batch.size(0), samples_remaining_to_explain)
            
            if samples_to_process_this_batch <= 0:
                break

            s_processed = saliency_generator.process_batch(
                non_image_data_batch[:samples_to_process_this_batch], 
                image_data_batch[:samples_to_process_this_batch] if image_data_batch is not None else None, 
                labels_batch[:samples_to_process_this_batch], 
                saliency_output_dir, processed_samples_count, samples_to_process_this_batch
            )

            if grad_cam_generator and image_data_batch is not None:
                gc_processed = grad_cam_generator.process_batch(
                    non_image_data_batch[:samples_to_process_this_batch], 
                    image_data_batch[:samples_to_process_this_batch], 
                    labels_batch[:samples_to_process_this_batch],
                    grad_cam_output_dir, processed_samples_count, samples_to_process_this_batch
                )

            if grad_cam_pp_generator and image_data_batch is not None:
                gcpp_processed = grad_cam_pp_generator.process_batch(
                    non_image_data_batch[:samples_to_process_this_batch], 
                    image_data_batch[:samples_to_process_this_batch], 
                    labels_batch[:samples_to_process_this_batch],
                    grad_cam_pp_output_dir, processed_samples_count, samples_to_process_this_batch
                )

            if gmar_generator:
                gmar_processed = gmar_generator.process_batch(
                    non_image_data_batch[:samples_to_process_this_batch], 
                    image_data_batch[:samples_to_process_this_batch] if image_data_batch is not None else None, 
                    labels_batch[:samples_to_process_this_batch],
                    gmar_output_dir, processed_samples_count, samples_to_process_this_batch
                )
            
            pbar.update(samples_to_process_this_batch)
            processed_samples_count += samples_to_process_this_batch

    logging.info(f"\nFinished generating individual explainability maps. Total samples processed for explanation: {processed_samples_count}.")

    saliency_stats = saliency_generator.get_final_stats(saliency_output_dir)
    saliency_stats_path = os.path.join(saliency_output_dir, "saliency_stats.txt")
    with open(saliency_stats_path, "w") as f:
        f.write("Saliency Maps Statistics:\n")
        f.write(f"Samples analyzed: {saliency_stats.get('total_samples', 0)}\n")
        f.write(f"Correct predictions: {saliency_stats.get('correct_predictions', 0)}\n")
        f.write(f"Accuracy: {saliency_stats.get('accuracy', 0):.4f}\n\n")
        if 'feature_importance' in saliency_stats and saliency_stats['feature_importance']:
            f.write("Top Non-Image Features (Average Absolute Gradient Magnitude):\n")
            for feature, importance in list(saliency_stats['feature_importance'].items())[:20]:
                f.write(f"{feature}: {importance:.6f}\n")
        f.write("\nModality Importance:\n")
        f.write(f"  Non-Image: {saliency_stats['modality_importance']['non_image']['percentage']:.2f}%\n")
        f.write(f"  Image: {saliency_stats['modality_importance']['image']['percentage']:.2f}%\n")
    logging.info(f"Saliency maps statistics saved to {saliency_stats_path}")

    if grad_cam_generator:
        grad_cam_stats = grad_cam_generator.get_final_stats()
        grad_cam_stats_path = os.path.join(grad_cam_output_dir, "grad_cam_stats.txt")
        with open(grad_cam_stats_path, "w") as f:
            f.write("Grad-CAM Statistics:\n")
            f.write(f"Samples processed: {grad_cam_stats.get('total_samples', 0)}\n")
            f.write(f"Samples failed: {grad_cam_stats.get('failed_samples', 0)}\n")
            f.write(f"Accuracy: {grad_cam_stats.get('accuracy', 0):.2f}%\n")
        logging.info(f"Grad-CAM statistics saved to {grad_cam_stats_path}")
        grad_cam_generator.cleanup()

    if grad_cam_pp_generator:
        grad_cam_pp_stats = grad_cam_pp_generator.get_final_stats()
        grad_cam_pp_stats_path = os.path.join(grad_cam_pp_output_dir, "grad_cam_plus_plus_stats.txt")
        with open(grad_cam_pp_stats_path, "w") as f:
            f.write("Grad-CAM++ Statistics:\n")
            f.write(f"Samples processed: {grad_cam_pp_stats.get('total_samples', 0)}\n")
            f.write(f"Samples failed: {grad_cam_pp_stats.get('failed_samples', 0)}\n")
            f.write(f"Accuracy: {grad_cam_pp_stats.get('accuracy', 0):.2f}%\n")
        logging.info(f"Grad-CAM++ statistics saved to {grad_cam_pp_stats_path}")
        grad_cam_pp_generator.cleanup()

    if gmar_generator:
        gmar_stats = gmar_generator.get_final_stats()
        gmar_stats_path = os.path.join(gmar_output_dir, "gmar_stats.txt")
        with open(gmar_stats_path, "w") as f:
            f.write("GMAR Statistics:\n")
            f.write(f"Samples processed: {gmar_stats.get('heatmap_generated', 0)}\n")
            f.write(f"Samples failed: {gmar_stats.get('failed_samples', 0)}\n")
            f.write(f"Accuracy: {gmar_stats.get('accuracy', 0):.2f}%\n")
        logging.info(f"GMAR statistics saved to {gmar_stats_path}")
        gmar_generator.cleanup()

    saliency_generator.cleanup()

    logging.info("\nGenerating confusion matrix and classification report for the full test set...")
    eval_metrics_output_dir = os.path.join(output_dir, "evaluation_metrics")
    os.makedirs(eval_metrics_output_dir, exist_ok=True)
    
    confidences, predictions, labels = get_confidences_predictions_labels(model, test_loader, device)
    
    plot_confusion_matrix_and_stats(predictions, labels, 
                                    output_dir=eval_metrics_output_dir, 
                                    class_names=class_names)
    
    plot_confidence_histogram(confidences, predictions, labels, 
                             output_dir=eval_metrics_output_dir, 
                             class_names=class_names)
    
    plot_confidence_per_class(confidences, predictions, labels,
                             output_dir=eval_metrics_output_dir,
                             class_names=class_names)
    
    ece = plot_calibration_curve(confidences, predictions, labels,
                                output_dir=eval_metrics_output_dir)
    
    calibration_stats_path = os.path.join(eval_metrics_output_dir, "calibration_stats.txt")
    with open(calibration_stats_path, "w") as f:
        f.write("Calibration Statistics:\n")
        f.write(f"Expected Calibration Error (ECE): {ece:.4f}\n")
        f.write(f"Mean Confidence: {np.mean(confidences):.4f}\n")
        f.write(f"Mean Accuracy: {np.mean(predictions == labels):.4f}\n")
    
    logging.info("Explainability evaluation finished.")

if __name__ == "__main__":
    default_output_dir_from_training = os.path.join(os.path.dirname(__file__), "..", "output")
    model_weights_path = os.path.join(default_output_dir_from_training, "3class_results/best_model.pth") 
    
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    
    with open(config_path, 'r') as f:
        loaded_config_for_paths = yaml.safe_load(f)
    test_csv_path = loaded_config_for_paths['data']['test_csv_path'] 
    
    output_dim = loaded_config_for_paths['classifier']['output_dim']
    if output_dim == 3:
        explainability_output_dir = os.path.join(os.path.dirname(__file__), "..", "explainability_results", "3class_model")
        logging.info("Using 3-class model explainability output directory")
    else:
        explainability_output_dir = os.path.join(os.path.dirname(__file__), "..", "explainability_results", f"{output_dim}class_model")
        logging.info(f"Using {output_dim}-class model explainability output directory")
    
    num_samples_for_maps = 20

    if not os.path.exists(model_weights_path):
        logging.error(f"Model weights not found at {model_weights_path}")
        logging.error("Please update the model_weights_path variable in run_explainability.py or ensure 'output/3class_results/best_model.pth' exists from a training run.")
        sys.exit(1)
    
    if not os.path.exists(test_csv_path):
        logging.error(f"Test CSV not found at {test_csv_path} (from config: {config_path})")
        sys.exit(1)

    run_explainability_eval(
        model_weights_path=model_weights_path,
        test_csv_path=test_csv_path,
        config_path=config_path,
        output_dir=explainability_output_dir,
        num_explain_samples=num_samples_for_maps
    ) 