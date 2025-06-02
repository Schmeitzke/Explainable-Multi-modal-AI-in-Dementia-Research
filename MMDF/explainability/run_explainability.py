import torch
import os
import logging
import pandas as pd
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mmdf_model import MMDF
import config as mmdf_config
from data_loader import get_dataloaders
from explainability.saliency_maps import SaliencyMapGenerator
from explainability.confidence_diagram import get_confidences_predictions_labels, plot_confusion_matrix_and_stats
from explainability.grad_cam import GradCAMGenerator
from explainability.grad_cam_plus_plus import GradCAMPlusPlusGenerator
from explainability.gmar import GMARGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_explainability_eval(model_weights_path, test_csv_path, image_base_dir, base_output_dir, num_explain_samples, model_modality_mode: str):
    specific_output_dir = os.path.join(base_output_dir, f"mmdf_explainability_output_{model_modality_mode}")
    os.makedirs(specific_output_dir, exist_ok=True)

    logging.info("Starting MMDF Model Explainability Evaluation...")
    logging.info(f"Model Weights: {model_weights_path}")
    logging.info(f"Model Modality Mode: {model_modality_mode}")
    logging.info(f"Test CSV: {test_csv_path}")
    logging.info(f"Image Base Directory: {image_base_dir}")
    logging.info(f"Output Directory: {specific_output_dir}")
    logging.info(f"Number of samples for explainability methods: {num_explain_samples}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    try:
        test_loader, _ = get_dataloaders(
            train_csv_path=test_csv_path,
            test_csv_path=test_csv_path, 
            image_base_dir=image_base_dir,
            image_modalities=mmdf_config.image_modalities,
            batch_size=mmdf_config.batch_size,
            num_clinical_features=mmdf_config.num_clinical_features,
            mri_image_size=mmdf_config.mri_image_size,
            num_classes=mmdf_config.num_classes,
        )
    except ValueError as e:
        logging.error(f"Error initializing test data loader: {e}")
        return
    except Exception as e:
        logging.error(f"An unexpected error occurred during test data loading: {e}")
        return

    if len(test_loader.dataset) == 0:
        logging.error("Test dataset is empty. Please check CSV path and image data. Aborting.")
        return
    logging.info(f"Total test samples available: {len(test_loader.dataset)}, processing up to {num_explain_samples} for explanations.")

    try:
        test_df_header = pd.read_csv(test_csv_path, nrows=0)
        clinical_feature_names = test_df_header.columns[2 : 2 + mmdf_config.num_clinical_features].tolist()
        if len(clinical_feature_names) != mmdf_config.num_clinical_features:
            logging.warning(f"Expected {mmdf_config.num_clinical_features} clinical features, found {len(clinical_feature_names)} from CSV. Using generic names.")
            clinical_feature_names = [f"ClinicalFeature_{j+1}" for j in range(mmdf_config.num_clinical_features)]
    except Exception as e:
        logging.error(f"Error reading clinical feature names from {test_csv_path}: {e}. Using generic names.")
        clinical_feature_names = [f"ClinicalFeature_{j+1}" for j in range(mmdf_config.num_clinical_features)]

    model = MMDF().to(device)
    try:
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
        logging.info(f"Successfully loaded model weights from {model_weights_path}")
    except FileNotFoundError:
        logging.error(f"Error: Model weights file not found at {model_weights_path}. Aborting.")
        return
    except Exception as e:
        logging.error(f"Error loading model state_dict: {e}. Aborting.")
        return
    model.eval()

    class_names = mmdf_config.class_names.copy()
    logging.info(f"Using class names from config: {class_names} (classification mode: {'binary' if mmdf_config.perform_binary_classification_cn_ad else 'multiclass'})")
    
    if len(class_names) != mmdf_config.num_classes:
        logging.warning(f"Mismatch between class_names length ({len(class_names)}) and num_classes ({mmdf_config.num_classes}). Using generic names as fallback.")
        class_names = [f"Class {i}" for i in range(mmdf_config.num_classes)]
    
    saliency_output_dir = os.path.join(specific_output_dir, "saliency_maps")
    os.makedirs(saliency_output_dir, exist_ok=True)
    saliency_generator = SaliencyMapGenerator(model, device, feature_columns=clinical_feature_names, class_names=class_names, model_modality_mode=model_modality_mode)

    grad_cam_output_dir = os.path.join(specific_output_dir, "grad_cam_maps")
    os.makedirs(grad_cam_output_dir, exist_ok=True)
    grad_cam_generator = None
    try:
        target_layer_for_grad_cam = model.image_net.patch_embed.proj
        grad_cam_generator = GradCAMGenerator(model, device, target_layer_for_grad_cam, class_names=class_names, model_modality_mode=model_modality_mode)
        logging.info(f"GradCAMGenerator initialized. Target layer: {target_layer_for_grad_cam}")
    except AttributeError as e:
        logging.error(f"Could not init GradCAMGenerator due to missing target layer: {e}. Grad-CAM will be skipped.")

    grad_cam_pp_output_dir = os.path.join(specific_output_dir, "grad_cam_plus_plus_maps")
    os.makedirs(grad_cam_pp_output_dir, exist_ok=True)
    grad_cam_pp_generator = None
    try:
        target_layer_for_grad_cam_pp = model.image_net.patch_embed.proj
        grad_cam_pp_generator = GradCAMPlusPlusGenerator(model, device, target_layer_for_grad_cam_pp, class_names=class_names, model_modality_mode=model_modality_mode)
        logging.info(f"GradCAMPlusPlusGenerator initialized. Target layer: {target_layer_for_grad_cam_pp}")
    except AttributeError as e:
        logging.error(f"Could not init GradCAMPlusPlusGenerator due to missing target layer: {e}. Grad-CAM++ will be skipped.")

    gmar_output_dir = os.path.join(specific_output_dir, "gmar_maps")
    os.makedirs(gmar_output_dir, exist_ok=True)
    gmar_generator = None
    try:
        vit_encoder_layers = model.image_net.encoders
        if vit_encoder_layers and len(vit_encoder_layers) > 0:
            gmar_generator = GMARGenerator(model, device, vit_encoder_layers, class_names=class_names, model_modality_mode=model_modality_mode)
            logging.info(f"GMARGenerator initialized for {len(vit_encoder_layers)} ViT encoder layers.")
        else:
            logging.warning("Could not initialize GMARGenerator: ViT encoder layers not found or empty. GMAR will be skipped.")
    except Exception as e:
        logging.error(f"Could not init GMARGenerator: {e}. GMAR will be skipped.", exc_info=True)

    logging.info(f"\nStarting generation of explainability maps for up to {num_explain_samples} samples...")
    
    processed_samples_count = 0
    
    if num_explain_samples is not None:
        total_batches_to_iterate = (num_explain_samples + test_loader.batch_size - 1) // test_loader.batch_size

    with tqdm(total=num_explain_samples if num_explain_samples is not None else len(test_loader.dataset), desc="Processing Samples for XAI") as pbar:
        for batch_idx, (clinical_data_batch, mri_data_batch, _, labels_batch) in enumerate(test_loader):
            if num_explain_samples is not None and processed_samples_count >= num_explain_samples:
                break

            samples_remaining_to_target = float('inf')
            if num_explain_samples is not None:
                samples_remaining_to_target = num_explain_samples - processed_samples_count
            
            samples_to_process_this_batch = min(labels_batch.size(0), samples_remaining_to_target)
            if samples_to_process_this_batch <=0:
                 break

            s_processed = saliency_generator.process_batch(
                clinical_data_batch, mri_data_batch, labels_batch, 
                saliency_output_dir, processed_samples_count, samples_to_process_this_batch
            )

            if grad_cam_generator:
                gc_processed = grad_cam_generator.process_batch(
                    clinical_data_batch, mri_data_batch, labels_batch,
                    grad_cam_output_dir, processed_samples_count, samples_to_process_this_batch
                )

            if grad_cam_pp_generator:
                gcpp_processed = grad_cam_pp_generator.process_batch(
                    clinical_data_batch, mri_data_batch, labels_batch,
                    grad_cam_pp_output_dir, processed_samples_count, samples_to_process_this_batch
                )

            if gmar_generator:
                gmar_processed = gmar_generator.process_batch(
                    clinical_data_batch, mri_data_batch, labels_batch,
                    gmar_output_dir, processed_samples_count, samples_to_process_this_batch
                )
            
            pbar.update(samples_to_process_this_batch)
            processed_samples_count += samples_to_process_this_batch

    logging.info(f"\nFinished generating individual explainability maps. Total samples aimed for processing: {processed_samples_count}.")

    saliency_stats = saliency_generator.get_final_stats(saliency_output_dir)
    saliency_stats_path = os.path.join(saliency_output_dir, "saliency_stats.txt")
    with open(saliency_stats_path, "w") as f:
        f.write("Saliency Maps Statistics:\n")
        f.write(f"Samples analyzed: {saliency_stats.get('total_samples', 0)}\n")
        f.write(f"Correct predictions on analyzed samples: {saliency_stats.get('correct_predictions', 0)}\n")
        f.write(f"Accuracy on analyzed samples: {saliency_stats.get('accuracy', 0):.4f}\n\n")
        if 'feature_importance' in saliency_stats and saliency_stats['feature_importance']:
            f.write("Top Clinical Features (Average Absolute Gradient Magnitude):\n")
            for feature, importance in list(saliency_stats['feature_importance'].items())[:20]:
                f.write(f"{feature}: {importance:.6f}\n")
        else:
            f.write("Clinical feature importance data not available or not generated.\n")
        f.write("\nModality Importance (Average Absolute Gradient Magnitude & Percentage):\n")
        f.write(f"  Clinical: Abs={saliency_stats['modality_importance']['clinical']['absolute']:.4f}, Pct={saliency_stats['modality_importance']['clinical']['percentage']:.2f}%\n")
        f.write(f"  MRI: Abs={saliency_stats['modality_importance']['mri']['absolute']:.4f}, Pct={saliency_stats['modality_importance']['mri']['percentage']:.2f}%\n")
    logging.info(f"Saliency maps statistics saved to {saliency_stats_path}")

    if grad_cam_generator:
        grad_cam_stats = grad_cam_generator.get_final_stats()
        grad_cam_stats_path = os.path.join(grad_cam_output_dir, "grad_cam_stats.txt")
        with open(grad_cam_stats_path, "w") as f:
            f.write("Grad-CAM Statistics:\n")
            f.write(f"Samples processed: {grad_cam_stats.get('total_samples', 0)}\n")
            f.write(f"Samples failed: {grad_cam_stats.get('failed_samples', 0)}\n")
            f.write(f"Correct predictions on processed samples: {grad_cam_stats.get('correct_predictions', 0)}\n")
            if grad_cam_stats.get('total_samples', 0) > 0:
                f.write(f"Accuracy on processed samples: {grad_cam_stats.get('accuracy', 0):.2f}%\n")
            else:
                f.write("Accuracy on processed samples: N/A\n")
        logging.info(f"Grad-CAM statistics saved to {grad_cam_stats_path}")
        grad_cam_generator.cleanup()
    else:
        logging.warning("Grad-CAM was not run.")

    if grad_cam_pp_generator:
        grad_cam_pp_stats = grad_cam_pp_generator.get_final_stats()
        grad_cam_pp_stats_path = os.path.join(grad_cam_pp_output_dir, "grad_cam_plus_plus_stats.txt")
        with open(grad_cam_pp_stats_path, "w") as f:
            f.write("Grad-CAM++ Statistics:\n")
            f.write(f"Samples processed: {grad_cam_pp_stats.get('total_samples', 0)}\n")
            f.write(f"Samples failed: {grad_cam_pp_stats.get('failed_samples', 0)}\n")
            f.write(f"Correct predictions on processed samples: {grad_cam_pp_stats.get('correct_predictions', 0)}\n")
            if grad_cam_pp_stats.get('total_samples', 0) > 0:
                f.write(f"Accuracy on processed samples: {grad_cam_pp_stats.get('accuracy', 0):.2f}%\n")
            else:
                f.write("Accuracy on processed samples: N/A\n")
        logging.info(f"Grad-CAM++ statistics saved to {grad_cam_pp_stats_path}")
        grad_cam_pp_generator.cleanup()
    else:
        logging.warning("Grad-CAM++ was not run.")
    
    saliency_generator.cleanup()

    if gmar_generator:
        gmar_stats = gmar_generator.get_final_stats()
        gmar_stats_path = os.path.join(gmar_output_dir, "gmar_stats.txt")
        with open(gmar_stats_path, "w") as f:
            f.write("GMAR Statistics:\n")
            f.write(f"Samples processed: {gmar_stats.get('total_samples', 0)}\n")
            f.write(f"Samples failed: {gmar_stats.get('failed_samples', 0)}\n")
            f.write(f"Correct predictions on processed samples: {gmar_stats.get('correct_predictions', 0)}\n")
            if gmar_stats.get('total_samples', 0) > 0:
                f.write(f"Accuracy on processed samples: {gmar_stats.get('accuracy', 0):.2f}%\n")
            else:
                f.write("Accuracy on processed samples: N/A\n")
        logging.info(f"GMAR statistics saved to {gmar_stats_path}")
        gmar_generator.cleanup()
    else:
        logging.warning("GMAR was not run.")

    logging.info("\nGenerating confusion matrix and classification report...")
    eval_metrics_output_dir = os.path.join(specific_output_dir, "evaluation_metrics")
    os.makedirs(eval_metrics_output_dir, exist_ok=True)

    conf_test_loader, _ = get_dataloaders(
            train_csv_path=test_csv_path, test_csv_path=test_csv_path, image_base_dir=image_base_dir,
            image_modalities=mmdf_config.image_modalities, batch_size=mmdf_config.batch_size,
            num_clinical_features=mmdf_config.num_clinical_features, mri_image_size=mmdf_config.mri_image_size,
            num_classes=mmdf_config.num_classes
    )
    _, predictions, labels = get_confidences_predictions_labels(model, conf_test_loader, device, model_modality_mode=model_modality_mode)
    
    plot_confusion_matrix_and_stats(predictions, labels, 
                                    output_dir=eval_metrics_output_dir, 
                                    class_names=class_names)
    
    logging.info("Explainability evaluation finished.")

if __name__ == "__main__":
    model_weights = ""
    
    csv_test_path = mmdf_config.test_csv_path
    
    img_base_dir = mmdf_config.image_data_base_dir
    
    output_results_dir = ""
    num_explain = 20

    if "image" in model_weights.lower():
        mode = "image_only"
    else:
        mode = "full"

    run_explainability_eval(
        model_weights_path=model_weights,
        test_csv_path=csv_test_path,
        image_base_dir=img_base_dir,
        base_output_dir=output_results_dir,
        num_explain_samples=num_explain,
        model_modality_mode=mode
    ) 