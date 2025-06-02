import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
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
    Helper class to compute Grad-CAM++.
    Manages hooks to capture activations and gradients from a target layer.
    Uses the common formulation for Grad-CAM++ weights based on powers of first-order gradients.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module, model_modality_mode: str = "full"):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.hook_handles = []
        self.model_modality_mode = model_modality_mode
        self._register_hooks()

    def _register_hooks(self):
        """Registers forward hook to capture activations and enable gradient retention."""
        def _forward_hook_fn(module, input, output):
            self.activations = output
            if output.requires_grad:
                output.retain_grad()

        if not isinstance(self.target_layer, nn.Module):
            logger.error(f"Target layer is not an nn.Module: {type(self.target_layer)}")
            raise ValueError("Target layer for GradCAM++ must be an nn.Module.")

        handle_fwd = self.target_layer.register_forward_hook(_forward_hook_fn)
        self.hook_handles.append(handle_fwd)
        logger.debug(f"Forward hook registered for {self.target_layer} in GradCAMPlusPlus")

    def _get_cam_plus_plus(self, mri_data_single: torch.Tensor, clinical_data_single: torch.Tensor, target_class_idx: int = None) -> tuple[np.ndarray | None, int, float]:
        """
        Computes the Grad-CAM++ heatmap for a single sample.
        Assumes model is in eval mode and on the correct device.
        """
        self.model.zero_grad()
        self.activations = None

        current_clinical_input = clinical_data_single if self.model_modality_mode in ["full", "clinical_only"] else None
        current_mri_input = mri_data_single if self.model_modality_mode in ["full", "image_only"] else None

        main_logits_output, image_only_logits_output, clinical_only_logits_output, _ = self.model(current_clinical_input, current_mri_input)

        target_logits_for_explanation = None
        if self.model_modality_mode == "image_only":
            if image_only_logits_output is None: raise ValueError("GradCAM++: Image-only mode, but image_only_logits are None.")
            target_logits_for_explanation = image_only_logits_output
        elif self.model_modality_mode == "clinical_only":
            if clinical_only_logits_output is None: raise ValueError("GradCAM++: Clinical-only mode, but clinical_only_logits are None.")
            target_logits_for_explanation = clinical_only_logits_output
        else:
            if main_logits_output is None: raise ValueError("GradCAM++: Full mode, but main_logits are None.")
            target_logits_for_explanation = main_logits_output

        if self.activations is None:
            logger.error("Activations not captured by forward hook. Check target layer and model graph.")
            pred_class = torch.argmax(target_logits_for_explanation, dim=1).item() if target_logits_for_explanation is not None else -1
            confidence = torch.max(torch.softmax(target_logits_for_explanation, dim=1)).item() if target_logits_for_explanation is not None else 0.0
            return None, pred_class, confidence

        if target_class_idx is None:
            target_class_idx = torch.argmax(target_logits_for_explanation, dim=1).item()

        score = target_logits_for_explanation[0, target_class_idx]
        score.backward()

        if self.activations.grad is None:
            logger.error("Gradients for activations (dScore/dActivations) not computed. Check retain_grad() and backward pass integrity.")
            confidence = torch.max(torch.softmax(target_logits_for_explanation, dim=1)).item()
            return None, target_class_idx, confidence

        grads = self.activations.grad
        activations = self.activations.detach()

        grads_power_2 = grads**2
        grads_power_3 = grads_power_2 * grads

        sum_A_k = torch.sum(activations, dim=(2, 3), keepdim=True)

        alpha_numerator = grads_power_2
        alpha_denominator = 2 * grads_power_2 + sum_A_k * grads_power_3 + 1e-7
        alpha = alpha_numerator / alpha_denominator

        relu_grads = F.relu(grads)
        weights_k = torch.sum(alpha * relu_grads, dim=(2, 3))

        sum_relu_grads_per_channel = torch.sum(relu_grads, dim=(2,3), keepdim=False)
        epsilon = 1e-7
        valid_channels = sum_relu_grads_per_channel > epsilon
        weights_k = torch.where(
            valid_channels,
            weights_k / (sum_relu_grads_per_channel + epsilon),
            weights_k
        )
        weights_k = weights_k.squeeze(0)

        cam = torch.sum(weights_k.unsqueeze(-1).unsqueeze(-1) * activations.squeeze(0), dim=0)
        cam = F.relu(cam)

        if cam.numel() > 0 and cam.max() > 1e-6:
            cam = cam / cam.max()
        else:
            cam = torch.zeros_like(cam)

        confidence = torch.max(torch.softmax(target_logits_for_explanation, dim=1)).item()
        return cam.cpu().numpy(), target_class_idx, confidence

    def remove_hooks(self):
        """Removes all registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        logger.debug("GradCAM++ hooks removed.")


class GradCAMPlusPlusGenerator:
    """
    Generates Grad-CAM++ visualizations for the MMDF model, focusing on the MRI modality.
    """
    def __init__(self, model: nn.Module, device: torch.device, target_layer_module: nn.Module, class_names: list[str] | None = None, model_modality_mode: str = "full"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.target_layer_module = target_layer_module
        self.class_names = class_names if class_names else ['Class 0', 'Class 1', 'Class 2']
        self.model_modality_mode = model_modality_mode

        self.grad_cam_plus_plus_instance = GradCAMPlusPlus(self.model, self.target_layer_module, self.model_modality_mode)

        self.stats = {
            'total_samples': 0,
            'correct_predictions': 0,
            'failed_samples': 0
        }

    def generate_heatmap_for_sample(self, clinical_data: torch.Tensor, mri_data: torch.Tensor, target_class: int = None) -> tuple[np.ndarray | None, int, float]:
        """
        Generates a single Grad-CAM++ heatmap for a given sample.
        Input tensors should be for a single sample (batch size 1) and on the correct device.
        """
        clinical_data_single = clinical_data.to(self.device)
        mri_data_single = mri_data.to(self.device)
        self.model.to(self.device)

        heatmap_np, predicted_class_idx, confidence = self.grad_cam_plus_plus_instance._get_cam_plus_plus(
            mri_data_single, clinical_data_single, target_class
        )
        return heatmap_np, predicted_class_idx, confidence

    def _plot_grad_cam_plus_plus_visualization(self, original_mri_slices: np.ndarray, heatmap_np: np.ndarray,
                                     output_dir: str, sample_idx: int,
                                     pred_class_idx: int, true_label_idx: int, confidence: float):
        """
        Plots and saves the Grad-CAM++ visualization for a single sample.
        """
        num_slices = original_mri_slices.shape[0]
        if num_slices != 3:
            logger.warning(f"Expected 3 MRI slices for plotting, got {num_slices}. Skipping Grad-CAM++ plot for sample {sample_idx}.")
            return

        if heatmap_np is None or heatmap_np.size == 0:
            logger.warning(f"Invalid or empty Grad-CAM++ heatmap for sample {sample_idx}. Skipping plot.")
            return

        fig, axes = plt.subplots(num_slices, 3, figsize=(12, 3 * num_slices + 1))
        if num_slices == 1: axes = np.array([axes])

        slice_titles = ['Slice 1 (e.g., Sagittal)', 'Slice 2 (e.g., Coronal)', 'Slice 3 (e.g., Axial)']
        pred_name = self.class_names[pred_class_idx] if pred_class_idx < len(self.class_names) else f"Cls {pred_class_idx}"
        true_name = self.class_names[true_label_idx] if true_label_idx < len(self.class_names) else f"Cls {true_label_idx}"
        fig.suptitle(f"Grad-CAM++ Sample {sample_idx} - Pred: {pred_name} ({confidence:.2f}), True: {true_name}", fontsize=14)

        for i in range(num_slices):
            mri_slice = original_mri_slices[i]
            resized_heatmap = resize(heatmap_np, (mri_slice.shape[0], mri_slice.shape[1]),
                                     anti_aliasing=True, preserve_range=True, mode='reflect')
            if resized_heatmap.max() > 1e-6: resized_heatmap /= resized_heatmap.max()
            else: resized_heatmap = np.zeros_like(resized_heatmap)

            axes[i, 0].imshow(mri_slice, cmap='gray')
            axes[i, 0].set_title(f"{slice_titles[i]}")
            axes[i, 0].axis('off')

            axes[i, 1].imshow(resized_heatmap, cmap='jet', vmin=0, vmax=1)
            axes[i, 1].set_title("Grad-CAM++")
            axes[i, 1].axis('off')

            axes[i, 2].imshow(mri_slice, cmap='gray')
            axes[i, 2].imshow(resized_heatmap, cmap='jet', alpha=0.5, vmin=0, vmax=1)
            axes[i, 2].set_title("Overlay")
            axes[i, 2].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plot_path = os.path.join(output_dir, f"grad_cam_plus_plus_sample_{sample_idx}.png")
        try:
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        except Exception as e:
            logger.error(f"Failed to save Grad-CAM++ plot for sample {sample_idx} to {plot_path}: {e}")
        plt.close(fig)

    def process_batch(self, clinical_data_batch: torch.Tensor, mri_data_batch: torch.Tensor, labels_batch: torch.Tensor, 
                        output_dir: str, global_sample_offset: int, samples_to_process_in_this_batch: int) -> int:
        """
        Processes a single batch of data for Grad-CAM++, generates visualizations, and updates stats.
        Args:
            clinical_data_batch: Batch of clinical data.
            mri_data_batch: Batch of MRI data.
            labels_batch: Batch of labels.
            output_dir: Directory to save visualizations.
            global_sample_offset: The starting global index for samples in this batch.
            samples_to_process_in_this_batch: Max number of samples from this batch to process.
        Returns:
            Number of samples actually processed from this batch.
        """
        if not self.grad_cam_plus_plus_instance.hook_handles:
            logger.debug("Re-registering GradCAM++ hooks before processing batch.")
            self.grad_cam_plus_plus_instance._register_hooks()

        batch_size = clinical_data_batch.size(0)
        num_processed_this_call = 0

        for i in range(min(batch_size, samples_to_process_in_this_batch)):
            current_global_sample_idx = global_sample_offset + i

            current_clinical_sample = clinical_data_batch[i:i+1]
            current_mri_sample = mri_data_batch[i:i+1]
            true_label = labels_batch[i].item()

            try:
                heatmap_np, pred_class_idx, confidence = self.generate_heatmap_for_sample(
                    current_clinical_sample, current_mri_sample
                )
                if heatmap_np is not None:
                    self.stats['total_samples'] += 1
                    if pred_class_idx == true_label: self.stats['correct_predictions'] += 1
                    original_mri_for_plot = current_mri_sample.squeeze(0).cpu().numpy()
                    self._plot_grad_cam_plus_plus_visualization(original_mri_for_plot, heatmap_np,
                                                              output_dir, current_global_sample_idx,
                                                              pred_class_idx, true_label, confidence)
                else:
                    logger.warning(f"Grad-CAM++ heatmap generation failed for global_sample_idx {current_global_sample_idx}.")
                    self.stats['failed_samples'] +=1
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
