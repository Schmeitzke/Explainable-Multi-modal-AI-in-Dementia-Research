import torch
import torch.nn as nn
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

class GMARInternal:
    """
    Internal helper class for Gradient-Weighted Attention Rollout.
    Uses forward hooks for attention maps & backward hooks for MHA grad_input.
    """
    def __init__(self, model, vit_encoder_layers, device, model_modality_mode: str = "full"):
        self.model = model
        self.vit_encoder_layers = vit_encoder_layers
        self.device = device
        self._attention_maps_for_pass = []
        self._mha_grad_inputs = []
        self._hook_handles = []
        self.model_modality_mode = model_modality_mode

    def _forward_hook_fn(self, module, input, output):
        """Forward hook to capture attention weights."""
        if isinstance(output, tuple) and len(output) == 2:
            attn_output_weights = output[1]
            if attn_output_weights is not None:
                self._attention_maps_for_pass.append(attn_output_weights.detach().clone())
            else:
                 logger.warning(f"Forward hook captured None attention weights from {module}")
        else:
            logger.warning(f"Forward hook expected tuple output from {module}, got {type(output)}")

    def _backward_hook_fn(self, module, grad_input, grad_output):
        """Backward hook function to capture MHA grad_input."""
        if grad_input is not None and len(grad_input) > 0 and grad_input[0] is not None:
            self._mha_grad_inputs.append(grad_input[0].detach().clone())
        else:
            logger.warning(f"Backward hook for {module} received None or invalid grad_input: {grad_input}")
            self._mha_grad_inputs.append(None)

    def _remove_hooks(self):
        """Removes all registered hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

    def _get_attention_maps_and_gradients(self, clinical_data_single: torch.Tensor, mri_data_single: torch.Tensor, target_class_idx: int = None):
        """
        Performs forward/backward pass using hooks to capture attention maps (forward)
        and MHA grad_input (backward).
        """
        self.model.eval()
        self._attention_maps_for_pass = []
        self._mha_grad_inputs = []
        self._hook_handles = []

        num_registered = 0
        for i, encoder_block in enumerate(self.vit_encoder_layers):
            if hasattr(encoder_block, 'attn') and isinstance(encoder_block.attn, nn.MultiheadAttention):
                fwd_handle = encoder_block.attn.register_forward_hook(self._forward_hook_fn)
                bwd_handle = encoder_block.attn.register_full_backward_hook(self._backward_hook_fn)
                self._hook_handles.extend([fwd_handle, bwd_handle])
                num_registered += 1
            else:
                logger.warning(f"GMAR-Adapt: Encoder block {i} has no MHA layer. Skipping hook registration.")
        
        if num_registered == 0:
             raise RuntimeError("GMAR-Adapt: Could not register hooks on any MHA layers.")

        self.model.zero_grad()
        
        current_clinical_input = clinical_data_single if self.model_modality_mode in ["full", "clinical_only"] else None
        current_mri_input = mri_data_single if self.model_modality_mode in ["full", "image_only"] else None
        
        main_logits_output, image_only_logits_output, clinical_only_logits_output, _ = self.model(current_clinical_input, current_mri_input)

        target_logits_for_explanation = None
        if self.model_modality_mode == "image_only":
            if image_only_logits_output is None: 
                self._remove_hooks()
                raise ValueError("GMAR: Image-only mode, but image_only_logits are None.")
            target_logits_for_explanation = image_only_logits_output
        elif self.model_modality_mode == "clinical_only":
            if clinical_only_logits_output is None: 
                self._remove_hooks()
                raise ValueError("GMAR: Clinical-only mode, but clinical_only_logits are None.")
            target_logits_for_explanation = clinical_only_logits_output
        else:
            if main_logits_output is None: 
                self._remove_hooks()
                raise ValueError("GMAR: Full mode, but main_logits are None.")
            target_logits_for_explanation = main_logits_output

        if len(self._attention_maps_for_pass) != num_registered:
            self._remove_hooks()
            raise RuntimeError(
                f"GMAR-Adapt: Expected {num_registered} attention map sets from forward hooks, "
                f"but captured {len(self._attention_maps_for_pass)}."
            )

        if target_class_idx is None:
            target_class_idx = torch.argmax(target_logits_for_explanation, dim=1).item()
        score = target_logits_for_explanation[0, target_class_idx]

        score.backward(retain_graph=False)

        self._mha_grad_inputs.reverse()
        
        if len(self._mha_grad_inputs) != num_registered:
            self._remove_hooks()
            raise RuntimeError(f"GMAR-Adapt: Captured {len(self._mha_grad_inputs)} grad_inputs, expected {num_registered}.")

        gradients_found = True
        valid_grad_inputs = []
        for i, grad in enumerate(self._mha_grad_inputs):
            if grad is None:
                gradients_found = False
                logger.error(f"GMAR-Adapt ERROR: Backward hook for layer {i} captured None grad_input.")
            else:
                valid_grad_inputs.append(grad)

        self._remove_hooks()

        if not gradients_found:
             raise RuntimeError("GMAR-Adapt: Backward hook captured None grad_input for one or more MHA layers.")

        confidence = torch.max(torch.softmax(target_logits_for_explanation, dim=1)).item()
        return self._attention_maps_for_pass, valid_grad_inputs, target_class_idx, confidence

    def _calculate_layer_relevance(self, mha_grad_inputs: list):
        """
        Calculates a relevance score for each layer based on the L1 norm of MHA grad_input.
        Args:
            mha_grad_inputs: List of tensors, each (B=1, N, D) - grad w.r.t MHA input.
        Returns:
            List of scalar relevance scores, one per layer.
        """
        layer_relevance = []
        for grad_input_BND in mha_grad_inputs:
            relevance = torch.sum(torch.abs(grad_input_BND.squeeze(0))).item()
            layer_relevance.append(relevance)
        
        total_relevance = sum(layer_relevance)
        if total_relevance < 1e-7:
             num_layers = len(mha_grad_inputs)
             return [1.0 / num_layers if num_layers > 0 else 0.0] * num_layers
        else:
             normalized_relevance = [r / total_relevance for r in layer_relevance]
             return normalized_relevance

    def _perform_relevance_weighted_rollout(self, attention_maps_per_layer: list, layer_relevance_scores: list):
        """
        Performs attention rollout, weighting each layer's contribution by its relevance score.
        Averages attention heads within each layer before applying relevance weight.
        Args:
            attention_maps_per_layer: List of (B=1, H, N, N) tensors (A_h^(l)).
            layer_relevance_scores: List of scalar relevance scores (normalized).
        Returns:
            A 2D numpy array representing the heatmap.
        """
        if not attention_maps_per_layer or not layer_relevance_scores or len(attention_maps_per_layer) != len(layer_relevance_scores):
            logger.warning("GMAR-Adapt: Mismatched inputs for relevance weighted rollout.")
            return None
        
        N = attention_maps_per_layer[0].size(-1)
        A_rollout = torch.eye(N, N, device=self.device)
        identity_matrix = torch.eye(N, N, device=self.device)

        for i, layer_attn_maps_B_H_N_N in enumerate(attention_maps_per_layer):
            relevance_l = layer_relevance_scores[i]
            
            A_avg_layer = torch.mean(layer_attn_maps_B_H_N_N, dim=1, keepdim=False)
            A_avg_layer_NN = A_avg_layer.squeeze(0)

            weighted_layer_contribution = identity_matrix + (relevance_l * A_avg_layer_NN)
            weighted_layer_contribution = identity_matrix + (relevance_l * A_avg_layer_NN)
            A_rollout = torch.matmul(A_rollout, weighted_layer_contribution)

        heatmap_vector = A_rollout[0, 1:] 

        try:
            patch_embed_module = self.model.image_net.patch_embed
            img_h, img_w = patch_embed_module.img_h_w_tuple
            patch_size = patch_embed_module.p_val
            num_patches_h = img_h // patch_size
            num_patches_w = img_w // patch_size
            if heatmap_vector.numel() != num_patches_h * num_patches_w:
                raise ValueError("Mismatch between heatmap size and patch grid size")
            heatmap_2d = heatmap_vector.reshape(num_patches_h, num_patches_w)
        except Exception as e:
            logger.error(f"GMAR-Adapt: Error reshaping heatmap - {e}. Check patch embedding info.")
            return None

        if heatmap_2d.numel() > 0:
            min_val = heatmap_2d.min()
            max_val = heatmap_2d.max()
            if (max_val - min_val).abs() > 1e-7:
                heatmap_2d = (heatmap_2d - min_val) / (max_val - min_val)
            else:
                heatmap_2d = torch.zeros_like(heatmap_2d) if min_val.abs() < 1e-7 else torch.ones_like(heatmap_2d) * 0.5
        return heatmap_2d.cpu().numpy()

    def compute_gmar(self, clinical_data_single: torch.Tensor, mri_data_single: torch.Tensor, target_class_idx: int = None):
        """
        Main method for Gradient-Weighted Attention Rollout.
        """
        pred_class, confidence = None, None
        heatmap_2d = None
        try:
            attention_maps, mha_grad_inputs, pred_class, confidence = self._get_attention_maps_and_gradients(
                clinical_data_single, mri_data_single, target_class_idx
            )
            
            if mha_grad_inputs:
                layer_relevance = self._calculate_layer_relevance(mha_grad_inputs)
                heatmap_2d = self._perform_relevance_weighted_rollout(attention_maps, layer_relevance)
            else:
                 logger.warning("GMAR-Adapt: No valid grad_inputs captured, cannot compute heatmap.")

        except RuntimeError as e:
            logger.error(f"GMAR-Adapt: Gradient capture/calculation failed: {e}")
            with torch.no_grad():
                current_clinical_input = clinical_data_single if self.model_modality_mode in ["full", "clinical_only"] else None
                current_mri_input = mri_data_single if self.model_modality_mode in ["full", "image_only"] else None

                main_logits_output, image_only_logits_output, clinical_only_logits_output, _ = self.model(current_clinical_input, current_mri_input)
                
                target_logits_for_fallback_pred = None
                if self.model_modality_mode == "image_only":
                    target_logits_for_fallback_pred = image_only_logits_output
                elif self.model_modality_mode == "clinical_only":
                    target_logits_for_fallback_pred = clinical_only_logits_output
                else:
                    target_logits_for_fallback_pred = main_logits_output
                
                if target_logits_for_fallback_pred is not None:
                    pred_class = torch.argmax(target_logits_for_fallback_pred, dim=1).item()
                    confidence = torch.max(torch.softmax(target_logits_for_fallback_pred, dim=1)).item()
                else:
                    pred_class = -1
                    confidence = 0.0

            heatmap_2d = None
        
        return heatmap_2d, pred_class, confidence

class GMARGenerator:
    """
    Generates Gradient-Weighted Attention Rollout visualizations.
    """
    def __init__(self, model: nn.Module, device: torch.device, vit_encoder_layers: nn.ModuleList, class_names: list[str] | None = None, model_modality_mode: str = "full"):
        self.model = model.to(device) 
        self.model.eval()
        self.device = device
        self.vit_encoder_layers = vit_encoder_layers
        self.class_names = class_names if class_names else ['Class 0', 'Class 1', 'Class 2']
        self.model_modality_mode = model_modality_mode
        self.gmar_calculator = GMARInternal(self.model, self.vit_encoder_layers, self.device, self.model_modality_mode)
        self.stats = {'total_samples_processed': 0, 'heatmap_generated': 0, 'correct_predictions': 0, 'failed_samples': 0}

    def generate_heatmap_for_sample(self, clinical_data: torch.Tensor, mri_data: torch.Tensor, target_class: int = None):
        clinical_data_single = clinical_data.to(self.device)
        mri_data_single = mri_data.to(self.device)
        heatmap_np, predicted_class_idx, confidence = self.gmar_calculator.compute_gmar(
            clinical_data_single, mri_data_single, target_class
        )
        return heatmap_np, predicted_class_idx, confidence

    def _plot_gmar_visualization(self, original_mri_slices: np.ndarray, heatmap_2d_np: np.ndarray,
                                 output_dir: str, sample_idx: int,
                                 pred_class_idx: int, true_label_idx: int, confidence: float):
        num_slices = original_mri_slices.shape[0]
        if num_slices != 3:
            logger.warning(f"Expected 3 MRI slices, got {num_slices}. Skip plot sample {sample_idx}.")
            return
        if heatmap_2d_np is None or heatmap_2d_np.size == 0:
            logger.warning(f"Invalid/empty heatmap for sample {sample_idx}. Skip plot.")
            return
        fig, axes = plt.subplots(num_slices, 3, figsize=(12, 3 * num_slices + 1.5))
        if num_slices == 1: axes = np.array([axes]).reshape(1,3)
        slice_titles = ['Slice 1 (e.g., Sagittal)', 'Slice 2 (e.g., Coronal)', 'Slice 3 (e.g., Axial)']
        pred_name = self.class_names[pred_class_idx] if 0 <= pred_class_idx < len(self.class_names) else f"Cls {pred_class_idx}"
        true_name = self.class_names[true_label_idx] if 0 <= true_label_idx < len(self.class_names) else f"Cls {true_label_idx}"
        fig.suptitle(f"Grad-Weighted Attn Rollout Sample {sample_idx} - Pred: {pred_name} ({confidence:.2f}), True: {true_name}", fontsize=14, y=0.98)
        for i in range(num_slices):
            mri_slice = original_mri_slices[i]
            resized_heatmap = resize(heatmap_2d_np,
                                     (mri_slice.shape[0], mri_slice.shape[1]),
                                     anti_aliasing=True, preserve_range=True, mode='reflect')
            if resized_heatmap.max() > 1e-7:
                resized_heatmap = (resized_heatmap - resized_heatmap.min()) / (resized_heatmap.max() - resized_heatmap.min())
            else:
                resized_heatmap = np.zeros_like(resized_heatmap)
            axes[i, 0].imshow(mri_slice, cmap='gray')
            axes[i, 0].set_title(f"{slice_titles[i]}")
            axes[i, 0].axis('off')
            axes[i, 1].imshow(resized_heatmap, cmap='hot', vmin=0, vmax=1)
            axes[i, 1].set_title("Grad-Weighted Heatmap")
            axes[i, 1].axis('off')
            axes[i, 2].imshow(mri_slice, cmap='gray')
            axes[i, 2].imshow(resized_heatmap, cmap='hot', alpha=0.6, vmin=0, vmax=1)
            axes[i, 2].set_title("Overlay")
            axes[i, 2].axis('off')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plot_path = os.path.join(output_dir, f"gmar_sample_{sample_idx}.png")
        try:
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        except Exception as e:
            logger.error(f"Failed to save plot for sample {sample_idx} to {plot_path}: {e}")
        plt.close(fig)

    def process_batch(self, clinical_data_batch: torch.Tensor, mri_data_batch: torch.Tensor, labels_batch: torch.Tensor,
                        output_dir: str, global_sample_offset: int, samples_to_process_in_this_batch: int) -> int:
        batch_size = clinical_data_batch.size(0)
        num_processed_this_call = 0
        for i in range(min(batch_size, samples_to_process_in_this_batch)):
            current_global_sample_idx = global_sample_offset + i
            self.stats['total_samples_processed'] += 1
            
            current_clinical_sample = clinical_data_batch[i:i+1]
            current_mri_sample = mri_data_batch[i:i+1]
            true_label = labels_batch[i].item()
            try:
                heatmap_np, pred_class_idx, confidence = self.generate_heatmap_for_sample(
                    current_clinical_sample, current_mri_sample
                )

                if heatmap_np is not None:
                     self.stats['heatmap_generated'] += 1
                     if pred_class_idx == true_label:
                         self.stats['correct_predictions'] += 1
                     original_mri_for_plot = current_mri_sample.squeeze(0).cpu().numpy()
                     self._plot_gmar_visualization(original_mri_for_plot, heatmap_np,
                                                   output_dir, current_global_sample_idx,
                                                   pred_class_idx, true_label, confidence)
                else:
                    logger.warning(f"Adapted GMAR heatmap generation failed for sample {current_global_sample_idx}.")
                    self.stats['failed_samples'] += 1
                    
            except Exception as e:
                logger.error(f"Unexpected error during GMAR-Adapt processing for sample {current_global_sample_idx}: {e}", exc_info=True)
                self.stats['failed_samples'] += 1
                
            num_processed_this_call += 1
        return num_processed_this_call

    def get_final_stats(self) -> dict:
        total_generated = self.stats['heatmap_generated']
        if total_generated > 0:
            self.stats['accuracy'] = (self.stats['correct_predictions'] / total_generated) * 100
        else:
            self.stats['accuracy'] = 0.0
        
        logger.info(f"GMAR-Adapt final stats: Attempted {self.stats['total_samples_processed']} samples.")
        logger.info(f"  Heatmaps Generated Successfully: {self.stats['heatmap_generated']}")
        logger.info(f"  Failures (Capture or Processing): {self.stats['failed_samples']}")
        if self.stats['total_samples_processed'] != self.stats['heatmap_generated'] + self.stats['failed_samples']:
             logger.warning("Stat counts mismatch (processed != generated + failed)")
             
        if total_generated > 0:
            logger.info(f"  Accuracy on samples w/ successful heatmap: {self.stats['accuracy']:.2f}%")
        return self.stats

    def cleanup(self):
        pass 