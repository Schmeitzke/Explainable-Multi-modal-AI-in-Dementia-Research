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

class GMARInternal:
    """
    Internal helper class for Gradient-Weighted Attention Rollout for ADTransformer.
    Uses forward hooks for attention maps & backward hooks for attention gradients.
    """
    def __init__(self, model, transformer_layers, device):
        self.model = model
        self.transformer_layers = transformer_layers
        self.device = device
        self._attention_maps_for_pass = []
        self._attention_grad_inputs = []
        self._hook_handles = []

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
        """Backward hook function to capture attention grad_input."""
        if grad_input is not None and len(grad_input) > 0 and grad_input[0] is not None:
            self._attention_grad_inputs.append(grad_input[0].detach().clone())
        else:
            logger.warning(f"Backward hook for {module} received None or invalid grad_input: {grad_input}")
            self._attention_grad_inputs.append(None)

    def _remove_hooks(self):
        """Removes all registered hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

    def _get_attention_maps_and_gradients(self, non_image_data_single: torch.Tensor, image_data_single: torch.Tensor, target_class_idx: int = None):
        """
        Performs forward/backward pass using hooks to capture attention maps (forward)
        and attention grad_input (backward).
        """
        self.model.eval()
        self._attention_maps_for_pass = []
        self._attention_grad_inputs = []
        self._hook_handles = []

        num_registered = 0
        for i, encoder_layer in enumerate(self.transformer_layers):
            if hasattr(encoder_layer, 'self_attn') and isinstance(encoder_layer.self_attn, nn.MultiheadAttention):
                fwd_handle = encoder_layer.self_attn.register_forward_hook(self._forward_hook_fn)
                bwd_handle = encoder_layer.self_attn.register_full_backward_hook(self._backward_hook_fn)
                self._hook_handles.extend([fwd_handle, bwd_handle])
                num_registered += 1
            else:
                logger.warning(f"GMAR: Encoder layer {i} has no self_attn layer. Skipping hook registration.")
        
        if num_registered == 0:
            raise RuntimeError("GMAR: Could not register hooks on any MultiheadAttention layers.")

        self.model.zero_grad()
        logits = self.model(non_image=non_image_data_single, image=image_data_single)

        if len(self._attention_maps_for_pass) != num_registered:
            self._remove_hooks()
            raise RuntimeError(
                f"GMAR: Expected {num_registered} attention map sets from forward hooks, "
                f"but captured {len(self._attention_maps_for_pass)}."
            )

        if target_class_idx is None:
            target_class_idx = torch.argmax(logits, dim=1).item()
        score = logits[0, target_class_idx]

        score.backward(retain_graph=False)

        self._attention_grad_inputs.reverse()
        
        if len(self._attention_grad_inputs) != num_registered:
            self._remove_hooks()
            raise RuntimeError(f"GMAR: Captured {len(self._attention_grad_inputs)} grad_inputs, expected {num_registered}.")

        gradients_found = True
        valid_grad_inputs = []
        for i, grad in enumerate(self._attention_grad_inputs):
            if grad is None:
                gradients_found = False
                logger.error(f"GMAR ERROR: Backward hook for layer {i} captured None grad_input.")
            else:
                valid_grad_inputs.append(grad)

        self._remove_hooks()

        if not gradients_found:
            raise RuntimeError("GMAR: Backward hook captured None grad_input for one or more attention layers.")

        confidence = torch.max(torch.softmax(logits, dim=1)).item()
        return self._attention_maps_for_pass, valid_grad_inputs, target_class_idx, confidence

    def _calculate_layer_relevance(self, attention_grad_inputs: list):
        """
        Calculates a relevance score for each layer based on the L1 norm of attention grad_input.
        """
        layer_relevance = []
        for grad_input_BND in attention_grad_inputs:
            relevance = torch.sum(torch.abs(grad_input_BND.squeeze(0))).item()
            layer_relevance.append(relevance)
        
        total_relevance = sum(layer_relevance)
        if total_relevance < 1e-7:
            num_layers = len(attention_grad_inputs)
            return [1.0 / num_layers if num_layers > 0 else 0.0] * num_layers
        else:
            normalized_relevance = [r / total_relevance for r in layer_relevance]
            return normalized_relevance

    def _perform_relevance_weighted_rollout(self, attention_maps_per_layer: list, layer_relevance_scores: list):
        """
        Performs attention rollout, weighting each layer's contribution by its relevance score.
        Averages attention heads within each layer before applying relevance weight.
        """
        if not attention_maps_per_layer or not layer_relevance_scores or len(attention_maps_per_layer) != len(layer_relevance_scores):
            logger.warning("GMAR: Mismatched inputs for relevance weighted rollout.")
            return None
        
        N = attention_maps_per_layer[0].size(-1)
        A_rollout = torch.eye(N, N, device=self.device)
        identity_matrix = torch.eye(N, N, device=self.device)

        for i, layer_attn_maps_B_H_N_N in enumerate(attention_maps_per_layer):
            relevance_l = layer_relevance_scores[i]
            
            A_avg_layer = torch.mean(layer_attn_maps_B_H_N_N, dim=1, keepdim=False)
            A_avg_layer_NN = A_avg_layer.squeeze(0)

            weighted_layer_contribution = (relevance_l * A_avg_layer_NN) + identity_matrix
            A_rollout = torch.matmul(A_rollout, weighted_layer_contribution)

        num_non_image_tokens = self.model.non_image_features if hasattr(self.model, 'non_image_features') else 0
        
        if N > num_non_image_tokens:
            image_tokens_start = num_non_image_tokens
            heatmap_vector = A_rollout[0, image_tokens_start:]
        else:
            heatmap_vector = A_rollout[0, 1:]

        if hasattr(self.model, 'image_emb') and self.model.image_emb is not None:
            try:
                num_image_tokens = None
                if hasattr(self.model.image_emb, 'actual_num_patches') and self.model.image_emb.actual_num_patches is not None:
                    num_image_tokens = self.model.image_emb.actual_num_patches
                elif hasattr(self.model.image_emb, 'expected_num_patches'):
                    num_image_tokens = self.model.image_emb.expected_num_patches
                else:
                    num_image_tokens = heatmap_vector.numel()
                    logger.warning(f"GMAR: Could not determine patch count, using all {num_image_tokens} tokens")
                
                if heatmap_vector.numel() >= num_image_tokens:
                    heatmap_vector = heatmap_vector[:num_image_tokens]
                    
                    sqrt_patches = int(np.sqrt(num_image_tokens))
                    if sqrt_patches * sqrt_patches == num_image_tokens:
                        heatmap_2d = heatmap_vector.reshape(sqrt_patches, sqrt_patches)
                    else:
                        h = int(np.sqrt(num_image_tokens))
                        w = (num_image_tokens + h - 1) // h
                        if h * w >= num_image_tokens:
                            padded_vector = torch.zeros(h * w, device=heatmap_vector.device)
                            padded_vector[:num_image_tokens] = heatmap_vector
                            heatmap_2d = padded_vector.reshape(h, w)
                        else:
                            heatmap_2d = heatmap_vector[:h*w].reshape(h, w)
                        logger.debug(f"GMAR: Using fallback grid reshape {h}x{w} for {num_image_tokens} tokens")
                else:
                    logger.warning(f"GMAR: Insufficient heatmap tokens ({heatmap_vector.numel()}) for expected patches ({num_image_tokens})")
                    total_needed = max(4, num_image_tokens)
                    sqrt_needed = int(np.sqrt(total_needed))
                    padded_vector = torch.zeros(sqrt_needed * sqrt_needed, device=heatmap_vector.device)
                    padded_vector[:heatmap_vector.numel()] = heatmap_vector
                    heatmap_2d = padded_vector.reshape(sqrt_needed, sqrt_needed)
                    logger.debug(f"GMAR: Used fallback padding to {sqrt_needed}x{sqrt_needed}")
                    
            except Exception as e:
                logger.error(f"GMAR: Error reshaping heatmap - {e}. Using fallback.")
                num_tokens = heatmap_vector.numel()
                if num_tokens >= 4:
                    sqrt_tokens = int(np.sqrt(num_tokens))
                    heatmap_2d = heatmap_vector[:sqrt_tokens*sqrt_tokens].reshape(sqrt_tokens, sqrt_tokens)
                else:
                    padded = torch.zeros(4, device=heatmap_vector.device)
                    padded[:num_tokens] = heatmap_vector
                    heatmap_2d = padded.reshape(2, 2)
        else:
            logger.warning("GMAR: No image embedding found in model, using fallback reshaping")
            num_tokens = heatmap_vector.numel()
            if num_tokens >= 4:
                sqrt_tokens = int(np.sqrt(num_tokens))
                heatmap_2d = heatmap_vector[:sqrt_tokens*sqrt_tokens].reshape(sqrt_tokens, sqrt_tokens)
            else:
                padded = torch.zeros(4, device=heatmap_vector.device)
                padded[:num_tokens] = heatmap_vector
                heatmap_2d = padded.reshape(2, 2)

        if heatmap_2d.numel() > 0:
            min_val = heatmap_2d.min()
            max_val = heatmap_2d.max()
            if max_val - min_val > 1e-7:
                heatmap_2d = (heatmap_2d - min_val) / (max_val - min_val)
            else:
                heatmap_2d = torch.zeros_like(heatmap_2d)
        return heatmap_2d.cpu().numpy()

    def compute_gmar(self, non_image_data_single: torch.Tensor, image_data_single: torch.Tensor, target_class_idx: int = None):
        """
        Main method for Gradient-Weighted Attention Rollout.
        """
        pred_class, confidence = None, None
        heatmap_2d = None
        try:
            attention_maps, attention_grad_inputs, pred_class, confidence = self._get_attention_maps_and_gradients(
                non_image_data_single, image_data_single, target_class_idx
            )
            
            if attention_grad_inputs:
                layer_relevance = self._calculate_layer_relevance(attention_grad_inputs)
                heatmap_2d = self._perform_relevance_weighted_rollout(attention_maps, layer_relevance)
            else:
                logger.warning("GMAR: No valid grad_inputs captured, cannot compute heatmap.")

        except RuntimeError as e:
            logger.error(f"GMAR: Gradient capture/calculation failed: {e}")
            with torch.no_grad():
                logits = self.model(non_image=non_image_data_single, image=image_data_single)
                if logits is not None:
                    pred_class = torch.argmax(logits, dim=1).item()
                    confidence = torch.max(torch.softmax(logits, dim=1)).item()
                else:
                    pred_class = -1
                    confidence = 0.0
            heatmap_2d = None
        
        return heatmap_2d, pred_class, confidence


class GMARGenerator:
    """
    Generates Gradient-Weighted Attention Rollout visualizations for ADTransformer.
    """
    def __init__(self, model: nn.Module, device: torch.device, transformer_layers: nn.ModuleList, class_names: list = None):
        self.model = model.to(device) 
        self.model.eval()
        self.device = device
        self.transformer_layers = transformer_layers
        self.class_names = class_names if class_names else ['CN', 'MCI', 'AD']
        self.gmar_calculator = GMARInternal(self.model, self.transformer_layers, self.device)
        
        self.stats = {'total_samples_processed': 0, 'heatmap_generated': 0, 'correct_predictions': 0, 'failed_samples': 0}

    def generate_heatmap_for_sample(self, non_image_data: torch.Tensor, image_data: torch.Tensor, target_class: int = None):
        non_image_data_single = non_image_data.to(self.device)
        image_data_single = image_data.to(self.device) if image_data is not None else None
        
        heatmap_np, predicted_class_idx, confidence = self.gmar_calculator.compute_gmar(
            non_image_data_single, image_data_single, target_class
        )
        return heatmap_np, predicted_class_idx, confidence

    def _plot_gmar_visualization(self, original_image_3d: np.ndarray, heatmap_2d_np: np.ndarray,
                                 output_dir: str, sample_idx: int,
                                 pred_class_idx: int, true_label_idx: int, confidence: float):
        """
        Plots GMAR visualization for 3D image data.
        """
        if len(original_image_3d.shape) != 4:
            logger.warning(f"Expected 4D image data (C, H, W, D), got {original_image_3d.shape}. Skip plot sample {sample_idx}.")
            return
        if heatmap_2d_np is None or heatmap_2d_np.size == 0:
            logger.warning(f"Invalid/empty heatmap for sample {sample_idx}. Skip plot.")
            return

        _, h, w, d = original_image_3d.shape
        
        mid_h = h // 2
        mid_w = w // 2
        mid_d = d // 2
        
        img_channel = original_image_3d[0]
        
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        
        pred_name = self.class_names[pred_class_idx] if 0 <= pred_class_idx < len(self.class_names) else f"Cls {pred_class_idx}"
        true_name = self.class_names[true_label_idx] if 0 <= true_label_idx < len(self.class_names) else f"Cls {true_label_idx}"
        fig.suptitle(f"Grad-Weighted Attn Rollout Sample {sample_idx} - Pred: {pred_name} ({confidence:.2f}), True: {true_name}", fontsize=14)

        view_titles = ['Sagittal View', 'Coronal View', 'Axial View']
        
        for i, (view_title, img_slice) in enumerate([
            (view_titles[0], img_channel[:, mid_w, :]),
            (view_titles[1], img_channel[mid_h, :, :]),
            (view_titles[2], img_channel[:, :, mid_d])
        ]):
            resized_heatmap = resize(heatmap_2d_np,
                                   (img_slice.shape[0], img_slice.shape[1]),
                                   anti_aliasing=True, preserve_range=True, mode='reflect')
            min_val = resized_heatmap.min()
            max_val = resized_heatmap.max()
            if max_val - min_val > 1e-7:
                resized_heatmap = (resized_heatmap - min_val) / (max_val - min_val)
            else:
                resized_heatmap = np.zeros_like(resized_heatmap)

            axes[i, 0].imshow(img_slice, cmap='gray')
            axes[i, 0].set_title(f"{view_title}")
            axes[i, 0].axis('off')

            axes[i, 1].imshow(resized_heatmap, cmap='hot', vmin=0, vmax=1)
            axes[i, 1].set_title("Grad-Weighted Heatmap")
            axes[i, 1].axis('off')

            axes[i, 2].imshow(img_slice, cmap='gray')
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

    def process_batch(self, non_image_data_batch: torch.Tensor, image_data_batch: torch.Tensor, labels_batch: torch.Tensor,
                        output_dir: str, global_sample_offset: int, samples_to_process_in_this_batch: int) -> int:
        batch_size = non_image_data_batch.size(0)
        num_processed_this_call = 0
        
        for i in range(min(batch_size, samples_to_process_in_this_batch)):
            current_global_sample_idx = global_sample_offset + i
            self.stats['total_samples_processed'] += 1
            
            current_non_image_sample = non_image_data_batch[i:i+1]
            current_image_sample = image_data_batch[i:i+1] if image_data_batch is not None else None
            true_label = labels_batch[i].item()
            
            try:
                heatmap_np, pred_class_idx, confidence = self.generate_heatmap_for_sample(
                    current_non_image_sample, current_image_sample
                )

                if heatmap_np is not None:
                    self.stats['heatmap_generated'] += 1
                    if pred_class_idx == true_label:
                        self.stats['correct_predictions'] += 1
                    
                    if current_image_sample is not None:
                        original_image_for_plot = current_image_sample.squeeze(0).cpu().numpy()
                        self._plot_gmar_visualization(original_image_for_plot, heatmap_np,
                                                      output_dir, current_global_sample_idx,
                                                      pred_class_idx, true_label, confidence)
                    else:
                        logger.warning(f"No image data for GMAR visualization of sample {current_global_sample_idx}")
                else:
                    logger.warning(f"GMAR heatmap generation failed for sample {current_global_sample_idx}.")
                    self.stats['failed_samples'] += 1
                    
            except Exception as e:
                logger.error(f"Unexpected error during GMAR processing for sample {current_global_sample_idx}: {e}", exc_info=True)
                self.stats['failed_samples'] += 1
                
            num_processed_this_call += 1
        return num_processed_this_call

    def get_final_stats(self) -> dict:
        total_generated = self.stats['heatmap_generated']
        if total_generated > 0:
            self.stats['accuracy'] = (self.stats['correct_predictions'] / total_generated) * 100
        else:
            self.stats['accuracy'] = 0.0
        
        logger.info(f"GMAR final stats: Attempted {self.stats['total_samples_processed']} samples.")
        logger.info(f"  Heatmaps Generated Successfully: {self.stats['heatmap_generated']}")
        logger.info(f"  Failures (Capture or Processing): {self.stats['failed_samples']}")
        
        if self.stats['total_samples_processed'] != self.stats['heatmap_generated'] + self.stats['failed_samples']:
            logger.warning("Stat counts mismatch (processed != generated + failed)")
             
        if total_generated > 0:
            logger.info(f"  Accuracy on samples w/ successful heatmap: {self.stats['accuracy']:.2f}%")
        return self.stats

    def cleanup(self):
        pass 