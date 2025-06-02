import logging
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.gridspec as gridspec

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class SaliencyMapGenerator:
    """
    Generates saliency maps for MMDF model predictions.
    """
    def __init__(self, model, device, feature_columns=None, class_names=None, model_modality_mode: str = "full"):
        """
        Initialize the saliency map generator.

        Args:
            model: The trained MMDF model.
            device: Device to run computations on (cuda or cpu).
            feature_columns: List of names for clinical features.
            class_names: List of class names for labeling.
            model_modality_mode: Mode of the model ("full", "image_only", "clinical_only").
        """
        self.model = model
        self.device = device
        self.model.eval()
        self.feature_columns = feature_columns if feature_columns else []
        self.class_names = class_names if class_names else ['Class 0', 'Class 1', 'Class 2']
        self.model_modality_mode = model_modality_mode

        self.stats = {
            'total_samples': 0,
            'correct_predictions': 0,
            'feature_importance': {name: 0.0 for name in self.feature_columns} if self.feature_columns else {},
            'modality_importance': {
                'clinical': {'absolute': 0.0, 'percentage': 0.0},
                'mri': {'absolute': 0.0, 'percentage': 0.0}
            }
        }
        self._temp_modality_percentage_sum = {
            'clinical': 0.0,
            'mri': 0.0
        }

    def _get_gradients(self, clinical_data, mri_data, target_class=None):
        """
        Calculate gradients of the output with respect to inputs.

        Args:
            clinical_data: Clinical features tensor.
            mri_data: MRI data tensor.
            target_class: Class to calculate gradients for (uses predicted class if None).

        Returns:
            Tuple of (clinical_gradients, mri_gradients, predicted_class, prediction_confidence)
        """
        clinical_data = clinical_data.clone().detach().to(self.device)
        clinical_data.requires_grad = True

        mri_data = mri_data.clone().detach().to(self.device)
        mri_data.requires_grad = True

        was_training = self.model.training
        self.model.eval()

        current_clinical_input = clinical_data if self.model_modality_mode in ["full", "clinical_only"] else None
        current_mri_input = mri_data if self.model_modality_mode in ["full", "image_only"] else None

        with torch.enable_grad():
            main_logits_output, image_only_logits_output, clinical_only_logits_output, _ = self.model(current_clinical_input, current_mri_input)

            target_logits_for_explanation = None
            if self.model_modality_mode == "image_only":
                if image_only_logits_output is None:
                    self.model.train(was_training)
                    raise ValueError("Saliency: Image-only mode, but image_only_logits are None.")
                target_logits_for_explanation = image_only_logits_output
            elif self.model_modality_mode == "clinical_only":
                if clinical_only_logits_output is None:
                    self.model.train(was_training)
                    raise ValueError("Saliency: Clinical-only mode, but clinical_only_logits are None.")
                target_logits_for_explanation = clinical_only_logits_output
            else:
                if main_logits_output is None:
                    self.model.train(was_training)
                    raise ValueError("Saliency: Full mode, but main_logits are None. Ensure both modalities are passed if appropriate.")
                target_logits_for_explanation = main_logits_output
            
            probs = torch.softmax(target_logits_for_explanation, dim=1)
            confidence, pred_class = torch.max(probs, dim=1)

            if target_class is None:
                target_class = pred_class.item()

            self.model.zero_grad()
            target_score = target_logits_for_explanation[0, target_class]
            target_score.backward()

        self.model.train(was_training)

        clinical_grad = torch.zeros_like(clinical_data)
        if current_clinical_input is not None and clinical_data.grad is not None:
            clinical_grad = clinical_data.grad.detach()
        
        mri_grad = torch.zeros_like(mri_data)
        if current_mri_input is not None and mri_data.grad is not None:
            mri_grad = mri_data.grad.detach()

        return clinical_grad, mri_grad, pred_class.item(), confidence.item()

    def _calculate_modality_importance(self, clinical_grad, mri_grad):
        """
        Calculate the relative importance of clinical vs MRI modalities.
        """
        clinical_importance = torch.mean(torch.abs(clinical_grad)).item()
        mri_importance = torch.mean(torch.abs(mri_grad)).item()

        total_importance = clinical_importance + mri_importance
        if total_importance > 0:
            clinical_percentage = (clinical_importance / total_importance) * 100
            mri_percentage = (mri_importance / total_importance) * 100
        else:
            clinical_percentage = 50
            mri_percentage = 50

        return {
            'clinical': {'absolute': clinical_importance, 'percentage': clinical_percentage},
            'mri': {'absolute': mri_importance, 'percentage': mri_percentage}
        }

    def _plot_modality_comparison(self, fig, ax, modality_importance):
        modalities = ['Clinical Features', 'MRI Features']
        percentages = [
            modality_importance['clinical']['percentage'],
            modality_importance['mri']['percentage']
        ]
        colors = ['cornflowerblue', 'orangered']
        bars = ax.barh(modalities, percentages, color=colors)
        for bar in bars:
            width = bar.get_width()
            ax.text(
                min(width + 1, 105),
                bar.get_y() + bar.get_height()/2,
                f"{width:.1f}%",
                va='center',
                ha='left' if width < 50 else 'right',
                color='black' if width < 90 else 'white'
            )
        ax.set_xlabel('Relative Importance (%)')
        ax.set_title('Modality Importance Comparison')
        ax.set_xlim(0, 110)
        ax.text(
            0, -0.25,
            f"Absolute importance - Clinical: {modality_importance['clinical']['absolute']:.4f}, " +
            f"MRI: {modality_importance['mri']['absolute']:.4f}",
            transform=ax.transAxes,
            fontsize=8
        )

    def _plot_feature_saliency(self, fig, ax, features, gradients, feature_names,
                               predicted_class, true_class, confidence):
        """Plot saliency for clinical features."""
        abs_gradients = np.abs(gradients)
        norm_gradients = abs_gradients / np.max(abs_gradients) if np.max(abs_gradients) > 0 else abs_gradients
        
        indices = np.argsort(norm_gradients)[::-1]
        top_n = min(20, len(indices))

        if not feature_names:
            feature_names = [f"Feature_{i+1}" for i in range(len(features))]

        top_feature_names = [feature_names[i] for i in indices[:top_n]]
        top_gradients = norm_gradients[indices[:top_n]]
        top_values = features[indices[:top_n]]

        bars = ax.barh(np.arange(top_n), top_gradients, color='cornflowerblue')
        for i, (value, bar) in enumerate(zip(top_values, bars)):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f"{value:.2f}", va='center')

        ax.set_yticks(np.arange(top_n))
        ax.set_yticklabels(top_feature_names)
        ax.set_xlabel('Relative Importance')
        ax.set_title(f"Prediction: {self.class_names[predicted_class]} ({confidence:.2f}), True: {self.class_names[true_class]}")
        ax.set_xlim(0, 1.1)

    def _plot_image_saliency(self, fig, ax_gridspec, image_data, image_gradients):
        """Plot saliency for MRI data (coronal, sagittal, axial views)."""
        if image_data.shape[0] != 3:
            print(f"Warning: Expected 3 channels for MRI for visualization, got {image_data.shape[0]}. Skipping image saliency plot.")
            ax = fig.add_subplot(ax_gridspec)
            ax.text(0.5, 0.5, "Image saliency N/A\n(Expected 3 channels)", ha='center', va='center')
            ax.axis('off')
            return

        slice_titles = ['Slice 1 / Sagittal-like', 'Slice 2 / Coronal-like', 'Slice 3 / Axial-like']

        grad_magnitude_slices = [np.abs(image_gradients[i]) for i in range(3)]

        gs_views = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=ax_gridspec, wspace=0.2)

        for i in range(3):
            ax = fig.add_subplot(gs_views[0, i])
            img_slice_to_show = image_data[i]
            if img_slice_to_show.max() > 1.0 and img_slice_to_show.min() >= 0:
                 img_slice_to_show = img_slice_to_show / 255.0 if img_slice_to_show.max() > 2.0 else img_slice_to_show / img_slice_to_show.max()
            img_slice_to_show = np.clip(img_slice_to_show, 0, 1)

            ax.imshow(img_slice_to_show, cmap='gray')
            saliency_slice_to_show = grad_magnitude_slices[i]
            if saliency_slice_to_show.max() > 0:
                saliency_slice_to_show = saliency_slice_to_show / saliency_slice_to_show.max()
            
            ax.imshow(saliency_slice_to_show, cmap='hot', alpha=0.5)
            ax.set_title(slice_titles[i])
            ax.axis('off')

    def process_batch(self, clinical_data_batch: torch.Tensor, mri_data_batch: torch.Tensor, labels_batch: torch.Tensor,
                        output_dir: str, global_sample_offset: int, samples_to_process_in_this_batch: int) -> int:
        """
        Processes a single batch of data for Saliency Maps, generates visualizations, and updates stats.
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
        batch_size = clinical_data_batch.size(0)
        num_processed_this_call = 0
        
        if not self.feature_columns and clinical_data_batch.shape[1] > 0 and not self.stats['feature_importance']:
            num_feats = clinical_data_batch.shape[1]
            self.feature_columns = [f"ClinicalFeature_{j}" for j in range(num_feats)]
            self.stats['feature_importance'] = {name: 0.0 for name in self.feature_columns}
            logger.info(f"Initialized feature_columns for SaliencyMapGenerator based on batch data: {self.feature_columns}")


        for i in range(min(batch_size, samples_to_process_in_this_batch)):
            current_global_sample_idx = global_sample_offset + i

            sample_clinical = clinical_data_batch[i:i+1].to(self.device)
            sample_mri = mri_data_batch[i:i+1].to(self.device)
            true_label = labels_batch[i].item()

            try:
                clinical_grad, mri_grad, pred_class, confidence = self._get_gradients(
                    sample_clinical, sample_mri
                )
                
                modality_importance = self._calculate_modality_importance(clinical_grad, mri_grad)

                self.stats['total_samples'] += 1
                if pred_class == true_label:
                    self.stats['correct_predictions'] += 1
                
                self.stats['modality_importance']['clinical']['absolute'] += modality_importance['clinical']['absolute']
                self.stats['modality_importance']['mri']['absolute'] += modality_importance['mri']['absolute']
                self._temp_modality_percentage_sum['clinical'] += modality_importance['clinical']['percentage']
                self._temp_modality_percentage_sum['mri'] += modality_importance['mri']['percentage']


                fig = plt.figure(figsize=(15, 15))
                plt.subplots_adjust(hspace=0.4)
                gs = fig.add_gridspec(3, 1, height_ratios=[0.5, 1, 1.2])

                ax_modality = fig.add_subplot(gs[0])
                self._plot_modality_comparison(fig, ax_modality, modality_importance)

                ax_features = fig.add_subplot(gs[1])
                self._plot_feature_saliency(
                    fig, ax_features,
                    sample_clinical[0].cpu().detach().numpy(),
                    clinical_grad[0].cpu().numpy(),
                    self.feature_columns,
                    pred_class, true_label, confidence
                )
                
                self._plot_image_saliency(
                    fig, gs[2],
                    sample_mri[0].cpu().detach().numpy(),
                    mri_grad[0].cpu().numpy()
                )
                
                os.makedirs(output_dir, exist_ok=True)
                plt.savefig(os.path.join(output_dir, f"saliency_sample_{current_global_sample_idx}.png"), dpi=150)
                plt.close(fig)

                if self.feature_columns and self.stats['feature_importance']:
                    abs_grads_clinical = np.abs(clinical_grad[0].cpu().numpy())
                    for j, feature_name in enumerate(self.feature_columns):
                        if j < len(abs_grads_clinical):
                             self.stats['feature_importance'][feature_name] += abs_grads_clinical[j]
            except Exception as e:
                logger.error(f"Error processing saliency for global_sample_idx {current_global_sample_idx}: {e}", exc_info=True)
            
            num_processed_this_call += 1
        
        return num_processed_this_call

    def get_final_stats(self, output_dir_for_summary_plots: str) -> dict:
        """Calculates final statistics, generates summary plots, and returns the stats dictionary."""
        if self.stats['total_samples'] > 0:
            if self.feature_columns and self.stats['feature_importance']:
                for feature_name in self.stats['feature_importance']:
                    self.stats['feature_importance'][feature_name] /= self.stats['total_samples']
                self.stats['feature_importance'] = dict(
                    sorted(self.stats['feature_importance'].items(), key=lambda item: item[1], reverse=True)
                )
            
            clinical_abs_avg = self.stats['modality_importance']['clinical']['absolute'] / self.stats['total_samples']
            mri_abs_avg = self.stats['modality_importance']['mri']['absolute'] / self.stats['total_samples']
            self.stats['modality_importance']['clinical']['absolute'] = clinical_abs_avg
            self.stats['modality_importance']['mri']['absolute'] = mri_abs_avg

            self.stats['modality_importance']['clinical']['percentage'] = self._temp_modality_percentage_sum['clinical'] / self.stats['total_samples']
            self.stats['modality_importance']['mri']['percentage'] = self._temp_modality_percentage_sum['mri'] / self.stats['total_samples']

            os.makedirs(output_dir_for_summary_plots, exist_ok=True)
            self._plot_feature_importance_summary(self.stats.get('feature_importance', {}), output_dir_for_summary_plots)
            self._plot_modality_importance_summary(self.stats['modality_importance'], output_dir_for_summary_plots)
        
        if self.stats['total_samples'] > 0:
            self.stats['accuracy'] = self.stats['correct_predictions'] / self.stats['total_samples']
        else:
            self.stats['accuracy'] = 0.0
            
        logger.info(f"Saliency Map final stats: Processed {self.stats['total_samples']} samples.")
        if self.stats['total_samples'] > 0:
            logger.info(f"Saliency Map accuracy on processed samples: {self.stats['accuracy']:.4f}")

        return self.stats
    
    def cleanup(self):
        """Placeholder for cleanup, if any future resources are used (e.g. hooks)."""
        logger.debug("Cleaning up SaliencyMapGenerator (no specific resources to free currently).")
        pass

    def _plot_feature_importance_summary(self, feature_importance, output_dir):
        if not feature_importance:
            print("No feature importance data to plot for summary.")
            return
            
        fig = plt.figure(figsize=(12, 8))
        top_features = list(feature_importance.keys())[:20]
        importances = [feature_importance[f] for f in top_features]
        
        plt.subplots_adjust(left=0.3)
        y_pos = np.arange(len(top_features))
        plt.barh(y_pos, importances, color='cornflowerblue')
        plt.yticks(y_pos, top_features)
        plt.xlabel('Average Absolute Gradient Magnitude')
        plt.title('Top 20 Most Important Clinical Features')
        plt.savefig(os.path.join(output_dir, 'clinical_feature_importance_summary.png'), dpi=150)
        plt.close(fig)

    def _plot_modality_importance_summary(self, modality_importance, output_dir):
        fig = plt.figure(figsize=(10, 6))
        modalities = ['Clinical Features', 'MRI Features']
        percentages = [
            modality_importance['clinical']['percentage'],
            modality_importance['mri']['percentage']
        ]
        colors = ['cornflowerblue', 'orangered']
        ax = fig.add_subplot(1, 1, 1)
        wedges, texts, autotexts = ax.pie(
            percentages, labels=modalities, colors=colors,
            autopct='%1.1f%%', startangle=90, shadow=False, explode=(0.05, 0.05)
        )
        plt.setp(autotexts, size=10, weight="bold")
        plt.setp(texts, size=12)
        ax.set_title('Average Modality Importance Across All Samples', size=14)
        fig.text(
            0.5, 0.01,
            f"Avg abs importance - Clinical: {modality_importance['clinical']['absolute']:.4f}, " +
            f"MRI: {modality_importance['mri']['absolute']:.4f}",
            ha='center', fontsize=9
        )
        plt.savefig(os.path.join(output_dir, 'modality_importance_summary.png'), dpi=150, bbox_inches='tight')
        plt.close(fig) 