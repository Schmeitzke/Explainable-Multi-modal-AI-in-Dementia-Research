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
    Generates saliency maps for ADTransformer model predictions.
    """
    def __init__(self, model, device, feature_columns=None, class_names=None):
        """
        Initialize the saliency map generator.

        Args:
            model: The trained ADTransformer model.
            device: Device to run computations on (cuda or cpu).
            feature_columns: List of names for non-image features.
            class_names: List of class names for labeling.
        """
        self.model = model
        self.device = device
        self.model.eval()
        self.feature_columns = feature_columns if feature_columns else []
        self.class_names = class_names if class_names else ['CN', 'MCI', 'AD']

        self.stats = {
            'total_samples': 0,
            'correct_predictions': 0,
            'feature_importance': {name: 0.0 for name in self.feature_columns} if self.feature_columns else {},
            'modality_importance': {
                'non_image': {'absolute': 0.0, 'percentage': 0.0},
                'image': {'absolute': 0.0, 'percentage': 0.0}
            }
        }
        self._temp_modality_percentage_sum = {
            'non_image': 0.0,
            'image': 0.0
        }

    def _get_gradients(self, non_image_data, image_data=None, target_class=None):
        """
        Calculate gradients of the output with respect to inputs.

        Args:
            non_image_data: Non-image features tensor.
            image_data: Image data tensor.
            target_class: Class to calculate gradients for (uses predicted class if None).

        Returns:
            Tuple of (non_image_gradients, image_gradients, predicted_class, prediction_confidence)
        """
        non_image_data = non_image_data.clone().detach().to(self.device)
        non_image_data.requires_grad = True

        image_grad = None
        if image_data is not None:
            image_data = image_data.clone().detach().to(self.device)
            image_data.requires_grad = True

        was_training = self.model.training
        self.model.eval()

        with torch.enable_grad():
            logits = self.model(non_image=non_image_data, image=image_data)
            
            probs = torch.softmax(logits, dim=1)
            confidence, pred_class = torch.max(probs, dim=1)

            if target_class is None:
                target_class = pred_class.item()

            self.model.zero_grad()
            score = logits[0, target_class]
            score.backward()

        self.model.train(was_training)

        non_image_grad = torch.zeros_like(non_image_data)
        if non_image_data.grad is not None:
            non_image_grad = non_image_data.grad.detach()
        
        if image_data is not None and image_data.grad is not None:
            image_grad = image_data.grad.detach()

        return non_image_grad, image_grad, pred_class.item(), confidence.item()

    def _calculate_modality_importance(self, non_image_grad, image_grad):
        """
        Calculate the relative importance of non-image vs image modalities.
        """
        non_image_importance = torch.mean(torch.abs(non_image_grad)).item()
        image_importance = 0.0
        if image_grad is not None:
            image_importance = torch.mean(torch.abs(image_grad)).item()

        total_importance = non_image_importance + image_importance
        if total_importance > 0:
            non_image_percentage = (non_image_importance / total_importance) * 100
            image_percentage = (image_importance / total_importance) * 100
        else:
            non_image_percentage = 50
            image_percentage = 50

        return {
            'non_image': {'absolute': non_image_importance, 'percentage': non_image_percentage},
            'image': {'absolute': image_importance, 'percentage': image_percentage}
        }

    def _plot_modality_comparison(self, fig, ax, modality_importance):
        modalities = ['Non-Image Features', 'Image Features']
        percentages = [
            modality_importance['non_image']['percentage'],
            modality_importance['image']['percentage']
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
            f"Absolute importance - Non-Image: {modality_importance['non_image']['absolute']:.4f}, " +
            f"Image: {modality_importance['image']['absolute']:.4f}",
            transform=ax.transAxes,
            fontsize=8
        )

    def _plot_feature_saliency(self, ax, gradients, feature_names,
                               predicted_class, true_class, confidence):
        """Plot bar chart of top feature importances."""
        abs_gradients = np.abs(gradients)
        
        if len(abs_gradients) == 0 or abs_gradients.max() == 0:
            ax.text(0.5, 0.5, "No gradients available", ha='center', va='center')
            ax.axis('off')
            return

        norm_gradients = abs_gradients / abs_gradients.max()
        
        top_n = min(20, len(norm_gradients))
        indices = np.argsort(norm_gradients)[::-1][:top_n]

        top_feature_names = [feature_names[i] for i in indices[:top_n]]

        ax.set_yticks(np.arange(top_n))
        ax.set_yticklabels(top_feature_names)
        ax.set_xlabel('Relative Importance')
        
        pred_class_name = (self.class_names[predicted_class] 
                          if self.class_names and predicted_class < len(self.class_names) 
                          else f"Class {predicted_class}")
        true_class_name = (self.class_names[true_class] 
                          if self.class_names and true_class < len(self.class_names) 
                          else f"Class {true_class}")
        
        ax.set_title(f"Prediction: {pred_class_name} ({confidence:.2f}), True: {true_class_name}")
        ax.set_xlim(0, 1.1)

    def _plot_image_saliency(self, fig, ax_gridspec, image_data, image_gradients):
        """Plot saliency for 3D image data (sagittal, coronal, axial views)."""
        if len(image_data.shape) != 4:
            print(f"Warning: Expected 4D image data (C, H, W, D), got {image_data.shape}. Skipping image saliency plot.")
            ax = fig.add_subplot(ax_gridspec)
            ax.text(0.5, 0.5, "Image saliency N/A\n(Expected 4D data)", ha='center', va='center')
            ax.axis('off')
            return

        c, h, w, d = image_data.shape
        
        grad_magnitude = np.sum(np.abs(image_gradients), axis=0)
        
        mid_h = h // 2
        mid_w = w // 2
        mid_d = d // 2
        
        img_channel = image_data[0]
        
        gs_views = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=ax_gridspec, wspace=0.2)

        ax1 = fig.add_subplot(gs_views[0, 0])
        sagittal_slice = img_channel[:, mid_w, :]
        sagittal_grad = grad_magnitude[:, mid_w, :]
        
        ax1.imshow(sagittal_slice, cmap='gray')
        ax1.imshow(sagittal_grad, cmap='hot', alpha=0.5)
        ax1.set_title('Sagittal View')
        ax1.axis('off')

        ax2 = fig.add_subplot(gs_views[0, 1])
        coronal_slice = img_channel[mid_h, :, :]
        coronal_grad = grad_magnitude[mid_h, :, :]
        
        ax2.imshow(coronal_slice, cmap='gray')
        ax2.imshow(coronal_grad, cmap='hot', alpha=0.5)
        ax2.set_title('Coronal View')
        ax2.axis('off')

        ax3 = fig.add_subplot(gs_views[0, 2])
        axial_slice = img_channel[:, :, mid_d]
        axial_grad = grad_magnitude[:, :, mid_d]
        
        ax3.imshow(axial_slice, cmap='gray')
        ax3.imshow(axial_grad, cmap='hot', alpha=0.5)
        ax3.set_title('Axial View')
        ax3.axis('off')

    def process_batch(self, non_image_data_batch: torch.Tensor, image_data_batch: torch.Tensor, labels_batch: torch.Tensor,
                        output_dir: str, global_sample_offset: int, samples_to_process_in_this_batch: int) -> int:
        """
        Processes a single batch of data for Saliency Maps, generates visualizations, and updates stats.
        """
        batch_size = non_image_data_batch.size(0)
        num_processed_this_call = 0
        
        if not self.feature_columns and non_image_data_batch.shape[1] > 0 and not self.stats['feature_importance']:
            num_feats = non_image_data_batch.shape[1]
            self.feature_columns = [f"NonImageFeature_{j}" for j in range(num_feats)]
            self.stats['feature_importance'] = {name: 0.0 for name in self.feature_columns}
            logger.info(f"Initialized feature_columns for SaliencyMapGenerator: {self.feature_columns}")

        for i in range(min(batch_size, samples_to_process_in_this_batch)):
            current_global_sample_idx = global_sample_offset + i

            sample_non_image = non_image_data_batch[i:i+1].to(self.device)
            sample_image = image_data_batch[i:i+1].to(self.device) if image_data_batch is not None else None
            true_label = labels_batch[i].item()

            try:
                non_image_grad, image_grad, pred_class, confidence = self._get_gradients(
                    sample_non_image, sample_image
                )
                
                modality_importance = self._calculate_modality_importance(non_image_grad, image_grad)

                self.stats['total_samples'] += 1
                if pred_class == true_label:
                    self.stats['correct_predictions'] += 1
                
                self.stats['modality_importance']['non_image']['absolute'] += modality_importance['non_image']['absolute']
                self.stats['modality_importance']['image']['absolute'] += modality_importance['image']['absolute']
                self._temp_modality_percentage_sum['non_image'] += modality_importance['non_image']['percentage']
                self._temp_modality_percentage_sum['image'] += modality_importance['image']['percentage']

                fig = plt.figure(figsize=(15, 15))
                plt.subplots_adjust(hspace=0.4)
                
                if image_grad is not None:
                    gs = fig.add_gridspec(3, 1, height_ratios=[0.5, 1, 1.2])
                else:
                    gs = fig.add_gridspec(2, 1, height_ratios=[0.5, 1])

                ax_modality = fig.add_subplot(gs[0])
                self._plot_modality_comparison(fig, ax_modality, modality_importance)

                ax_features = fig.add_subplot(gs[1])
                self._plot_feature_saliency(
                    fig, ax_features,
                    sample_non_image[0].cpu().detach().numpy(),
                    non_image_grad[0].cpu().numpy(),
                    self.feature_columns,
                    pred_class, true_label, confidence
                )
                
                if image_grad is not None:
                    self._plot_image_saliency(
                        fig, gs[2],
                        sample_image[0].cpu().detach().numpy(),
                        image_grad[0].cpu().numpy()
                    )
                
                os.makedirs(output_dir, exist_ok=True)
                plt.savefig(os.path.join(output_dir, f"saliency_sample_{current_global_sample_idx}.png"), dpi=150)
                plt.close(fig)

                if self.feature_columns and self.stats['feature_importance']:
                    abs_grads_non_image = np.abs(non_image_grad[0].cpu().numpy())
                    for j, feature_name in enumerate(self.feature_columns):
                        if j < len(abs_grads_non_image):
                             self.stats['feature_importance'][feature_name] += abs_grads_non_image[j]
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
            
            non_image_abs_avg = self.stats['modality_importance']['non_image']['absolute'] / self.stats['total_samples']
            image_abs_avg = self.stats['modality_importance']['image']['absolute'] / self.stats['total_samples']
            self.stats['modality_importance']['non_image']['absolute'] = non_image_abs_avg
            self.stats['modality_importance']['image']['absolute'] = image_abs_avg

            self.stats['modality_importance']['non_image']['percentage'] = self._temp_modality_percentage_sum['non_image'] / self.stats['total_samples']
            self.stats['modality_importance']['image']['percentage'] = self._temp_modality_percentage_sum['image'] / self.stats['total_samples']

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
        plt.title('Top 20 Most Important Non-Image Features')
        plt.savefig(os.path.join(output_dir, 'non_image_feature_importance_summary.png'), dpi=150)
        plt.close(fig)

    def _plot_modality_importance_summary(self, modality_importance, output_dir):
        fig = plt.figure(figsize=(10, 6))
        modalities = ['Non-Image Features', 'Image Features']
        percentages = [
            modality_importance['non_image']['percentage'],
            modality_importance['image']['percentage']
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
            f"Avg abs importance - Non-Image: {modality_importance['non_image']['absolute']:.4f}, " +
            f"Image: {modality_importance['image']['absolute']:.4f}",
            ha='center', fontsize=9
        )
        plt.savefig(os.path.join(output_dir, 'modality_importance_summary.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    def cleanup(self):
        """Cleanup method for consistency with other generators."""
        logger.info("SaliencyMapGenerator cleanup completed.")
        pass 