import torch
import torch.nn as nn
from MMDF import config
from MMDF.components.ma_cnn import MA1DCNN
from MMDF.components.mri_vit import MRI_ViT
from typing import Tuple, Optional

class MMDF(nn.Module):
    """
    Multimodal Medical Data Fusion model combining MA-1DCNN and MRI_ViT.
    Includes an auxiliary head for the image modality.
    """
    def __init__(self):
        super(MMDF, self).__init__()
        self.clinical_net = MA1DCNN()
        self.image_net = MRI_ViT()

        fusion_dims = config.fusion_hidden_dims
        dropout = config.fusion_dropout
        fusion_in_dim = config.ma_output_features_before_fusion + config.vit_output_features_before_fusion

        layers = []
        in_dim = fusion_in_dim
        for dim in fusion_dims:
            layers += [nn.Linear(in_dim, dim), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            in_dim = dim
        layers.append(nn.Linear(in_dim, config.num_classes))
        self.fusion_head = nn.Sequential(*layers)

        self.image_aux_head = nn.Linear(config.vit_output_features_before_fusion, config.num_classes)

        self.clinical_aux_head = nn.Linear(config.ma_output_features_before_fusion, config.num_classes)


    def forward(self, clinical_data: Optional[torch.Tensor], mri_data: Optional[torch.Tensor]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass for the MMDF model.
        Can handle scenarios where one modality might be missing (for single-modality training).

        Args:
            clinical_data: Tensor of shape (B, num_clinical_features) or None.
            mri_data: Tensor of shape (B, 3, H, W) or None.

        Returns:
            A tuple containing:
            - main_logits: Logits from the fusion head (B, num_classes) or None if not applicable.
            - image_only_logits: Logits from the image auxiliary head (B, num_classes) or None.
            - clinical_only_logits: Logits from the clinical auxiliary head (B, num_classes) or None.
            - last_attn_weights: Attention weights from the last ViT block or None.
        """
        feat1: Optional[torch.Tensor] = None
        feat2: Optional[torch.Tensor] = None
        last_attn_weights: Optional[torch.Tensor] = None
        main_logits: Optional[torch.Tensor] = None
        image_only_logits: Optional[torch.Tensor] = None
        clinical_only_logits: Optional[torch.Tensor] = None

        if clinical_data is not None:
            feat1 = self.clinical_net(clinical_data)
            clinical_only_logits = self.clinical_aux_head(feat1)

        if mri_data is not None:
            feat2, last_attn_weights = self.image_net(mri_data)
            image_only_logits = self.image_aux_head(feat2)
        
        if feat1 is not None and feat2 is not None:
            x = torch.cat([feat1, feat2], dim=1)
            main_logits = self.fusion_head(x)
        elif feat1 is not None and config.training_mode == "full": 
            pass 
        elif feat2 is not None and config.training_mode == "full":
            pass 

        return main_logits, image_only_logits, clinical_only_logits, last_attn_weights
