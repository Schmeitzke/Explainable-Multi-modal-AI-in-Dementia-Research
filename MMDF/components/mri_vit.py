import torch
import torch.nn as nn
from MMDF import config
from MMDF.components.vit_patch_embedding import PatchPositionEmbedding
from MMDF.components.transformer_encoder import TransformerEncoderBlock
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class MRI_ViT(nn.Module):
    """
    Vision Transformer for MRI slices.
    """
    def __init__(self):
        super(MRI_ViT, self).__init__()
        embed_dim = config.vit_embed_dim
        depth = config.vit_depth
        num_heads = config.vit_num_heads
        mlp_dim = config.vit_mlp_dim
        mlp_dropout = config.vit_mlp_dropout
        out_feats = config.vit_output_features_before_fusion

        self.patch_embed = PatchPositionEmbedding()
        self.encoders = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim, mlp_dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, out_feats)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass for MRI_ViT.
        Args:
            x: Input tensor of shape (batch, 3, H, W)
        Returns:
            A tuple containing:
            - out: Output features from the head (B, out_feats).
            - last_attn_weights: Attention weights from the last encoder block (B, num_heads, N+1, N+1) or None if depth=0.
        """
        logger.debug(f"MRI_ViT Input: Shape={x.shape}, Device={x.device}, RequiresGrad={x.requires_grad}")
        x = self.patch_embed(x)
        logger.debug(f"  After PatchEmbed: Shape={x.shape}, RequiresGrad={x.requires_grad}")
        
        last_attn_weights = None
        for i, blk in enumerate(self.encoders):
            logger.debug(f"    Entering Encoder Block {i}")
            x, attn_weights = blk(x)
            logger.debug(f"    Output from Encoder Block {i}: Shape={x.shape}, RequiresGrad={x.requires_grad}")
            if i == len(self.encoders) - 1:
                last_attn_weights = attn_weights
            
        x = self.norm(x)
        logger.debug(f"  After final Norm: Shape={x.shape}, RequiresGrad={x.requires_grad}")
        cls = x[:, 0]
        logger.debug(f"  CLS token extracted: Shape={cls.shape}, RequiresGrad={cls.requires_grad}")
        out = self.head(cls)
        logger.debug(f"MRI_ViT Output (after head): Shape={out.shape}, Device={out.device}, RequiresGrad={out.requires_grad}")
        
        return out, last_attn_weights
