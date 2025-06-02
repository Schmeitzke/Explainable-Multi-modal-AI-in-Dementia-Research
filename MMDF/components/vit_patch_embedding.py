import torch
import torch.nn as nn
from MMDF import config
import logging

logger = logging.getLogger(__name__)

class PatchPositionEmbedding(nn.Module):
    """
    Combines patch embedding and positional embeddings for MRI_ViT.
    """
    def __init__(self):
        super(PatchPositionEmbedding, self).__init__()
        img_h, img_w = config.mri_image_size
        p = config.vit_patch_size
        self.embed_dim = config.vit_embed_dim
        
        self.img_h_w_tuple = config.mri_image_size
        self.p_val = config.vit_patch_size
        
        self.num_patches = (img_h // p) * (img_w // p)
        logger.info(f"Calculated num_patches: {self.num_patches} for image size {img_h}x{img_w} and patch size {p}")
        
        self.proj = nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=p,
            stride=p
        )
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.embed_dim)
        )
        
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        logger.info(f"Initialized pos_embed with shape: {self.pos_embed.shape}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3, H, W)
        Returns:
            Tensor of shape (batch, num_patches+1, embed_dim)
        """
        B = x.size(0)
        logger.debug(f"Input shape: {x.shape}")
        
        x = self.proj(x)
        logger.debug(f"After projection: {x.shape}")
        
        x = x.flatten(2)
        x = x.transpose(1, 2)
        logger.debug(f"After flatten and transpose: {x.shape}")
        
        actual_num_patches = x.shape[1]
        if actual_num_patches != self.num_patches:
            raise RuntimeError(f"Expected {self.num_patches} patches but got {actual_num_patches}. "
                             f"Check image size {self.img_h_w_tuple} and patch size {self.p_val}")
        
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        logger.debug(f"After adding CLS token: {x.shape}")
        
        if x.shape[1] != self.pos_embed.shape[1]:
            raise RuntimeError(f"Sequence length mismatch: x has {x.shape[1]} tokens but pos_embed has {self.pos_embed.shape[1]}")
        
        x = x + self.pos_embed
        logger.debug(f"PatchPositionEmbedding Output: Shape={x.shape}, Device={x.device}, RequiresGrad={x.requires_grad}")
        return x
