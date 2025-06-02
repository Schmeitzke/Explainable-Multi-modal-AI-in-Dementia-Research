import torch
import torch.nn as nn
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class TransformerEncoderBlock(nn.Module):
    """
    Single Transformer Encoder block with MSA and MLP, including LayerNorm and residuals.
    """
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 mlp_dim: int,
                 mlp_dropout: float):
        super(TransformerEncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim,
                                          num_heads,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(mlp_dropout)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y = self.norm1(x)
        attn_out, attn_weights = self.attn(y, y, y, need_weights=True)
        x = x + attn_out
        y2 = self.norm2(x)
        mlp_out = self.mlp(y2)
        x = x + mlp_out
        return x, attn_weights
