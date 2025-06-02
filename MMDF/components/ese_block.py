import torch
import torch.nn as nn
from MMDF import config

class ESEBlock1D(nn.Module):
    """
    Enhanced Squeeze-and-Excitation (ESE) block for 1D feature maps.
    """
    def __init__(self, channels: int):
        super(ESEBlock1D, self).__init__()
        self.channels = channels
        reduction = config.ma_ese_reduction_ratio
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // reduction, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for ESEBlock1D.
        Args:
            x: Input tensor of shape (batch, channels, length)
        Returns:
            Reweighted tensor of same shape.
        """
        w = self.avg_pool(x)
        w = self.fc(w)
        return x * w
