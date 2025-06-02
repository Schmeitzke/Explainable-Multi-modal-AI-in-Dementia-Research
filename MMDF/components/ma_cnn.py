import torch
import torch.nn as nn
from MMDF import config
from MMDF.components.ese_block import ESEBlock1D

class MultiScaleConv1D(nn.Module):
    """
    Multi-scale convolution module with parallel branches of different kernel sizes.
    """
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConv1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels // 3, kernel_size=1, padding=0)
        self.conv3 = nn.Conv1d(in_channels, out_channels // 3, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_channels - 2 * (out_channels // 3), kernel_size=5, padding=2)
        
    def forward(self, x):
        out1 = self.conv1(x)
        out3 = self.conv3(x)
        out5 = self.conv5(x)
        return torch.cat([out1, out3, out5], dim=1)

class MA1DCNN(nn.Module):
    """
    Multi-scale Attention-embedded 1D CNN for clinical data.
    Implements the architecture described in the paper with multi-scale convolutions.
    """
    def __init__(self):
        super(MA1DCNN, self).__init__()
        in_channels = 1
        kernels = config.ma_conv_kernels
        kernel_sizes = config.ma_kernel_sizes
        pool = config.ma_pool_size
        
        self.stages = nn.ModuleList()
        
        for i, out_ch in enumerate(kernels):
            ks2 = kernel_sizes[i*2 + 1]
            
            multi_scale_conv = MultiScaleConv1D(in_channels, out_ch)
            bn1 = nn.BatchNorm1d(out_ch)
            relu1 = nn.ReLU(inplace=True)
            
            conv2 = nn.Conv1d(out_ch, out_ch, ks2, padding=ks2//2)
            bn2 = nn.BatchNorm1d(out_ch)
            relu2 = nn.ReLU(inplace=True)
            
            pool_layer = nn.MaxPool1d(pool)
            ese_block = ESEBlock1D(out_ch)
            
            stage = nn.Sequential(
                multi_scale_conv,
                bn1,
                relu1,
                conv2,
                bn2,
                relu2,
                pool_layer,
                ese_block
            )
            self.stages.append(stage)
            in_channels = out_ch
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_channels, config.ma_output_features_before_fusion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        
        for stage in self.stages:
            x = stage(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out
