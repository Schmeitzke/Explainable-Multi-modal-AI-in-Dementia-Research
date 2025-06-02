import torch.nn as nn

class PatchCNN(nn.Module):
    """
    PatchCNN for processing 3D image patches in the ADTransformer model.
    
    This component uses a shared CNN to process all patches, which is the standard
    approach in transformer vision models. All patches are processed through the
    same CNN weights, then reshaped into patch embeddings.
    """
    def __init__(self, in_channels=1, patch_size=48, num_patches=36, 
                 conv_channels=None, conv_kernel_size=3, pool_size=2, embed_dim=64):
        super(PatchCNN, self).__init__()
        if conv_channels is None:
            conv_channels = [8, 16, 32, 64]
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size,)*3
        
        self.expected_num_patches = num_patches
        self.embed_dim = embed_dim

        self.shared_cnn = self._make_shared_cnn(in_channels, conv_channels, conv_kernel_size, pool_size)
        
        self.proj = None
        if conv_channels[-1] != embed_dim:
            self.proj = nn.Linear(conv_channels[-1], embed_dim)
    
    def _make_shared_cnn(self, in_channels, conv_channels, conv_kernel_size, pool_size):
        """Create a shared CNN that processes all patches with the same weights"""
        layers = []
        input_ch = in_channels
        for out_ch in conv_channels:
            pad = conv_kernel_size // 2 if isinstance(conv_kernel_size, int) else tuple(k // 2 for k in conv_kernel_size)
            layers.append(nn.Conv3d(input_ch, out_ch, kernel_size=conv_kernel_size, padding=pad))
            layers.append(nn.BatchNorm3d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool3d(kernel_size=pool_size, stride=pool_size))
            input_ch = out_ch
        layers.append(nn.AdaptiveAvgPool3d(output_size=1))
        layers.append(nn.Flatten())
        return nn.Sequential(*layers)
    
    def forward(self, x):
        B, C, H, W, D = x.shape
        ph, pw, pd = self.patch_size
        assert H % ph == 0 and W % pw == 0 and D % pd == 0, "Image dimensions must be divisible by patch_size"
        nH, nW, nD = H // ph, W // pw, D // pd
        total_patches = nH * nW * nD
        
        if total_patches != self.expected_num_patches:
            print(f"WARNING: Configuration specified {self.expected_num_patches} patches but actual patches is {total_patches}")
            print(f"The model will use {total_patches} patches based on image dimensions {(H, W, D)} and patch size {(ph, pw, pd)}")
        
        x_reshaped = x.view(B, C, nH, ph, nW, pw, nD, pd)
        x_reshaped = x_reshaped.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        x_patches = x_reshaped.view(B, total_patches, C, ph, pw, pd)
        
        x_patches_flat = x_patches.view(B * total_patches, C, ph, pw, pd)
        
        patch_features = self.shared_cnn(x_patches_flat)
        
        if self.proj is not None:
            patch_features = self.proj(patch_features)
        
        tokens = patch_features.view(B, total_patches, self.embed_dim)
        return tokens
