from typing import Tuple, List

num_clinical_features: int = 94
mri_image_size: Tuple[int, int] = (144, 144)

perform_binary_classification_cn_ad: bool = True
binary_class_names: List[str] = ["CN", "AD"]
multiclass_class_names: List[str] = ["CN", "MCI", "AD"]

class_names: List[str] = binary_class_names if perform_binary_classification_cn_ad else multiclass_class_names
num_classes: int = len(class_names)

train_csv_path: str = ""
test_csv_path: str = ""
image_data_base_dir: str = ""
image_modalities: List[str] = ["T1", "MPRAGE", "FLAIR"]

ma_conv_kernels: List[int] = [32, 64, 128, 64]
ma_kernel_sizes: List[int] = [3, 5, 1, 5, 3, 5, 1, 5]
ma_pool_size: int = 2
ma_ese_reduction_ratio: int = 4
ma_output_features_before_fusion: int = 3

vit_patch_size: int = 6
vit_embed_dim: int = 64
vit_depth: int = 8
vit_num_heads: int = 8
vit_mlp_dim: int = vit_embed_dim * 4
vit_mlp_dropout: float = 0.1
vit_output_features_before_fusion: int = 3

fusion_hidden_dims: List[int] = [64, 32]
fusion_dropout: float = 0.1

loss_sce_alpha: float = 1.0
loss_sce_beta: float = 1.0
loss_efl_gamma: float = 2.0
loss_adaptive_weight_alpha: float = 0.01
loss_initial_weight1: float = 0.5
loss_initial_weight2: float = 0.5

aux_loss_weight: float = 0.4

guidance_loss_weight: float = 0.3

learning_rate: float = 0.001
batch_size: int = 16
epochs: int = 10
optimizer: str = 'Adam'
lr_scheduler: str = 'CosineDecay'
weight_decay: float = 0.0

training_mode: str = "full"  # Options: "full", "image_only", "clinical_only"