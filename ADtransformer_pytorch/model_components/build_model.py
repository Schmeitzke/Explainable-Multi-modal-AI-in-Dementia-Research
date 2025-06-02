import yaml
import torch
import torch.nn as nn
import logging

from model_components.embeddings import NonImageEmbedding, PositionalEncoding
from model_components.transformer_encoder import TransformerEncoder
from model_components.classifier import ClassificationHead
from model_components.patch_cnn import PatchCNN

def build_adni_transformer(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    nonimg_cfg = config["non_image"]
    img_cfg = config["image"]
    trans_cfg = config["transformer"]
    clf_cfg = config["classifier"]
    
    embed_dim = nonimg_cfg["embed_dim"]
    
    if img_cfg.get("use_images", False) and img_cfg["embed_dim"] != embed_dim:
        logging.warning(f"Image embed_dim ({img_cfg['embed_dim']}) differs from non_image embed_dim ({embed_dim}). "
                       f"Using non_image embed_dim ({embed_dim}) for consistency.")
        img_cfg["embed_dim"] = embed_dim
    
    use_images = img_cfg.get("use_images", False)
    
    non_image_emb = NonImageEmbedding(num_features=nonimg_cfg["num_features"], 
                                     embed_dim=embed_dim)
    
    image_emb = None
    if use_images:
        logging.info(f"[build_model.py] Initializing PatchCNN with in_channels: {img_cfg.get('in_channels')}")
        image_emb = PatchCNN(in_channels=img_cfg["in_channels"],
                           patch_size=img_cfg["patch_size"],
                           num_patches=img_cfg["num_patches"],
                           conv_channels=img_cfg["conv_channels"],
                           embed_dim=embed_dim)

    pos_enc = None

    transformer = TransformerEncoder(embed_dim=embed_dim,
                                   num_heads=trans_cfg["num_heads"],
                                   ffn_dim=trans_cfg["ffn_dim"],
                                   num_layers=trans_cfg["num_layers"],
                                   dropout=trans_cfg.get("dropout", 0.0))

    classifier = None

    model = ADTransformer(non_image_emb, image_emb, pos_enc, transformer, classifier, 
                         use_images, 
                         embed_dim, 
                         nonimg_cfg["num_features"],
                         clf_cfg["hidden_dim"],
                         clf_cfg["output_dim"],
                         clf_cfg.get("dropout", 0.0))
    
    return model

class ADTransformer(nn.Module):
    """
    ADTransformer model for Alzheimer's Disease classification.
    
    Note: This model uses a non-standard lazy initialization pattern for positional
    encoding and classifier components. These are intentionally initialized during the 
    first forward pass once the total number of tokens
    is known, allowing for dynamic model structure based on data characteristics.
    """
    def __init__(self, non_image_emb, image_emb, pos_enc, transformer, classifier, 
                 use_images=False, embed_dim=64, non_image_features=34, 
                 clf_hidden_dim=128, clf_output_dim=3, dropout=0.5):
        super(ADTransformer, self).__init__()
        self.non_image_emb = non_image_emb
        self.image_emb = image_emb
        self.pos_enc = pos_enc
        self.transformer = transformer
        self.classifier = classifier
        self.use_images = use_images
        
        self.embed_dim = embed_dim
        self.non_image_features = non_image_features
        self.clf_hidden_dim = clf_hidden_dim
        self.clf_output_dim = clf_output_dim
        self.dropout = dropout
        
        self.total_tokens = None
    
    def _init_positional_encoding(self, total_tokens):
        """Initialize positional encoding with the correct token count"""
        self.pos_enc = PositionalEncoding(num_tokens=total_tokens, embed_dim=self.embed_dim)
        if next(self.parameters(), None) is not None:
            device = next(self.parameters()).device
            self.pos_enc = self.pos_enc.to(device)
        
    def _init_classifier(self, total_tokens):
        """Initialize classifier with the correct input dimension"""
        classifier_input_dim = total_tokens * self.embed_dim
        self.classifier = ClassificationHead(
            input_dim=classifier_input_dim,
            hidden_dim=self.clf_hidden_dim,
            output_dim=self.clf_output_dim,
            dropout=self.dropout
        )
        if next(self.parameters(), None) is not None:
            device = next(self.parameters()).device
            self.classifier = self.classifier.to(device)
    
    def forward(self, non_image=None, image=None):
        non_img_tokens = self.non_image_emb(non_image)
        
        if self.use_images and image is not None:
            img_tokens = self.image_emb(image)
            
            if non_img_tokens.size(-1) != img_tokens.size(-1):
                raise RuntimeError(f"Embedding dimension mismatch: non_image tokens have dim {non_img_tokens.size(-1)}, "
                                 f"but image tokens have dim {img_tokens.size(-1)}. "
                                 f"Both must have embed_dim={self.embed_dim}")
            
            tokens = torch.cat([non_img_tokens, img_tokens], dim=1)
        else:
            tokens = non_img_tokens
        
        total_tokens = tokens.size(1)
        if self.pos_enc is None or self.total_tokens != total_tokens:
            self.total_tokens = total_tokens
            self._init_positional_encoding(total_tokens)
            self._init_classifier(total_tokens)
            print(f"Initialized model with {total_tokens} total tokens")
        
        tokens = self.pos_enc(tokens)
        
        encoded_tokens = self.transformer(tokens)
        
        fused_rep = encoded_tokens.view(encoded_tokens.size(0), -1)
        
        logits = self.classifier(fused_rep)
        return logits
