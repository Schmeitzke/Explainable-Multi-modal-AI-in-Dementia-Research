import torch
import torch.nn as nn

class NonImageEmbedding(nn.Module):
    def __init__(self, num_features, embed_dim):
        super(NonImageEmbedding, self).__init__()
        self.num_features = num_features
        self.linear = nn.Linear(1, embed_dim)
    
    def forward(self, x):
        assert x.size(1) == self.num_features, f"Expected {self.num_features} features, got {x.size(1)}"
        x_expanded = x.unsqueeze(-1)
        embeddings = self.linear(x_expanded)
        return embeddings

class PositionalEncoding(nn.Module):
    def __init__(self, num_tokens, embed_dim):
        super(PositionalEncoding, self).__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
    
    def forward(self, x):
        return x + self.pos_embed
