import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn import Linear

embed_dim=512
encoder_ffn_embed_dim = 1024
encoder_layers=3

class Atten(nn.Module):
    def __init__(self):
        super(Atten, self).__init__()

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x):
        return x

def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()

        self.self_atten = Atten()
        self.fc1 = Linear(embed_dim, encoder_ffn_embed_dim)
        self.fc2 = Linear( encoder_ffn_embed_dim,  embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(embed_dim) for i in range(2)])

    def forward(self, x):
        return x


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()

        self.self_atten = Atten()
        self.fc1 = Linear(embed_dim, encoder_ffn_embed_dim)
        self.fc2 = Linear( encoder_ffn_embed_dim,  embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(embed_dim) for i in range(2)])

    def forward(self, x):
        return x


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList([])
        self.layers.extend([
            EncoderLayer()
            for i in range( encoder_layers)
        ])

    def forward(self, x):
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList([])
        self.layers.extend([
            DecoderLayer()
            for i in range( encoder_layers)
        ])

    def forward(self, x):
        return x


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()

        self.lll=Linear(2,3)

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        return x



