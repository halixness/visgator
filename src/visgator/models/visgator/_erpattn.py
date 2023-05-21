"""
    For testing:
        size = 256
        x = torch.randn((10, 5, size))
        y = torch.zeros_like(x)
        attn = ERPAttention(input_size=size, hidden_size=256, heads=8)
        y = attn(x)
        bbox_token = y[0]

        print(y.shape) 
        print(bbox_token) 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MultiHeadAttention(nn.Module):
    """ Basic multi head self attention """
    def __init__(
            self, 
            input_size, 
            hidden_size,
            heads
        ):
        super(MultiHeadAttention, self).__init__()

        self.dq = self.dk = self.dv = hidden_size

        self.Wq = nn.Linear(input_size, self.dq * heads)
        self.Wk = nn.Linear(input_size, self.dk * heads)
        self.Wv = nn.Linear(input_size, self.dv * heads)
        self.Wo = nn.Linear(self.dv * heads, self.dv)

    def forward(self, x) -> torch.Tensor:
        z = F.softmax(self.Wq(x) @ torch.t(self.Wk(x)), dim=-1) / (self.dk)**(1/2) 
        z = z @ self.Wv(x)
        return self.Wo(z)
    

class ERPAttentionBlock(nn.Module):
    """ An encoder block (self attention) """
    def __init__(
            self, 
            input_size,
            hidden_size,
            heads
        ):
        super(ERPAttentionBlock, self).__init__()

        self.MHA = MultiHeadAttention(input_size, hidden_size, heads)
        self.lnorm = nn.LayerNorm([hidden_size])
        self.ffn = nn.Linear(hidden_size, hidden_size)

    def forward(self, x) -> torch.Tensor:
        z = self.MHA(x)
        z = self.lnorm(z + x) 
        return self.lnorm(self.ffn(z) + z)
        
        
class ERPAttention(nn.Module):
    """ Layered encoder blocks """
    def __init__(
            self, 
            input_size,
            hidden_size = 256,
            heads = 8,
            layers = 1
        ):
        super(ERPAttention, self).__init__()
        
        self.layers = nn.ModuleList(
            [ERPAttentionBlock(input_size, hidden_size, heads) for l in range(layers)]
        )
        self.bbox_token = torch.nn.Parameter(torch.randn(1, input_size))
        self.bbox_token.requires_grad = True

    def sentence_sin_embedding(self, shape):
        """ Basic sentence-wise sinusoidal embedding """
        sentences, tokens, embedding_size = shape
        embeddings = torch.zeros(shape)

        for pos in range(sentences):
            for t in range(tokens):
                for i in range(embedding_size):
                    if i % 2 == 0: embeddings[pos, t, i] = np.sin(pos/10000**(2*i / embedding_size))
                    else: embeddings[pos, t, i] = np.cos(pos/10000**(2*i / embedding_size))
        return embeddings

    def forward(self, x) -> torch.Tensor:

        x += self.sentence_sin_embedding(x.shape)
        x = torch.flatten(x, start_dim=0, end_dim=1)
        x = torch.concat((self.bbox_token, x))
        
        for l in self.layers:
            x = l(x)
        return x
