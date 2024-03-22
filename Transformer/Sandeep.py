import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size(-1)
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    attention = F.softmax(scaled, dim=-1)
    if mask is not None:
        attention = attention.masked_fill(mask == float('-inf'), 0.0)
    values = torch.matmul(attention, v)
    return values, attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model , 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, mask):
        batch_size, sequence_length, d_model = x.size()
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.permute(0, 2, 1, 3)
        out = self.linear_layer(values.reshape(batch_size, sequence_length, -1))
        return out


class PositionwiseFeedForward():
    values=0

class MultiHeadCrossAttention():
    values=0
