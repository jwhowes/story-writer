import torch
import torch.nn.functional as F

from torch import nn
from math import sqrt


def generate_theta(d_model, n_heads, max_length, base_theta=10000):
    assert d_model % n_heads == 0
    assert (d_model // n_heads) % 2 == 0
    d_attn = d_model // n_heads
    theta = (
            1 / (base_theta ** (2 * torch.arange(d_attn // 2) / d_attn))
    )
    theta = torch.outer(torch.arange(max_length), theta)

    return torch.polar(torch.ones_like(theta), theta)


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):

        return x * self.weight * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_size=None):
        super(SwiGLU, self).__init__()
        if hidden_size is None:
            hidden_size = 4 * d_model

        self.gate_proj = nn.Linear(d_model, hidden_size, bias=False)
        self.hidden_proj = nn.Linear(d_model, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, d_model, bias=False)

    def forward(self, x):
        return self.out_proj(
            F.silu(self.gate_proj(x)) * self.hidden_proj(x)
        )


class Attention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super(Attention, self).__init__()
        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.scale = sqrt(d_model // n_heads)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def rope(x, theta):
        B, L, n_heads, _ = x.shape
        x = torch.view_as_complex(
            x.view(B, L, n_heads, -1, 2)
        )

        return torch.view_as_real(
            x * theta.view(1, L, 1, -1)
        ).flatten(3)

    def forward(self, x, attention_mask=None, theta=None):
        B, L, _ = x.shape

        q = self.W_q(x).view(B, L, self.n_heads, -1)
        k = self.W_k(x).view(B, L, self.n_heads, -1)
        v = self.W_v(x).view(B, L, self.n_heads, -1).transpose(1, 2)

        if theta is not None:
            q = self.rope(q, theta)
            k = self.rope(k, theta)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / self.scale

        if attention_mask is not None:
            attn = attn + attention_mask.view(-1, 1, L, L)

        x = self.dropout(
            F.softmax(attn, dim=-1)
        ) @ v

        return self.W_o(
            x
            .transpose(1, 2)
            .contiguous()
            .view(B, L, -1)
        )


class Block(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout=0.0, resid_dropout=0.0, norm_eps=1e-6):
        super(Block, self).__init__()
        self.attn = Attention(d_model, n_heads, attn_dropout)
        self.attn_norm = RMSNorm(d_model, norm_eps)

        self.ffn = SwiGLU(d_model)
        self.ffn_dropout = nn.Dropout(resid_dropout)
        self.ffn_norm = RMSNorm(d_model, norm_eps)

    def forward(self, x, attention_mask=None, theta=None):
        x = x + self.attn(self.attn_norm(x), attention_mask=attention_mask, theta=theta)

        return x + self.ffn_dropout(self.ffn(self.ffn_norm(x)))
