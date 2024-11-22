import torch
import torch.nn.functional as F

from torch import nn
from diffusers import DDPMScheduler

from .util import SwiGLU, Attention, generate_theta
from math import sqrt


class RMSFiLM(nn.Module):
    def __init__(self, d_model, d_t, eps=1e-6):
        super(RMSFiLM, self).__init__()
        self.eps = eps

        self.gamma = nn.Linear(d_t, d_model)

    def forward(self, x, t):
        B = x.shape[0]
        g = self.gamma(t).view(B, 1, -1)

        return x * g * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class SinusoidalEmbedding(nn.Module):
    def __init__(self, d_model, base_theta=10000):
        super(SinusoidalEmbedding, self).__init__()
        assert d_model % 2 == 0
        self.register_buffer(
            "theta",
            1.0 / (base_theta ** (2 * torch.arange(d_model // 2) / d_model)),
            persistent=False
        )

    def forward(self, x):
        B = x.shape[0]
        x = x.view(-1, 1).float() * self.theta

        return torch.stack((
            torch.cos(x),
            torch.sin(x)
        ), dim=-1).view(B, -1)


class TimeConditionalBlock(nn.Module):
    def __init__(self, d_model, d_t, n_heads, norm_eps=1e-6, attn_dropout=0.0, resid_dropout=0.0):
        super(TimeConditionalBlock, self).__init__()
        self.attn = Attention(d_model, n_heads, attn_dropout)
        self.attn_norm = RMSFiLM(d_model, d_t, eps=norm_eps)

        self.ffn = SwiGLU(d_model)
        self.ffn_norm = RMSFiLM(d_model, d_t, eps=norm_eps)
        self.ffn_dropout = nn.Dropout(resid_dropout)

    def forward(self, x, t, theta=None):
        x = x + self.attn(
            self.attn_norm(x, t), theta=theta
        )

        return x + self.ffn_dropout(self.ffn(self.ffn_norm(x, t)))


class DiffusionEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(DiffusionEmbedding, self).__init__()
        self.scale = sqrt(d_model)

        self.emb = nn.Embedding(vocab_size, d_model)

    def interpolate(self, logits, norm=True):
        x = F.softmax(logits, dim=-1) @ self.emb.weight

        if norm:
            x = F.normalize(x, dim=-1, p=2) * self.scale

        return x

    def forward(self, x, norm=True):
        x = self.emb(x)

        if norm:
            x = F.normalize(x, dim=-1, p=2) * self.scale

        return x


class DiffusionModel(nn.Module):
    def __init__(
            self, vocab_size, d_model, d_t, n_layers, n_heads, max_length=512,
            attn_dropout=0.0, resid_dropout=0.0, norm_eps=1e-6, base_theta=10000
    ):
        super(DiffusionModel, self).__init__()
        self.register_buffer(
            "theta",
            generate_theta(d_model, n_heads, max_length, base_theta=base_theta),
            persistent=False
        )

        self.t_model = nn.Sequential(
            SinusoidalEmbedding(d_t, base_theta),
            SwiGLU(d_t)
        )

        self.embedding = DiffusionEmbedding(vocab_size, d_model // 2)
        self.scheduler = DDPMScheduler()
        self.T = self.scheduler.num_train_timesteps

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(TimeConditionalBlock(d_model, d_t, n_heads, norm_eps, attn_dropout, resid_dropout))

        self.head = nn.Linear(d_model, vocab_size)

    def pred_logits(self, z_t, clean, t):
        x = torch.concatenate((
            z_t,
            clean
        ), dim=-1)
        t_emb = self.t_model(t)

        for layer in self.layers:
            x = layer(x, t_emb, theta=self.theta)

        return self.head(x)

    def forward(self, tokens, clean_mask):
        B = tokens.shape[0]

        t = torch.randint(0, self.T, (B,), device=tokens.device)

        z_0 = self.embedding(tokens)
        noise = torch.randn_like(z_0)
        z_t = self.scheduler.add_noise(z_0, noise, t)

        z_0[~clean_mask] = 0.0

        logits = self.pred_logits(z_t, z_0, t)

        return F.cross_entropy(logits.transpose(1, 2), tokens)
