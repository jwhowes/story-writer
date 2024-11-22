import torch

from torch import nn

from .util import Block, generate_theta


class GPT(nn.Module):
    def __init__(
            self, vocab_size, d_model, n_layers, n_heads, max_length=512,
            attn_dropout=0.0, resid_dropout=0.0, norm_eps=1e-6, base_theta=10000
    ):
        super(GPT, self).__init__()
        self.register_buffer(
            "theta",
            generate_theta(d_model, n_heads, max_length, base_theta=base_theta),
            persistent=False
        )

        self.register_buffer(
            "mask",
            torch.triu(
                torch.full((max_length, max_length), float('-inf')),
                diagonal=1
            ),
            persistent=False
        )

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(Block(d_model, n_heads, attn_dropout, resid_dropout, norm_eps))

        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens):
        B, L = tokens.shape
        x = self.embedding(tokens)

        for layer in self.layers:
            x = layer(x, attention_mask=self.mask[:L, :L], theta=self.theta[:L])

        return self.head(x)
