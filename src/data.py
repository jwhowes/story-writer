import polars as pl
import numpy as np
import torch.nn.functional as F
import torch

from torch.utils.data import Dataset
from random import random


class StoryDataset(Dataset):
    def __init__(self, tokenizer, df_path="datasets/tiny-stories.gzip", p_uncond=0.1, p_autoreg=0.5):
        self.p_uncond = p_uncond
        self.p_autoreg = p_autoreg

        self.texts = pl.read_parquet(df_path)["text"]
        self.tokenizer = tokenizer
        self.L = self.tokenizer.model_max_length

        self.bos_token = self.tokenizer.cls_token_id
        self.eos_token = self.tokenizer.sep_token_id
        self.pad_token = self.tokenizer.pad_token_id

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizer(
            self.texts[idx],
            return_tensors="pt",
            add_special_tokens=False
        )["input_ids"].squeeze()

        L = tokens.shape[0]
        if L >= self.L - 2:
            start_idx = np.random.randint(0, L - self.L - 1)
            tokens = tokens[start_idx: start_idx + self.L - 2]
        else:
            tokens = F.pad(tokens, (self.L - L - 2, 0), value=self.pad_token)

        tokens = torch.concatenate((
            torch.full((1,), self.bos_token),
            tokens,
            torch.full((1,), self.eos_token)
        ))

        clean_mask = torch.zeros(self.L, dtype=torch.bool)
        if random() >= self.p_uncond:
            num_clean = np.random.randint(self.L)
            if random() < self.p_autoreg:
                indices = torch.arange(self.L)[:num_clean]
            else:
                indices = torch.randperm(self.L)[:num_clean]

            clean_mask[indices] = True

        return tokens, clean_mask
