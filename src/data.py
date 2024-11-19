import polars as pl
import torch

from torch.utils.data import Dataset
from transformers import BatchEncoding


class StoryDataset(Dataset):
    def __init__(self, tokenizer, df_path="datasets/tiny-stories.gzip"):
        self.texts = pl.read_parquet(df_path)["text"]
        self.tokenizer = tokenizer

        self.bos_token = self.tokenizer.cls_token_id
        self.eos_token = self.tokenizer.sep_token_id
        self.pad_token = self.tokenizer.pad_token_id

    def __len__(self):
        return len(self.texts)

    def collate(self, text):
        return self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")

    def __getitem__(self, idx):
        return self.texts[idx]
