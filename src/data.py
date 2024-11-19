import polars as pl
import numpy as np

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class StoryDataset(Dataset):
    def __init__(self, tokenizer, df_path="datasets/tiny-stories.gzip"):
        self.texts = pl.read_parquet(df_path)["text"]
        self.tokenizer = tokenizer

        self.bos_token = self.tokenizer.cls_token_id
        self.eos_token = self.tokenizer.sep_token_id
        self.pad_token = self.tokenizer.pad_token_id

    def __len__(self):
        return len(self.texts)

    def collate(self, tokens):
        return pad_sequence(tokens, batch_first=True, padding_value=0)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.texts[idx], return_tensors="pt")["input_ids"].squeeze()

        L = tokens.shape[0]
        if L > self.tokenizer.model_max_length:
            start_idx = np.random.randint(0, L - self.tokenizer.model_max_length + 1)
            tokens = tokens[start_idx: start_idx + self.tokenizer.model_max_length]

        return tokens
