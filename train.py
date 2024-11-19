import torch
import warnings

from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from accelerate import Accelerator
from torch import nn

from src.data import StoryDataset
from src.model import GPT


def train(model, dataloader):
    num_epochs = 5

    accelerator = Accelerator()

    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    lr_scheduler = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=0,
        num_training_steps=num_epochs * len(dataloader)
    )

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    model, dataloader, opt, lr_scheduler = accelerator.prepare(
        model, dataloader, opt, lr_scheduler
    )

    for epoch in range(num_epochs):
        print(f"EPOCH {epoch + 1} / {num_epochs}")
        for i, tokens in enumerate(dataloader):
            opt.zero_grad()

            logits = model(tokens[:, :-1])

            loss = criterion(logits.transpose(1, 2), tokens[:, 1:])
            loss.backward()

            opt.step()
            lr_scheduler.step()

            if i % 10 == 0:
                print(f"\t{i} / {len(dataloader)} iters.\tLoss: {loss.item():.4f}")


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

    dataset = StoryDataset(tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=24,
        shuffle=True,
        pin_memory=True,
        collate_fn=dataset.collate
    )

    model = GPT(
        vocab_size=tokenizer.vocab_size,
        d_model=768,
        n_layers=12,
        n_heads=12,
        max_length=tokenizer.model_max_length
    )

    train(model, dataloader)
