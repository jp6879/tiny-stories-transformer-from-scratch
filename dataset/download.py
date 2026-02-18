import os
import torch
from torch.utils.data import Dataset


def download_tinystories(data_dir="data", num_train=10000, num_val=1000):
    """Grab a subset of TinyStories from HuggingFace and cache it locally."""

    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, "train.txt")
    val_path = os.path.join(data_dir, "val.txt")

    if os.path.exists(train_path) and os.path.exists(val_path):
        print("Dataset already downloaded, loading from cache...")
        train_text = open(train_path, encoding="utf-8").read()
        val_text = open(val_path, encoding="utf-8").read()
        print(f"  {len(train_text):,} train chars, {len(val_text):,} val chars")
        return train_text, val_text

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Need the 'datasets' library to download TinyStories.\n"
            "Run: pip install datasets"
        )

    print(f"Downloading TinyStories (train={num_train}, val={num_val})...")
    ds = load_dataset("roneneldan/TinyStories", streaming=True)

    train_stories = []
    for i, row in enumerate(ds["train"]):
        if i >= num_train:
            break
        train_stories.append(row["text"])
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{num_train} train stories...")

    val_stories = []
    for i, row in enumerate(ds["validation"]):
        if i >= num_val:
            break
        val_stories.append(row["text"])

    train_text = "\n".join(train_stories)
    val_text = "\n".join(val_stories)

    with open(train_path, "w", encoding="utf-8") as f:
        f.write(train_text)
    with open(val_path, "w", encoding="utf-8") as f:
        f.write(val_text)

    print(f"Saved {len(train_stories)} train + {len(val_stories)} val stories")
    return train_text, val_text


class TextDataset(Dataset):
    """Splits a flat token-ID sequence into (input, target) chunks for
    next-token prediction training."""

    def __init__(self, token_ids, seq_len):
        self.seq_len = seq_len
        self.data = torch.tensor(token_ids, dtype=torch.long)

    def __len__(self):
        return max(0, (len(self.data) - 1) // self.seq_len)

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.data[start : start + self.seq_len]
        y = self.data[start + 1 : start + self.seq_len + 1]
        return x, y
