"""Memory-mapped next-token datasets for low-RAM training."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, Subset


class MemmapTokenDataset(Dataset):
    def __init__(self, path: str, block_size: int):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Token file does not exist: {self.path}")
        self.tokens = np.memmap(self.path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        if len(self.tokens) <= block_size:
            raise ValueError(f"Token file {self.path} is shorter than block size")

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, index):
        # Copy the small window so PyTorch does not hold a read-only NumPy view.
        window = np.array(self.tokens[index : index + self.block_size + 1], dtype=np.int64)
        return torch.from_numpy(window[:-1]), torch.from_numpy(window[1:])


class BilingualSampler(Sampler[int]):
    """Choose a language, then a random valid window inside its token span."""

    def __init__(self, dataset: MemmapTokenDataset, english_tokens: int, english_probability: float):
        if not (dataset.block_size < english_tokens < len(dataset.tokens) - dataset.block_size):
            raise ValueError("English token boundary is outside the usable dataset")
        if not 0.0 <= english_probability <= 1.0:
            raise ValueError("english_probability must be between 0 and 1")
        self.dataset = dataset
        self.english_tokens = english_tokens
        self.english_probability = english_probability

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        remaining = len(self)
        while remaining > 0:
            chunk = min(remaining, 1024)
            choose_english = torch.rand(chunk) < self.english_probability
            en_indices = torch.randint(0, self.english_tokens - self.dataset.block_size, (chunk,))
            zh_indices = torch.randint(self.english_tokens, len(self.dataset), (chunk,))
            indices = torch.where(choose_english, en_indices, zh_indices)
            yield from indices.tolist()
            remaining -= chunk


def make_loader(path: str, block_size: int, batch_size: int, shuffle: bool, num_workers: int,
                english_tokens: int = 0, english_probability: float = 0.5):
    dataset = MemmapTokenDataset(path, block_size)
    sampler = BilingualSampler(dataset, english_tokens, english_probability) if shuffle and english_tokens else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


def make_eval_loader(path: str, block_size: int, batch_size: int, batches: int, num_workers: int,
                     start: int = 0, end: int = None):
    """Evaluate fixed windows spread across the whole validation corpus."""
    dataset = MemmapTokenDataset(path, block_size)
    end = len(dataset) if end is None else end
    if not 0 <= start < end <= len(dataset):
        raise ValueError("Invalid validation token span")
    sample_count = min(end - start, batch_size * batches)
    indices = np.linspace(start, end - 1, sample_count, dtype=np.int64).tolist()
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
