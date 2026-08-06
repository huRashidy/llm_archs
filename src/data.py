"""
Data Pipeline Module for TinyStories
====================================
Handles loading, tokenization, and dynamic batching of text data for Causal Language Modeling (CLM).
"""

from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class TinyStoriesDataset(Dataset):
    """
    Dataset wrapper that tokenizes TinyStories text into fixed-length sequence blocks.
    """
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizerBase, 
        split: str = "train", 
        seq_len: int = 512,
        max_samples: Optional[int] = None
    ):
        self.seq_len = seq_len
        self.tokenizer = tokenizer

        # Load raw dataset from HuggingFace
        raw_ds = load_dataset("roneneldan/TinyStories", split=split)
        
        if max_samples is not None:
            raw_ds = raw_ds.select(range(min(max_samples, len(raw_ds))))

        print(f"Tokenizing {len(raw_ds)} samples for '{split}' split...")
        
        # Tokenize entire split into flattened token stream
        tokenized = tokenizer(
            raw_ds["text"],
            truncation=False,
            padding=False,
            add_special_tokens=True,
            return_attention_mask=False
        )
        
        # Concatenate all token lists into one long array
        all_tokens = []
        for ids in tokenized["input_ids"]:
            all_tokens.extend(ids)

        # Chunk into fixed sequence blocks of length seq_len + 1 (for input x and target y)
        total_len = len(all_tokens)
        block_size = seq_len + 1
        num_blocks = total_len // block_size

        # Truncate leftovers and convert to 2D tensor of shape (num_blocks, seq_len + 1)
        token_array = torch.tensor(all_tokens[: num_blocks * block_size], dtype=torch.long)
        self.chunks = token_array.view(num_blocks, block_size)

        print(f"✅ Prepared {len(self.chunks)} sequence chunks of length {seq_len}.")

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: Input token IDs of shape (seq_len,)
            y: Target token IDs shifted by 1 of shape (seq_len,)
        """
        chunk = self.chunks[idx]
        x = chunk[:-1]  # Tokens from index 0 to seq_len - 1
        y = chunk[1:]   # Tokens from index 1 to seq_len
        return x, y


def get_dataloader(
    seq_len: int = 512,
    batch_size: int = 16,
    tokenizer_name: str = "gpt2",
    split: str = "train",
    max_samples: Optional[int] = None,
    num_workers: int = 2
) -> Tuple[DataLoader, PreTrainedTokenizerBase]:
    """
    Factory helper to instantiate tokenizer and PyTorch DataLoader.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = TinyStoriesDataset(
        tokenizer=tokenizer,
        split=split,
        seq_len=seq_len,
        max_samples=max_samples,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return dataloader, tokenizer


if __name__ == "__main__":
    # Quick sanity test of data loading logic
    loader, tok = get_dataloader(seq_len=128, batch_size=4, max_samples=100)
    for x_batch, y_batch in loader:
        print("x_batch shape:", x_batch.shape)  # Expected: (4, 128)
        print("y_batch shape:", y_batch.shape)  # Expected: (4, 128)
        print("Decoded sample prompt:", tok.decode(x_batch[0][:20]))
        break