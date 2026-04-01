import json
import os

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


# PubMedQA Dataset class for PyTorch
class MedicalQADataset(Dataset):
    """PyTorch dataset wrapper for PubMedQA examples."""

    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = {"no": 0, "yes": 1, "maybe": 2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        question = example["question"]
        context = " ".join(example["context"]["contexts"])

        encoded = self.tokenizer(
            question,
            context,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Some tokenizers may not return token_type_ids; create zeros in that case.
        token_type_ids = encoded.get("token_type_ids")
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(encoded["input_ids"])

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "token_type_ids": token_type_ids.squeeze(0),
            "label": torch.tensor(self.label_map[example["final_decision"]], dtype=torch.long),
        }


# Create train/val/test splits
def create_splits(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """Shuffle train split and create train/val/test partitions."""
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    full_data = list(dataset["train"])
    rng = np.random.default_rng(seed)
    indices = np.arange(len(full_data))
    rng.shuffle(indices)

    shuffled_data = [full_data[i] for i in indices]

    train_end = int(len(shuffled_data) * train_ratio)
    val_end = train_end + int(len(shuffled_data) * val_ratio)

    train_data = shuffled_data[:train_end]
    val_data = shuffled_data[train_end:val_end]
    test_data = shuffled_data[val_end:]

    print(f"Train split size: {len(train_data)}")
    print(f"Validation split size: {len(val_data)}")
    print(f"Test split size: {len(test_data)}")

    return train_data, val_data, test_data


# Create DataLoaders
def create_dataloaders(train_data, val_data, test_data, tokenizer, batch_size=8, max_length=512):
    """Build DataLoader objects for each data split."""
    train_dataset = MedicalQADataset(train_data, tokenizer, max_length=max_length)
    val_dataset = MedicalQADataset(val_data, tokenizer, max_length=max_length)
    test_dataset = MedicalQADataset(test_data, tokenizer, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


# Test the dataset pipeline
if __name__ == "__main__":
    # Keep requested imports explicitly used while staying side-effect free.
    _ = json
    _ = os

    dataset = load_dataset("pubmed_qa", "pqa_labeled")
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")

    train_data, val_data, test_data = create_splits(dataset)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data,
        val_data,
        test_data,
        tokenizer,
        batch_size=8,
    )

    _ = val_loader
    _ = test_loader

    batch = next(iter(train_loader))
    print(f"input_ids shape: {batch['input_ids'].shape}")
    print(f"attention_mask shape: {batch['attention_mask'].shape}")
    print(f"label: {batch['label']}")
    print("DataLoader pipeline working correctly!")
