"""PyTorch Dataset for tokenized legal facts."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class LegalChargeDataset(Dataset):
    def __init__(
        self,
        facts: List[str],
        labels: List[np.ndarray],
        tokenizer_name: str,
        max_length: int,
        has_label: Optional[List[bool]] = None,
    ):
        self.facts = facts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.has_label = has_label if has_label is not None else [True] * len(facts)

    def __len__(self) -> int:
        return len(self.facts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.facts[idx]
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.from_numpy(self.labels[idx]),
            "has_label": torch.tensor(1.0 if self.has_label[idx] else 0.0, dtype=torch.float32),
        }
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"].squeeze(0)
        return item


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {
        "input_ids": torch.stack([b["input_ids"] for b in batch], dim=0),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch], dim=0),
        "labels": torch.stack([b["labels"] for b in batch], dim=0),
        "has_label": torch.stack([b["has_label"] for b in batch], dim=0),
    }
    if "token_type_ids" in batch[0]:
        out["token_type_ids"] = torch.stack([b["token_type_ids"] for b in batch], dim=0)
    return out
