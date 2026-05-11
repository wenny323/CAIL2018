"""Encoder + additive attention pooling + FC head (multi-label logits)."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel


class BertAttentionClassifier(nn.Module):
    """
    PLM encoder (BERT/ELECTRA/RoBERTa via AutoModel) + Bahdanau-style
    attention pooling over sequence + two-layer classifier.
    """

    def __init__(self, num_classes: int, pretrained_model_name: str, dropout: float = 0.3, fc_hidden: int = 512):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        self.hidden_size = int(self.encoder.config.hidden_size)
        d = self.hidden_size
        self.attn_proj = nn.Linear(d, d)
        self.attn_scorer = nn.Linear(d, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, num_classes)

    def attention_pooling(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # hidden_states: [B, L, d], attention_mask: [B, L] with 1 for real tokens
        u = self.attn_scorer(torch.tanh(self.attn_proj(hidden_states))).squeeze(-1)
        u = u.masked_fill(attention_mask == 0, -1e9)
        alpha = torch.softmax(u, dim=-1)
        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
        ctx = torch.bmm(alpha.unsqueeze(1), hidden_states).squeeze(1)
        return ctx

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs)
        h = out.last_hidden_state
        pooled = self.attention_pooling(h, attention_mask)
        x = self.dropout(torch.relu(self.fc1(pooled)))
        logits = self.fc2(x)
        return logits
