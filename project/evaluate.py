"""Multilabel metrics: macro/micro F1 and subset accuracy."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def logits_to_preds(logits: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Sigmoid then threshold -> binary indicators."""
    prob = 1.0 / (1.0 + np.exp(-logits))
    return (prob >= threshold).astype(np.int32)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    y_true, y_pred: shape [N, C] binary (0/1).
    subset accuracy: fraction of samples where all C labels match exactly.
    """
    macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    micro = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    subset_acc = float((y_true == y_pred).all(axis=1).mean()) if y_true.size else 0.0
    return {
        "macro_f1": macro,
        "micro_f1": micro,
        "subset_accuracy": subset_acc,
    }
