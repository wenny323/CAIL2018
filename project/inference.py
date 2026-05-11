"""Run inference with a saved checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

import config as cfg
from dataset import LegalChargeDataset, collate_batch
from evaluate import logits_to_preds
from model import BertAttentionClassifier
from utils import _maybe_opencc_s2t, get_torch_device


def load_artifacts(
    checkpoint_path: Path, device: torch.device
) -> Tuple[BertAttentionClassifier, dict]:
    try:
        blob = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        blob = torch.load(checkpoint_path, map_location=device)
    num_classes = int(blob["num_classes"])
    model = BertAttentionClassifier(
        num_classes=num_classes,
        pretrained_model_name=blob["pretrained_model_name"],
        dropout=cfg.DROPOUT,
        fc_hidden=cfg.FC_HIDDEN,
    )
    model.load_state_dict(blob["model_state_dict"])
    model.to(device)
    model.eval()
    return model, blob


def predict_texts(
    texts: List[str],
    model: BertAttentionClassifier,
    blob: dict,
    device: torch.device,
) -> List[List[str]]:
    id2label = {int(k): v for k, v in blob["id2label"].items()}
    threshold = float(blob.get("threshold", cfg.PRED_THRESHOLD))
    tok = blob.get("tokenizer_name") or blob["pretrained_model_name"]
    labels_zero = [np.zeros(len(id2label), dtype=np.float32) for _ in texts]
    ds = LegalChargeDataset(texts, labels_zero, tok, int(blob.get("max_len", cfg.MAX_LEN)), has_label=[False] * len(texts))
    loader = torch.utils.data.DataLoader(
        ds, batch_size=max(1, min(8, len(texts))), collate_fn=collate_batch, shuffle=False
    )
    out: List[List[str]] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
            logits = model(input_ids, attention_mask, token_type_ids).cpu().numpy()
            preds = logits_to_preds(logits, threshold=threshold)
            for row in preds:
                names = [id2label[i] for i, v in enumerate(row) if v == 1]
                out.append(names)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=cfg.BEST_MODEL_PATH)
    parser.add_argument("--text", type=str, default="", help="Single fact string to classify")
    args = parser.parse_args()

    device = get_torch_device()
    print(f"Device: {device}")
    model, blob = load_artifacts(args.checkpoint, device)

    if args.text:
        texts = [_maybe_opencc_s2t(args.text)]
        for names in predict_texts(texts, model, blob, device):
            print("Predicted accusations:", names if names else "(none above threshold)")
        return

    print("Enter fact lines (empty line to exit):")
    while True:
        line = input().strip()
        if not line:
            break
        line = _maybe_opencc_s2t(line)
        for names in predict_texts([line], model, blob, device):
            print("Predicted accusations:", names if names else "(none above threshold)")


if __name__ == "__main__":
    main()
