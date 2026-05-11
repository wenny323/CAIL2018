"""Training loop with differential LR, linear warmup, best checkpoint by valid macro F1."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

import config as cfg
from dataset import LegalChargeDataset, collate_batch
from evaluate import compute_metrics, logits_to_preds
from model import BertAttentionClassifier
from utils import (
    build_filtered_dataset,
    build_label_maps,
    get_torch_device,
    load_records,
    save_label_map,
    set_seed,
)


def _tokenizer_name() -> str:
    return cfg.TOKENIZER_NAME or cfg.PRETRAINED_MODEL_NAME


def _prepare_encoder(model: BertAttentionClassifier) -> None:
    if cfg.USE_GRADIENT_CHECKPOINTING:
        model.encoder.gradient_checkpointing_enable()


def train_one_epoch(
    model: BertAttentionClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    loss_fn: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for batch in tqdm(loader, desc="train", leave=False):
        labels = batch["labels"].to(device)
        m = batch["has_label"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids, attention_mask, token_type_ids)
        loss_vec = loss_fn(logits, labels)
        loss_per_sample = loss_vec.mean(dim=1)
        loss = (loss_per_sample * m).sum() / m.sum().clamp(min=1.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += float(loss.item()) * input_ids.size(0)
        n += input_ids.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate_model(
    model: BertAttentionClassifier,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> dict:
    model.eval()
    ys, ps = [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        sel = batch["has_label"] > 0.5
        if not sel.any():
            continue
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        logits = model(input_ids, attention_mask, token_type_ids)
        labels = batch["labels"].to(device)
        logits_np = logits[sel].cpu().numpy()
        labels_np = labels[sel].cpu().numpy()
        pred_np = logits_to_preds(logits_np, threshold=threshold)
        ys.append(labels_np)
        ps.append(pred_np)
    if not ys:
        return {"macro_f1": 0.0, "micro_f1": 0.0, "subset_accuracy": 0.0}

    y_true = np.vstack(ys)
    y_pred = np.vstack(ps)
    return compute_metrics(y_true, y_pred)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=cfg.TRAIN_FILE)
    parser.add_argument("--valid", type=Path, default=cfg.VALID_FILE)
    parser.add_argument("--test", type=Path, default=cfg.TEST_FILE)
    args = parser.parse_args()

    set_seed(cfg.SEED)
    device = get_torch_device()
    print(f"Device: {device}")

    train_records = load_records(args.train)
    valid_records = load_records(args.valid)
    test_records = load_records(args.test)

    label2id, id2label = build_label_maps(train_records, min_freq=cfg.MIN_LABEL_FREQ)
    save_label_map(label2id, cfg.LABEL_MAP_PATH)

    tf, tl, th = build_filtered_dataset(train_records, label2id)
    vf, vl, vh = build_filtered_dataset(valid_records, label2id)
    tef, tel, teh = build_filtered_dataset(test_records, label2id)

    num_classes = len(label2id)
    tok_name = _tokenizer_name()

    train_ds = LegalChargeDataset(tf, tl, tok_name, cfg.MAX_LEN, has_label=th)
    if len(train_ds) == 0:
        raise RuntimeError("No training samples after filtering. Check data paths and MIN_LABEL_FREQ.")
    valid_ds = LegalChargeDataset(vf, vl, tok_name, cfg.MAX_LEN, has_label=vh)
    test_ds = LegalChargeDataset(tef, tel, tok_name, cfg.MAX_LEN, has_label=teh)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=0,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=0,
    )

    model = BertAttentionClassifier(
        num_classes=num_classes,
        pretrained_model_name=cfg.PRETRAINED_MODEL_NAME,
        dropout=cfg.DROPOUT,
        fc_hidden=cfg.FC_HIDDEN,
    ).to(device)
    _prepare_encoder(model)

    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = [
        p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder.")
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": cfg.ENCODER_LR},
            {"params": head_params, "lr": cfg.HEAD_LR},
        ],
        weight_decay=cfg.WEIGHT_DECAY,
    )

    num_training_steps = len(train_loader) * cfg.EPOCHS
    num_warmup_steps = int(num_training_steps * cfg.WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    best_path = cfg.BEST_MODEL_PATH

    for epoch in range(1, cfg.EPOCHS + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, loss_fn)
        metrics = evaluate_model(model, valid_loader, device, cfg.PRED_THRESHOLD)
        print(
            f"Epoch {epoch}/{cfg.EPOCHS} train_loss={tr_loss:.4f} "
            f"valid macro_f1={metrics['macro_f1']:.4f} micro_f1={metrics['micro_f1']:.4f} "
            f"subset_acc={metrics['subset_accuracy']:.4f}"
        )
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label2id": label2id,
                    "id2label": id2label,
                    "threshold": cfg.PRED_THRESHOLD,
                    "pretrained_model_name": cfg.PRETRAINED_MODEL_NAME,
                    "tokenizer_name": tok_name,
                    "num_classes": num_classes,
                    "max_len": cfg.MAX_LEN,
                },
                best_path,
            )
            print(f"  saved best checkpoint -> {best_path}")

    if best_path.is_file():
        try:
            blob = torch.load(best_path, map_location=device, weights_only=False)
        except TypeError:
            blob = torch.load(best_path, map_location=device)
    else:
        blob = None
    if blob:
        model.load_state_dict(blob["model_state_dict"])

    if any(teh):
        tm = evaluate_model(model, test_loader, device, cfg.PRED_THRESHOLD)
        print(
            f"Test (gold available): macro_f1={tm['macro_f1']:.4f} "
            f"micro_f1={tm['micro_f1']:.4f} subset_accuracy={tm['subset_accuracy']:.4f}"
        )
    else:
        print("Test set has no gold labels; skipping test metric computation.")


if __name__ == "__main__":
    main()
