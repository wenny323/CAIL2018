"""Load CAIL JSON/JSONL, label vocabulary, multi-hot labels, optional OpenCC."""
from __future__ import annotations

import gzip
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from config import LABEL_MAP_PATH, MIN_LABEL_FREQ, SEED, USE_OPENCC_S2T


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize_label(s: str) -> str:
    return s.replace("[", "").replace("]", "").strip()


def _maybe_opencc_s2t(text: str) -> str:
    if not USE_OPENCC_S2T:
        return text
    try:
        from opencc import OpenCC  # type: ignore
    except ImportError as e:
        raise ImportError("USE_OPENCC_S2T=True requires opencc-python-reimplemented") from e
    cc = OpenCC("s2t")
    return cc.convert(text)


def load_records(path: Path) -> List[Dict[str, Any]]:
    """Load CAIL records: JSONL (one JSON per line) or a JSON array file (.json or .json.gz)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            text = f.read().strip()
    else:
        text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array in {path}")
        return data
    records: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def iter_accusations(records: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for r in records:
        meta = r.get("meta") or {}
        acc = meta.get("accusation")
        if not acc:
            continue
        if isinstance(acc, str):
            out.append(_normalize_label(acc))
        else:
            for a in acc:
                out.append(_normalize_label(str(a)))
    return out


def build_label_maps(
    train_records: List[Dict[str, Any]], min_freq: int = MIN_LABEL_FREQ
) -> Tuple[Dict[str, int], Dict[int, str]]:
    counts = Counter(iter_accusations(train_records))
    kept = [lab for lab, c in counts.items() if c >= min_freq]
    kept.sort()
    label2id = {lab: i for i, lab in enumerate(kept)}
    id2label = {i: lab for lab, i in label2id.items()}
    if not label2id:
        raise RuntimeError(
            f"No labels with frequency >= {min_freq}. Lower MIN_LABEL_FREQ or check train data."
        )
    return label2id, id2label


def record_to_multihot(
    record: Dict[str, Any], label2id: Dict[str, int]
) -> Optional[np.ndarray]:
    meta = record.get("meta") or {}
    acc = meta.get("accusation")
    if not acc:
        return None
    if isinstance(acc, list):
        names = [_normalize_label(str(a)) for a in acc]
    else:
        names = [_normalize_label(str(acc))]
    vec = np.zeros(len(label2id), dtype=np.float32)
    any_kept = False
    for name in names:
        if name in label2id:
            vec[label2id[name]] = 1.0
            any_kept = True
    if not any_kept:
        return None
    return vec


def has_gold_labels(record: Dict[str, Any]) -> bool:
    meta = record.get("meta")
    if not meta or "accusation" not in meta:
        return False
    acc = meta.get("accusation")
    if acc is None:
        return False
    if isinstance(acc, list):
        return len(acc) > 0
    return bool(str(acc).strip())


def build_filtered_dataset(
    records: List[Dict[str, Any]], label2id: Dict[str, int], apply_opencc: bool = USE_OPENCC_S2T
) -> Tuple[List[str], List[np.ndarray], List[bool]]:
    """Return parallel lists: facts, label vectors (or empty for no-label), has_label flags."""
    facts: List[str] = []
    labels: List[np.ndarray] = []
    has_label: List[bool] = []
    for r in records:
        fact = r.get("fact", "")
        if not isinstance(fact, str):
            fact = str(fact)
        fact = _maybe_opencc_s2t(fact) if apply_opencc else fact
        gold = has_gold_labels(r)
        if gold:
            vec = record_to_multihot(r, label2id)
            if vec is None:
                continue
            facts.append(fact)
            labels.append(vec)
            has_label.append(True)
        else:
            facts.append(fact)
            labels.append(np.zeros(len(label2id), dtype=np.float32))
            has_label.append(False)
    return facts, labels, has_label


def save_label_map(label2id: Dict[str, int], path: Optional[Path] = None) -> None:
    path = path or LABEL_MAP_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    id2label = {i: lab for lab, i in label2id.items()}
    torch.save({"label2id": label2id, "id2label": id2label}, path)


def load_label_map(path: Optional[Path] = None) -> Tuple[Dict[str, int], Dict[int, str]]:
    path = path or LABEL_MAP_PATH
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location="cpu")
    return blob["label2id"], blob["id2label"]
