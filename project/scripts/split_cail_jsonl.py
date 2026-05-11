#!/usr/bin/env python3
"""Split a single JSONL file into data_train / data_valid / data_test (practice-style names)."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input_jsonl", type=Path, help="Source JSONL (one JSON object per line)")
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    p.add_argument("--train-ratio", type=float, default=0.9)
    p.add_argument("--valid-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    lines = [ln.strip() for ln in args.input_jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
    for ln in lines:
        json.loads(ln)

    rnd = random.Random(args.seed)
    idx = list(range(len(lines)))
    rnd.shuffle(idx)

    n = len(lines)
    n_train = int(n * args.train_ratio)
    n_valid = int(n * args.valid_ratio)
    n_test = n - n_train - n_valid

    train_i = set(idx[:n_train])
    valid_i = set(idx[n_train : n_train + n_valid])
    test_i = set(idx[n_train + n_valid :])

    def write_subset(name: str, subset: set[int]) -> None:
        out = args.out_dir / name
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for i in sorted(subset):
                f.write(lines[i] + "\n")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_subset("data_train.json", train_i)
    write_subset("data_valid.json", valid_i)
    write_subset("data_test.json", test_i)
    print(f"Wrote train/valid/test counts: {n_train}, {n_valid}, {n_test} -> {args.out_dir}")


if __name__ == "__main__":
    main()
