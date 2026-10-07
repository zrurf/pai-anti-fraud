"""训练共用工具：随机种子、设备、类权重、JSONL 读写"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def class_weights_from(labels: list[int], num_classes: int) -> torch.Tensor:
    """auto 类权重：按频率倒数归一化（N / (C · count_c)），再除以均值使权重均值为 1。"""
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.clip(counts, 1, None)
    weights = len(labels) / (num_classes * counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
