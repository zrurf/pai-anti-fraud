"""数据集读取与 tokenize

统一 JSONL 格式（逐行）：
    {"text": "...", "label": 0..3 | -1, "subtype": "...", "source": "...", "split": "train|val|test"}
label: 0=正常 1=推销 2=刷单 3=钓鱼；-1 表示未映射（仅用于二分类/预训练）。
"""

from __future__ import annotations

import json
from typing import Any, Callable, Iterable

import torch


def read_jsonl(path: str) -> list[dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_split(path: str, labeled_only: bool = True) -> list[dict[str, Any]]:
    records = read_jsonl(path)
    if labeled_only:
        records = [r for r in records if r.get("label", -1) >= 0]
    return records


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, records: list[dict[str, Any]]):
        assert all(r.get("label", -1) >= 0 for r in records), "训练/评估数据必须已映射标签"
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.records[idx]


def make_collate_fn(tokenizer, max_length: int, augmentation: Callable[[str], str] | None = None):
    # 返回 DataLoader 的 collate_fn。augmentation 为逐文本增强函数（各专家的数据侧重）

    def collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts = [b["text"] for b in batch]
        if augmentation is not None:
            texts = [augmentation(t) for t in texts]
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}

    return collate


class TinyTokenizer:
    """离线测试用微型分词器（按空白切词的词表映射），模拟 HF tokenizer 接口。

    仅用于本地冒烟测试（model_name="tiny-random"），不可用于真实训练。
    """

    PAD, UNK, CLS, SEP = 0, 1, 2, 3

    def __init__(self):
        self.vocab: dict[str, int] = {"[PAD]": self.PAD, "[UNK]": self.UNK, "[CLS]": self.CLS, "[SEP]": self.SEP}
        for i in range(496):
            self.vocab[f"w{i}"] = i + 4

    def _encode_one(self, text: str, max_length: int) -> tuple[list[int], list[int]]:
        tokens = text.split()[: max_length - 2]
        ids = [self.CLS] + [self.vocab.get(t, self.UNK) for t in tokens] + [self.SEP]
        return ids, [1] * len(ids)

    def __call__(self, texts: Iterable[str], padding=True, truncation=True, max_length=128, return_tensors="pt", **_):
        encoded = [self._encode_one(t, max_length) for t in texts]
        longest = max(len(ids) for ids, _ in encoded)
        input_ids, attention_mask = [], []
        for ids, mask in encoded:
            pad = longest - len(ids)
            input_ids.append(ids + [self.PAD] * pad)
            attention_mask.append(mask + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
