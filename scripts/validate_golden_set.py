"""黄金测试集校验器

黄金测试集（data/golden_test_set.jsonl）是人工审核的校园风格评估集，绝不进入训练。
本脚本在构建后校验其可用性：JSONL 格式、标签合法性、规模与类别分布、文件内重复、
与正式训练池 train/val/test 的文本泄漏（完全一致即视为泄漏）。

校验通过（exit 0）才可用于评估；任何 ERROR 项都会以非零码退出。

用法（仓库根目录）：
    python -m scripts.validate_golden_set [--golden data/golden_test_set.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

LABEL_NAMES = {0: "正常", 1: "推销", 2: "刷单", 3: "钓鱼"}
POOL_FILES = ["train.jsonl", "val.jsonl", "test.jsonl"]


def read_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"ERROR 第 {i} 行不是合法 JSON：{e}")
                sys.exit(1)
    return records


def normalize(text: str) -> str:
    """泄漏比对用的归一化：去空白差异即可，不做更激进的归一。"""
    return "".join(text.split())


def main() -> None:
    parser = argparse.ArgumentParser(description="黄金测试集校验")
    parser.add_argument("--golden", default="data/golden_test_set.jsonl")
    parser.add_argument("--pool-dir", default="data/processed")
    args = parser.parse_args()

    if not Path(args.golden).exists():
        print(f"ERROR 找不到 {args.golden}（黄金测试集尚未构建，生成规范见 docs/黄金测试集LLM生成Prompt.md）")
        sys.exit(1)

    records = read_jsonl(args.golden)
    errors: list[str] = []
    warnings: list[str] = []

    # 1) 规模
    if len(records) < 200:
        warnings.append(f"样本量 {len(records)} < 规划的 200 条")
    else:
        print(f"OK 样本量 {len(records)}")

    # 2) 字段与标签
    labels = []
    for i, r in enumerate(records, 1):
        if "text" not in r or not isinstance(r["text"], str) or not r["text"].strip():
            errors.append(f"第 {i} 行缺少非空 text 字段")
        if "label" not in r:
            errors.append(f"第 {i} 行缺少 label 字段")
            continue
        if r["label"] not in LABEL_NAMES:
            errors.append(f"第 {i} 行 label={r['label']} 不在 0–3")
            continue
        labels.append(r["label"])

    # 3) 类别分布（黄金集应四类均衡，各 ≥ 25 条）
    dist = Counter(labels)
    for lab in sorted(LABEL_NAMES):
        n = dist.get(lab, 0)
        print(f"    {LABEL_NAMES[lab]}(label={lab}): {n} 条")
        if n < 25:
            warnings.append(f"{LABEL_NAMES[lab]} 仅 {n} 条（规划为四类均衡，每类 ≥25）")

    # 4) 文件内重复
    texts = [normalize(r["text"]) for r in records if isinstance(r.get("text"), str)]
    dup = {t for t, c in Counter(texts).items() if c > 1}
    if dup:
        errors.append(f"文件内存在 {len(dup)} 条重复文本")
    else:
        print("OK 文件内无重复")

    # 5) 与训练池泄漏比对
    pool_texts: set[str] = set()
    for name in POOL_FILES:
        pool_path = Path(args.pool_dir) / name
        if not pool_path.exists():
            warnings.append(f"训练池文件缺失，跳过泄漏比对：{pool_path}")
            continue
        for line in open(pool_path, encoding="utf-8"):
            if line.strip():
                pool_texts.add(normalize(json.loads(line)["text"]))
    leaked = pool_texts & set(texts)
    if leaked:
        errors.append(f"与训练池存在 {len(leaked)} 条文本泄漏（黄金集绝不能进入训练数据）")
    else:
        print(f"OK 与训练池 {len(POOL_FILES)} 个划分无文本泄漏")

    for w in warnings:
        print(f"WARN {w}")
    for e in errors:
        print(f"ERROR {e}")
    if errors:
        print("校验未通过 ❌")
        sys.exit(1)
    print("校验通过 ✅（仅提示项可酌情处理）")


if __name__ == "__main__":
    main()
