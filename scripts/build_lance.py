"""把处理好的 JSONL 统一打包为 Lance 列式格式

用法：
  python -m scripts.build_lance --input data/processed --out data/processed/messages.lance
输入目录下所有 *.jsonl 会被合并打包。
"""

from __future__ import annotations

import argparse
from pathlib import Path

FIELDS = ["id", "text", "label", "subtype", "source", "split", "meta"]


def main() -> None:
    parser = argparse.ArgumentParser(description="JSONL → Lance")
    parser.add_argument("--input", required=True, help="包含 *.jsonl 的目录（或单个文件）")
    parser.add_argument("--out", required=True, help="输出 .lance 路径")
    args = parser.parse_args()

    try:
        import lance
        import pyarrow as pa
    except ImportError as e:
        raise SystemExit(f"缺少依赖（pip install pylance pyarrow）：{e}")

    from antifraud.data.dataset import read_jsonl

    input_path = Path(args.input)
    files = sorted(input_path.glob("*.jsonl")) if input_path.is_dir() else [input_path]
    records: list[dict] = []
    for f in files:
        for r in read_jsonl(str(f)):
            records.append({k: r.get(k) for k in FIELDS})
    print(f"合并 {len(files)} 个文件，共 {len(records)} 条")

    import json as _json

    columns = {
        "id": [r["id"] or "" for r in records],
        "text": [r["text"] or "" for r in records],
        "label": [r["label"] if r["label"] is not None else -1 for r in records],
        "subtype": [r["subtype"] or "" for r in records],
        "source": [r["source"] or "" for r in records],
        "split": [r["split"] or "" for r in records],
        "meta": [_json.dumps(r["meta"], ensure_ascii=False) if isinstance(r["meta"], dict) else (r["meta"] or "") for r in records],
    }
    table = pa.table(columns)
    # lance>=12 移除了 write_table，改用 write_dataset
    lance.write_dataset(table, args.out, mode="overwrite")
    print(f"已写入 {args.out}")


if __name__ == "__main__":
    main()
