"""对说话人分离转写结果做 PII 脱敏（正则 + 分句 NER 姓名），产出合并文件供 build_processed 使用

用法：python -m scripts.mask_pii_teleantifraud
输入：tmp/teleantifraud_asr_spk_*.jsonl（说话人分离转写）
输出：tmp/teleantifraud_asr_spk_masked.jsonl（text 为脱敏后文本；pii 字段为替换统计）
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from antifraud.data.pii import _apply_spans, _regex_spans, load_ner, pii_counts

OUT = ROOT / "tmp" / "teleantifraud_asr_spk_masked.jsonl"
SENT_SPLIT = re.compile(r"(?<=[。！？；!?])")


def sentence_spans(text: str, ner) -> list[tuple[int, int, str]]:
    """分句送 NER（短句姓名召回更高），把命中姓名的句子内偏移映射回全文偏移。"""
    spans: list[tuple[int, int, str]] = []
    sentences: list[tuple[int, str]] = []
    pos = 0
    for piece in SENT_SPLIT.split(text):
        if piece:
            sentences.append((pos, piece))
        pos += len(piece)
    if not sentences:
        return spans
    results = ner([s for _, s in sentences], batch_size=32)
    for (offset, _), ents in zip(sentences, results):
        for ent in ents:
            if str(ent.get("entity_group", "")).lower() in {"name", "per"}:
                spans.append((offset + ent["start"], offset + ent["end"], "[姓名]"))
    return spans


def main() -> None:
    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ner = load_ner(device)
    print(f"NER 加载完成（{device}）")

    shard_files = sorted(ROOT.glob("tmp/teleantifraud_asr_spk_*.jsonl"))
    rows = []
    seen = set()
    for f in shard_files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            if r["audio_path"] in seen:
                continue
            seen.add(r["audio_path"])
            rows.append(r)
    print(f"读取 {len(rows)} 条转写（来自 {len(shard_files)} 个分片文件）")

    n_pii_rows = 0
    with open(OUT, "w", encoding="utf-8") as fout:
        for i, r in enumerate(rows):
            text = r.get("text", "")
            if text:
                spans = _regex_spans(text) + sentence_spans(text, ner)
                masked = _apply_spans(text, spans)
                counts = pii_counts(text, masked)
                if counts:
                    n_pii_rows += 1
            else:
                masked, counts = text, {}
            fout.write(
                json.dumps(
                    {
                        "audio_path": r["audio_path"],
                        "text": masked,
                        "duration_s": r.get("duration_s", 0.0),
                        "n_spk": r.get("n_spk", 0),
                        "pii": counts,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            if (i + 1) % 1000 == 0:
                print(f"进度 {i + 1}/{len(rows)}（含 PII 行 {n_pii_rows}）", flush=True)
    print(f"脱敏完成：{len(rows)} 条，其中 {n_pii_rows} 条含被脱敏内容 → {OUT}")


if __name__ == "__main__":
    sys.exit(main())
