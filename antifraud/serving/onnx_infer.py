"""ONNXRuntime 推理封装

输出：类别、置信度、各专家门控权重、各专家独立意见、证据/归因字段。

- predict()：快路径。evidence 为关键词匹配（人工维护词表），实时接口用。
- explain()：归因路径。核心是 attribute_occlusion()——Occlusion 留一法 token 级归因，
  纯 ONNX 实现（不需要梯度/torch）：按 tokenizer 的 word 分组（中文单字、英文/数字整词）
  逐组把 token 替换为 [UNK]，全部副本拼一个 batch 一次前向，
  归因分数 = 基线预测概率 − 遮蔽后预测概率（drop 越大该词越重要）。
  tokens 附带字符区间；merged 为相邻字合并后的词组显示版。
  tokenizer 不支持 word 级映射（如离线测试用 TinyTokenizer）时降级为关键词证据。
- 若导出图带 attn_scores 输出（注意力池化专家的 token 权重，见 scripts/export_onnx.py），
  explain() 额外返回 expert_attention 第二视角；其解释力随该专家的门控权重降低而减弱，
  消费方应结合 expert_weights 解读。

专家/类别命名从 predict_meta.json 读取（gate/expert_logits 第 i 维对应 expert_names[i]，
专家顺序与标签顺序无关）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

LABEL_NAMES = ["正常", "推销", "刷单", "钓鱼"]
EXPERT_NAMES = ["推销", "刷单", "钓鱼", "正常"]

# 证据关键词占位（人工维护，覆盖三类诈骗的典型话术；归因降级时也用它兜底）
KEYWORD_EVIDENCE: dict[str, list[str]] = {
    "推销": ["报备", "学姐", "学长", "优惠", "推广", "下单", "低价", "秒杀", "上新"],
    "刷单": ["刷单", "返利", "佣金", "任务", "做单", "垫付", "小额", "升级", "接单"],
    "钓鱼": ["教务处", "辅导员", "学号", "身份证", "银行卡", "点击链接", "核实", "逾期", "教务", "注册"],
}


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _softmax_batch(x: np.ndarray) -> np.ndarray:
    z = np.exp(x - x.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


def _to_numpy(t) -> np.ndarray:
    """tokenizer 返回值统一为 numpy（HF 返回 numpy，TinyTokenizer 等测试桩返回 torch 张量）。"""
    return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)


def _merge_adjacent(tokens: list[dict], score_key: str) -> list[dict]:
    """把字符区间恰好相接的相邻词条合并为一条（分数累加），供前端按词组高亮。

    中文按字打分，"教/务/处" 连续入选时合并显示为"教务处"；只做展示聚合，
    结果按分数降序返回，原始打分不变。
    """
    merged: list[dict] = []
    for t in sorted(tokens, key=lambda x: x["start"]):
        if merged and t["start"] == merged[-1]["end"]:
            prev = merged[-1]
            prev["text"] += t["text"]
            prev["end"] = t["end"]
            prev[score_key] = round(prev[score_key] + t[score_key], 4)
        else:
            merged.append(dict(t))
    merged.sort(key=lambda x: -x[score_key])
    return merged


class OnnxPredictor:
    def __init__(
        self,
        onnx_path: str,
        tokenizer_dir: str,
        label_names: list[str] | None = None,
        expert_names: list[str] | None = None,
        max_length: int = 128,
        tokenizer=None,
    ):
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(tokenizer_dir)
        meta_path = Path(tokenizer_dir) / "predict_meta.json"
        self.attn_expert: str | None = None
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            label_names = label_names or meta.get("label_names")
            expert_names = expert_names or meta.get("expert_names")
            self.attn_expert = meta.get("attn_expert")
        self.label_names = label_names or LABEL_NAMES
        self.expert_names = expert_names or self.label_names
        self.max_length = max_length
        self.output_names = {o.name for o in self.session.get_outputs()}

    # 主预测

    def predict(self, text: str) -> dict:
        enc = self.tokenizer([text], padding=True, truncation=True, max_length=self.max_length, return_tensors="np")
        inputs = {
            "input_ids": _to_numpy(enc["input_ids"]).astype(np.int64),
            "attention_mask": _to_numpy(enc["attention_mask"]).astype(np.int64),
        }
        outputs = self.session.run(None, inputs)
        graph_names = [o.name for o in self.session.get_outputs()]
        by_name = dict(zip(graph_names, outputs))
        logits = by_name["logits"]
        gate = by_name.get("gate")
        expert_logits = by_name.get("expert_logits")

        prob = _softmax(logits[0].astype(np.float64))
        idx = int(prob.argmax())
        label_name = self.label_names[idx]

        result = {
            "label": idx,
            "label_name": label_name,
            "confidence": float(prob[idx]),
            "probabilities": {self.label_names[i]: float(prob[i]) for i in range(len(self.label_names))},
        }
        if gate is not None and expert_logits is not None:
            result["expert_weights"] = {self.expert_names[i]: float(gate[0][i]) for i in range(len(self.expert_names))}
            result["expert_opinions"] = {
                self.expert_names[i]: self.label_names[int(expert_logits[0][i].argmax())]
                for i in range(len(self.expert_names))
            }
        result["evidence"] = self._keyword_hits(text, label_name)
        return result

    def _keyword_hits(self, text: str, label_name: str) -> list[str]:
        return [kw for kw in KEYWORD_EVIDENCE.get(label_name, []) if kw in text]

    # Occlusion 留一法归因

    def attribute_occlusion(self, text: str, top_k: int = 8) -> dict:
        """Token 级归因（Occlusion 留一法，纯 ONNX，一次 [N+1, L] 批量前向）。

        返回 {"source": "occlusion", ...}；tokenizer 能力不足时返回
        {"source": "keyword_fallback", "tokens": [], "note": ...}（由 explain() 填充关键词）。
        """
        try:
            enc = self.tokenizer(
                [text], truncation=True, max_length=self.max_length,
                return_offsets_mapping=True, return_tensors="np",
            )
        except (TypeError, ValueError, NotImplementedError, KeyError):
            return {"source": "keyword_fallback", "tokens": [], "note": "tokenizer 不支持 offset mapping"}
        if not hasattr(enc, "word_ids"):
            return {"source": "keyword_fallback", "tokens": [], "note": "tokenizer 不支持 word 级映射"}

        word_ids = enc.word_ids(0)
        offsets = _to_numpy(enc["offset_mapping"])[0]
        input_ids = _to_numpy(enc["input_ids"]).astype(np.int64)
        attention_mask = _to_numpy(enc["attention_mask"]).astype(np.int64)
        unk_id = getattr(self.tokenizer, "unk_token_id", None)
        if unk_id is None:
            return {"source": "keyword_fallback", "tokens": [], "note": "tokenizer 无 [UNK] id"}

        # word 分组（word_id -> token 位置 + 字符区间）；None/零宽 span 为特殊 token，跳过
        groups: dict[int, list[int]] = {}
        for pos, wid in enumerate(word_ids):
            if wid is not None:
                groups.setdefault(int(wid), []).append(pos)
        items: list[tuple[list[int], int, int]] = []
        for _, positions in sorted(groups.items()):
            spans = [offsets[p] for p in positions]
            start, end = int(min(s for s, _ in spans)), int(max(e for _, e in spans))
            if start == 0 and end == 0:
                continue
            items.append((positions, start, end))
        if not items:
            return {"source": "keyword_fallback", "tokens": [], "note": "无可遮蔽的词"}

        n = len(items)
        batch_ids = np.repeat(input_ids, n + 1, axis=0)
        batch_mask = np.repeat(attention_mask, n + 1, axis=0)
        for i, (positions, _, _) in enumerate(items):
            batch_ids[i + 1, positions] = unk_id

        base_logits = self.session.run(["logits"], {"input_ids": input_ids, "attention_mask": attention_mask})[0]
        base_prob = _softmax(base_logits[0].astype(np.float64))
        target = int(base_prob.argmax())
        base_p = float(base_prob[target])

        logits = self.session.run(["logits"], {"input_ids": batch_ids, "attention_mask": batch_mask})[0]
        probs = _softmax_batch(logits.astype(np.float64))
        drops = base_p - probs[1:, target]

        ranked = sorted(zip(items, drops), key=lambda t: -t[1])[: max(1, top_k)]
        tokens = [
            {"text": text[s:e], "start": s, "end": e, "drop": round(float(d), 4)}
            for (positions, s, e), d in ranked
        ]
        return {
            "source": "occlusion",
            "target_label_name": self.label_names[target],
            "baseline_prob": round(base_p, 4),
            "tokens": tokens,
            "merged": _merge_adjacent(tokens, "drop"),
            "n_words": n,
            "n_forward_batch": n + 1,
        }

    # 专家注意力第二视角

    def _attention_top_tokens(self, text: str, top_k: int = 8) -> dict | None:
        """读取导出图中的 attn_scores（注意力池化专家的 token 权重），返回权重最高的词。"""
        if self.attn_expert is None or "attn_scores" not in self.output_names:
            return None
        enc = self.tokenizer(
            [text], truncation=True, max_length=self.max_length,
            return_offsets_mapping=True, return_tensors="np",
        )
        offsets = _to_numpy(enc["offset_mapping"])[0]
        input_ids = _to_numpy(enc["input_ids"]).astype(np.int64)
        attention_mask = _to_numpy(enc["attention_mask"]).astype(np.int64)
        attn = self.session.run(["attn_scores"], {"input_ids": input_ids, "attention_mask": attention_mask})[0][0]

        order = np.argsort(-attn)
        tokens = []
        for pos in order:
            s, e = int(offsets[pos][0]), int(offsets[pos][1])
            if s == 0 and e == 0:
                continue
            tokens.append({"text": text[s:e], "start": s, "end": e, "weight": round(float(attn[pos]), 4)})
            if len(tokens) >= top_k:
                break
        if not tokens:
            return None
        gate_weight = None
        if self.attn_expert in self.expert_names:
            idx = self.expert_names.index(self.attn_expert)
            outputs = self.session.run(["gate"], {"input_ids": input_ids, "attention_mask": attention_mask})[0]
            gate_weight = float(outputs[0][idx])
        return {
            "expert": self.attn_expert,
            "gate_weight": gate_weight,
            "tokens": tokens,
            "merged": _merge_adjacent(tokens, "weight"),
        }

    # 归因入口

    def explain(self, text: str, top_k: int = 8) -> dict:
        """主预测结果 + token 级归因。归因失败自动降级关键词，不阻塞主结果。"""
        result = self.predict(text)
        fallback = [{"text": w} for w in self._keyword_hits(text, result["label_name"])]
        try:
            attribution = self.attribute_occlusion(text, top_k=top_k)
            if attribution.get("source") == "keyword_fallback" and not attribution.get("tokens"):
                attribution["tokens"] = fallback
        except Exception as exc:  # noqa: BLE001 — 任何归因异常都降级，不影响主预测
            attribution = {"source": "keyword_fallback", "tokens": fallback, "note": f"occlusion 归因失败：{exc}"}
        result["attribution"] = attribution
        try:
            att = self._attention_top_tokens(text, top_k=top_k)
            if att is not None:
                result["expert_attention"] = att
        except Exception:  # noqa: BLE001 — 第二视角失败不影响主结果
            pass
        return result
