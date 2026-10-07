"""导出 ONNX：加载阶段三 checkpoint → 合并 PEFT 权重 → 单图导出（backbone + 专家 + 路由器）

用法（仓库根目录）：
  python -m scripts.export_onnx --checkpoint checkpoints/stage3/model.pt
  python -m scripts.export_onnx
产物：
  checkpoints/onnx/model.onnx          推理图（输入 input_ids/attention_mask；
                                       输出 logits/gate/expert_logits，若存在注意力池化专家另含 attn_scores）
  checkpoints/onnx/predict_meta.json   类别与专家命名元数据 + attn_expert（OnnxPredictor 自动读取）
  checkpoints/onnx/                    同时保存 tokenizer，供 OnnxPredictor 使用
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from antifraud.config import load_config
from antifraud.models import build_tokenizer, merge_peft
from antifraud.training.pipeline import build_model, load_checkpoint, resolve_model_name


class ExportWrapper(torch.nn.Module):
    """把模型输出摊平成 ONNX 输出张量；若存在注意力池化专家，额外导出其 token 级注意力权重。

    注意：这里手动展开 MoEFraudClassifier.forward（单次 encode），避免 torch.onnx 追踪出
    两份 encoder 子图；注意力权重用与 AttentionPooling 完全相同的公式、同一模块参数重算，
    不改动模型与训练路径。输出顺序：logits, gate, expert_logits[, attn_scores]。
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.attn_expert_idx = [i for i, e in enumerate(model.experts) if e.pooling_name == "attention"]

    def forward(self, input_ids, attention_mask):
        model = self.model
        hidden = model.encode(input_ids, attention_mask)
        expert_logits = torch.stack([e(hidden, attention_mask) for e in model.experts], dim=1)
        gate = model.router(hidden, attention_mask)
        fused = (gate.unsqueeze(-1) * expert_logits).sum(dim=1)
        outputs = [fused, gate, expert_logits]
        for i in self.attn_expert_idx:
            q = model.experts[i].pooling.query
            scores = (hidden @ q) / q.shape[0] ** 0.5
            scores = scores.masked_fill(attention_mask == 0, torch.finfo(scores.dtype).min)
            outputs.append(scores.softmax(dim=-1))
        return tuple(outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 ONNX 模型")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/stage3/model.pt")
    parser.add_argument("--out-dir", dest="out_dir", default="checkpoints/onnx")
    parser.add_argument("--model-name", dest="model_name", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_name = resolve_model_name(cfg, args.model_name)

    # checkpoint 内记录了训练时的 model_name，优先使用
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    name = ckpt.get("model_name", model_name)

    model = build_model(cfg, model_name=name)
    load_checkpoint(model, args.checkpoint, strict=True)
    model.backbone = merge_peft(model.backbone)
    model.eval()
    print(f"[export] 已加载 {args.checkpoint}（model_name={name}），PEFT 已合并")

    tokenizer = build_tokenizer(name)
    sample = tokenizer(["这是一个测试样本"], padding=True, truncation=True, max_length=cfg["data"]["max_length"], return_tensors="pt")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "model.onnx"

    wrapper = ExportWrapper(model)
    has_attn = bool(wrapper.attn_expert_idx)
    output_names = ["logits", "gate", "expert_logits"] + (["attn_scores"] if has_attn else [])
    kwargs = dict(
        input_names=["input_ids", "attention_mask"],
        output_names=output_names,
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
            "gate": {0: "batch"},
            "expert_logits": {0: "batch"},
        },
        opset_version=17,
    )
    if has_attn:
        kwargs["dynamic_axes"]["attn_scores"] = {0: "batch", 1: "seq"}
    try:
        torch.onnx.export(wrapper, (sample["input_ids"], sample["attention_mask"]), str(onnx_path), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(wrapper, (sample["input_ids"], sample["attention_mask"]), str(onnx_path), **kwargs)
    print(f"[export] 已导出 {onnx_path}（输出：{', '.join(output_names)}）")

    # 专家/标签命名元数据：推理层据此正确标注各专家输出（gate/expert_logits 第 i 维 = 专家 i）
    meta = {
        "model_name": name,
        "label_names": cfg["data"]["label_names"],
        "expert_names": [e["name"] for e in cfg["experts"]],
    }
    if has_attn:
        meta["attn_expert"] = cfg["experts"][wrapper.attn_expert_idx[0]]["name"]
    (out_dir / "predict_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[export] 元数据已保存到 {out_dir / 'predict_meta.json'}")

    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(out_dir)
        print(f"[export] tokenizer 已保存到 {out_dir}")


if __name__ == "__main__":
    main()
