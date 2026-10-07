"""测试集正式评估：Accuracy / Macro-F1 / 按类 P-R-F1 / 混淆矩阵 / ECE / Macro-F1 bootstrap 置信区间

支持 --mode stage1（单头 forward_stage1）与 --mode stage3（多专家门控融合）；
--compare <另一个 checkpoint> 时在同一测试集上做 McNemar 检验与配对 bootstrap Macro-F1 差值区间。
结果写入 results/（JSON），供课程报告直接引用。

用法示例：
    python -m scripts.eval_testset --checkpoint checkpoints/stage1/model.pt --mode stage1 \
        --test data/processed/test.jsonl --out results/stage1_test.json
    python -m scripts.eval_testset --checkpoint checkpoints/stage3/model.pt --mode stage3 \
        --test data/processed/test.jsonl --out results/stage3_test.json \
        --compare checkpoints/stage1/model.pt --compare-mode stage1
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from tqdm import tqdm

from antifraud.config import load_config
from antifraud.data.dataset import load_split
from antifraud.training.metrics import (
    expected_calibration_error,
    macro_f1,
    mcnemar_test,
    per_class_prf,
)
from antifraud.training.pipeline import build_model, load_checkpoint, make_dataloader


def predict(model, loader, mode: str, device: torch.device, fusion: str = "gated") -> tuple[list[int], list[int], np.ndarray]:
    """返回 (y_true, y_pred, 最大softmax置信度)。

    fusion 仅在 mode=stage3 时生效：gated=门控加权（默认）| equal=等权平均（消融 A1）| top1=硬选择（消融 A2）。
    """
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    confs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"推理[{mode}/{fusion}]", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            if mode == "stage1":
                logits = model.forward_stage1(input_ids, attention_mask)
            else:
                out = model(input_ids, attention_mask)
                if fusion == "equal":
                    logits = out["expert_logits"].mean(dim=1)
                elif fusion == "top1":
                    hard = torch.nn.functional.one_hot(out["gate"].argmax(dim=-1), out["gate"].shape[-1]).float()
                    logits = (hard.unsqueeze(-1) * out["expert_logits"]).sum(dim=1)
                else:
                    logits = out["logits"]
            probs = torch.softmax(logits.float(), dim=-1)
            conf, pred = probs.max(dim=-1)
            y_true.extend(batch["labels"].tolist())
            y_pred.extend(pred.cpu().tolist())
            confs.append(conf.cpu().numpy())
    return y_true, y_pred, np.concatenate(confs)


def bootstrap_macro_f1_ci(
    y_true: list[int], y_pred: list[int], n_boot: int = 1000, seed: int = 42
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    y_arr = np.asarray(y_true)
    p_arr = np.asarray(y_pred)
    n = len(y_arr)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        scores.append(macro_f1(y_arr[idx].tolist(), p_arr[idx].tolist()))
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return {"mean": float(np.mean(scores)), "ci95_low": float(lo), "ci95_high": float(hi)}


def paired_bootstrap_f1_diff(
    y_true: list[int],
    pred_a: list[int],
    pred_b: list[int],
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """配对 bootstrap：Macro-F1(B) - Macro-F1(A) 的 95% 区间（A=compare，B=主模型）。"""
    rng = np.random.default_rng(seed)
    y_arr = np.asarray(y_true)
    a_arr = np.asarray(pred_a)
    b_arr = np.asarray(pred_b)
    n = len(y_arr)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        f1_a = macro_f1(y_arr[idx].tolist(), a_arr[idx].tolist())
        f1_b = macro_f1(y_arr[idx].tolist(), b_arr[idx].tolist())
        diffs.append(f1_b - f1_a)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"mean": float(np.mean(diffs)), "ci95_low": float(lo), "ci95_high": float(hi)}


def confusion_matrix(y_true: list[int], y_pred: list[int], num_classes: int) -> list[list[int]]:
    cm = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def evaluate_ckpt(
    model,
    loader,
    mode: str,
    device: torch.device,
    label_names: list[str],
    n_boot: int,
    fusion: str = "gated",
) -> dict:
    y_true, y_pred, conf = predict(model, loader, mode, device, fusion=fusion)
    correct = np.asarray([int(t == p) for t, p in zip(y_true, y_pred)])
    ece = expected_calibration_error(conf, correct)
    return {
        "mode": mode,
        "accuracy": float(np.mean(correct)),
        "macro_f1": macro_f1(y_true, y_pred),
        "per_class": per_class_prf(y_true, y_pred, label_names),
        "confusion_matrix": confusion_matrix(y_true, y_pred, len(label_names)),
        "ece": float(ece),
        "macro_f1_bootstrap": bootstrap_macro_f1_ci(y_true, y_pred, n_boot=n_boot),
        "_y_pred": y_pred,
        "_y_true": y_true,
    }


def infer_peft_method(state: dict) -> str:
    """从 checkpoint 的键名推断 PEFT 类型（DoRA 含 magnitude 向量，LoRA 只有 A/B）。"""
    keys = list(state.keys())
    if any("lora_magnitude_vector" in k for k in keys):
        return "dora"
    if any(".lora_A." in k or ".lora_B." in k for k in keys):
        return "lora"
    return "none"


def main() -> None:
    parser = argparse.ArgumentParser(description="测试集正式评估")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/stage3/model.pt")
    parser.add_argument("--mode", choices=["stage1", "stage3"], default="stage3")
    parser.add_argument("--test", default="data/processed/test.jsonl")
    parser.add_argument("--out", default="results/stage3_test.json")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=128)
    parser.add_argument("--n-boot", dest="n_boot", type=int, default=1000)
    parser.add_argument("--compare", default=None, help="对照 checkpoint（同一测试集，做 McNemar + 配对 bootstrap）")
    parser.add_argument("--compare-mode", dest="compare_mode", choices=["stage1", "stage3"], default="stage1")
    parser.add_argument("--fusion", choices=["gated", "equal", "top1"], default="gated",
                        help="stage3 融合方式：gated=门控（默认）；equal/top1 为消融 A1/A2（不重训，仅改融合）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = load_split(args.test)
    label_names = cfg["data"]["label_names"]
    y_truth = [r["label"] for r in records]
    print(f"[eval] mode={args.mode} test={args.test} n={len(records)} device={device}", flush=True)

    def load(path: str):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        meta_name = ckpt.get("model_name")
        method = infer_peft_method(ckpt["model"] if "model" in ckpt else ckpt)
        cfg["peft"]["method"] = method
        cfg["peft"]["enabled"] = method != "none"
        model = build_model(cfg, model_name=meta_name)
        load_checkpoint(model, path)
        print(f"[eval] 已加载 {path}（model_name={meta_name}, peft={method}）", flush=True)
        return model.to(device)

    model = load(args.checkpoint)
    from antifraud.models import build_tokenizer

    tokenizer = build_tokenizer(model.backbone.name_or_path)
    loader = make_dataloader(records, tokenizer, cfg, args.batch_size)
    result = evaluate_ckpt(model, loader, args.mode, device, label_names, args.n_boot, fusion=args.fusion)
    result["fusion"] = args.fusion
    y_pred_main = result.pop("_y_pred")
    result.pop("_y_true")

    if args.compare:
        model_b = load(args.compare)
        loader_b = make_dataloader(records, tokenizer, cfg, args.batch_size)
        res_b = evaluate_ckpt(model_b, loader_b, args.compare_mode, device, label_names, args.n_boot)
        y_pred_b = res_b.pop("_y_pred")
        res_b.pop("_y_true")
        result["compare"] = {
            "checkpoint": args.compare,
            "mode": args.compare_mode,
            **res_b,
            "mcnemar_vs_main": mcnemar_test(y_truth, y_pred_b, y_pred_main),
            "paired_f1_diff_main_minus_compare": paired_bootstrap_f1_diff(y_truth, y_pred_b, y_pred_main),
        }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(
        f"[eval] accuracy={result['accuracy']:.4f} macro_f1={result['macro_f1']:.4f} "
        f"CI95=[{result['macro_f1_bootstrap']['ci95_low']:.4f}, {result['macro_f1_bootstrap']['ci95_high']:.4f}] "
        f"ece={result['ece']:.4f}",
        flush=True,
    )
    print(f"[eval] 结果已保存到 {args.out}", flush=True)


if __name__ == "__main__":
    main()
