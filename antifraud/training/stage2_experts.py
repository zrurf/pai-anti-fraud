"""阶段二 Adapt：冻结 backbone，训练 4 个异构专家

每个专家：
- 损失 = 加权交叉熵，对自己负责的类别加权 lambda=3；
- 数据侧重 = WeightedRandomSampler 对负责类别 3 倍过采样；
- 增强 = 按专家配置的增强组合（configs/default.yaml 的 experts[].augment）。

用法（仓库根目录）：
  python -m antifraud.training.stage2_experts \
      --stage1 checkpoints/stage1/model.pt \
      --train data/processed/train.jsonl --val data/processed/val.jsonl \
      --out checkpoints/stage2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from antifraud.config import load_config
from antifraud.training.losses import weighted_ce
from antifraud.training.metrics import macro_f1
from antifraud.training.pipeline import (
    build_model,
    freeze_backbone,
    freeze_module,
    load_checkpoint,
    make_dataloader,
    resolve_class_weights,
    resolve_model_name,
    save_checkpoint,
    unfreeze_module,
)
from antifraud.training.utils import get_device, set_seed, write_json


@torch.no_grad()
def evaluate_expert(model, loader, device, expert_idx: int) -> dict:
    model.eval()
    ys, ps = [], []
    for batch in loader:
        out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        logits = out["expert_logits"][:, expert_idx]
        ps += logits.argmax(dim=-1).tolist()
        ys += batch["labels"].tolist()
    return {"macro_f1": macro_f1(ys, ps), "labels": ys, "predictions": ps}


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段二：训练异构专家")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--stage1", required=True, help="阶段一单头模型 checkpoint")
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-name", dest="model_name", default=None)
    parser.add_argument("--tokenizer-name", dest="tokenizer_name", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--resume", dest="resume", default=None, help="断点续训：指向之前的 --out 目录，跳过已完成专家并续训未完成的")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = get_device()

    from antifraud.data.dataset import load_split
    from antifraud.models import build_tokenizer

    name = resolve_model_name(cfg, args.model_name)
    tokenizer = build_tokenizer(name, args.tokenizer_name)
    model = build_model(cfg, model_name=name)
    load_checkpoint(model, args.stage1, strict=True)
    freeze_backbone(model)
    model.backbone.eval()  # 冻结的 backbone 固定为 eval 模式：关闭其内部 dropout，保证表示确定性
    model.to(device)
    print(f"[stage2] 已从 {args.stage1} 载入 backbone，backbone 已冻结（eval 模式）")

    train_records = load_split(args.train)
    val_records = load_split(args.val)
    class_weights = resolve_class_weights(cfg, [r["label"] for r in train_records])
    if class_weights is not None:
        class_weights = class_weights.to(device)

    epochs = args.epochs or cfg["train"]["stage2"]["epochs"]
    batch_size = args.batch_size or cfg["train"]["stage2"]["batch_size"]
    lr = cfg["train"]["stage2"]["lr"]
    lam = cfg["loss"]["lambda_self"]
    num_classes = cfg["data"]["num_classes"]

    history = {}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 断点续训：读取已完成专家的最优权重并跳过
    done_experts: set[int] = set()
    if args.resume:
        resume_dir = Path(args.resume)
        if resume_dir.is_file():
            resume_dir = resume_dir.parent
        done_file = resume_dir / "experts_done.json"
        if done_file.exists():
            done_info = json.loads(done_file.read_text(encoding="utf-8"))
            done_experts = set(done_info.get("experts_done", []))
            for idx_done, meta in done_info.get("history", {}).items():
                history[idx_done] = meta
            for idx_done in sorted(done_experts):
                expert_ckpt = resume_dir / f"expert_{idx_done}.pt"
                if expert_ckpt.exists():
                    state = torch.load(expert_ckpt, map_location="cpu", weights_only=False)["expert_state"]
                    model.experts[idx_done].load_state_dict(state)
                freeze_module(model.experts[idx_done])
                print(f"[stage2] 断点续训：专家 {idx_done}（{cfg['experts'][idx_done]['name']}）已恢复最优权重并冻结", flush=True)
            if not done_experts:
                print(f"[stage2] 未找到已完成专家记录（{done_file}），从头训练", flush=True)
        else:
            print(f"[stage2] 未找到断点记录 {done_file}，从头训练", flush=True)

    for idx, expert_cfg in enumerate(cfg["experts"]):
        if idx in done_experts:
            continue
        # 专家按名称经 self_class 绑定负责类别（专家列表顺序与标签顺序无关）
        self_class = expert_cfg.get("self_class", idx)
        assert 0 <= self_class < num_classes, f"专家 {expert_cfg['name']} 的 self_class 越界"
        unfreeze_module(model.experts[idx])
        print(f"\n[stage2] 训练专家 {idx}（{expert_cfg['name']}，负责类别 {self_class}，λ={lam}）", flush=True)

        sampler_weights = [lam if r["label"] == self_class else 1.0 for r in train_records]
        train_loader = make_dataloader(
            train_records, tokenizer, cfg, batch_size=batch_size,
            sampler_weights=sampler_weights, augment_kinds=expert_cfg.get("augment"),
        )
        val_loader = make_dataloader(val_records, tokenizer, cfg, batch_size=batch_size)
        optimizer = torch.optim.AdamW(model.experts[idx].parameters(), lr=lr, weight_decay=cfg["train"]["weight_decay"])

        best_f1 = -1.0
        best_state = None
        for epoch in range(epochs):
            model.experts[idx].train()
            total, count = 0.0, 0
            for batch in tqdm(train_loader, desc=f"expert[{expert_cfg['name']}] epoch {epoch + 1}/{epochs}"):
                out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                logits = out["expert_logits"][:, idx]
                loss = weighted_ce(logits, batch["labels"].to(device), class_weights=class_weights, lambda_self=lam, self_class=self_class)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item()
                count += 1
            metrics = evaluate_expert(model, val_loader, device, idx)
            print(f"[stage2] epoch {epoch + 1} loss={total / max(1, count):.4f} expert_val_macro_f1={metrics['macro_f1']:.4f}")
            if metrics["macro_f1"] > best_f1:
                best_f1 = metrics["macro_f1"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.experts[idx].state_dict().items()}

        if best_state is not None:
            model.experts[idx].load_state_dict(best_state)
        history[expert_cfg["name"]] = {"best_val_macro_f1": best_f1}
        freeze_module(model.experts[idx])  # 训练完冻结，进入下一个专家
        # 逐专家落盘：中断后 --resume 可跳过已完成专家
        torch.save(
            {"expert_state": model.experts[idx].state_dict(), "expert_idx": idx, "best_val_macro_f1": best_f1},
            out_dir / f"expert_{idx}.pt",
        )
        write_json(out_dir / "experts_done.json", {"experts_done": sorted(done_experts | {idx}), "history": history})
        done_experts.add(idx)

    save_checkpoint(model, f"{args.out}/model.pt", label_names=cfg["data"]["label_names"], model_name=name)
    write_json(f"{args.out}/metrics.json", {"experts": history})
    print(f"\n[stage2] 全部专家训练完成，checkpoint 已保存到 {args.out}")


if __name__ == "__main__":
    main()
