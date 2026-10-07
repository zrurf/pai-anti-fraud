"""阶段三 Route：冻结 backbone 与专家，训练路由器（可选联合微调）

- 路由训练：损失 = 融合输出加权交叉熵（只用全局类权重 w_c）+ 负载均衡损失（alpha 全程固定）。
- 联合微调（--joint）：解冻路由器与各专家模块（含注意力池化的可学习参数），
  backbone 保持冻结，少量轮次对齐特征。
- 监控指标：各专家激活比例 f_i 与平均门控概率 p_i（路由效率）。

用法：
  python -m antifraud.training.stage3_router \
      --stage2 checkpoints/stage2/model.pt \
      --train data/processed/train.jsonl --val data/processed/val.jsonl \
      --out checkpoints/stage3 [--joint]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from antifraud.config import load_config
from antifraud.training.losses import load_balance_loss, weighted_ce
from antifraud.training.metrics import expected_calibration_error, macro_f1
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
def evaluate_fusion(model, loader, device) -> dict:
    model.eval()
    ys, ps, confs, gates = [], [], [], []
    for batch in loader:
        out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        prob = out["logits"].softmax(dim=-1)
        ps += prob.argmax(dim=-1).tolist()
        ys += batch["labels"].tolist()
        confs += prob.max(dim=-1).values.tolist()
        gates += out["gate"].tolist()
    gates = np.asarray(gates)
    assign = gates.argmax(axis=1)
    f_i = np.bincount(assign, minlength=gates.shape[1]) / len(gates)  # 硬分配频率
    p_i = gates.mean(axis=0)  # 平均门控概率
    correct = (np.asarray(ys) == np.asarray(ps)).astype(float)
    return {
        "macro_f1": macro_f1(ys, ps),
        "ece": expected_calibration_error(np.asarray(confs), correct),
        "gate_activation_f": f_i.tolist(),  # 各专家硬分配频率
        "gate_probability_p": p_i.tolist(),  # 各专家平均门控概率
        "labels": ys,
        "predictions": ps,
    }


def train_phase(
    model,
    train_loader,
    device,
    cfg,
    epochs,
    lr,
    class_weights,
    train_experts: bool,
    save_path: str | None = None,
    resume: str | None = None,
    desc: str = "route",
) -> None:
    """train_experts=False：仅路由器在训练模式（backbone/专家保持 eval，关闭冻结模块内部的 dropout）。
    train_experts=True：联合微调，路由器与各专家模块均训练。
    save_path：每 epoch 落盘训练状态（供 --resume 断点续训）。"""
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=cfg["train"]["weight_decay"])
    start_epoch = 0
    alpha = cfg["loss"]["alpha_bal"]
    if resume:
        resume_path = Path(resume)
        if resume_path.is_dir():
            resume_path = resume_path / Path(save_path).name if save_path else resume_path / "last.pt"
        if resume_path.exists():
            ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt["epoch"]
            print(f"[{desc}] 从 epoch {start_epoch} 断点续训", flush=True)
        else:
            print(f"[{desc}] 未找到断点 {resume_path}，从头训练", flush=True)
    for epoch in range(start_epoch, epochs):
        model.router.train()
        if train_experts:
            for expert in model.experts:
                expert.train()
        total, count = 0.0, 0
        for batch in tqdm(train_loader, desc=f"{desc} epoch {epoch + 1}/{epochs}"):
            out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            loss = weighted_ce(out["logits"], batch["labels"].to(device), class_weights=class_weights) + load_balance_loss(out["gate"], alpha=alpha)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            count += 1
        print(f"[{desc}] epoch {epoch + 1} loss={total / max(1, count):.4f}", flush=True)
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1}, save_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段三：训练路由器（+ 可选联合微调）")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--stage2", required=True, help="阶段二 checkpoint")
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-name", dest="model_name", default=None)
    parser.add_argument("--tokenizer-name", dest="tokenizer_name", default=None)
    parser.add_argument("--joint", action="store_true", help="路由收敛后进行少量轮次联合微调")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--resume", dest="resume", default=None, help="断点续训：指向之前的 --out 目录（续 3a 路由轮次；联合微调中断则续 last_joint.pt）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = get_device()

    from antifraud.data.dataset import load_split
    from antifraud.models import build_tokenizer

    name = resolve_model_name(cfg, args.model_name)
    tokenizer = build_tokenizer(name, args.tokenizer_name)
    model = build_model(cfg, model_name=name)
    load_checkpoint(model, args.stage2, strict=True)
    freeze_backbone(model)
    for expert in model.experts:
        freeze_module(expert)
    model.backbone.eval()   # 冻结的 backbone 固定为 eval 模式
    for expert in model.experts:
        expert.eval()       # 阶段 3a 中冻结的专家同样固定为 eval 模式
    model.to(device)
    print(f"[route] 已从 {args.stage2} 载入；backbone 与专家全部冻结（eval 模式），仅训练路由器")

    train_records = load_split(args.train)
    val_records = load_split(args.val)
    class_weights = resolve_class_weights(cfg, [r["label"] for r in train_records])
    if class_weights is not None:
        class_weights = class_weights.to(device)

    s3 = cfg["train"]["stage3"]
    batch_size = args.batch_size or s3["batch_size"]
    train_loader = make_dataloader(train_records, tokenizer, cfg, batch_size=batch_size, shuffle=True)
    val_loader = make_dataloader(val_records, tokenizer, cfg, batch_size=batch_size)

    print("[route] == 阶段 3a：仅训练路由器 ==", flush=True)
    train_phase(
        model, train_loader, device, cfg, s3["router_epochs"], s3["router_lr"], class_weights,
        train_experts=False, save_path=f"{args.out}/last_route.pt", resume=args.resume, desc="route",
    )
    metrics = evaluate_fusion(model, val_loader, device)
    print(f"[route] 验证 macro_f1={metrics['macro_f1']:.4f} ece={metrics['ece']:.4f}")
    print(f"[route] 专家激活比例 f_i={np.round(metrics['gate_activation_f'], 3).tolist()}")
    print(f"[route] 平均门控概率 p_i={np.round(metrics['gate_probability_p'], 3).tolist()}")

    if args.joint:
        print("[route] == 阶段 3b：联合微调（解冻路由器 + 各专家模块，backbone 仍冻结）==", flush=True)
        for expert in model.experts:
            unfreeze_module(expert)
        train_phase(
            model, train_loader, device, cfg, s3["joint_epochs"], s3["joint_lr"], class_weights,
            train_experts=True, save_path=f"{args.out}/last_joint.pt", resume=f"{args.out}/last_joint.pt", desc="joint",
        )
        metrics = evaluate_fusion(model, val_loader, device)
        print(f"[route] 联合微调后 macro_f1={metrics['macro_f1']:.4f} ece={metrics['ece']:.4f}")
        print(f"[route] 专家激活比例 f_i={np.round(metrics['gate_activation_f'], 3).tolist()}")

    save_checkpoint(model, f"{args.out}/model.pt", label_names=cfg["data"]["label_names"], model_name=name)
    write_json(
        f"{args.out}/metrics.json",
        {
            "macro_f1": metrics["macro_f1"],
            "ece": metrics["ece"],
            "gate_activation_f": metrics["gate_activation_f"],
            "gate_probability_p": metrics["gate_probability_p"],
            "joint": args.joint,
        },
    )
    print(f"[route] 完成，checkpoint 已保存到 {args.out}")


if __name__ == "__main__":
    main()
