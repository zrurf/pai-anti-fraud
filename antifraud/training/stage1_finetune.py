"""阶段一 Branch：MLM 继续预训练 + 四分类单头监督微调

主路径 = MLM 继续预训练 → 四分类单头微调（可选的二分类中间预训练只用于消融 A5 对照组）。
单头模型的产出即对比实验中的"MacBERT+PEFT 单头"基线（主模型 B/C）。

用法：
  # 1) MLM 继续预训练（80 万短信语料，每行 "label\\ttext"）
  python -m antifraud.training.stage1_finetune --task mlm \
      --corpus "rawdata/SpamMessage-master/data/带标签短信.txt" --out checkpoints/stage1_mlm

  # 2) 四分类单头微调（--init-from 可选接入 MLM 产物）
  python -m antifraud.training.stage1_finetune --task clf \
      --train data/processed/train.jsonl --val data/processed/val.jsonl \
      --out checkpoints/stage1 [--init-from checkpoints/stage1_mlm/model.pt]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from antifraud.config import load_config
from antifraud.models.backbone import resolve_model_path
from antifraud.training.pipeline import (
    build_model,
    linear_warmup_scheduler,
    load_backbone_weights,
    make_dataloader,
    resolve_class_weights,
    resolve_model_name,
    save_checkpoint,
)
from antifraud.training.losses import weighted_ce
from antifraud.training.metrics import macro_f1
from antifraud.training.utils import get_device, set_seed, write_json


def mlm_collate_fn(tokenizer, max_length: int, mlm_probability: float):
    """手工实现 15% MLM 掩码（不依赖特定版本的数据整理器）。特殊 token 不参与掩码。"""
    special_ids = {
        sid
        for sid in (
            getattr(tokenizer, "pad_token_id", None),
            getattr(tokenizer, "cls_token_id", None),
            getattr(tokenizer, "sep_token_id", None),
        )
        if sid is not None
    }
    mask_id = getattr(tokenizer, "mask_token_id", None)
    if mask_id is None:
        mask_id = getattr(tokenizer, "UNK", 1)  # tiny-random 回退：用 [UNK] 代替 [MASK]

    def collate(batch: list[dict]) -> dict:
        texts = [b["text"] for b in batch]
        enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]
        labels = input_ids.clone()
        prob = torch.full(labels.shape, mlm_probability)
        for sid in special_ids:
            prob[input_ids == sid] = 0.0
        masked = torch.bernoulli(prob).bool() & attention_mask.bool()
        labels[~masked] = -100
        input_ids[masked] = mask_id
        return {"input_ids": input_ids, "attention_mask": attention_mask, "mlm_labels": labels}

    return collate


def run_mlm(cfg: dict, args) -> None:
    from torch.utils.data import DataLoader
    from transformers import AutoModelForMaskedLM, BertConfig, BertForMaskedLM

    from antifraud.models import build_tokenizer

    set_seed(cfg.get("seed", 42))
    name = resolve_model_name(cfg, args.model_name)
    tokenizer = build_tokenizer(name, args.tokenizer_name)
    if name == "tiny-random":
        mlm_model = BertForMaskedLM(
            BertConfig(vocab_size=500, hidden_size=64, num_hidden_layers=2, num_attention_heads=2, intermediate_size=128, max_position_embeddings=256)
        )
    else:
        mlm_model = AutoModelForMaskedLM.from_pretrained(resolve_model_path(name))
        if cfg.get("peft", {}).get("enabled") and cfg["peft"].get("method", "lora") != "none":
            from peft import LoraConfig, get_peft_model

            lora = LoraConfig(
                r=cfg["peft"]["r"],
                lora_alpha=cfg["peft"]["alpha"],
                lora_dropout=cfg["peft"]["dropout"],
                target_modules=list(cfg["peft"]["target_modules"]),
                use_dora=cfg["peft"]["method"] == "dora",
            )
            mlm_model = get_peft_model(mlm_model, lora)

    texts: list[dict] = []
    with open(args.corpus, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text = line.split("\t", 1)[1] if "\t" in line else line  # 80万短信语料格式 "label\ttext"
            if text:
                texts.append({"text": text})
            if args.max_samples and len(texts) >= args.max_samples:
                break
    print(f"[MLM] 语料 {len(texts)} 条")

    loader = DataLoader(texts, batch_size=args.batch_size, shuffle=True, collate_fn=mlm_collate_fn(tokenizer, cfg["data"]["max_length"], cfg["train"]["stage1"]["mlm_probability"]))
    optimizer = torch.optim.AdamW((p for p in mlm_model.parameters() if p.requires_grad), lr=args.lr, weight_decay=cfg["train"]["weight_decay"])
    device = get_device()
    mlm_model.to(device)

    steps = args.epochs * len(loader)
    scheduler = linear_warmup_scheduler(optimizer, int(cfg["train"]["warmup_ratio"] * steps), steps)

    start_epoch = 0
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.is_dir():
            resume_path = resume_path / "last.pt"
        if resume_path.exists():
            ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
            mlm_model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"]
            print(f"[MLM] 从 epoch {start_epoch} 断点续训", flush=True)
        else:
            print(f"[MLM] 未找到断点 {resume_path}，从头训练", flush=True)

    for epoch in range(start_epoch, args.epochs):
        mlm_model.train()
        total, count = 0.0, 0
        for batch in tqdm(loader, desc=f"MLM epoch {epoch + 1}/{args.epochs}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = mlm_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["mlm_labels"])
            optimizer.zero_grad()
            out.loss.backward()
            optimizer.step()
            scheduler.step()
            total += out.loss.item()
            count += 1
        print(f"[MLM] epoch {epoch + 1} loss = {total / max(1, count):.4f}", flush=True)
        Path(args.out).mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": mlm_model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch + 1},
            f"{args.out}/last.pt",
        )

    save_checkpoint(mlm_model, f"{args.out}/model.pt")
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(args.out)
    print(f"[MLM] 已保存到 {args.out}/model.pt")


@torch.no_grad()
def evaluate_stage1(model, loader, device) -> dict:
    model.eval()
    ys, ps, confs = [], [], []
    for batch in loader:
        logits = model.forward_stage1(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        prob = logits.softmax(dim=-1)
        ps += prob.argmax(dim=-1).tolist()
        ys += batch["labels"].tolist()
        confs += prob.max(dim=-1).values.tolist()
    return {"macro_f1": macro_f1(ys, ps), "predictions": ps, "labels": ys, "confidences": confs}


def run_clf(cfg: dict, args) -> None:
    from antifraud.data.dataset import load_split
    from antifraud.models import build_tokenizer
    from antifraud.training.pipeline import load_checkpoint, make_dataloader

    set_seed(cfg.get("seed", 42))
    name = resolve_model_name(cfg, args.model_name)
    tokenizer = build_tokenizer(name, args.tokenizer_name)
    model = build_model(cfg, model_name=name)

    if args.init_from:
        matched, total = load_backbone_weights(model, args.init_from)
        print(f"[clf] 从 {args.init_from} 装入 backbone 权重：{matched}/{total} 个键匹配")

    train_records = load_split(args.train)
    val_records = load_split(args.val)
    labels = [r["label"] for r in train_records]
    class_weights = resolve_class_weights(cfg, labels)
    if class_weights is not None:
        class_weights = class_weights.to(get_device())
    print(f"[clf] 训练 {len(train_records)} 条 / 验证 {len(val_records)} 条")

    train_loader = make_dataloader(train_records, tokenizer, cfg, batch_size=args.batch_size, shuffle=True)
    val_loader = make_dataloader(val_records, tokenizer, cfg, batch_size=args.batch_size)

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=cfg["train"]["weight_decay"])
    device = get_device()
    model.to(device)

    steps = args.epochs * len(train_loader)
    scheduler = linear_warmup_scheduler(optimizer, int(cfg["train"]["warmup_ratio"] * steps), steps)

    best_f1, history = -1.0, []
    start_epoch = 0
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.is_dir():
            resume_path = resume_path / "last.pt"
        if resume_path.exists():
            ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"]
            best_f1 = ckpt["best_f1"]
            history = ckpt.get("history", [])
            print(f"[clf] 从 epoch {start_epoch} 断点续训（best_f1={best_f1:.4f}）", flush=True)
        else:
            print(f"[clf] 未找到断点 {resume_path}，从头训练", flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total, count = 0.0, 0
        for batch in tqdm(train_loader, desc=f"clf epoch {epoch + 1}/{args.epochs}"):
            logits = model.forward_stage1(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            loss = weighted_ce(logits, batch["labels"].to(device), class_weights=class_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item()
            count += 1
        metrics = evaluate_stage1(model, val_loader, device)
        history.append({"epoch": epoch + 1, "train_loss": total / max(1, count), "val_macro_f1": metrics["macro_f1"]})
        print(f"[clf] epoch {epoch + 1} loss={history[-1]['train_loss']:.4f} val_macro_f1={metrics['macro_f1']:.4f}", flush=True)
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            save_checkpoint(model, f"{args.out}/model.pt", label_names=cfg["data"]["label_names"], model_name=name)
            if hasattr(tokenizer, "save_pretrained"):
                tokenizer.save_pretrained(args.out)
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch + 1, "best_f1": best_f1, "history": history},
            f"{args.out}/last.pt",
        )

    write_json(f"{args.out}/metrics.json", {"best_val_macro_f1": best_f1, "history": history})
    print(f"[clf] 最佳 val_macro_f1={best_f1:.4f}，checkpoint 已保存到 {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段一：MLM 继续预训练 / 四分类单头微调")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--task", choices=["mlm", "clf"], required=True)
    parser.add_argument("--corpus", help="mlm: 纯文本语料路径（每行文本或 label\\ttext）")
    parser.add_argument("--train", help="clf: 训练集 JSONL")
    parser.add_argument("--val", help="clf: 验证集 JSONL")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-name", dest="model_name", default=None, help="覆盖 backbone（tiny-random 用于离线冒烟测试）")
    parser.add_argument("--tokenizer-name", dest="tokenizer_name", default=None)
    parser.add_argument("--init-from", dest="init_from", default=None, help="clf: 可选，从 MLM checkpoint 装入 backbone 权重")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-samples", dest="max_samples", type=int, default=0)
    parser.add_argument("--resume", dest="resume", default=None, help="断点续训：checkpoint 文件或含 last.pt 的输出目录")
    args = parser.parse_args()

    cfg = load_config(args.config)
    args.epochs = args.epochs or (cfg["train"]["stage1"]["mlm_epochs"] if args.task == "mlm" else cfg["train"]["stage1"]["clf_epochs"])
    args.batch_size = args.batch_size or cfg["train"]["stage1"]["batch_size"]
    args.lr = args.lr or (cfg["train"]["stage1"]["mlm_lr"] if args.task == "mlm" else cfg["train"]["stage1"]["clf_lr"])

    if args.task == "mlm":
        assert args.corpus, "--task mlm 需要 --corpus"
        run_mlm(cfg, args)
    else:
        assert args.train and args.val, "--task clf 需要 --train 与 --val"
        run_clf(cfg, args)


if __name__ == "__main__":
    main()
