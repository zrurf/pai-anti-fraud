"""三阶段训练的共用构件：模型构建、DataLoader、冻结/解冻、checkpoint 读写。"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from antifraud.data.augment import build_augment_fn
from antifraud.data.dataset import TextDataset, make_collate_fn
from antifraud.models import MoEFraudClassifier, build_backbone

TINY_RANDOM_HIDDEN = 64  # tiny-random 模型的 hidden_size（仅冒烟测试用）


def resolve_model_name(cfg: dict, model_name: str | None) -> str:
    return model_name or cfg["model"]["name"]


def backbone_dim_of(model_name: str, cfg: dict) -> int:
    return TINY_RANDOM_HIDDEN if model_name == "tiny-random" else cfg["model"]["hidden_size"]


def build_model(cfg: dict, model_name: str | None = None) -> MoEFraudClassifier:
    name = resolve_model_name(cfg, model_name)
    backbone = build_backbone(name, cfg.get("peft"))
    return MoEFraudClassifier(
        backbone,
        backbone_dim=backbone_dim_of(name, cfg),
        num_classes=cfg["data"]["num_classes"],
        experts_cfg=cfg["experts"],
        router_cfg=cfg["router"],
        dropout=cfg["model"]["dropout"],
    )


def make_dataloader(
    records: list[dict],
    tokenizer,
    cfg: dict,
    batch_size: int,
    shuffle: bool = False,
    sampler_weights: list[float] | None = None,
    augment_kinds: list[str] | None = None,
) -> DataLoader:
    dataset = TextDataset(records)
    collate = make_collate_fn(tokenizer, cfg["data"]["max_length"], augmentation=build_augment_fn(augment_kinds or []))
    if sampler_weights is not None:
        sampler = WeightedRandomSampler(sampler_weights, num_samples=len(dataset), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


def freeze_module(module: torch.nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)


def unfreeze_module(module: torch.nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(True)


def freeze_backbone(model: MoEFraudClassifier) -> None:
    freeze_module(model.backbone)


def save_checkpoint(model: torch.nn.Module, path: str, **extra) -> None:
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), **extra}, path)


def load_checkpoint(model: torch.nn.Module, path: str, strict: bool = True) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=strict)
    return ckpt


def _strip_state_prefixes(key: str) -> str:
    for prefix in ("base_model.model.", "bert."):
        while key.startswith(prefix):
            key = key[len(prefix) :]
    return key


def load_backbone_weights(model: MoEFraudClassifier, path: str) -> tuple[int, int]:
    """尽力把预训练 checkpoint（如阶段一 MLM 产物）的权重装入 backbone。

    不同来源的 state_dict 键前缀不同（"bert." / "base_model.model." 等），源与目标两侧都
    逐前缀归一后做键匹配（只归一源侧会因目标键带 PEFT 前缀而 0 匹配），返回 (匹配数, 总数)。
    不匹配的键（如 MLM 头）会被跳过。
    """
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    target = model.backbone.state_dict()
    source = {_strip_state_prefixes(k): v for k, v in state.items()}
    matched = 0
    for key, value in target.items():
        src = source.get(_strip_state_prefixes(key))
        if src is not None and src.shape == value.shape:
            target[key] = src
            matched += 1
    model.backbone.load_state_dict(target)
    return matched, len(state)


def resolve_class_weights(cfg: dict, labels: list[int]) -> torch.Tensor | None:
    from antifraud.training.utils import class_weights_from

    setting = cfg["loss"]["class_weights"]
    if setting == "auto":
        return class_weights_from(labels, cfg["data"]["num_classes"])
    if isinstance(setting, list):
        return torch.tensor(setting, dtype=torch.float32)
    return None


def linear_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def fn(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)
