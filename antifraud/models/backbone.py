"""backbone 构建：MacBERT + 可选 LoRA / DoRA（PEFT）

method: "lora" | "dora" | "none"（全量微调）
导出 ONNX 前用 merge_peft() 将低秩权重合并回原权重
model_name 传 "tiny-random" 可构建一个随机初始化的微型 BERT，供离线冒烟测试使用（不下载任何权重，不可用于真实训练）
"""

from __future__ import annotations

from pathlib import Path

import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, BertConfig, BertModel

# 本地模型目录：models/<别名>/。存在时优先离线加载，不访问 Hugging Face Hub，使训练/评估/部署在无网机器上可用；目录缺失时回退到 Hub id（联网按原逻辑下载或读缓存）
LOCAL_MODEL_ROOT = Path(__file__).resolve().parents[2] / "models"
MODEL_ALIASES = {
    "hfl/chinese-macbert-base": "chinese-macbert-base",
}


def resolve_model_path(model_name: str) -> str:
    # 把 HF 模型 id 解析为本地目录（若存在），实现离线优先
    if Path(model_name).is_dir():
        return model_name
    alias = MODEL_ALIASES.get(model_name)
    if alias and (LOCAL_MODEL_ROOT / alias).is_dir():
        return str(LOCAL_MODEL_ROOT / alias)
    return model_name


def build_backbone(model_name: str, peft_cfg: dict | None = None) -> nn.Module:
    if model_name == "tiny-random":
        config = BertConfig(
            vocab_size=500,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=128,
            max_position_embeddings=256,
        )
        return BertModel(config)

    model = AutoModel.from_pretrained(resolve_model_path(model_name))
    if peft_cfg and peft_cfg.get("enabled") and peft_cfg.get("method", "lora") != "none":
        from peft import LoraConfig, get_peft_model

        lora = LoraConfig(
            r=peft_cfg.get("r", 8),
            lora_alpha=peft_cfg.get("alpha", 16),
            lora_dropout=peft_cfg.get("dropout", 0.1),
            target_modules=list(peft_cfg.get("target_modules", ["query", "key", "value"])),
            use_dora=peft_cfg.get("method") == "dora",
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()
    return model


def build_tokenizer(model_name: str, tokenizer_name: str | None = None):
    if model_name == "tiny-random":
        from antifraud.data.dataset import TinyTokenizer

        return TinyTokenizer()
    return AutoTokenizer.from_pretrained(tokenizer_name or resolve_model_path(model_name))


def merge_peft(model: nn.Module) -> nn.Module:
    # 合并 LoRA/DoRA 权重到 backbone，用于 ONNX 导出（合并后参数量与推理延迟与全量微调一致）
    try:
        from peft import PeftModel

        if isinstance(model, PeftModel):
            return model.merge_and_unload()
    except ImportError:
        pass
    return model
