"""异构专家网络

每个专家 = 独立池化 + MLP 分类头，输出 C 维 logits
所有专家损失统一为加权交叉熵（不异构化损失函数），
分工机制 = 对自己负责类别的损失加权 lambda（默认 3.0）+ 数据侧重采样（见训练策略）。
"""

import torch
import torch.nn as nn

from .pooling import POOLING_OUTPUT_MULT, build_pooling


class ExpertHead(nn.Module):
    """单个专家。num_hidden_layers 为隐藏层层数（输出层另计），与 configs/default.yaml 专家表对应：
    推销 2 层(512)、刷单 3 层(256)、钓鱼 2 层(512)、正常 2 层(256)。
    """

    def __init__(
        self,
        name: str,
        pooling: str,
        hidden_dim: int,
        num_hidden_layers: int,
        backbone_dim: int,
        num_classes: int = 4,
        dropout: float = 0.1,
        augment: list[str] | None = None,
    ):
        super().__init__()
        self.name = name
        self.pooling_name = pooling
        self.augment = augment or []
        self.pooling = build_pooling(pooling, backbone_dim)
        in_dim = backbone_dim * POOLING_OUTPUT_MULT[pooling]
        blocks: list[nn.Module] = []
        d = in_dim
        for _ in range(num_hidden_layers):
            blocks += [nn.Linear(d, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden_dim
        blocks.append(nn.Linear(d, num_classes))
        self.mlp = nn.Sequential(*blocks)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.pooling(hidden, attention_mask))


def build_experts(experts_cfg: list[dict], backbone_dim: int, num_classes: int, dropout: float) -> nn.ModuleList:
    return nn.ModuleList(
        ExpertHead(
            name=e["name"],
            pooling=e["pooling"],
            hidden_dim=e["hidden"],
            num_hidden_layers=e["layers"],
            backbone_dim=backbone_dim,
            num_classes=num_classes,
            dropout=dropout,
            augment=e.get("augment"),
        )
        for e in experts_cfg
    )
