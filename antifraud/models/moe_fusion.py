"""异构专家融合

数学形式：
    z_i = E_i(p_i(H))                       每个专家独立输出 C 维 logits
    g   = softmax(router([mean;max] 池化))   路由器输出 M 维权重
    z   = Σ_i g_i · z_i                     门控加权融合
    y   = softmax(z)

forward_stage1() 是阶段一的单分类头路径（均值池化 + Linear），该头即为对比实验中的"MacBERT 单头"基线模型。
"""

import torch
import torch.nn as nn

from .experts import build_experts
from .pooling import MeanPooling
from .router import Router

DEFAULT_EXPERTS = [
    {"name": "推销", "pooling": "mean", "hidden": 512, "layers": 2, "self_class": 1},
    {"name": "刷单", "pooling": "attention", "hidden": 256, "layers": 3, "self_class": 2},
    {"name": "钓鱼", "pooling": "cls", "hidden": 512, "layers": 2, "self_class": 3},
    {"name": "正常", "pooling": "mean_max", "hidden": 256, "layers": 2, "self_class": 0},
]


class MoEFraudClassifier(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        backbone_dim: int = 768,
        num_classes: int = 4,
        experts_cfg: list[dict] | None = None,
        router_cfg: dict | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.pool_mean = MeanPooling()
        self.drop = nn.Dropout(dropout)
        self.stage1_head = nn.Linear(backbone_dim, num_classes)
        self.experts = build_experts(experts_cfg or DEFAULT_EXPERTS, backbone_dim, num_classes, dropout)
        router_cfg = router_cfg or {}
        self.router = Router(backbone_dim, len(self.experts), router_cfg.get("dim", 256), router_cfg.get("dropout", dropout))

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state

    def forward_stage1(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # 阶段一：单分类头四分类，产出即对比基线主模型
        pooled = self.drop(self.pool_mean(self.encode(input_ids, attention_mask), attention_mask))
        return self.stage1_head(pooled)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        # 阶段二/三与推理：多专家门控融合路径
        hidden = self.encode(input_ids, attention_mask)
        expert_logits = torch.stack([e(hidden, attention_mask) for e in self.experts], dim=1)  # [B, M, C]
        gate = self.router(hidden, attention_mask)  # [B, M]
        fused = (gate.unsqueeze(-1) * expert_logits).sum(dim=1)  # z = Σ g_i · z_i
        return {"logits": fused, "expert_logits": expert_logits, "gate": gate}
