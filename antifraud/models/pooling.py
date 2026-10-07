"""池化算子

backbone 输出 token 级隐藏状态序列 H（[batch, seq_len, dim]），
各专家按异构性配置选择自己的池化策略，
路由器使用均值+最大池化拼接（兼顾平铺与峰值两种统计视角）。
"""

import torch
import torch.nn as nn


class MeanPooling(nn.Module):
    """按 attention_mask 加权求平均，输出 [batch, dim]。"""

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


class MaxPooling(nn.Module):
    """按 attention_mask 取逐维最大值，输出 [batch, dim]。"""

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).bool()
        neg = torch.finfo(hidden.dtype).min
        return hidden.masked_fill(~mask, neg).max(dim=1).values


class CLSPooling(nn.Module):
    """取 [CLS] token 表示，输出 [batch, dim]。"""

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return hidden[:, 0]


class AttentionPooling(nn.Module):
    """单头注意力池化：可学习查询向量对每个 token 打分，softmax 后加权求和。

    注意：含可学习参数。阶段三联合微调时必须随专家 MLP 一起解冻。
    """

    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim) / dim**0.5)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        scores = (hidden @ self.query) / self.query.shape[0] ** 0.5
        scores = scores.masked_fill(attention_mask == 0, torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1).unsqueeze(-1)
        return (hidden * weights).sum(dim=1)


class MeanMaxPooling(nn.Module):
    """均值池化与最大池化拼接，输出 [batch, 2*dim]。"""

    def __init__(self):
        super().__init__()
        self.mean = MeanPooling()
        self.max = MaxPooling()

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.mean(hidden, attention_mask), self.max(hidden, attention_mask)], dim=-1)


POOLING_OUTPUT_MULT = {"mean": 1, "attention": 1, "cls": 1, "mean_max": 2}


def build_pooling(name: str, dim: int) -> nn.Module:
    cls = {
        "mean": MeanPooling,
        "attention": AttentionPooling,
        "cls": CLSPooling,
        "mean_max": MeanMaxPooling,
    }[name]
    return cls(dim) if name == "attention" else cls()
