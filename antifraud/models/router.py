"""路由器

输入 h_r = [均值池化; 最大池化]（2d 维），经投影 + LayerNorm + Dropout 后
进入 2 层 MLP，softmax 输出 M 维专家权重。温度固定为 1，不引入温度调度：
路由分布的锐化由负载均衡损失系数 alpha 统一调控。
"""

import torch
import torch.nn as nn

from .pooling import MeanMaxPooling


class Router(nn.Module):
    def __init__(self, backbone_dim: int, num_experts: int, dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.pooling = MeanMaxPooling()
        self.proj = nn.Linear(backbone_dim * 2, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, num_experts))

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        u = self.drop(self.norm(self.proj(self.pooling(hidden, attention_mask))))
        return self.mlp(u).softmax(dim=-1)
