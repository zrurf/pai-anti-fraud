"""损失函数

- weighted_ce: 加权交叉熵。全局类权重 w_c 处理类别不平衡；
  专家分工权重 lambda_{i,i}=3（对自己负责的类别加权）仅用于阶段二。
- load_balance_loss: Switch Transformer 形式的负载均衡损失。
  f_i 由 argmax 硬分配统计（不参与梯度），梯度仅经软路由概率 p_i 回传。
"""

import torch
import torch.nn.functional as F


def weighted_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    lambda_self: float | None = None,
    self_class: int | None = None,
) -> torch.Tensor:
    per_sample = F.cross_entropy(logits, labels, weight=class_weights, reduction="none")
    if lambda_self is not None and self_class is not None:
        multiplier = torch.where(labels == self_class, torch.tensor(lambda_self, device=labels.device), torch.tensor(1.0, device=labels.device))
        per_sample = per_sample * multiplier
    return per_sample.mean()


def load_balance_loss(gate: torch.Tensor, alpha: float = 0.01) -> torch.Tensor:
    """gate: [batch, M] 的软路由概率。L_bal = alpha * M * Σ_i f_i · p_i。"""
    num_experts = gate.shape[1]
    assign = gate.argmax(dim=-1)  # 硬分配，detached（不参与梯度）
    f = torch.bincount(assign, minlength=num_experts).float() / gate.shape[0]
    p = gate.mean(dim=0)
    return alpha * num_experts * (f * p).sum()
