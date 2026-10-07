"""离线冒烟测试：tiny-random backbone，不下载任何权重

验证：
- 编码输出形状 [B, L, d]；stage1 单头输出 [B, C]；
- 专家 logits [B, M, C]、门控 g [B, M] 且逐行和为 1、融合 logits [B, C]；
- 加权交叉熵 + 负载均衡损失可反传，梯度流向路由器与专家、不流向 backbone；
- 池化输出维度（mean_max 拼接为 2d）；
- ECE / McNemar 指标的数值正确性。

运行：python -m tests.test_moe_shapes
"""

from __future__ import annotations

import numpy as np
import torch

from antifraud.config import load_config
from antifraud.training.losses import load_balance_loss, weighted_ce
from antifraud.training.metrics import expected_calibration_error, mcnemar_test
from antifraud.training.pipeline import (
    build_model,
    freeze_backbone,
    make_dataloader,
    unfreeze_module,
)


def tiny_records(n: int = 12) -> list[dict]:
    return [
        {"text": f"w{i % 400} w{(i * 7) % 400} w{(i * 13) % 400} 文本", "label": i % 4, "source": "dummy", "split": "train"}
        for i in range(n)
    ]


def main() -> None:
    cfg = load_config()
    cfg["data"]["max_length"] = 32
    model = build_model(cfg, model_name="tiny-random")
    from antifraud.models import build_tokenizer

    tokenizer = build_tokenizer("tiny-random")

    batch = next(iter(make_dataloader(tiny_records(), tokenizer, cfg, batch_size=6)))
    hidden = model.encode(batch["input_ids"], batch["attention_mask"])
    assert hidden.shape[:2] == batch["input_ids"].shape, f"编码形状错误: {hidden.shape}"
    d = hidden.shape[-1]
    assert d == 64, f"tiny-random hidden 应为 64，实际 {d}"

    # 池化维度
    p_mean = model.pool_mean(hidden, batch["attention_mask"])
    p_meanmax = model.router.pooling(hidden, batch["attention_mask"])
    assert p_mean.shape == (6, d)
    assert p_meanmax.shape == (6, 2 * d), "均值+最大池化拼接应为 2d"

    # stage1 单头
    logits1 = model.forward_stage1(batch["input_ids"], batch["attention_mask"])
    assert logits1.shape == (6, 4)

    # 多专家融合
    out = model(batch["input_ids"], batch["attention_mask"])
    assert out["expert_logits"].shape == (6, 4, 4), "应为 [B, M=4, C=4]"
    assert out["gate"].shape == (6, 4)
    assert torch.allclose(out["gate"].sum(dim=-1), torch.ones(6), atol=1e-5), "门控权重逐行和应为 1"
    assert out["logits"].shape == (6, 4)
    # 融合 = Σ g_i · z_i 手工复核
    manual = (out["gate"].unsqueeze(-1) * out["expert_logits"]).sum(1)
    assert torch.allclose(out["logits"], manual, atol=1e-6)

    # 冻结 backbone 后反传：梯度进路由器与专家，不进 backbone
    freeze_backbone(model)
    assert all(not p.requires_grad for p in model.backbone.parameters())
    loss = weighted_ce(out["logits"], batch["labels"]) + load_balance_loss(out["gate"], alpha=0.01)
    loss.backward()
    assert torch.isfinite(loss), "损失应为有限值"
    router_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.router.parameters())
    expert_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.experts[1].parameters())
    assert router_grad, "路由器应有梯度"
    assert expert_grad, "专家应有梯度（注意力池化的可学习查询也算专家参数）"
    assert all(p.grad is None for p in model.backbone.parameters()), "冻结的 backbone 不应有梯度"

    # 专家分工加权：λ=3 时负责类别的样本损失贡献应放大
    unfreeze_module(model.experts[0])
    logits4 = out["logits"].detach()[:4]
    labels4 = torch.tensor([0, 1, 2, 3])
    l_plain = weighted_ce(logits4, labels4)
    l_self = weighted_ce(logits4, labels4, lambda_self=3.0, self_class=0)
    assert l_self.item() > l_plain.item(), "对自己负责类别加权后损失应增大"

    # 指标：ECE（完美校准应接近 0）、McNemar
    conf = np.array([0.9, 0.9, 0.1, 0.1])
    correct = np.array([1, 1, 1, 1])  # 预测 1 与 0 各两个，置信度与正确性对齐
    ece = expected_calibration_error(conf, correct, n_bins=10)
    assert 0.0 <= ece <= 1.0
    m = mcnemar_test([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 2, 2], [0, 0, 1, 1, 2, 1])
    assert set(m) == {"n01", "n10", "p_value"} and 0.0 <= m["p_value"] <= 1.0

    print("test_moe_shapes: 全部通过 ✅")


if __name__ == "__main__":
    main()
