"""评估指标：Macro-F1、ECE、McNemar 检验"""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score, precision_recall_fscore_support


def macro_f1(y_true: list[int], y_pred: list[int]) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def per_class_prf(y_true: list[int], y_pred: list[int], label_names: list[str]) -> dict[str, dict[str, float]]:
    # 显式传入 labels：数据只覆盖部分类别时（如初期只有正常/钓鱼），仍按完整标签表输出
    labels = list(range(len(label_names)))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        label_names[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])}
        for i in range(len(label_names))
    }


def expected_calibration_error(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """ECE = Σ_k |B_k|/N · |acc(B_k) − conf(B_k)|，K 为置信度区间数（与专家数量 M 无关）。"""
    conf = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for k in range(n_bins):
        mask = (conf > edges[k]) & (conf <= edges[k + 1])
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def mcnemar_test(y_true: list[int], pred_a: list[int], pred_b: list[int]) -> dict[str, float]:
    """比较模型 A 与 B 在同一测试集上的差异显著性（精确二项检验）。

    只统计两者预测不一致的样本：n01 = A 对 B 错，n10 = A 错 B 对。
    """
    y = np.asarray(y_true)
    a = np.asarray(pred_a)
    b = np.asarray(pred_b)
    n01 = int(((a == y) & (b != y)).sum())
    n10 = int(((a != y) & (b == y)).sum())
    n = n01 + n10
    if n == 0:
        return {"n01": n01, "n10": n10, "p_value": 1.0}
    p_value = float(binomtest(n01, n, 0.5).pvalue)
    return {"n01": n01, "n10": n10, "p_value": p_value}
