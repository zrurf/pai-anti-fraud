"""轻量文本增强

各专家的数据侧重增强，按专家名注册增强组合
"""

from __future__ import annotations

import random

EMOJIS = "😀😅😂🥰😍🤔😊🙃😴🤑👍🔥💰🎁🎉"

# 简易同形/近形替换表（对抗鲁棒性评估也用同一张表）
HOMOGLYPHS = {
    "微": "溦", "信": "伩", "返": "反", "款": "欸", "卡": "乍",
    "单": "兽", "刷": "唰", "学": "茡", "教": "敎", "务": "務",
}


def emoji_insert(text: str, p: float = 0.12) -> str:
    # 推销专家
    out = []
    for ch in text:
        out.append(ch)
        if random.random() < p:
            out.append(random.choice(EMOJIS))
    return "".join(out)


def homoglyph_substitute(text: str, p: float = 0.05) -> str:
    # 同形字替换
    return "".join(HOMOGLYPHS.get(ch, ch) if random.random() < p else ch for ch in text)


def random_space(text: str, p: float = 0.04) -> str:
    # 随机空格插入
    return "".join(ch + " " if random.random() < p else ch for ch in text)


AUGMENT_REGISTRY: dict[str, callable] = {
    "emoji_insert": emoji_insert,
    "homoglyph": homoglyph_substitute,
    "random_space": random_space,
}


def build_augment_fn(kinds: list[str]) -> callable | None:
    if not kinds:
        return None
    fns = [AUGMENT_REGISTRY[k] for k in kinds]

    def apply(text: str) -> str:
        for fn in fns:
            text = fn(text)
        return text

    return apply
