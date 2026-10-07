"""PII 脱敏

mask_regex：高置信度正则规则：URL/邮箱/身份证/手机号/固话/银行卡/关键词账号（微信号/QQ/验证码等）
mask_pii：在正则之上叠加 NER 姓名识别（uer/roberta-base-finetuned-cluener2020-chinese，CLUENER"name"类），可选传入 ner pipeline（GPU/CPU 均可）；不传则只做正则

替换占位符：[电话] [证件号] [银行卡] [网址] [邮箱] [账号] [姓名] [数字]
金额/数量类的 4-12 位短数字保留（"转账5000元"），避免破坏训练信号
"""

from __future__ import annotations

import re

PHONE = re.compile(r"(?<!\d)(?:1[3-9]\d{9}|0\d{2,3}-?\d{7,8}|400-?\d{3,4}-?\d{3,4})(?!\d)")
ID_CARD = re.compile(r"(?<!\d)(?:\d{17}[\dXx]|\d{15})(?!\d)")
BANK_CARD = re.compile(r"(?<!\d)\d{13,19}(?!\d)")
URL = re.compile(r"(?:https?://|www\.)[^\s，。；、,;；]{2,}", re.I)
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
ACCOUNT = re.compile(
    r"(?:微信号?|微信是|加微信?|加微|加V|V信|vx|vx号|WX|QQ号?|扣扣|账号|帐号|卡号|卡是|密码|验证码|取件码)[:：是,，]?[\s]*([A-Za-z0-9_\-]{4,25})"
)
LONG_DIGIT = re.compile(r"(?<!\d)\d{12}(?!\d)")  # 12 位残留长数字串（未命中上述规则的兜底）

NER_LABELS = {"name", "per"}  # CLUENER 的 "name" 与通用模型的 "PER"


def _regex_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for m in URL.finditer(text):
        spans.append((m.start(), m.end(), "[网址]"))
    for m in EMAIL.finditer(text):
        spans.append((m.start(), m.end(), "[邮箱]"))
    for m in ID_CARD.finditer(text):
        spans.append((m.start(), m.end(), "[证件号]"))
    for m in PHONE.finditer(text):
        spans.append((m.start(), m.end(), "[电话]"))
    for m in BANK_CARD.finditer(text):
        spans.append((m.start(), m.end(), "[银行卡]"))
    for m in ACCOUNT.finditer(text):
        spans.append((m.start(1), m.end(1), "[账号]"))
    for m in LONG_DIGIT.finditer(text):
        spans.append((m.start(), m.end(), "[数字]"))
    return spans


def _apply_spans(text: str, spans: list[tuple[int, int, str]]) -> str:
    # 按位置应用替换，重叠时保留先出现（更靠左）的规则
    spans = sorted(spans, key=lambda x: (x[0], -(x[1] - x[0])))
    out: list[str] = []
    last = 0
    for s, e, ph in spans:
        if s < last:
            continue
        out.append(text[last:s])
        out.append(ph)
        last = e
    out.append(text[last:])
    return "".join(out)


def mask_regex(text: str) -> str:
    return _apply_spans(text, _regex_spans(text))


def load_ner(device: str = "cpu"):
    # 加载 CLUENER 中文 NER（首次需联网下载，已预缓存可离线）
    from transformers import pipeline

    return pipeline(
        "ner",
        model="uer/roberta-base-finetuned-cluener2020-chinese",
        aggregation_strategy="simple",
        device=device,
    )


def mask_pii(text: str, ner=None) -> str:
    # 正则 +NER 姓名识别的完整脱敏
    spans = _regex_spans(text)
    if ner is not None:
        for ent in ner(text):
            if str(ent.get("entity_group", "")).lower() in NER_LABELS:
                spans.append((ent["start"], ent["end"], "[姓名]"))
    return _apply_spans(text, spans)


def pii_counts(before: str, after: str) -> dict[str, int]:
    # 统计脱敏前后各占位符出现次数
    counts: dict[str, int] = {}
    for ph in ("[电话]", "[证件号]", "[银行卡]", "[网址]", "[邮箱]", "[账号]", "[姓名]", "[数字]"):
        n = after.count(ph) - before.count(ph)
        if n > 0:
            counts[ph] = n
    return counts
