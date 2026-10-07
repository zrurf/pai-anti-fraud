"""各公开数据集与本项目四分类标签的映射表

映射原则：话术类型差异大的样本不强行映射（label=-1，仅作二分类/预训练用）
"""

from __future__ import annotations

LABEL_NAMES = ["正常", "推销", "刷单", "钓鱼"]

# CCL23-Eval 任务6 的 12 类（受害人笔录，存在文风偏移）
CCL23_MAP: dict[str, int] = {
    "刷单返利": 2,
    "虚假购物服务": 1,  # 可选映射
    "冒充电商物流客服": 3,
    "冒充公检法": 3,
    "冒充领导熟人": 3,
    "冒充军警购物": 3,
    # 贷款代办信用卡 / 虚假网络投资理财 / 虚假征信 / 网游虚假交易 / 网络婚恋交友 / 网黑案件 → 不映射
}

# 阿里云天池电信五分类（label00-04 文件名 → 标签）
TIANCHI5_MAP: dict[str, int] = {
    "label00": 0,  # 正常文本
    "label01": 3,  # 公安（公检法）诈骗 → 钓鱼
    "label02": -1,  # 贷款诈骗：与刷单话术差异大，不映射
    "label03": 3,  # 冒充客服 → 钓鱼
    "label04": 3,  # 冒充领导熟人 → 钓鱼
}

# TeleAntiFraud SFT 细分标注（answers 字段）→ 四分类。
# 映射规则见 DATASETS.md。
TELEANTIFRAUD_MAP: dict[str, int] = {
    # 7 类正常场景通话 → 正常（近领域负样本）
    "订餐服务": 0, "咨询客服": 0, "预约服务": 0, "交通咨询": 0,
    "日常购物": 0, "打车服务": 0, "外卖服务": 0,
    # 冒充/身份类诈骗 → 钓鱼
    "客服诈骗": 3, "银行诈骗": 3, "钓鱼诈骗": 3, "身份盗窃": 3,
    # 投资诈骗 / 彩票诈骗 / 绑架诈骗：与校园三类话术差异大，不映射
    # 二分类的 fraud / normal：无细分标注，不映射（fraud 仅作二分类正样本）
}

# FBS_SMS_Dataset 的 14 类文件名规则（实际目录名为下划线分隔，如 AD_Loan、FR_Phishing(Bank)）
FBS_PREFIX_MAP: list[tuple[str, int]] = [
    ("AD", 1),  # AD_Loan / AD_Retail / AD_Real_estate / AD_Network_service / AD_Other → 推销
    ("FR_Phishing", 3),  # FR_Phishing(Bank) / FR_Phishing(Other) → 钓鱼
    # FR_Financial / FR_Other / IL_*（赌博、发票等非法类）/ Other → 不映射
]


def map_ccl23(native_label: str) -> int:
    return CCL23_MAP.get(native_label.strip(), -1)


# FGRC-SCD 风险类别（与 CCL23 任务6 的 12 类同一 taxonomy，已实查数据确认；文档"标签映射"表）
FGRC_MAP: dict[str, int] = {
    "无风险": 0,
    "刷单返利类": 2,  # FGRC-SCD 无此类，CCL2023 镜像的主要类别
    "冒充电商物流客服类": 3,
    "冒充公检法及政府机关类": 3,
    "冒充领导、熟人类": 3,
    "冒充军警购物类诈骗": 3,
    "虚假购物、服务类": 1,  # 可选映射（推广式诱导话术），消融时可按 subtype 剔除
    # 虚假网络投资理财类 / 虚假信用服务类 / 网络婚恋、交友类 / 网黑案件 / 网络游戏产品虚假交易类 / 贷款代办信用卡类 → 不映射
}

# anti_fraud_dataset 的 CSV 文件名（诈骗类型）→ 标签；其余类型话术差异大不映射
ANTI_FRAUD_MAP: dict[str, int] = {
    "兼职刷单诈骗": 2,
    "冒充客服诈骗": 3,
}


def map_tianchi5(file_label: str) -> int:
    return TIANCHI5_MAP.get(file_label.strip(), -1)


def map_teleantifraud(native_answer: str) -> int:
    return TELEANTIFRAUD_MAP.get(native_answer.strip(), -1)


def map_fgrc(native_label: str) -> int:
    return FGRC_MAP.get(native_label.strip(), -1)


def map_anti_fraud(native_type: str) -> int:
    return ANTI_FRAUD_MAP.get(native_type.strip(), -1)


def map_fbs(dir_name: str) -> int:
    for prefix, label in FBS_PREFIX_MAP:
        if dir_name.startswith(prefix):
            return label
    return -1


def map_binary(is_fraud: bool) -> int:
    """二分类（欺诈/正常）到四分类空间的占位：欺诈样本映射为 -1，待细分。"""
    return -1 if is_fraud else 0
