"""生成占位用的假数据（离线冒烟测试与流水线联调）

真实数据到位后，用 scripts/download_datasets.py + antifraud/data/label_map.py
构建 data/processed/ 下的正式 train/val/test.jsonl，再删除本脚本产物。

用法：python -m scripts.make_dummy_data --out data/processed_dummy --per-class 60
"""

from __future__ import annotations

import argparse
import random

from antifraud.training.utils import write_jsonl

TEMPLATES: dict[int, list[str]] = {
    0: [
        "今天的{course}课程通知：下午{n}点在{room}上课，请同学们准时参加。",
        "社团招新宣讲将于周{n}晚七点在{room}举行，欢迎大家来交流。",
        "期中考试范围已发布在课程群里，请同学们复习第一到第{n}章。",
        "图书馆提醒：你借阅的《人工智能原理》{n}月{n}日前需要归还。",
    ],
    1: [
        "同学你好，我是{room}学院的学姐，店里上新了一批{course}资料，添加微信领取新生优惠，已经向群管理员报备了。",
        "低价出售{course}网课会员，限时{n}折，需要的私聊我，学长学姐都在用。",
        "姐弟们，我这边有公众号推广任务，动动手指点个关注就好，已向管理员报备，感谢支持～",
    ],
    2: [
        "动动手指就能赚钱！下载{app}完成小额任务立返{n}元，佣金随任务等级逐步升级，当日结算。",
        "招收线上刷单学员，一单一结，首单垫付{n}元返{n}元，多做多得，名额有限速来。",
        "在家兼职，手机接单，佣金{n}到{n}元不等，完成任务后本金和佣金一起返还。",
    ],
    3: [
        "【教务处通知】检测到你的学号信息异常，请立即点击链接填写身份证与银行卡号进行核实，逾期将影响本学期注册。",
        "同学你好，我是你的辅导员，学校正在统计助学金信息，请把学号、身份证号发我，今晚{n}点前截止。",
        "【信息中心】您的校园邮箱密码即将过期，请点击链接进入门户重置，否则账号将被冻结。",
    ],
}

SLOTS = {
    "course": ["高等数学", "人工智能原理", "数据结构", "线性代数", "大学英语"],
    "room": ["A101", "B203", "实验楼302", "教五", "线上腾讯会议"],
    "app": ["赚客", "兼职猫", "某宝特价版", "刷单助手", "轻松赚"],
}


def make_record(label: int, rng: random.Random) -> dict:
    template = rng.choice(TEMPLATES[label])
    text = template.format(n=rng.randint(1, 30), **{k: rng.choice(v) for k, v in SLOTS.items()})
    return {"text": text, "label": label, "subtype": "dummy", "source": "dummy", "split": ""}


def main() -> None:
    parser = argparse.ArgumentParser(description="生成占位假数据")
    parser.add_argument("--out", default="data/processed_dummy")
    parser.add_argument("--per-class", dest="per_class", type=int, default=60, help="训练集每类条数")
    parser.add_argument("--val-per-class", dest="val_per_class", type=int, default=15)
    parser.add_argument("--test-per-class", dest="test_per_class", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    for split, n in (("train", args.per_class), ("val", args.val_per_class), ("test", args.test_per_class)):
        records = [make_record(label, rng) | {"split": split} for label in range(4) for _ in range(n)]
        rng.shuffle(records)
        write_jsonl(f"{args.out}/{split}.jsonl", records)
        print(f"{split}: {len(records)} 条 → {args.out}/{split}.jsonl")


if __name__ == "__main__":
    main()
