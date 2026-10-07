"""LLM 合成数据生成（OpenAI 兼容接口）

调用 OpenAI 兼容接口（DeepSeek 等），按类别模板批量生成训练样本。
环境变量：
  LLM_API_BASE   如 https://api.deepseek.com/v1
  LLM_API_KEY    密钥
  LLM_MODEL      如 deepseek-chat

用法：
  python -m scripts.generate_synthetic --per-class 200 --out data/processed/synthetic.jsonl

安全与合规约束：
- 只生成"弱化、虚构、不可执行"样本；不含真实链接/学号/身份证/银行卡；
- 不生成可直接实施诈骗的操作指南；涉及色情擦边变体保持最小必要规模；
- 产出需经敏感信息扫描 + 去重 + 人工抽检（TODO：BERTScore 相似度过滤）。
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request

from antifraud.training.utils import write_jsonl

# 每类 1-2 个 prompt 模板；变体维度覆盖话术/身份/紧迫感等轴
PROMPTS: dict[int, list[str]] = {
    0: [  # 正常消息（含近领域正常：正式通知、正常推广——降低误报的关键）
        "请写一条校园群里的正常消息（label=0 正常）：课程/考试/活动通知，或正式的社团公告，或合法商家的正常推广。"
        "要求口语自然、无诱导、无索取敏感信息，40-120字，直接输出文本。",
        "请写一条'看起来像诈骗但实际正常'的校园消息（label=0 正常）：例如含链接的正常学术邮件、"
        "正式落款的真实通知、真实的新生群推销报备公告。40-120字，直接输出文本。",
    ],
    1: [  # 推销诈骗（弱化、虚构、不可执行）
        "请虚构一条校园群推销诈骗话术样本（label=1 推销，仅用于训练诈骗分类器，弱化且不可执行）："
        "冒充学长/学姐私聊推销资料或商品，使用'已向群管理员报备'等迷惑措辞，可含表情符号。"
        "不得包含真实联系方式或链接。40-120字，直接输出文本。",
    ],
    2: [  # 刷单诈骗
        "请虚构一条刷单诈骗话术样本（label=2 刷单，仅用于训练诈骗分类器，弱化且不可执行）："
        "'小额返利诱导→任务逐步升级'结构，导流到虚构的 App/群。不得包含真实链接、真实收款方式。"
        "40-120字，直接输出文本。",
    ],
    3: [  # 钓鱼（冒充学校/教师）
        "请虚构一条冒充学校/教师的钓鱼信息样本（label=3 钓鱼，仅用于训练诈骗分类器，弱化且不可执行）："
        "正式公文口吻，伪造教务处/辅导员落款，索取学号、身份证等信息，可用虚构链接占位符如【链接】。"
        "不得包含真实链接或真实个人信息。40-120字，直接输出文本。",
    ],
}

LABEL_NAMES = ["正常", "推销", "刷单", "钓鱼"]


def call_llm(prompt: str, base: str, key: str, model: str, temperature: float = 1.0) -> str:
    body = json.dumps(
        {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": temperature}
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base.rstrip('/')}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM 合成数据生成")
    parser.add_argument("--per-class", dest="per_class", type=int, default=200)
    parser.add_argument("--out", default="data/processed/synthetic.jsonl")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sleep", type=float, default=0.5, help="请求间隔秒数")
    args = parser.parse_args()

    base, key, model = os.environ.get("LLM_API_BASE", ""), os.environ.get("LLM_API_KEY", ""), os.environ.get("LLM_MODEL", "")
    if not (base and key and model):
        raise SystemExit("请设置环境变量 LLM_API_BASE / LLM_API_KEY / LLM_MODEL")

    seen: set[str] = set()
    records: list[dict] = []
    for label, prompts in PROMPTS.items():
        for i in range(args.per_class):
            prompt = prompts[i % len(prompts)]
            try:
                text = call_llm(prompt, base, key, model, args.temperature)
            except Exception as e:  # noqa: BLE001
                print(f"[{LABEL_NAMES[label]}] 第 {i + 1} 条失败：{e}，重试一次")
                time.sleep(3)
                try:
                    text = call_llm(prompt, base, key, model, args.temperature)
                except Exception as e2:  # noqa: BLE001
                    print(f"放弃：{e2}")
                    continue
            text = text.strip().strip("\"“”")
            if not text or text in seen:
                continue
            seen.add(text)
            records.append({"text": text, "label": label, "subtype": "synthetic", "source": "llm", "split": ""})
            if (i + 1) % 50 == 0:
                print(f"[{LABEL_NAMES[label]}] {i + 1}/{args.per_class}")
            time.sleep(args.sleep)

    write_jsonl(args.out, records)
    print(f"合成 {len(records)} 条（去重后）→ {args.out}")
    print("TODO：敏感信息扫描（真实手机号/链接/身份证正则）、BERTScore 与真实样本相似度过滤、人工抽检。")


if __name__ == "__main__":
    main()
