"""公开数据集下载与获取指引（各数据集的许可证与引用义务见 DATASETS.md）

用法（仓库根目录）：
  python -m scripts.download_datasets                # 打印所有数据集状态与指引
  python -m scripts.download_datasets --auto         # 自动下载可直接获取的部分
  python -m scripts.download_datasets --only chifraud,fbs_sms --auto
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

RAW_DIR = Path("rawdata")

# name -> (状态, 说明, 自动下载命令)。状态：AUTO = 脚本可直接下载
DATASETS: dict[str, tuple[str, str, list[str] | None]] = {
    "tianchi5": (
        "LOCAL",
        "电信诈骗五分类（天池 dataset/201837）：约 6.4 万条短信，5 类。已从官方渠道下载，"
        "解包 tmp/Telecom_Fraud_Texts_5-main.zip → rawdata/tianchi5/ 即可",
        None,
    ),
    "teleantifraud": (
        "AUTO",
        "TeleAntiFraud-28k（HF JimmyMa99/TeleAntiFraud，Apache 2.0，gated 已授权同意）："
        "电信诈骗通话 ASR 文本 + 音频，含欺诈/正常与细分类标注。仅需文本时只拉取文本子目录",
        ["hf_download", "JimmyMa99/TeleAntiFraud", str(RAW_DIR / "teleantifraud")],
    ),
    "chifraud": (
        "AUTO",
        "ChiFraud（COLING 2025）：59,106 条欺诈文本(11类) + 352,328 条正常，CC BY-NC 4.0，用于 MLM/二分类",
        ["git", "clone", "--depth", "1", "https://github.com/xuemingxxx/ChiFraud", str(RAW_DIR / "chifraud")],
    ),
    "spam80w": (
        "AUTO",
        "80万中文短信（hrwhisper/SpamMessage）：MLM 继续预训练语料，无许可证标注",
        ["git", "clone", "--depth", "1", "https://github.com/hrwhisper/SpamMessage", str(RAW_DIR / "spam80w")],
    ),
    "fbs_sms": (
        "AUTO",
        "FBS_SMS_Dataset（CCS 2020）：14K 伪基站短信 14 类，使用需引用论文，用于推销类补充",
        ["git", "clone", "--depth", "1", "https://github.com/Cypher-Z/FBS_SMS_Dataset", str(RAW_DIR / "fbs_sms")],
    ),
    "spamhunter": (
        "AUTO",
        "SpamHunter（CCS 2022）：21,918 条多语言 SMS（中文占比很小），用于跨语言鲁棒性",
        ["git", "clone", "--depth", "1", "https://github.com/opmusic/SpamHunter_dataset", str(RAW_DIR / "spamhunter")],
    ),
    "fgrc_scd": (
        "AUTO",
        "FGRC-SCD（HF Abooooo/FGRC-SCD，CC BY 4.0）：中文风险短信/对话，四分类监督主力数据之一",
        ["hf_download", "Abooooo/FGRC-SCD", str(RAW_DIR / "fgrc_scd")],
    ),
    "anti_fraud": (
        "AUTO",
        "anti_fraud_dataset（HF quanshiyun/anti_fraud_dataset，GPL-3.0）：30,308 行诈骗对话",
        ["hf_download", "quanshiyun/anti_fraud_dataset", str(RAW_DIR / "anti_fraud")],
    ),
    "llmgen_phishing": (
        "AUTO",
        "LLMGen-Phishing-Email-Dataset（HF Dizzzy0x00/...，Apache 2.0）：6,776 条（中文 3,030）钓鱼邮件",
        ["hf_download", "Dizzzy0x00/LLMGen-Phishing-Email-Dataset", str(RAW_DIR / "llmgen_phishing")],
    ),
    "phishmmf": (
        "AUTO",
        "PhishMMF（GitHub 12345677876/PhishMMF，MIT）：11,672 条邮件（含原始文本），只用其邮件文本部分",
        ["git", "clone", "--depth", "1", "https://github.com/12345677876/PhishMMF", str(RAW_DIR / "phishmmf")],
    ),
    "scamgen": (
        "AUTO",
        "ScamGen（Mendeley DOI 10.17632/dkypjhkmgb.1，CC BY 4.0）：中文电话诈骗半合成语料，5 类意图标注。"
        "直接下载地址 https://data.mendeley.com/datasets/dkypjhkmgb/1/files "
        "（Mendeley 需逐文件点击下载，故不自动化），放入 data/raw/scamgen/",
        None,
    ),
    "zenodo_phishing": (
        "AUTO",
        "Phishing Email Curated（Zenodo DOI 10.5281/zenodo.8339691，CC BY 4.0，全英文，跨语言泛化用）："
        "逐文件直链下载 https://zenodo.org/records/8339691 （共 449MB，按需取用）",
        None,
    ),
}

# 数据真实存在，但需要注册或申请，按项目约束不使用（仅作记录）
ABANDONED: dict[str, str] = {
    "CCL23-Eval 任务 6（GJSeason/CCL2023-FCC）": "需 CodaLab 注册 + 人工审核（1-2 工作日），放弃申请",
    "DataCon2023 邮件安全赛道": "官方渠道需参赛注册，赛后开放状态未确认；第三方镜像无标签，价值有限，不使用",
}


def hf_download(repo_id: str, out_dir: str) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id, local_dir=out_dir, repo_type="dataset")


def run_one(name: str, do_download: bool) -> None:
    status, desc, cmd = DATASETS[name]
    print(f"\n=== {name} [{status}] ===\n{desc}")
    if cmd is None or not do_download:
        return
    out = Path(cmd[-1])
    if out.exists():
        print(f"已存在，跳过：{out}")
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if cmd[0] == "hf_download":
            hf_download(cmd[1], cmd[2])
        else:
            subprocess.run(cmd, check=True)
        print(f"下载完成：{out}")
    except Exception as e:  # noqa: BLE001
        print(f"下载失败（{e}）——请按上面的指引手动获取")


def main() -> None:
    parser = argparse.ArgumentParser(description="数据集下载与指引")
    parser.add_argument("--auto", action="store_true", help="自动下载所有可直接下载的数据集")
    parser.add_argument("--only", default=None, help="逗号分隔的数据集名过滤")
    args = parser.parse_args()

    names = list(DATASETS)
    if args.only:
        names = [n.strip() for n in args.only.split(",") if n.strip() and n.strip() in DATASETS]
    for name in names:
        run_one(name, args.auto)

    print("\n=== 已放弃的数据集（需注册/申请，按项目约束不使用）===")
    for name, reason in ABANDONED.items():
        print(f"- {name}：{reason}")


if __name__ == "__main__":
    main()
