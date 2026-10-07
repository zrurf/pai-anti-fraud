# 数据集清单与许可证

本项目的全部训练/评估数据均来自公开数据集，**不进入版本控制**（见 `.gitignore`），由 `scripts/build_processed.py` 从 `rawdata/` 原始文件整理生成。本文档说明每个数据集的来源、许可证、角色与映射规则要点，保证项目在只有代码的情况下可以自解释地重建数据。

标签体系（四分类）：**0=正常，1=推销，2=刷单，3=钓鱼**；`-1` 表示未映射（仅用于 MLM/二分类预训练或跨语言材料）。

数据角色：
- **clf**：进入四分类训练池（train/val/test）；
- **mlm**：仅用于 MLM 继续预训练 / 二分类中间预训练语料；
- **xlang**：跨语言/补充材料，不参与训练池。

## 目录约定

| 目录 | 内容 | 是否入库 |
| :--- | :--- | :--- |
| `rawdata/` | 各数据集原始下载件（压缩包/音频/原始标注） | 否 |
| `dataset/` | 整理后的 Lance 列式归档（逐源一个 `.lance` + 训练池 `messages.lance`） | 否 |
| `data/processed/` | `train.jsonl` / `val.jsonl` / `test.jsonl` 及逐源中间 JSONL | 否 |

重建命令：先按本文档获取原始数据放入 `rawdata/`，再运行 `python -m scripts.build_processed`（解析 → 映射 → 去重 → 分层划分 → JSONL + Lance）。

## 四分类训练池构成

当前版本（分层划分，种子 42）：**train 270,471 / val 28,395 / test 28,366，合计 327,232 条**。

| 类别 | 样本量 | 来源 |
| :--- | ---: | :--- |
| 0 正常 | 160,330 | FGRC-SCD 无风险、天池 label00（运营商营销短信=近领域负样本）、TeleAntiFraud 正常场景通话、spam10000 闲聊、LLMGen 中文正常邮件 |
| 1 推销 | 53,939 | CCL2023 虚假购物服务、FGRC-SCD 虚假购物服务、FBS_SMS AD_*、spam10000 群聊推广 |
| 2 刷单 | 28,492 | CCL2023 刷单返利类 28,353（**受害人笔录**，与消息文体存在域偏移）+ anti_fraud 兼职刷单对话 139 |
| 3 钓鱼 | 84,471 | 天池冒充 3 类、FGRC-SCD 冒充 4 类、CCL2023 冒充 4 类、TeleAntiFraud 冒充类通话、中文钓鱼邮件（LLMGen/PhishMMF）、FBS FR_Phishing |

已知数据局限（使用结果时须一并说明）：

1. **刷单类 val/test 各仅 14 条**：CCL2023 镜像只提供了官方 train 文件（28,353 条全部进入训练集），分层划分只作用于其余来源（anti_fraud 的 139 条对话按 80/10/10 分层）。因此测试集上刷单类的按类指标统计上脆弱，报告必须注明样本量。
2. **文体偏移**：CCL2023 为受害人笔录（第三人称叙述）、TeleAntiFraud 为通话转写（说话人轮次格式）、FGRC-SCD 由案件数据合成，均与"校园消息"目标文体存在域偏移。
3. **类别高度不平衡**（正常 ≫ 推销 > 钓鱼 ≫ 刷单），由 `configs/default.yaml` 的 `loss.class_weights: auto` 与阶段二的 WeightedRandomSampler 处理。

## 数据集明细

### 进入训练池（clf）

| 数据集 | 来源与获取 | 许可证 | 条目数 | 映射要点 |
| :--- | :--- | :--- | ---: | :--- |
| 天池五分类短信 | 阿里天池 `dataset/201837`（需天池账号） | GPL-3.0；使用需引用 Appl. Sci. 2024 相关论文 | 11,556 | label00→0（多为运营商营销，近领域负样本）；label01/03/04（冒充公检法/客服/领导熟人）→3；label02 贷款→不映射。label01–03 为 UTF-8、label04 为 GBK |
| TeleAntiFraud | HF `JimmyMa99/TeleAntiFraud`（gated，需同意条款） | Apache 2.0；ACM MM 2025 论文 | 13,711 | 仅文本部分不含通话转写，转写由 `scripts/asr_teleantifraud_spk.py`（paraformer-zh + fsmn-vad + ct-punc + cam++ 说话人分离）完成，并经 `scripts/mask_pii_teleantifraud.py` 做 PII 脱敏（正则 + CLUENER NER）。细分标注：客服/银行/钓鱼/身份盗窃→3；7 类正常场景→0；投资/彩票/绑架→不映射 |
| CCL2023 电信诈骗分类 | 百度 AI Studio 镜像 `datasetdetail/215947`（需百度账号；官方 CodaLab 渠道需注册审核） | 镜像页面标注 CC0；建议引用 CCL2023 评测论文 | 82,178（有标签） | 刷单返利类→2；冒充 4 类→3；虚假购物服务→1。**只有官方 train 文件带标签**（见上文刷单类局限） |
| FGRC-SCD | HF `Abooooo/FGRC-SCD` | CC BY 4.0 | 352,988 | 无风险→0；冒充电商物流客服/公检法/领导熟人/军警购物→3；虚假购物服务→1；投资理财/信用服务/婚恋/网黑→不映射。含短信与【坐席】/【客户】对话两种文体 |
| anti_fraud_dataset | HF `quanshiyun/anti_fraud_dataset` | GPL-3.0 | 2,212 | 12 类 CSV 按文件名映射：兼职刷单诈骗→2（139 例）；冒充客服诈骗→3（200 例）；其余 10 类"宁缺毋滥"不映射 |
| LLMGen-Phishing | HF `Dizzzy0x00/LLMGen-Phishing-Email-Dataset` | Apache 2.0 | 6,776 | 中文部分：label 1→3、label 0→0；英文部分→不映射。LLM 生成数据 |
| FBS_SMS | GitHub `Cypher-Z/FBS_SMS_Dataset` | 论文 CCS 2020，使用需引用 | 13,024 | 14 类文件（预分词、含 DIGIT/URL 占位符）：AD_*→1；FR_Phishing*→3；其余→不映射 |
| PhishMMF | GitHub `12345677876/PhishMMF` | MIT | 6,778 | datacon2023 中文钓鱼邮件（自带 label）→3（去重后 783 条独特文本，群发副本多）；英文邮件→不映射 |
| chinese-spam-10000 | HF `reatiny/chinese-spam-10000`（gated） | 见上游仓库 | 9,684 | spam→1（群聊推广，含色情引流/代刷）；闲聊→0；官方 train/dev/test 划分保留 |

### 仅用于 MLM / 二分类预训练（mlm）

| 数据集 | 来源 | 许可证 | 条目数 | 用途 |
| :--- | :--- | :--- | ---: | :--- |
| 80万短信 SpamMessage | GitHub `hrwhisper/SpamMessage` | 未标注（学术用途） | 978,922 | MLM 继续预训练主语料（`带标签短信.txt`，每行 "label\ttext"）；原生 0/1 标签存 subtype 备用 |
| ChiFraud | GitHub `xuemingxxx/ChiFraud` | CC BY-NC 4.0（**禁商用**） | 391,382 | 11 类：0=正常（33.2 万）→0、1–10 欺诈→-1，构成二分类中间预训练语料（消融 A5 对照组用） |

### 跨语言/补充材料（xlang，label=-1，不参与训练池）

| 数据集 | 来源 | 许可证 | 条目数 | 说明 |
| :--- | :--- | :--- | ---: | :--- |
| SpamHunter | GitHub `opmusic/SpamHunter_dataset` | CCS 2022 论文 | 25,063 | 多语言短信，中文仅数百条 |
| ScamGen | Mendeley `10.17632/dkypjhkmgb.1` | CC BY 4.0 | 1,574,038 | 中文电话诈骗话术变体（模板增强），5 意图标签与本系统 taxonomy 不对应，不映射 |
| Zenodo 邮件合集 | Zenodo `8339691` | CC BY 4.0 | 206,742 | 11 个英文邮件 CSV（TREC 05/06/07、CEAS_08、Enron、SpamAssassin、Nigerian Fraud 等），跨语言泛化用 |
| anti_fraud_sft（净增部分） | GitHub `golfxiao/anti_fraud_sft` | 同上游 anti_fraud（GPL-3.0） | 40 | 12 类对话与 anti_fraud_dataset 逐字节相同（已核验未重复入库），净增仅 40 个正常商务会议长对话 |

### 模型权重

| 模型 | 来源 | 许可证 | 用途 |
| :--- | :--- | :--- | :--- |
| chinese-macbert-base | HF `hfl/chinese-macbert-base` | Apache 2.0 | backbone；本地副本放 `models/chinese-macbert-base/`（离线运行），不进入版本控制 |
| cluener NER | HF `uer/roberta-base-finetuned-cluener2020-chinese` | Apache 2.0 | 仅 PII 脱敏（TeleAntiFraud 入库前的数据处理步骤）用 |

## 黄金测试集（预留）

`data/golden_test_set.jsonl` 为人工审核的校园风格黄金测试集（规划 200 条，四类均衡、含"像诈骗的正常消息"难负例），**绝不进入训练**，用于域偏移预检。文件尚待构建：

- 生成规范与 LLM 生成 prompt 见 `docs/黄金测试集LLM生成Prompt.md`；
- 构建后用 `python -m scripts.validate_golden_set.py` 校验格式、标签分布与训练池泄漏。

## 引用义务汇总

正式报告/论文中需引用：天池五分类（Appl. Sci. 2024）、TeleAntiFraud（ACM MM 2025）、CCL2023 评测（CCL 2023）、FBS_SMS（CCS 2020）、SpamHunter（CCS 2022）、FGRC-SCD（CC BY 4.0 数据引用）、MacBERT（找原文引用信息）。ChiFraud 为 CC BY-NC 4.0，**仅限非商业用途**。
