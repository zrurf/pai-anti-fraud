# 基于多专家门控融合 BERT 的校园网络诈骗文本识别系统

**方法概要**：MacBERT backbone + LoRA/DoRA 参数高效微调；核心创新是 4 个异构专家（各自池化策略 mean/attention/cls/mean_max + 各自负责类别的损失加权 λ=3）+ 可训练门控路由器软融合，构成输出级 Mixture-of-Experts；三阶段训练（Branch–Adapt–Route）；与传统 TF-IDF 基线对比；ONNX 导出 + FastAPI 部署。完整技术说明（架构、训练策略、复现步骤）见 `docs/技术文档.md`。

## 主要结果
>种子 `42`，test n=28,366，完整数字与置信区间见 `results/对比测试报告.md`

| 方案 | Accuracy | Macro-F1 | 95% CI |
| :--- | ---: | ---: | :---: |
| TF-IDF + 逻辑回归 | 0.8619 | 0.6609 | — |
| MacBERT+LoRA 单头 | 0.8941 | 0.7953 | [0.742, 0.838] |
| MacBERT+DoRA 单头（阶段一） | 0.8939 | 0.7998 | [0.745, 0.843] |
| **完整系统 MoE 门控融合（阶段三）** | **0.9110** | **0.8835** | **[0.846, 0.909]** |

## 项目结构

```
antifraud/                  # Python 包
├── config.py               # YAML 配置加载（configs/default.yaml 是唯一超参来源）
├── data/
│   ├── dataset.py          # JSONL 读取 / tokenize / collate（TinyTokenizer 仅供离线测试）
│   ├── label_map.py        # 各公开数据集 → 四分类标签映射
│   ├── pii.py              # PII 脱敏（正则 + CLUENER NER，数据处理阶段用）
│   └── augment.py          # 数据增强：emoji 插入 / 同形字 / 空格（阶段二专家数据侧重）
├── models/
│   ├── backbone.py         # MacBERT + LoRA/DoRA（PEFT）；resolve_model_path 本地优先（离线可用）
│   ├── pooling.py          # mean / attention / cls / mean_max 四种池化（专家异构性之一）
│   ├── experts.py          # 4 个异构专家 MLP 头
│   ├── router.py           # 路由器（投影 + 2层MLP + softmax，温度固定为 1）
│   └── moe_fusion.py       # 整体模型：backbone → 专家 + 路由器 → 门控融合；stage1 单头
├── training/
│   ├── losses.py           # 加权 CE（全局 w_c × 专家 λ=3）+ 负载均衡损失（Switch 形式）
│   ├── metrics.py          # Macro-F1 / 按类 P-R-F1 / ECE / McNemar
│   ├── pipeline.py         # 三阶段共用构件（dataloader / checkpoint / 权重装入）
│   ├── stage1_finetune.py  # 阶段一：MLM 继续预训练 + 四分类单头微调（= 对比基线）
│   ├── stage2_experts.py   # 阶段二：冻结 backbone，逐个训练异构专家
│   └── stage3_router.py    # 阶段三：路由训练（+ 可选联合微调）
├── baselines/
│   └── tfidf_baseline.py   # TF-IDF + 朴素贝叶斯 / 逻辑回归（传统方法对比）
└── serving/
    ├── onnx_infer.py       # ONNXRuntime 推理封装（类别/置信度/专家权重/专家意见）
    └── api.py              # FastAPI（POST /predict）
configs/
├── default.yaml            # 全部超参（experts[].self_class 显式绑定专家↔类别）
├── stage1_lora.yaml        # LoRA 单头对比（唯一区别 peft.method: lora）
└── seed_*.yaml             # 多种子重复实验（仅 seed 字段不同）
scripts/                    # 数据整理 / 评估 / 导出等脚本（见 scripts/README.md）
tests/                      # 离线冒烟测试（不需要任何下载）
docs/                       # 技术文档、黄金测试集生成规范（docs/决策文档/ 为过程存档，不入库）
release/                    # 部署工件（release/model/ 为最终交付模型，入库）
models/                     # 本地 backbone 权重（不入库，见"离线运行"）
DATASETS.md                 # 数据集清单、来源、许可证、引用义务
results/                    # 实验输出（metrics JSON 不入库；*.md 报告入库）
```

## 安装

```bash
pip install -r requirements.txt
```

GPU 训练请按 PyTorch 官网指令安装与本机 CUDA 匹配的 torch 版本（本项目在 RTX 4060 Laptop 8GB + torch 2.14/cu130 上验证：stage1 batch 32、stage2/3 batch 64 均可）。

## 离线运行

**本仓库设计为模型本地化、全程不依赖网络**：

1. 把 `hfl/chinese-macbert-base` 的模型文件（config.json / pytorch_model.bin / vocab.txt / tokenizer*.json 等）放到 `models/chinese-macbert-base/`；
2. 代码中的 `resolve_model_path()`（antifraud/models/backbone.py）会自动把 HF 模型 id 重定向到该本地目录；目录不存在时才回退到 Hub 在线加载。

因此训练、评估、ONNX 导出、API 推理在无网机器上均可运行。冒烟测试则完全不需要任何模型文件：

```bash
python -m tests.test_moe_shapes        # 模型与损失正确性（tiny-random，无下载）
python -m tests.test_pipeline_smoke    # 三阶段 + ONNX + 推理端到端冒烟
```

## 数据准备

数据集清单（来源、许可证、映射规则、已知局限）见 [DATASETS.md](DATASETS.md)。数据与模型权重不进入版本控制。获取原始数据放入 `rawdata/` 后：

```bash
python -m scripts.build_processed       # 解析→映射→去重→分层划分→ data/processed/*.jsonl + dataset/*.lance
```

训练池规模：train 270,471 / val 28,395 / test 28,366（正常 160,330 / 推销 53,939 / 刷单 28,492 / 钓鱼 84,471）。

## 正式训练（三阶段 Branch–Adapt–Route）

```bash
# 1) 阶段一 Branch：MLM 继续预训练（80 万短信语料）
python -m antifraud.training.stage1_finetune --task mlm \
    --corpus "rawdata/SpamMessage-master/data/带标签短信.txt" \
    --out checkpoints/stage1_mlm --max-samples 200000

# 2) 阶段一 Branch：四分类单头微调（产出即对比基线"MacBERT+PEFT 单头"）
python -m antifraud.training.stage1_finetune --task clf \
    --train data/processed/train.jsonl --val data/processed/val.jsonl \
    --out checkpoints/stage1 --init-from checkpoints/stage1_mlm/model.pt

# 3) 阶段二 Adapt：冻结 backbone，逐个训练 4 个异构专家（加权 CE λ=3 + 数据侧重采样）
python -m antifraud.training.stage2_experts --stage1 checkpoints/stage1/model.pt \
    --train data/processed/train.jsonl --val data/processed/val.jsonl --out checkpoints/stage2

# 4) 阶段三 Route：路由训练 + 联合微调
python -m antifraud.training.stage3_router --stage2 checkpoints/stage2/model.pt \
    --train data/processed/train.jsonl --val data/processed/val.jsonl \
    --out checkpoints/stage3 --joint
```

- 每个阶段在每个 epoch 后保存 best（`model.pt`）与续训状态（`last*.pt`）；checkpoint 内含 `model_name` 与 `label_names` 元数据。
- **断点续训**：中断后原命令加 `--resume <out目录>`（stage1 从 last.pt 的 epoch 继续，注意 `--epochs` 需 ≥ 已完成 epoch 数；stage2 逐专家续训并跳过已完成专家；stage3 续 last_route.pt / last_joint.pt）。
- 换种子/换 PEFT 方法：通过 `--config` 指定 `configs/seed_*.yaml` 或 `configs/stage1_lora.yaml`，输出目录另存，不要覆盖已有 checkpoint。

## 评估、消融与统计检验

```bash
# 传统基线（分钟级）
python -m antifraud.baselines.tfidf_baseline --train data/processed/train.jsonl \
    --test data/processed/test.jsonl --out results/tfidf_baseline.json

# 测试集正式评估：accuracy / Macro-F1 / 按类 P-R-F1 / 混淆矩阵 / ECE / bootstrap 95% CI
python -m scripts.eval_testset --checkpoint checkpoints/stage1/model.pt --mode stage1 \
    --out results/stage1_test.json
python -m scripts.eval_testset --checkpoint checkpoints/stage3/model.pt --mode stage3 \
    --out results/stage3_test.json --compare checkpoints/stage1/model.pt   # 额外输出 McNemar + 配对 bootstrap

# 融合方式消融（不重训，仅改融合：A1 等权 / A2 top-1 硬选择）
python -m scripts.eval_testset --checkpoint checkpoints/stage3/model.pt --mode stage3 \
    --fusion equal --out results/ablation_a1_equal.json
python -m scripts.eval_testset --checkpoint checkpoints/stage3/model.pt --mode stage3 \
    --fusion top1 --out results/ablation_a2_top1.json
```

多种子协议：同一数据划分文件（划分不随种子变化），仅训练随机性不同；使用 `configs/seed_*.yaml` 全链路重跑后取均值±标准差。

公平评估原则：所有方案输入一致（纯文本）、共享同一 train/val/test 划分、报告 bootstrap 置信区间与 McNemar 检验而非单次最优。

## 部署

```bash
python -m scripts.export_onnx                  # 产出 checkpoints/onnx/model.onnx + predict_meta.json + tokenizer
mkdir -p release/model && cp checkpoints/onnx/{model.onnx,predict_meta.json,tokenizer.json,tokenizer_config.json} release/model/
python -m uvicorn antifraud.serving.api:app --host 127.0.0.1 --port 8000
# POST /predict {"text": "..."} → label / confidence / probabilities / expert_weights / expert_opinions / evidence
# POST /explain {"text": "...", "top_k": 8} → /predict 全部字段 + attribution（Occlusion token 级归因）
#                                            + expert_attention（刷单专家注意力第二视角）
```

部署工件默认从 `release/model/` 加载（可用环境变量 `ANTIFRAUD_ONNX` / `ANTIFRAUD_TOKENIZER` 覆盖）。
生产服务器的最小交付与部署步骤详见 [release/README.md](release/README.md)（只需 `release/model/` + `antifraud/serving/`，无需 torch/GPU）。
归因实现：`/predict` 走关键词证据（快路径）；`/explain` 走 **Occlusion 留一法 token 级归因**（逐词替换为 [UNK] 后
[N+1,L] 批量前向，按预测概率下降幅度排序，纯 ONNX 不需要梯度），tokenizer 不支持 word 级映射时自动降级关键词；
若导出图含 `attn_scores`（注意力池化专家），`/explain` 附带 `expert_attention` 第二视角。
ONNX 推理封装（`antifraud/serving/onnx_infer.py`）读取 `predict_meta.json` 还原类别名与专家名。

## 黄金测试集

`data/golden_test_set.jsonl` 是人工审核的校园风格评估集（规划 200 条，含"像诈骗的正常消息"难负例），**绝不进入训练**，用于域偏移预检。位置已预留、暂未构建：

- LLM 生成规范与 prompt：`docs/黄金测试集LLM生成Prompt.md`；
- 生成后校验：`python -m scripts.validate_golden_set.py`（格式 / 标签分布 / 与训练池泄漏检查）。

## 引用与许可

第三方数据集与模型的许可证、引用义务集中在 [DATASETS.md](DATASETS.md)；其中 ChiFraud 为 CC BY-NC 4.0（仅限非商业用途），天池五分类与 FBS_SMS 等有论文引用义务。项目代码与数据使用者请遵守各来源的原始许可。
