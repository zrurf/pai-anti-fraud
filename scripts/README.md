# scripts/ 目录说明

所有脚本均在**仓库根目录**以 `python -m scripts.<名称>` 运行（保证相对路径与包内导入一致）。按用途分组：

## 数据整理（重建训练池的必经路径）

| 脚本 | 作用 |
| :--- | :--- |
| `build_processed.py` | 核心整理器：解析 `rawdata/` 各来源 → 按 `antifraud/data/label_map.py` 映射四分类标签 → 全局去重 → 分层划分（种子 42）→ 写 `data/processed/*.jsonl` 与 `dataset/*.lance`。支持 `--only <源>` 单源重建（聚合时始终读磁盘上全部 clf 源，不会清空全局划分）；`--promote-asr-review` 将人工审查通过的 ASR 转写行放行进训练池 |
| `build_lance.py` | 把 JSONL 转为 Lance 列式归档（`dataset/*.lance`） |

## 数据获取与 ASR（历史用过的编排脚本，重建数据时按需使用）

| 脚本 | 作用 |
| :--- | :--- |
| `download_datasets.py` | 打印全部数据集的来源、状态与获取指引（`--auto` 自动下载可直连部分） |
| `download_audio_chunked.py` / `download_audio_parallel.py` | HF 大文件（TeleAntiFraud audio.zip 12.7GB）的分段续传/并行下载器；HF CDN 单连接会断流，hf CLI/curl 不可靠时的兜底 |
| `repair_zip_and_extract.py` | 坏 zip 成员重取 + CRC 全量校验 + 补齐解压 |
| `post_download_asr.py` | "等下载完成 → 解压 → ASR → 重建"的编排器 |
| `asr_teleantifraud.py` | TeleAntiFraud 通话批量转写（SenseVoiceSmall 版，初版） |
| `asr_teleantifraud_spk.py` | 说话人分离版转写（paraformer-zh + fsmn-vad + ct-punc + cam++，双分片 `--shard/--num-shards`），最终采用版本 |
| `mask_pii_teleantifraud.py` | 转写文本 PII 脱敏（正则 + 分句 CLUENER NER），依赖 `antifraud/data/pii.py` |

注意：ASR 与脱敏仅是**数据入库前的处理步骤**，训练与部署完全不需要重跑；转写产物已包含在训练池中。

## 训练辅助

| 脚本 | 作用 |
| :--- | :--- |
| `make_dummy_data.py` | 生成占位假数据（联调/冒烟测试用） |
| `generate_synthetic.py` | LLM 合成数据生成（OpenAI 兼容接口，需环境变量提供 API；消融 A6 材料） |

## 评估与部署

| 脚本 | 作用 |
| :--- | :--- |
| `eval_testset.py` | 测试集正式评估：accuracy / Macro-F1 / 按类 P-R-F1 / 混淆矩阵 / ECE / bootstrap 95% 置信区间。`--compare <ckpt>` 输出两模型 McNemar 检验与配对 bootstrap Macro-F1 差值；`--fusion equal|top1` 在不重训的情况下做融合方式消融（A1 等权 / A2 top-1）。结果 JSON 写入 `results/` |
| `export_onnx.py` | 加载 stage3 checkpoint → PEFT 权重合并 → 导出 `checkpoints/onnx/model.onnx`，同时写 `predict_meta.json`（类别名/专家名）与 tokenizer，供 `antifraud/serving/` 离线推理 |
| `validate_golden_set.py` | 黄金测试集校验：JSONL 格式、标签合法性、行数与类别分布、文件内重复、与 train/val/test 的文本泄漏检查 |
