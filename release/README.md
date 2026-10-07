# 发布目录：模型部署工件

本目录存放**交付部署**的模型工件与说明。训练工作区在 `checkpoints/`（含续训状态、逐阶段 checkpoint），
部署只需要这里的 `model/`。工件由 `scripts/export_onnx.py` 产出（`checkpoints/onnx/`）后复制到本目录。

## model/ 内容

| 文件 | 说明 |
| :--- | :--- |
| `model.onnx` | 合并后的完整推理图（MacBERT backbone + 4 异构专家 + 门控路由器；PEFT 权重已合并，单图自包含） |
| `predict_meta.json` | 元数据：类别名（label_names）与专家名（expert_names），推理封装据此命名输出 |
| `tokenizer.json` / `tokenizer_config.json` | tokenizer 文件，从本地目录加载，**运行时无需联网** |

## 部署步骤

1. 打包本目录（`model/`）+ 服务代码 `antifraud/serving/`（连同 `antifraud/__init__.py`、`antifraud/serving/__init__.py`）到服务器；
2. 服务器安装依赖（**不需要 torch / GPU**）：

   ```bash
   pip install onnxruntime fastapi uvicorn transformers numpy
   ```

3. 启动服务（默认路径即指向 `release/model/`，可用环境变量覆盖）：

   ```bash
   uvicorn antifraud.serving.api:app --host 0.0.0.0 --port 8000
   # 多 worker：--workers 4
   # 自定义路径：ANTIFRAUD_ONNX=/opt/m/model.onnx ANTIFRAUD_TOKENIZER=/opt/m/
   ```

4. 验证：

   ```bash
   curl http://127.0.0.1:8000/health
   # → {"status":"ok"}
   curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
        -d '{"text": "刷单返利兼职，日结300，先垫付后返款"}'
   # → {"label": ..., "label_name": ..., "confidence": ..., "probabilities": {...},
   #     "expert_weights": {...}, "expert_opinions": {...}, "evidence": [...]}
   ```

## 接口说明

异步任务协议：`/predict` 与 `/explain` 只提交任务、立即返回 ticket；结果通过 `/tasks/{ticket}` 轮询获取。
任务状态：`queued` → `running` → `done`（含 `result`）/ `failed`（含 `error`）；已完成结果保留 1 小时后自动清理。

| 端点 | 方法 | 说明 |
| :--- | :--- | :--- |
| `/health` | GET | 健康检查 |
| `/predict` | POST | 提交预测任务 `{"text": "..."}` |
| `/explain` | POST | 提交归因任务 `{"text": "...", "top_k": 8}` |
| `/tasks/{ticket}` | GET | 查询任务状态与结果 |

`/predict` 的 result 字段：

- `label` / `label_name`：预测类别（0=正常 1=推销 2=刷单 3=钓鱼）；
- `confidence` / `probabilities`：最大置信度与四类概率；
- `expert_weights`：路由器门控权重（4 专家软融合权重之和为 1，可解释性字段）；
- `expert_opinions`：每个专家独立给出的类别判断；
- `evidence`：关键词证据（人工维护词表，快路径兜底）。

`/explain` 的 result 在上述字段之上追加：

- `attribution`：**Occlusion 留一法 token 级归因**（纯 ONNX，一次 [N+1, L] 批量前向）。
  `source="occlusion"` 时：`tokens` 为按重要性排序的 top-k 词条（含字符区间 start/end，
  `drop` = 遮蔽该词后目标类别概率的下降幅度，越大越关键；中文按字、英文/数字按整词）；
  `merged` 为相邻字合并后的词组显示版（如"教/务/处"→"教务处"，分数累加，前端可直接用于高亮）；
  tokenizer 能力不足或归因异常时 `source="keyword_fallback"` 降级，不阻塞主结果。
- `expert_attention`（若导出图含 attn_scores）：注意力池化专家（刷单专家）的 token 权重第二视角，
  含 `gate_weight` 供判断解释力强弱——该专家门控权重低时此视角仅作参考；`merged` 同上。

## 已知性能与注意

- `/predict` 提交到出结果 CPU ~30ms、`/explain` ~0.4–1.5s（长文本更慢）；两者均为异步任务，轮询间隔建议 0.1–1s；并发 worker 数用环境变量 `ANTIFRAUD_WORKERS` 调整（默认 2）；
- 已知域偏移：训练数据中刷单类为受害人笔录文体，对校园风格的刷单话术可能倾向判为"推销"（置信度高时建议人工复核边界样本）；
- 版本对应：本工件由仓库 `checkpoints/stage3/model.pt`（种子 42 主路径）导出，导出脚本与依赖版本见仓库 README。
