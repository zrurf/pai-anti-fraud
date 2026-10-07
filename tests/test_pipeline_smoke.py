"""端到端流水线冒烟测试（离线，tiny-random backbone + 占位假数据）

  假数据 → 阶段一单头微调 → 阶段二专家训练 → 阶段三路由训练(+联合) → ONNX 导出 → 推理

全部以子进程调用真实 CLI（python -m ...），验证流水线脚本本身可运行。
运行：python -m tests.test_pipeline_smoke
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        print(result.stdout[-3000:])
        print(result.stderr[-3000:])
        raise SystemExit(f"命令失败：{' '.join(cmd)}")
    tail = (result.stdout or "").strip().splitlines()[-3:]
    print("\n".join(tail))


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = str(Path(tmp) / "smoke")
        common = ["--model-name", "tiny-random", "--batch-size", "8"]

        run([sys.executable, "-m", "scripts.make_dummy_data", "--out", f"{tmp}/data", "--per-class", "24", "--val-per-class", "6", "--test-per-class", "6"])

        # 传统基线（sklearn）
        run([sys.executable, "-m", "antifraud.baselines.tfidf_baseline", "--train", f"{tmp}/data/train.jsonl", "--test", f"{tmp}/data/val.jsonl", "--out", f"{tmp}/results/tfidf.json"])

        # 阶段一：单头四分类
        run([sys.executable, "-m", "antifraud.training.stage1_finetune", "--task", "clf", "--train", f"{tmp}/data/train.jsonl", "--val", f"{tmp}/data/val.jsonl", "--out", f"{tmp}/stage1", *common, "--epochs", "1"])

        # 阶段二：专家训练
        run([sys.executable, "-m", "antifraud.training.stage2_experts", "--stage1", f"{tmp}/stage1/model.pt", "--train", f"{tmp}/data/train.jsonl", "--val", f"{tmp}/data/val.jsonl", "--out", f"{tmp}/stage2", *common, "--epochs", "1"])

        # 阶段三：路由训练 + 联合微调
        run([sys.executable, "-m", "antifraud.training.stage3_router", "--stage2", f"{tmp}/stage2/model.pt", "--train", f"{tmp}/data/train.jsonl", "--val", f"{tmp}/data/val.jsonl", "--out", f"{tmp}/stage3", *common, "--joint"])

        # ONNX 导出 + 推理
        run([sys.executable, "-m", "scripts.export_onnx", "--checkpoint", f"{tmp}/stage3/model.pt", "--out-dir", f"{tmp}/onnx", "--model-name", "tiny-random"])
        check_onnx(tmp)
        check_explain_degrade(tmp)

    print("\ntest_pipeline_smoke: 全部通过 ✅")


def check_onnx(tmp: str) -> None:
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(f"{tmp}/onnx/model.onnx", providers=["CPUExecutionProvider"])
    # tiny 词表只有 w0..w495，用词表内 token 构造输入；按名取输出（图可能含 attn_scores 等额外输出）
    ids = np.array([[2, 10, 20, 30, 3]], dtype=np.int64)
    mask = np.ones_like(ids)
    names = [o.name for o in session.get_outputs()]
    logits, gate, expert_logits = session.run(["logits", "gate", "expert_logits"], {"input_ids": ids, "attention_mask": mask})
    assert logits.shape == (1, 4), f"logits 形状错误: {logits.shape}"
    assert gate.shape == (1, 4)
    assert expert_logits.shape == (1, 4, 4)
    assert abs(float(gate.sum()) - 1.0) < 1e-4, "门控权重和应为 1"
    if "attn_scores" in names:
        attn = session.run(["attn_scores"], {"input_ids": ids, "attention_mask": mask})[0]
        assert attn.shape == ids.shape, f"attn_scores 形状错误: {attn.shape}"
        assert abs(float(attn.sum()) - 1.0) < 1e-4, "注意力权重和应为 1"
        print(f"attn_scores 输出正常（sum={float(attn.sum()):.4f}）")
    print(f"ONNX 推理正常：logits={np.round(logits[0], 3).tolist()} gate={np.round(gate[0], 3).tolist()}")


def check_explain_degrade(tmp: str) -> None:
    """TinyTokenizer 无 word 级映射 → explain() 应降级为关键词证据，且不阻塞主结果。"""
    sys.path.insert(0, str(ROOT))
    from antifraud.data.dataset import TinyTokenizer
    from antifraud.serving.onnx_infer import OnnxPredictor

    predictor = OnnxPredictor(f"{tmp}/onnx/model.onnx", f"{tmp}/onnx", tokenizer=TinyTokenizer())
    out = predictor.explain("w10 w20 w30 刷单 返利 垫付")
    assert out["attribution"]["source"] == "keyword_fallback", f"应降级关键词，实际 {out['attribution']['source']}"
    assert "label_name" in out and "confidence" in out
    print(f"explain 降级分支正常：source={out['attribution']['source']} tokens={[t['text'] for t in out['attribution']['tokens']]}")


if __name__ == "__main__":
    main()
