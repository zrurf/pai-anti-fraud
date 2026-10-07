"""检测 API

任务式异步协议：POST /predict、POST /explain 只提交任务并立即返回 ticket，
结果通过 GET /tasks/{ticket} 轮询获取（queued → running → done/failed）。
归因等重操作不再长时间占用 HTTP 连接，客户端也不会因网关超时丢结果。

启动（模型工件在 release/model/，由 scripts/export_onnx.py 产出后复制）：
  uvicorn antifraud.serving.api:app --host 127.0.0.1 --port 8000
环境变量（可选）：
  ANTIFRAUD_ONNX / ANTIFRAUD_TOKENIZER  模型路径（默认 release/model/）
  ANTIFRAUD_WORKERS                     归因/预测工作线程数（默认 2）
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from antifraud.serving.onnx_infer import OnnxPredictor

app = FastAPI(title="校园诈骗文本识别 API", version="0.1.0")

_predictor: OnnxPredictor | None = None
_predictor_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=int(os.environ.get("ANTIFRAUD_WORKERS", "2")))
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_JOB_TTL_SECONDS = 3600.0  # 已完成任务保留时长，超时清理防内存增长


def get_predictor() -> OnnxPredictor:
    global _predictor
    with _predictor_lock:
        if _predictor is None:
            onnx_path = os.environ.get("ANTIFRAUD_ONNX", "release/model/model.onnx")
            tokenizer_dir = os.environ.get("ANTIFRAUD_TOKENIZER", "release/model")
            _predictor = OnnxPredictor(onnx_path, tokenizer_dir)
        return _predictor


@app.on_event("startup")
def warmup() -> None:
    """启动时预加载模型，避免首个请求承担加载耗时。"""
    try:
        get_predictor()
        print("[api] 模型加载完成")
    except Exception as exc:  # noqa: BLE001 — 启动失败要可见，但不能崩掉进程（便于排查路径配置）
        print(f"[api] 模型加载失败（请求任务时会重试）：{exc}")


def _cleanup_expired_locked() -> None:
    now = time.time()
    expired = [t for t, j in _jobs.items() if j.get("finished_at") and now - j["finished_at"] > _JOB_TTL_SECONDS]
    for t in expired:
        _jobs.pop(t, None)


def _submit(fn, *args) -> str:
    ticket = uuid.uuid4().hex
    with _jobs_lock:
        _cleanup_expired_locked()
        _jobs[ticket] = {"status": "queued", "submitted_at": time.time()}

    def run() -> None:
        with _jobs_lock:
            _jobs[ticket]["status"] = "running"
        try:
            result = fn(*args)
            with _jobs_lock:
                _jobs[ticket].update(status="done", result=result, finished_at=time.time())
        except Exception as exc:  # noqa: BLE001 — 任务失败记录到 ticket，不抛出线程
            with _jobs_lock:
                _jobs[ticket].update(status="failed", error=str(exc), finished_at=time.time())

    _executor.submit(run)
    return ticket


class PredictRequest(BaseModel):
    text: str


class ExplainRequest(BaseModel):
    text: str
    top_k: int = 8


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def submit_predict(req: PredictRequest) -> dict:
    ticket = _submit(get_predictor().predict, req.text)
    return {"ticket": ticket, "status": "queued"}


@app.post("/explain")
def submit_explain(req: ExplainRequest) -> dict:
    top_k = max(1, min(req.top_k, 32))
    ticket = _submit(get_predictor().explain, req.text, top_k)
    return {"ticket": ticket, "status": "queued"}


@app.get("/tasks/{ticket}")
def get_task(ticket: str) -> dict:
    with _jobs_lock:
        job = _jobs.get(ticket)
        if job is None:
            raise HTTPException(status_code=404, detail="任务不存在或已过期")
        resp = {"ticket": ticket, "status": job["status"]}
        if job["status"] == "done":
            resp["result"] = job["result"]
        elif job["status"] == "failed":
            resp["error"] = job["error"]
        return resp
