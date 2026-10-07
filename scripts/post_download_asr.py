"""下载完成后的自动接力：等 audio.zip → 解压 → 全量 ASR → 重建 Lance

与 scripts/download_audio_parallel.py 配合使用（等它拼出 audio.zip 并清理 part 文件后自动开始）。
产物：tmp/teleantifraud_asr.jsonl（供人工审查）+ dataset/teleantifraud.lance（split=asr_review）。
"""

import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "rawdata" / "teleantifraud"
ZIP = BASE / "audio.zip"
TARGET = 12_735_906_663
MAX_WAIT_S = 6 * 3600


def wait_for_zip() -> None:
    print("[orchestrator] 等待 audio.zip 下载完成…", flush=True)
    t0 = time.time()
    while time.time() - t0 < MAX_WAIT_S:
        parts = list(BASE.glob("audio.zip.part*")) + list(BASE.glob("audio.zip.aria2"))
        # 注意：aria2 预分配会让文件提前"长得"像下载完了，必须等控制文件消失且 zip 可读
        if ZIP.exists() and ZIP.stat().st_size == TARGET and not parts:
            try:
                with zipfile.ZipFile(ZIP) as zf:
                    if len(zf.namelist()) > 0:
                        print("[orchestrator] audio.zip 就绪", flush=True)
                        return
            except Exception:
                pass
        time.sleep(60)
    raise SystemExit("[orchestrator] 等待超时（下载未在期限内完成）")


def extract() -> None:
    audio_dir = BASE / "audio"
    if audio_dir.exists() and len(list(audio_dir.glob("*"))) > 10:
        print("[orchestrator] 已解压过，跳过", flush=True)
        return
    print("[orchestrator] 解压 audio.zip（约 13GB）…", flush=True)
    with zipfile.ZipFile(ZIP) as zf:
        zf.extractall(BASE)
    n = sum(1 for _ in BASE.rglob("*.mp3"))
    print(f"[orchestrator] 解压完成，MP3 数 {n}", flush=True)


def run_asr() -> None:
    print("[orchestrator] 开始全量 ASR（13,711 条，预计 2-4 小时）…", flush=True)
    r = subprocess.run([sys.executable, "-m", "scripts.asr_teleantifraud"], cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"[orchestrator] ASR 失败，退出码 {r.returncode}")


def rebuild_lance() -> None:
    print("[orchestrator] 重建 JSONL 与 Lance（asr_review 待审行不入训练集）…", flush=True)
    r = subprocess.run([sys.executable, "-m", "scripts.build_processed"], cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"[orchestrator] build_processed 失败")

    # 释放转写用 wav 缓存（可由 mp3 重新生成）
    wav_cache = ROOT / "tmp" / "asr_wav"
    if wav_cache.exists():
        shutil.rmtree(wav_cache)
        print("[orchestrator] 已清理 tmp/asr_wav 缓存", flush=True)

    # 汇总
    out = ROOT / "tmp" / "teleantifraud_asr.jsonl"
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    ok = [r for r in rows if r.get("text")]
    empty = [r for r in rows if not r.get("text")]
    total_sec = sum(r.get("duration_s", 0) for r in rows)
    print(f"[orchestrator] ASR 结果：成功 {len(ok)} / 空 {len(empty)} / 总时长 {total_sec/3600:.1f} 小时", flush=True)
    for r in ok[:3]:
        print(f"  样例 {r['audio_path']}: {r['text'][:60]}", flush=True)


if __name__ == "__main__":
    wait_for_zip()
    extract()
    run_asr()
    rebuild_lance()
    print("[orchestrator] 全部完成", flush=True)
