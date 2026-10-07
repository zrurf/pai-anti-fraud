"""TeleAntiFraud 音频批量转写（SenseVoiceSmall + fsmn-vad，CUDA）

HF 公开版不含通话转写文本，话术原文在 audio.zip 的 13,711 个通话音频中。
本脚本：mp3 →(ffmpeg)→ wav(16k mono) →(SenseVoiceSmall)→ 文本，
逐块追加写入 tmp/teleantifraud_asr.jsonl，可随时中断续跑（按 audio_path 去重续传）。

用法（仓库根目录）：
  python -m scripts.asr_teleantifraud --limit 8        # 小样试跑
  python -m scripts.asr_teleantifraud                  # 全量（约 2-4 小时，GPU）

产出：tmp/teleantifraud_asr.jsonl，每行 {"audio_path", "text", "duration_s"}
转写结果属机器生成，须人工审查后再进入训练（见 scripts/build_processed.py 的合并逻辑）。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "rawdata" / "teleantifraud"
OUT = ROOT / "tmp" / "teleantifraud_asr.jsonl"
WAV_CACHE = ROOT / "tmp" / "asr_wav"
CHUNK = 32  # 每批文件数（funasr 内部再按 batch_size_s 二次组批）

RICH_TAG = re.compile(r"<\|[^|]*\|>")


def collect_audio_paths() -> list[str]:
    """从 binary + sft 文件收集全部唯一音频相对路径。"""
    paths: set[str] = set()
    for fname in ("binary_classification/train.json", "binary_classification/test.json"):
        for r in json.loads((RAW / fname).read_text(encoding="utf-8")):
            for m in r.get("prompt", []):
                if isinstance(m.get("content"), list):
                    for t in m["content"]:
                        if t.get("type") == "audio":
                            paths.add(t["audio_url"])
    for fname in ("sft/train.jsonl", "sft/test.jsonl"):
        for line in (RAW / fname).read_text(encoding="utf-8").splitlines():
            if line.strip():
                for a in json.loads(line).get("audios", []):
                    paths.add(a)
    return sorted(paths)


def load_done() -> set[str]:
    if not OUT.exists():
        return set()
    # 只有成功行算已完成；ffmpeg 失败等错误行在下次运行时重试
    return {
        json.loads(l)["audio_path"]
        for l in OUT.read_text(encoding="utf-8").splitlines()
        if l.strip() and "error" not in json.loads(l)
    }


def to_wav(mp3_path: Path, wav_path: Path, ffmpeg: str) -> bool:
    if wav_path.exists() and wav_path.stat().st_size > 1000:
        return True
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_wav = wav_path.with_suffix(".tmp.wav")
    r = subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(mp3_path), "-ac", "1", "-ar", "16000", str(tmp_wav)],
        capture_output=True,
    )
    if r.returncode != 0:
        return False
    tmp_wav.replace(wav_path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="TeleAntiFraud 批量 ASR")
    parser.add_argument("--limit", type=int, default=0, help="只转写前 N 条（试跑用）")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    import imageio_ffmpeg
    import soundfile as sf
    from funasr import AutoModel

    all_paths = collect_audio_paths()
    done = load_done()
    pending = [p for p in all_paths if p not in done]
    if args.limit:
        pending = pending[: args.limit]
    print(f"音频总数 {len(all_paths)}，已转写 {len(done)}，待转写 {len(pending)}")
    if not pending:
        print("无待转写音频，完成。")
        return

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    model = AutoModel(
        model="iic/SenseVoiceSmall",
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device="cuda:0",
        disable_update=True,
        disable_pbar=True,
    )
    print("模型加载完成，开始转写")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t_start, n_ok, n_fail = time.time(), 0, 0
    with open(out_path, "a", encoding="utf-8") as fout:
        for i in range(0, len(pending), CHUNK):
            chunk = pending[i : i + CHUNK]
            wavs: list[Path] = []
            valid: list[str] = []
            for rel in chunk:
                mp3 = RAW / rel
                wav = WAV_CACHE / rel.replace("/", "_").replace(".mp3", ".wav")
                if to_wav(mp3, wav, ffmpeg):
                    wavs.append(wav)
                    valid.append(rel)
                else:
                    fout.write(json.dumps({"audio_path": rel, "text": "", "duration_s": 0.0, "error": "ffmpeg_failed"}, ensure_ascii=False) + "\n")
                    n_fail += 1
            if not wavs:
                continue
            try:
                results = model.generate(
                    input=[str(w) for w in wavs],
                    cache={},
                    language="zh",
                    use_itn=True,
                    batch_size_s=60,
                    merge_vad=False,
                )
            except Exception as e:  # noqa: BLE001
                # 整批失败时逐条重试，定位坏文件
                print(f"[warn] 批 {i // CHUNK} 失败（{e}），改为逐条重试")
                results = []
                for rel, wav in zip(valid, wavs):
                    try:
                        r = model.generate(input=str(wav), cache={}, language="zh", use_itn=True, batch_size_s=60)
                        results.append(r[0])
                    except Exception as e2:  # noqa: BLE001
                        results.append({"text": "", "_skip": True})
                        fout.write(json.dumps({"audio_path": rel, "text": "", "duration_s": 0.0, "error": str(e2)[:200]}, ensure_ascii=False) + "\n")
                        n_fail += 1
            for rel, wav, res in zip(valid, wavs, results):
                if res.get("_skip"):
                    continue
                try:
                    info = sf.info(str(wav))
                    duration = info.frames / info.samplerate
                except Exception:  # noqa: BLE001
                    duration = 0.0
                text = RICH_TAG.sub("", res.get("text", "")).strip()
                fout.write(json.dumps({"audio_path": rel, "text": text, "duration_s": round(duration, 1)}, ensure_ascii=False) + "\n")
                n_ok += 1
            fout.flush()
            n_done = min(i + CHUNK, len(pending))
            rate = (time.time() - t_start) / max(1, n_ok + n_fail)
            eta_min = rate * (len(pending) - n_done) / 60
            print(f"进度 {n_done}/{len(pending)}（成功 {n_ok} 失败 {n_fail}）速率 {rate:.2f}s/条 ETA {eta_min:.0f} 分钟")

    print(f"转写完成：成功 {n_ok}，失败 {n_fail}，结果在 {out_path}")


if __name__ == "__main__":
    sys.exit(main())
