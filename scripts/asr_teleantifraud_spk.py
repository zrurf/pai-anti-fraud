"""TeleAntiFraud 转写

带说话人分离（paraformer-zh + fsmn-vad + ct-punc + cam++）
PII 脱敏由 scripts/mask_pii_teleantifraud.py 在本脚本完成后统一处理

用法（仓库根目录）：
  python -m scripts.asr_teleantifraud_spk --shard 0 --num-shards 2
  python -m scripts.asr_teleantifraud_spk --shard 1 --num-shards 2
两个分片可并行（各自 GPU 流水线），结果分别写入 tmp/teleantifraud_asr_spk_{shard}.jsonl，断点续传。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # 必须在 torch 初始化前设置

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "rawdata" / "teleantifraud"
WAV_CACHE = ROOT / "tmp" / "asr_wav_spk"
CHUNK = 16
CJK_SPACE = re.compile(r"(?<=[\u4e00-\u9fff。，！？；：、])\s+(?=[\u4e00-\u9fff。，！？；：、])")


def collect_audio_paths() -> list[str]:
    sys.path.insert(0, str(ROOT))
    from scripts.asr_teleantifraud import collect_audio_paths

    return collect_audio_paths()


def load_done(out_path: Path) -> set[str]:
    """合并所有分片的已完成清单，避免改分片数后重复计算。"""
    done: set[str] = set()
    for f in ROOT.glob("tmp/teleantifraud_asr_spk_*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if "error" not in r:
                done.add(r["audio_path"])
    return done


def to_wav(mp3_path: Path, wav_path: Path, ffmpeg: str) -> bool:
    if wav_path.exists() and wav_path.stat().st_size > 1000:
        return True
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_wav = wav_path.with_suffix(f".tmp{__import__('os').getpid()}.wav")  # 唯一临时名，避免与其他进程冲突
    r = subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(mp3_path), "-ac", "1", "-ar", "16000", str(tmp_wav)],
        capture_output=True,
    )
    if r.returncode != 0:
        return False
    try:
        tmp_wav.replace(wav_path)
    except PermissionError:
        # 另一进程刚完成同一文件：校验后直接采用
        time.sleep(0.3)
        if wav_path.exists() and wav_path.stat().st_size > 1000:
            tmp_wav.unlink(missing_ok=True)
            return True
        return False
    return True


def build_turn_text(sentence_info: list[dict]) -> tuple[str, int]:
    """句级 spk 标签 → 相邻同说话人合并为轮次，说话人按首次出现顺序重编号。"""
    if not sentence_info:
        return "", 0
    turns: list[tuple[int, list[str]]] = []
    for s in sentence_info:
        text = CJK_SPACE.sub("", s.get("text", "")).strip()
        if not text:
            continue
        spk = s.get("spk", 0)
        if turns and turns[-1][0] == spk:
            turns[-1][1].append(text)
        else:
            turns.append((spk, [text]))
    renumber: dict[int, int] = {}
    parts: list[str] = []
    for spk, texts in turns:
        if spk not in renumber:
            renumber[spk] = len(renumber) + 1
        parts.append(f"【说话人{renumber[spk]}】{''.join(texts)}")
    return "".join(parts), len(renumber)


def main() -> None:
    parser = argparse.ArgumentParser(description="TeleAntiFraud 说话人分离转写")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    import imageio_ffmpeg
    import soundfile as sf
    import torch
    from funasr import AutoModel

    out_path = ROOT / "tmp" / f"teleantifraud_asr_spk_{args.shard}.jsonl"
    all_paths = collect_audio_paths()
    done = load_done(out_path)
    mine = [p for p in all_paths if p not in done][ args.shard :: args.num_shards ]
    if args.limit:
        mine = mine[: args.limit]
    print(f"[shard{args.shard}] 总数 {len(all_paths)}，已完成 {len(done)}，本分片待处理 {len(mine)}", flush=True)
    if not mine:
        print("无待处理音频，完成。")
        return

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    model = AutoModel(
        model="paraformer-zh",
        vad_model="fsmn-vad",
        punc_model="ct-punc",
        spk_model="cam++",
        device="cuda:0",
        disable_update=True,
        disable_pbar=True,
    )
    print(f"[shard{args.shard}] 模型加载完成，开始转写", flush=True)

    cache = WAV_CACHE  # 共享预转码缓存（预转码完成后各分片只读）
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t_start, n_ok, n_fail = time.time(), 0, 0
    with open(out_path, "a", encoding="utf-8") as fout:
        for i in range(0, len(mine), CHUNK):
            chunk = mine[i : i + CHUNK]
            wavs, valid = [], []
            for rel in chunk:
                wav = cache / rel.replace("/", "_").replace(".mp3", ".wav")
                if to_wav(RAW / rel, wav, ffmpeg):
                    wavs.append(wav)
                    valid.append(rel)
                else:
                    fout.write(json.dumps({"audio_path": rel, "text": "", "duration_s": 0.0, "error": "ffmpeg_failed"}, ensure_ascii=False) + "\n")
                    n_fail += 1
            if not wavs:
                continue
            try:
                results = model.generate(input=[str(w) for w in wavs], cache={}, batch_size_s=100)
            except Exception as e:  # noqa: BLE001
                print(f"[shard{args.shard}] 批 {i // CHUNK} 失败（{str(e)[:120]}），逐条重试", flush=True)
                results = []
                for rel, wav in zip(valid, wavs):
                    try:
                        rr = model.generate(input=str(wav), cache={})
                        results.append(rr[0])
                    except Exception as e2:  # noqa: BLE001
                        results.append({"_skip": True})
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
                text, n_spk = build_turn_text(res.get("sentence_info", []))
                fout.write(json.dumps({"audio_path": rel, "text": text, "duration_s": round(duration, 1), "n_spk": n_spk}, ensure_ascii=False) + "\n")
                n_ok += 1
            fout.flush()
            torch.cuda.empty_cache()  # 可变 batch 形状易碎片化，逐批归还未用显存
            # wav 缓存保留（预转码共享缓存，磁盘充足），断点续传零转码开销
            n_done = min(i + CHUNK, len(mine))
            rate = (time.time() - t_start) / max(1, n_ok + n_fail)
            print(f"[shard{args.shard}] 进度 {n_done}/{len(mine)}（成功 {n_ok} 失败 {n_fail}）{rate:.2f}s/条 ETA {rate * (len(mine) - n_done) / 3600:.1f}h", flush=True)

    print(f"[shard{args.shard}] 转写完成：成功 {n_ok}，失败 {n_fail}")


if __name__ == "__main__":
    sys.exit(main())
