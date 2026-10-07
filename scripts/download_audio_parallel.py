"""audio.zip 并行分段下载器

用法：python -m scripts.download_audio_parallel
产出：rawdata/teleantifraud/audio.zip（拼接自 4 个 .partN，全部完成后自动删除 part 文件）
"""

import shutil
import sys
import threading
import time
import urllib.request
from pathlib import Path

URLS = [
    "https://hf-mirror.com/datasets/JimmyMa99/TeleAntiFraud/resolve/main/audio.zip",  # 国内直连（VPN 关闭时可用）
    "https://huggingface.co/datasets/JimmyMa99/TeleAntiFraud/resolve/main/audio.zip",  # 官方（VPN 开启时可用）
]
BASE = Path(__file__).resolve().parents[1] / "rawdata" / "teleantifraud"
OUT = BASE / "audio.zip"
TARGET = 12_735_906_663
CHUNK = 8 * 1024 * 1024  # 每请求 8MB
WORKERS = 4


def fetch_range(start: int, end: int, token: str) -> bytes:
    """严格取回 [start, end]：非 206 或 Content-Range 不符即抛异常，绝不返回未验证数据。
    依次尝试各镜像源（成功的源会被记住并优先复用）。"""
    global _preferred
    order = [_preferred] + [u for u in URLS if u != _preferred]
    last_err = None
    for url in order:
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {token}", "Range": f"bytes={start}-{end}", "User-Agent": "Mozilla/5.0"}
        )
        try:
            resp = urllib.request.urlopen(req, timeout=120)
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
        try:
            if resp.status != 206:
                raise RuntimeError(f"期待 206，实际 {resp.status}")
            content_range = resp.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {start}-"):
                raise RuntimeError(f"Content-Range 不符: {content_range}")
            data = resp.read()
            if len(data) != end - start + 1:
                raise RuntimeError(f"字节不足: 期待 {end - start + 1}，实际 {len(data)}")
            _preferred = url
            return data
        finally:
            resp.close()
    raise RuntimeError(f"所有源均失败: {last_err}")


_preferred = URLS[0]


def worker(part: int, lo: int, hi: int, token: str) -> None:
    """下载 [lo, hi) 到 audio.zip.part{part}，按已有大小断点续传。"""
    path = BASE / f"audio.zip.part{part}"
    while True:
        offset = path.stat().st_size if path.exists() else 0
        if lo + offset >= hi:
            print(f"[part{part}] 完成 {offset/1073741824:.2f}GB", flush=True)
            return
        end = min(lo + offset + CHUNK, hi) - 1
        try:
            data = fetch_range(lo + offset, end, token)
        except Exception as e:  # noqa: BLE001
            print(f"[part{part}] {lo + offset} 失败（{e}）重试", flush=True)
            time.sleep(5)
            continue
        with open(path, "ab") as f:
            f.write(data)


def main() -> None:
    from huggingface_hub import get_token

    token = get_token()
    if OUT.exists():
        print("删除旧文件，重新下载")
        OUT.unlink()

    bounds = [round(i * TARGET / WORKERS / CHUNK) * CHUNK for i in range(WORKERS + 1)]
    bounds[-1] = TARGET
    threads = []
    for i in range(WORKERS):
        t = threading.Thread(target=worker, args=(i, bounds[i], bounds[i + 1], token), daemon=True)
        t.start()
        threads.append(t)
    t0 = time.time()
    while any(t.is_alive() for t in threads):
        done = sum((BASE / f"audio.zip.part{i}").stat().st_size if (BASE / f"audio.zip.part{i}").exists() else 0 for i in range(WORKERS))
        rate = done / max(1e-9, time.time() - t0) / 1048576
        print(f"总进度 {done/1073741824:.2f}/{TARGET/1073741824:.2f} GB @ {rate:.1f}MB/s", flush=True)
        time.sleep(30)
    for t in threads:
        t.join()

    print("拼接 part 文件…")
    with open(OUT, "wb") as out:
        for i in range(WORKERS):
            part_path = BASE / f"audio.zip.part{i}"
            with open(part_path, "rb") as f:
                shutil.copyfileobj(f, out, 1024 * 1024)
    assert OUT.stat().st_size == TARGET, f"大小不符: {OUT.stat().st_size} ≠ {TARGET}"

    import zipfile

    zf = zipfile.ZipFile(OUT)
    names = zf.namelist()
    print(f"zip 中央目录 OK，成员数 {len(names)}")
    for i in range(WORKERS):
        (BASE / f"audio.zip.part{i}").unlink()
    print("下载与校验全部完成")


if __name__ == "__main__":
    sys.exit(main())
