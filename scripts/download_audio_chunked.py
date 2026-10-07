"""audio.zip 分段续传下载器 v2（带 206 强制校验与最终完整性验证）

v1 的教训：HF 的 resolve 会 302 到 CDN，个别请求的 Range 头可能被忽略而返回 200 全量流，
把文件写坏成"大小正确但中段错位"。v2 对每段强制断言 HTTP 206 + Content-Range 与请求一致，
任何异常一律丢弃该段重试，绝不在未验证的情况下写文件。
"""

import sys
import time
import urllib.request
from pathlib import Path

URL = "https://huggingface.co/datasets/JimmyMa99/TeleAntiFraud/resolve/main/audio.zip"
OUT = Path(__file__).resolve().parents[1] / "rawdata" / "teleantifraud" / "audio.zip"
TARGET = 12_735_906_663
CHUNK = 64 * 1024 * 1024  # 每段 64MB（更小段=单段重试成本更低）


def fetch_range(start: int, end: int, token: str) -> bytes:
    """严格取回 [start, end] 这段字节：非 206 或 Content-Range 不符即抛异常。"""
    req = urllib.request.Request(URL, headers={"Authorization": f"Bearer {token}", "Range": f"bytes={start}-{end}"})
    resp = urllib.request.urlopen(req, timeout=180)
    try:
        if resp.status != 206:
            raise RuntimeError(f"期待 206，实际 {resp.status}（Range 未被服务端接受）")
        content_range = resp.headers.get("Content-Range", "")
        if not content_range.startswith(f"bytes {start}-{end}/"):
            raise RuntimeError(f"Content-Range 不符: {content_range}")
        data = resp.read()
        if len(data) != end - start + 1:
            raise RuntimeError(f"字节不足: 期待 {end - start + 1}，实际 {len(data)}")
        return data
    finally:
        resp.close()


def main() -> None:
    from huggingface_hub import get_token

    token = get_token()
    # 历史遗留的坏文件一律删除重来
    if OUT.exists() and OUT.stat().st_size != TARGET:
        print(f"删除可疑的旧文件（{OUT.stat().st_size} 字节 ≠ {TARGET}）")
        OUT.unlink()
    while True:
        size = OUT.stat().st_size if OUT.exists() else 0
        if size >= TARGET:
            break
        end = min(size + CHUNK, TARGET) - 1
        t0 = time.time()
        try:
            data = fetch_range(size, end, token)
        except Exception as e:  # noqa: BLE001
            print(f"段 {size}-{end} 失败（{e}），5 秒后重试")
            time.sleep(5)
            continue
        with open(OUT, "ab") as f:
            f.write(data)
        rate = len(data) / max(1e-9, time.time() - t0) / 1048576
        print(f"段 {size}/{TARGET}：+{len(data)/1048576:.0f}MB @ {rate:.1f}MB/s", flush=True)

    # 终验：大小 + zip 中央目录
    assert OUT.stat().st_size == TARGET, f"大小不符: {OUT.stat().st_size}"
    import zipfile

    zf = zipfile.ZipFile(OUT)
    names = zf.namelist()
    print(f"zip 中央目录 OK，成员数 {len(names)}")
    print("下载与校验全部完成")


if __name__ == "__main__":
    sys.exit(main())
