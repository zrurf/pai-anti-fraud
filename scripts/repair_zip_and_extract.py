"""audio.zip 坏成员修复与全量解压

场景：aria2 分段下载完成后无内嵌校验，个别分段字节损坏（zip 中央目录有效、成员 CRC 失败）。
流程：枚举全部坏成员 → 按 zip 中央目录定位坏成员的字节区间 → 从镜像重取并解压校验 →
      提取其余成员（已解压且 CRC 正确的跳过）→ 最终全量 CRC 复验。
"""

import struct
import sys
import zipfile
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.download_audio_parallel import fetch_range

BASE = Path(__file__).resolve().parents[1] / "rawdata" / "teleantifraud"
ZIP = BASE / "audio.zip"


def member_bytes(zf: zipfile.ZipFile, info: zipfile.ZipInfo, token: str) -> bytes:
    """从远端精确重取某成员的原始内容（解析本地文件头拿数据偏移，fetch 后 zlib 解压并核对 CRC）。"""
    header_off = info.header_offset
    head = fetch_range(header_off, header_off + 512, token)
    if head[:4] != b"PK\x03\x04":
        raise RuntimeError(f"{info.filename} 本地文件头签名错误")
    name_len, extra_len = struct.unpack("<HH", head[26:30])
    data_off = header_off + 30 + name_len + extra_len
    comp = fetch_range(data_off, data_off + info.compress_size - 1, token)
    if info.compress_type == zipfile.ZIP_STORED:
        raw = comp
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        raw = zlib.decompress(comp, -15)
    else:
        raise RuntimeError(f"{info.filename} 不支持的压缩类型 {info.compress_type}")
    if zlib.crc32(raw) & 0xFFFFFFFF != info.CRC:
        raise RuntimeError(f"{info.filename} 重取后 CRC 仍不符")
    return raw


def crc_scan(zf: zipfile.ZipFile, infos: list[zipfile.ZipInfo], report_every: int = 2000) -> list[str]:
    bad: list[str] = []
    for i, info in enumerate(infos):
        try:
            with zf.open(info) as f:
                while f.read(1 << 20):
                    pass
        except Exception as e:  # noqa: BLE001
            bad.append(info.filename)
            print(f"[bad {len(bad)}] {info.filename}: {e}", flush=True)
        if (i + 1) % report_every == 0:
            print(f"  扫描 {i + 1}/{len(infos)}，坏 {len(bad)}", flush=True)
    return bad


def main() -> None:
    from huggingface_hub import get_token

    token = get_token()
    zf = zipfile.ZipFile(ZIP)
    infos = zf.infolist()
    print(f"成员总数 {len(infos)}")

    bad = crc_scan(zf, infos)
    print(f"扫描完成：坏成员 {len(bad)} 个")

    for name in bad:
        info = zf.getinfo(name)
        raw = member_bytes(zf, info, token)
        target = BASE / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        print(f"[fixed] {name} ({len(raw)} 字节)", flush=True)

    missing = 0
    for info in infos:
        target = BASE / info.filename
        if target.exists() and target.stat().st_size == info.file_size:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as f, open(target, "wb") as out:
            out.write(f.read())
        missing += 1
    print(f"补提取 {missing} 个文件")

    n_bad = len(crc_scan(zf, infos, report_every=5000))
    print(f"复验完成：坏成员 {n_bad} 个")
    if n_bad:
        raise SystemExit(1)
    print("zip 全部成员 CRC 校验通过 ✅")


if __name__ == "__main__":
    sys.exit(main())
