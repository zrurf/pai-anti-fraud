"""把 rawdata/ 下的原始数据集解析、映射标签、去重并划分，产出统一 JSONL 与 Lance

数据集清单（来源、许可证、逐源映射规则要点）见仓库根目录 DATASETS.md；
标签映射函数集中在 antifraud/data/label_map.py。覆盖的训练池来源（角色 clf）：
tianchi5 / teleantifraud（ASR 转写）/ ccl2023 / fgrc_scd / anti_fraud / llmgen_phishing /
fbs_sms / phishmmf / spam10000；另含 mlm 角色（spam80w / chifraud）与 xlang 角色
（spamhunter / scamgen / zenodo_phishing / anti_fraud_sft）的逐源归档。

用法（仓库根目录）：
  python -m scripts.build_processed                 # 解析全部并产出 JSONL + Lance
  python -m scripts.build_processed --only <源>     # 单源重建（聚合划分始终读磁盘上全部 clf 源）
  python -m scripts.build_processed --no-lance      # 只产出 JSONL
  python -m scripts.build_processed --promote-asr-review   # 放行 TeleAntiFraud 转写行（需人工审查通过）

产出：
  data/processed/<源>.jsonl         逐源全量（含未映射行 label=-1）
  data/processed/{train,val,test}.jsonl   已映射(label>=0)行的分层训练划分（种子 42）
  dataset/<源>.lance 与 messages.lance    Lance 归档（--no-lance 跳过）
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
from pathlib import Path

from antifraud.data.label_map import LABEL_NAMES, map_anti_fraud, map_fgrc, map_tianchi5, map_teleantifraud
from antifraud.training.utils import write_jsonl

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "rawdata"
PROCESSED = ROOT / "data" / "processed"
DATASET_DIR = ROOT / "dataset"
OUT_ASR = ROOT / "tmp" / "teleantifraud_asr.jsonl"

SPLIT_RATIO = {"train": 0.8, "val": 0.1, "test": 0.1}


def make_id(source: str, text: str, seq: int) -> str:
    """稳定 ID：来源 + 文本内容哈希前 12 位（去重重跑不漂移）+ 序号。"""
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
    return f"{source}-{digest}-{seq:04d}"


def dedupe_key(text: str) -> str:
    """去重键：压缩所有空白后取哈希，避免同文不同空格重复。"""
    return hashlib.md5(re.sub(r"\s+", "", text).encode("utf-8")).hexdigest()


def parse_tianchi5() -> list[dict]:
    """解析 rawdata/tianchi5/label0*-last.csv（逐文件探测编码：utf-8 → gbk）。"""
    records: list[dict] = []
    seen: set[str] = set()
    for path in sorted(RAW.glob("tianchi5/label0*-last.csv")):
        file_label = path.name[:7]  # "label00" .. "label04"
        label = map_tianchi5(file_label)
        raw = path.read_bytes()
        encoding = "utf-8"
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError:
            encoding = "gbk"
        reader = csv.DictReader(io.StringIO(raw.decode(encoding)))
        n_before = len(seen)
        for row in reader:
            text = (row.get("content") or "").strip()
            if len(text) < 5:
                continue
            key = dedupe_key(text)
            if key in seen:
                continue
            seen.add(key)
            records.append(
                {
                    "id": "",  # 写入前统一编号
                    "text": text,
                    "label": label,
                    "subtype": (row.get("label") or "").strip(),
                    "source": "tianchi5",
                    "split": "",
                    "meta": {"native_file": file_label, "encoding": encoding},
                }
            )
        print(f"[tianchi5] {path.name} enc={encoding} 映射={label} 新增 {len(seen) - n_before} 条")
    print(f"[tianchi5] 共 {len(records)} 条（去重后）")
    return records


def parse_teleantifraud() -> list[dict]:
    """解析 rawdata/teleantifraud：按通话音频聚合标签，挂接 ASR 转写文本。

    标签优先级：细分诈骗类型（客服/银行/钓鱼/身份盗窃→钓鱼）> 正常场景/'normal'→正常
    > 无细分 'fraud'→-1。转写来自 tmp/teleantifraud_asr.jsonl（scripts/asr_teleantifraud.py 产物），
    全部行 split=asr_review，人工审查通过前不进入训练。
    无转写的音频保留为档案行（text=''，label=-1）。
    """
    base = RAW / "teleantifraud"

    # 1) 每个音频的原生标签聚合
    audio_meta: dict[str, dict] = {}

    def add_label(audio: str, native: str) -> None:
        info = audio_meta.setdefault(audio, {"labels": set(), "subtypes": set()})
        info["labels"].add(native)
        info["subtypes"].add(native)

    for fname in ("binary_classification/train.json", "binary_classification/test.json"):
        path = base / fname
        if path.exists():
            for r in json.loads(path.read_text(encoding="utf-8")):
                for m in r.get("prompt", []):
                    if isinstance(m.get("content"), list):
                        for t in m["content"]:
                            if t.get("type") == "audio":
                                add_label(t["audio_url"], r.get("answer", ""))
    for fname in ("sft/train.jsonl", "sft/test.jsonl"):
        path = base / fname
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    for a in r.get("audios", []):
                        add_label(a, r.get("answers", ""))

    # 2) ASR 转写结果：按优先级合并（说话人分离版优先，逐音频回退到旧连续文本版）
    asr: dict[str, dict] = {}
    candidates = [
        (ROOT / "tmp" / "teleantifraud_asr_spk_masked.jsonl", "paraformer-zh+cam++(说话人分离)+NER脱敏"),
        (ROOT / "tmp" / "teleantifraud_asr_spk_0.jsonl", "paraformer-zh+cam++(说话人分离)"),
        (ROOT / "tmp" / "teleantifraud_asr_spk_1.jsonl", "paraformer-zh+cam++(说话人分离)"),
        (OUT_ASR, "sensevoice-small+fsmn-vad(连续文本)"),
    ]
    n_by_source: dict[str, int] = {}
    for path, label in candidates:
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if "error" in r:
                    continue
                if r["audio_path"] not in asr:  # 高优先级文件先读，先到先得
                    asr[r["audio_path"]] = r
                    n_by_source[label] = n_by_source.get(label, 0) + 1
    if asr:
        print(f"[teleantifraud] 转写合并：{n_by_source}")

    # 3) 逐音频聚合出四分类标签：细分诈骗类型 > 正常 > 未映射
    from antifraud.data.pii import mask_regex

    records: list[dict] = []
    for audio in sorted(audio_meta):
        info = audio_meta[audio]
        natives = sorted(info["subtypes"])
        mapped = [(n, map_teleantifraud(n)) for n in natives]
        fraud_types = [n for n, v in mapped if v == 3]
        normals = [n for n, v in mapped if v == 0]
        if fraud_types:
            label, subtype = 3, "+".join(fraud_types)
        elif normals:
            label, subtype = 0, "+".join(normals)
        else:
            label, subtype = -1, "+".join(natives) or "unknown"
        tr = asr.get(audio)
        # 转写文本一律过一遍正则脱敏（幂等）；NER 姓名脱敏已在 mask pass 中完成
        text = mask_regex((tr or {}).get("text", ""))
        meta = {"native_labels": natives, "audio_path": audio}
        if tr:
            meta["asr"] = (
                "paraformer-zh+cam++(说话人分离)" if "n_spk" in tr else "sensevoice-small+fsmn-vad(连续文本)"
            )
            meta["duration_s"] = tr.get("duration_s", 0.0)
            if tr.get("n_spk"):
                meta["n_spk"] = tr["n_spk"]
        records.append(
            {
                "id": "",
                "text": text,
                "label": label if text else -1,  # 无转写只作档案
                "subtype": subtype,
                "source": "teleantifraud",
                "split": "asr_review" if text else "untranscribed",
                "meta": meta,
            }
        )
    n_text = sum(1 for r in records if r["text"])
    print(f"[teleantifraud] 音频 {len(records)} 条，其中已转写 {n_text} 条（split=asr_review 待人工审查）")
    return records


PARSERS = {
    "tianchi5": parse_tianchi5,
    "teleantifraud": parse_teleantifraud,
    "chifraud": lambda: parse_chifraud(),
    "fgrc_scd": lambda: parse_fgrc_scd(),
    "anti_fraud": lambda: parse_anti_fraud(),
    "llmgen_phishing": lambda: parse_llmgen_phishing(),
    "fbs_sms": lambda: parse_fbs_sms(),
    "spam80w": lambda: parse_spam80w(),
    "phishmmf": lambda: parse_phishmmf(),
    "spamhunter": lambda: parse_spamhunter(),
    "scamgen": lambda: parse_scamgen(),
    "zenodo_phishing": lambda: parse_zenodo_phishing(),
    "anti_fraud_sft": lambda: parse_anti_fraud_sft(),
    "ccl2023": lambda: parse_ccl23(),
    "spam10000": lambda: parse_spam10000(),
}

# ChiFraud 的 11 类名称（dataset/class.txt），仅 0=正常 映射为四分类的正常
CHIFRAUD_CLASSES = [
    "正常", "赌博博彩", "招嫖色情", "办假证", "虚假办卡", "违禁药品交易",
    "违规提现", "虚假证明", "虚假手机卡", "地下黑贷", "新类型",
]


def parse_chifraud() -> list[dict]:
    """ChiFraud（COLING 2025）：59,106 条欺诈(11 类) + 352,328 条正常，TSV 格式 "Label_id\\tText"。

    按文档映射：仅 0=正常 → 四分类 0；其余 10 类欺诈 → -1（欺诈/正常二分类预训练正样本）。
    自带按时间划分 train/t2022/t2023，原样保留在 split（测试集晚于训练集原则）。
    角色 = mlm：不进四分类训练池，独立 lance 供 MLM/二分类使用。
    """
    base = RAW / "chifraud" / "dataset"
    records: list[dict] = []
    seen: set[str] = set()
    for native_split in ("train", "t2022", "t2023"):
        path = base / f"ChiFraud_{native_split}.csv"
        if not path.exists():
            print(f"[chifraud] 缺少 {path.name}，跳过")
            continue
        n_before = len(seen)
        for line in path.read_text(encoding="utf-8").splitlines()[1:]:
            if "\t" not in line:
                continue
            label_str, text = line.split("\t", 1)
            text = text.strip()
            if len(text) < 5:
                continue
            key = dedupe_key(text)
            if key in seen:
                continue
            seen.add(key)
            native = int(label_str)
            records.append(
                {
                    "id": "",
                    "text": text,
                    "label": 0 if native == 0 else -1,
                    "subtype": CHIFRAUD_CLASSES[native] if 0 <= native < 11 else str(native),
                    "source": "chifraud",
                    "split": native_split,
                    "meta": {"native_label": native},
                }
            )
        print(f"[chifraud] {native_split} 新增 {len(seen) - n_before} 条")
    print(f"[chifraud] 共 {len(records)} 条（角色=mlm/二分类）")
    return records


def parse_fbs_sms() -> list[dict]:
    """FBS_SMS_Dataset（CCS 2020）：14 类伪基站短信，每类一个 UTF-8 文件、每行一条。

    文本已预分词并用 DIGIT/URL/NAME/PLACE 等占位符脱敏。
    映射见 label_map.FBS_PREFIX_MAP：AD_*→1 推销，FR_Phishing*→3 钓鱼，其余→-1。
    """
    from antifraud.data.label_map import map_fbs

    base = RAW / "FBS_SMS_Dataset-master"
    records: list[dict] = []
    seen: set[str] = set()
    for f in sorted(p for p in base.iterdir() if p.is_file() and p.name != "README.md"):
        raw = f.read_bytes()
        encoding = "utf-8"
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError:
            encoding = "gbk"
        n_before = len(seen)
        for line in raw.decode(encoding).splitlines():
            text = line.strip()
            if len(text) < 5:
                continue
            key = dedupe_key(text)
            if key in seen:
                continue
            seen.add(key)
            records.append(
                {
                    "id": "",
                    "text": text,
                    "label": map_fbs(f.name),
                    "subtype": f.name,
                    "source": "fbs_sms",
                    "split": "",
                    "meta": {"native_file": f.name, "encoding": encoding},
                }
            )
        print(f"[fbs_sms] {f.name} 新增 {len(seen) - n_before} 条")
    print(f"[fbs_sms] 共 {len(records)} 条")
    return records


def parse_spam80w() -> list[dict]:
    """80 万中文短信（hrwhisper/SpamMessage）：垃圾约 8 万 + 正常约 72 万，0/1 标签。

    按文档定位仅用于 MLM 继续预训练，全部行 label=-1（原生 0/1 存 subtype 与 meta）。
    """
    base = RAW / "SpamMessage-master"
    if not base.exists():
        base = RAW / "spam80w"
    records: list[dict] = []
    seen: set[str] = set()
    for f in sorted(base.rglob("带标签短信*.txt")) + sorted(base.rglob("*.txt")):
        if f.name == "README.md" or "readme" in f.name.lower():
            continue
        raw = f.read_bytes()
        encoding = "utf-8"
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError:
            encoding = "gbk"
        n_before = len(seen)
        for line in raw.decode(encoding, errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            text = line.split("\t", 1)[1] if "\t" in line else line
            native = line.split("\t", 1)[0] if "\t" in line else ""
            text = text.strip()
            if len(text) < 5:
                continue
            key = dedupe_key(text)
            if key in seen:
                continue
            seen.add(key)
            records.append(
                {
                    "id": "",
                    "text": text,
                    "label": -1,  # MLM 语料：不进四分类
                    "subtype": f"短信-{native or '未知'}",
                    "source": "spam80w",
                    "split": "train",
                    "meta": {"native_label": native, "file": f.name, "encoding": encoding},
                }
            )
        if n_before != len(seen):
            print(f"[spam80w] {f.name} enc={encoding} 新增 {len(seen) - n_before} 条")
    print(f"[spam80w] 共 {len(records)} 条（角色=mlm，label 全部 -1）")
    return records


def parse_phishmmf() -> list[dict]:
    """PhishMMF（GitHub 12345677876/PhishMMF，MIT）：原始邮件文本。

    - datacon2023_1/2.jsonl：3,000 封**中文**钓鱼邮件（label 全 1）→ 3 钓鱼（文档映射表）；
    - CEAS_08_0.jsonl / SpamAssasin_0.jsonl：6,000 封英文正常邮件（label 0）→ -1（跨语言用）；
    - all.jsonl 是特征向量非邮件文本，跳过。
    """
    base = RAW / "phishmmf_data"

    def strip_html(s) -> str:
        import re

        if isinstance(s, dict):  # Body 可能是 {"text": ..., "html": {...}} 的嵌套结构
            for key in ("text", "html"):
                if isinstance(s.get(key), str) and s[key].strip():
                    s = s[key]
                    break
                elif isinstance(s.get(key), dict):
                    s = s[key]
        if not isinstance(s, str):
            return ""
        s = re.sub(r"<[^>]+>", " ", s)
        return re.sub(r"\s+", " ", s).strip()[:4000]

    def iter_json_objects(path: Path):
        """容错读取 JSONL：文件内存在裸换行（字符串值内）时，折叠换行后顺序解码。"""
        text = path.read_text(encoding="utf-8", errors="replace").replace("\r", " ").replace("\n", " ")
        dec = json.JSONDecoder()
        idx, n = 0, len(text)
        while idx < n:
            while idx < n and text[idx] in " \t":
                idx += 1
            if idx >= n:
                break
            try:
                obj, end = dec.raw_decode(text, idx)
                yield obj
                idx = end
            except json.JSONDecodeError:
                brace = text.find("{", idx + 1)
                if brace == -1:
                    break
                idx = brace

    records: list[dict] = []
    seen: set[str] = set()
    for fname, lang, do_map in (
        ("datacon2023_1.jsonl", "cn", True),
        ("datacon2023_2.jsonl", "cn", True),
        ("CEAS_08_0.jsonl", "en", False),
        ("SpamAssasin_0.jsonl", "en", False),
    ):
        path = base / fname
        if not path.exists():
            print(f"[phishmmf] 缺少 {fname}，跳过")
            continue
        n_before = len(seen)
        for r in iter_json_objects(path):
            meta = r.get("Metadata", {})
            subject = strip_html(r.get("Metadata", {}).get("Subject") or meta.get("Subject") or r.get("subject") or "")
            body = strip_html(r.get("Body") or r.get("body") or "")
            text = (subject + "\n" + body).strip()
            if len(text) < 5:
                continue
            key = dedupe_key(text)
            if key in seen:
                continue
            seen.add(key)
            native = r.get("label")
            label = {1: 3, 0: 0}.get(native, -1) if do_map else -1
            records.append(
                {
                    "id": "",
                    "text": text,
                    "label": label,
                    "subtype": f"{lang}-邮件",
                    "source": "phishmmf",
                    "split": "",
                    "meta": {"native_label": native, "lang": lang, "file": fname},
                }
            )
        print(f"[phishmmf] {fname} 新增 {len(seen) - n_before} 条")
    print(f"[phishmmf] 共 {len(records)} 条")
    return records


def parse_spamhunter() -> list[dict]:
    """SpamHunter（CCS 2022）：多语言 SMS 垃圾消息（中文占比很小），无四分类映射。

    每行一条短信（已用 X 占位符脱敏部分数字），全部 label=-1，角色=xlang。
    中文行用 subtype 标记，便于后续跨语言实验筛选。
    """
    import glob
    import re

    cjk = re.compile(r"[\u4e00-\u9fff]")
    records: list[dict] = []
    seen: set[str] = set()
    for f in sorted(glob.glob(str(RAW / "spamhunter_texts" / "*.txt"))):
        n_before = len(seen)
        for line in Path(f).read_text(encoding="utf-8", errors="replace").splitlines():
            text = line.strip()
            if len(text) < 5:
                continue
            key = dedupe_key(text)
            if key in seen:
                continue
            seen.add(key)
            lang = "zh" if cjk.search(text) else "other"
            records.append(
                {
                    "id": "",
                    "text": text,
                    "label": -1,
                    "subtype": f"spam-{lang}",
                    "source": "spamhunter",
                    "split": "train",
                    "meta": {"file": Path(f).name, "lang": lang},
                }
            )
        if n_before != len(seen):
            print(f"[spamhunter] {Path(f).name} 新增 {len(seen) - n_before} 条")
    print(f"[spamhunter] 共 {len(records)} 条（角色=xlang）")
    return records


def parse_scamgen() -> list[dict]:
    """ScamGen（Mendeley 10.17632/dkypjhkmgb.1）：中文电话诈骗话术，5 类意图。

    文件格式 "text\\tintent_id"（0=公检法 1=投机 2=轻率 3=冒充熟人 4=其他），
    含 Easy/Med/Hard 三难度与 mask 变体（互为重复，去重合并）。
    文档映射表未给 ScamGen 映射，全部 label=-1，角色=xlang（类型补充）。
    """
    import glob

    intents = {0: "公检法", 1: "投机", 2: "轻率", 3: "冒充熟人", 4: "其他"}
    records: list[dict] = []
    seen: set[str] = set()
    for f in sorted(glob.glob(str(RAW / "scamgen" / "**" / "*.txt"), recursive=True)):
        if "/.git/" in f.replace("\\", "/"):
            continue
        n_before = len(seen)
        for line in Path(f).read_text(encoding="utf-8", errors="replace").splitlines():
            if "\t" not in line:
                continue
            text, _, label_str = line.rpartition("\t")
            text = text.strip()
            if len(text) < 5:
                continue
            key = dedupe_key(text)
            if key in seen:
                continue
            seen.add(key)
            try:
                intent = int(label_str.strip())
            except ValueError:
                continue
            records.append(
                {
                    "id": "",
                    "text": text,
                    "label": -1,
                    "subtype": f"话术-{intents.get(intent, intent)}",
                    "source": "scamgen",
                    "split": "train",
                    "meta": {"native_intent": intent, "file": Path(f).name},
                }
            )
        if n_before != len(seen):
            print(f"[scamgen] {Path(f).name} 新增 {len(seen) - n_before} 条")
    print(f"[scamgen] 共 {len(records)} 条（角色=xlang）")
    return records


def parse_zenodo_phishing() -> list[dict]:
    """Zenodo 8339691（CC BY 4.0）：11 个经典英文钓鱼邮件 CSV（TREC/CEAS/Enron 等，约 428MB）。

    列 = sender/receiver/date/subject/body/label/urls。全英文，角色=xlang，label=-1（原生标签存 meta）。
    """
    import csv as _csv
    import re as _re

    _csv.field_size_limit(50_000_000)
    base = RAW / "zenodo_phishing"
    records: list[dict] = []
    seen: set[str] = set()
    for f in sorted(base.glob("*.csv")):
        n_before = len(seen)
        for row in _csv.DictReader(open(f, encoding="utf-8", errors="replace")):
            subject = _re.sub(r"\s+", " ", row.get("subject") or "").strip()
            body = _re.sub(r"<[^>]+>", " ", row.get("body") or "")
            body = _re.sub(r"\s+", " ", body).strip()[:4000]
            text = (subject + "\n" + body).strip()
            if len(text) < 5:
                continue
            key = dedupe_key(text)
            if key in seen:
                continue
            seen.add(key)
            native = (row.get("label") or "").strip()
            records.append(
                {
                    "id": "",
                    "text": text,
                    "label": -1,
                    "subtype": f"en-邮件-{f.stem}",
                    "source": "zenodo_phishing",
                    "split": "train",
                    "meta": {"native_label": native, "file": f.name},
                }
            )
        print(f"[zenodo] {f.name} 新增 {len(seen) - n_before} 条")
    print(f"[zenodo] 共 {len(records)} 条（角色=xlang）")
    return records


def parse_anti_fraud_sft() -> list[dict]:
    """golfxiao/anti_fraud_sft（GPL-3.0）的净增部分：meeting_label_false.csv。

    仓库的 12 类 csv_dialogs 与已入库的 anti_fraud_dataset **逐字节相同**（同一上游），不再重复入库；
    净增只有 7,370 条 is_fraud=False 的正常商务会议通话对话，作为正常类风格多样性补充（角色=xlang）。
    """
    import csv as _csv

    path = RAW / "anti_fraud_sft" / "csv_dialogs" / "meeting_label_false.csv"
    records: list[dict] = []
    if not path.exists():
        print("[anti_fraud_sft] 缺少 meeting_label_false.csv，跳过")
        return records
    seen: set[str] = set()
    cases: dict[str, list[dict]] = {}
    for r in _csv.DictReader(open(path, encoding="utf-8")):
        cases.setdefault(r["case"], []).append(r)
    for case_id, turns in cases.items():
        text = "\n".join(f"【{t['speaker']}】{t['content']}" for t in turns if t.get("content"))
        if len(text) < 5:
            continue
        key = dedupe_key(text)
        if key in seen:
            continue
        seen.add(key)
        records.append(
            {
                "id": "",
                "text": text,
                "label": -1,
                "subtype": "会议-正常对话",
                "source": "anti_fraud_sft",
                "split": "train",
                "meta": {"case": case_id, "turns": len(turns), "native_label": "False"},
            }
        )
    print(f"[anti_fraud_sft] 共 {len(records)} 段会议对话（角色=xlang）")
    return records


def parse_ccl23() -> list[dict]:
    """CCL2023-Eval 任务 6（刷单返利类 35,459 条为最大类）——需用户从 AI Studio 镜像手动下载。

    把下载的 JSON/JSONL 放入 rawdata/ccl2023/ 后本解析器自动生效。
    数据格式（官方 README）：{"案件编号": int, "案情描述": str, "案件类别": str}，
    类别与 FGRC-SCD 同一 taxonomy（复检后映射见 label_map.FGRC_MAP；刷单返利类→2 刷单）。
    注意：文本为受害人笔录（第三人称），与消息文体存在域偏移，用于训练时建议单独评估（见数据卡）。
    """
    from antifraud.data.label_map import map_fgrc

    base = RAW / "ccl2023"
    if not base.exists():
        print("[ccl2023] 未发现 rawdata/ccl2023/（请先从 AI Studio 镜像下载，见数据卡说明），跳过")
        return []

    def iter_records(path: Path):
        raw = path.read_bytes()
        encoding = "utf-8"
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError:
            encoding = "gbk"  # AI Studio 镜像为 GBK 编码
        text = raw.decode(encoding, errors="replace").strip()
        if not text:
            return
        if text.startswith("["):  # JSON 数组
            for obj in json.loads(text):
                yield obj
        else:  # JSONL / 每行一个 JSON
            for line in text.splitlines():
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue

    records: list[dict] = []
    seen: set[str] = set()
    for f in sorted(base.rglob("*.json")) + sorted(base.rglob("*.jsonl")) + sorted(base.rglob("*.txt")):
        n_before = len(seen)
        for obj in iter_records(f):
            desc = str(obj.get("案情描述") or obj.get("text") or "").strip()
            cat = str(obj.get("案件类别") or "").strip()
            if len(desc) < 10 or not cat:
                continue  # 无标签的竞赛 test 集不导入
            key = dedupe_key(desc)
            if key in seen:
                continue
            seen.add(key)
            name = f.stem.lower()
            split = "val" if "dev" in name else ("test" if "test" in name else ("train" if "train" in name else ""))
            records.append(
                {
                    "id": "",
                    "text": desc,
                    "label": map_fgrc(cat),
                    "subtype": cat,
                    "source": "ccl2023",
                    "split": split if map_fgrc(cat) >= 0 else "train",
                    "meta": {"case_id": obj.get("案件编号"), "file": f.name},
                }
            )
        print(f"[ccl2023] {f.name} 新增 {len(seen) - n_before} 条")
    print(f"[ccl2023] 共 {len(records)} 条")
    return records


def parse_spam10000() -> list[dict]:
    """reatiny/chinese-spam-10000（gated 已授权）：群聊风格垃圾/推广短信约 1 万条。

    内容高度契合校园场景（色情引流、扣号、刷单接单话术，按定义归入推销类）。
    label 1（垃圾/推广）→ 1 推销；label 0（普通闲聊）→ 0 正常；train/dev/test 保留官方划分。
    注意：spam 混有少量赌博类推广，无细分标签，如需剔除按 subtype 筛选。
    """
    import csv as _csv
    from collections import Counter as _Counter

    base = RAW / "chinese-spam-10000"
    records: list[dict] = []
    seen: set[str] = set()
    for fname, split in (("train.csv", "train"), ("dev.csv", "val"), ("test.csv", "test")):
        path = base / fname
        if not path.exists():
            print(f"[spam10000] 缺少 {fname}，跳过")
            continue
        n_before = len(seen)
        for r in _csv.DictReader(open(path, encoding="utf-8")):
            text = (r.get("text") or "").strip()
            if len(text) < 3:  # 短文本多，阈值放宽
                continue
            key = dedupe_key(text)
            if key in seen:
                continue
            seen.add(key)
            native = (r.get("label") or "").strip()
            records.append(
                {
                    "id": "",
                    "text": text,
                    "label": {"1": 1, "0": 0}.get(native, -1),
                    "subtype": "群聊垃圾推广" if native == "1" else "群聊闲聊",
                    "source": "spam10000",
                    "split": split,
                    "meta": {"native_label": native, "file": fname},
                }
            )
        print(f"[spam10000] {fname} 新增 {len(seen) - n_before} 条")
    print(f"[spam10000] 共 {len(records)} 条")
    return records


def parse_fgrc_scd() -> list[dict]:
    """FGRC-SCD（HF Abooooo/FGRC-SCD）：中文风险短信与对话，风险类别与 CCL23 12 类同 taxonomy。

    短信 = message/*.json（文本/风险类别/案件编号）；对话 = dialog/dialog/*.json（riskType/riskPoint/text）。
    映射见 label_map.FGRC_MAP：无风险→0，冒充 4 类→3，虚假购物服务→1（可选），其余→-1。
    """
    import glob

    records: list[dict] = []
    seen: set[str] = set()

    def add(text: str, native: str, subtype_tag: str, extra_meta: dict, path_name: str) -> None:
        text = text.strip()
        if len(text) < 5:
            return
        key = dedupe_key(text)
        if key in seen:
            return
        seen.add(key)
        records.append(
            {
                "id": "",
                "text": text,
                "label": map_fgrc(native),
                "subtype": f"{subtype_tag}-{native}",
                "source": "fgrc_scd",
                "split": "",
                "meta": {"risk_category": native, "file": path_name, **extra_meta},
            }
        )

    for f in sorted(glob.glob(str(RAW / "fgrc_scd" / "sms" / "message" / "*.json"))):
        path_name = Path(f).name
        data = json.loads(Path(f).read_text(encoding="utf-8"))
        n_before = len(seen)
        for r in data:
            add(r.get("文本", ""), r.get("风险类别", ""), "短信", {"case_id": r.get("案件编号")}, path_name)
        print(f"[fgrc_scd] sms/{path_name} 新增 {len(seen) - n_before} 条")
    for f in sorted(glob.glob(str(RAW / "fgrc_scd" / "dialog" / "dialog" / "*.json"))):
        path_name = Path(f).name
        data = json.loads(Path(f).read_text(encoding="utf-8"))
        n_before = len(seen)
        for r in data:
            add(r.get("text", ""), r.get("riskType", ""), "对话", {"risk_point": r.get("riskPoint", "")}, path_name)
        print(f"[fgrc_scd] dialog/{path_name} 新增 {len(seen) - n_before} 条")
    print(f"[fgrc_scd] 共 {len(records)} 条（去重后）")
    return records


def parse_anti_fraud() -> list[dict]:
    """anti_fraud_dataset（HF quanshiyun/anti_fraud_dataset）：按诈骗类型分文件的诈骗对话。

    CSV 列 = case/speaker/content/is_fraud（is_fraud=True 为骗子发言），每个 case 一段对话。
    映射见 label_map.ANTI_FRAUD_MAP：兼职刷单诈骗→2、冒充客服诈骗→3，其余类型→-1。
    """
    import glob
    import csv as _csv

    records: list[dict] = []
    seen: set[str] = set()
    for f in sorted(glob.glob(str(RAW / "anti_fraud" / "*诈骗*.csv"))):
        native_type = Path(f).stem.split("(")[0].strip("“”\" ")
        rows = list(_csv.DictReader(open(f, encoding="utf-8")))
        cases: dict[str, list[dict]] = {}
        for r in rows:
            cases.setdefault(r["case"], []).append(r)
        n_before = len(seen)
        for case_id, turns in cases.items():
            text = "\n".join(f"【{t['speaker']}】{t['content']}" for t in turns if t.get("content"))
            if len(text) < 5:
                continue
            key = dedupe_key(text)
            if key in seen:
                continue
            seen.add(key)
            records.append(
                {
                    "id": "",
                    "text": text,
                    "label": map_anti_fraud(native_type),
                    "subtype": native_type,
                    "source": "anti_fraud",
                    "split": "",
                    "meta": {"native_type": native_type, "case": case_id, "turns": len(turns)},
                }
            )
        print(f"[anti_fraud] {native_type} cases={len(cases)} 新增 {len(seen) - n_before} 条")
    print(f"[anti_fraud] 共 {len(records)} 条对话")
    return records


def parse_llmgen_phishing() -> list[dict]:
    """LLMGen-Phishing（HF Dizzzy0x00/...）：钓鱼邮件二分类，中文 3,030 + 英文 3,746 条。

    中文部分按 label 映射（1→3 钓鱼，0→0 正常）；英文部分 → -1（跨语言泛化用，不入中文训练池）。
    """
    import csv as _csv

    records: list[dict] = []
    seen: set[str] = set()
    for fname, lang, do_map in (("CN_Phishing_Email_dataset.csv", "cn", True), ("GPT_Phishing_Email_dataset.csv", "en", False)):
        path = RAW / "llmgen_phishing" / fname
        if not path.exists():
            print(f"[llmgen] 缺少 {fname}，跳过")
            continue
        n_before = len(seen)
        for r in _csv.DictReader(open(path, encoding="utf-8-sig")):
            text = (r.get("content") or "").strip()
            if len(text) < 5:
                continue
            key = dedupe_key(text)
            if key in seen:
                continue
            seen.add(key)
            native = (r.get("label") or "").strip()
            label = {"1": 3, "0": 0}.get(native, -1) if do_map else -1
            records.append(
                {
                    "id": "",
                    "text": text,
                    "label": label,
                    "subtype": f"{lang}-邮件",
                    "source": "llmgen_phishing",
                    "split": "",
                    "meta": {"native_label": native, "lang": lang},
                }
            )
        print(f"[llmgen] {fname} 新增 {len(seen) - n_before} 条")
    print(f"[llmgen] 共 {len(records)} 条")
    return records

# 数据角色：clf = 进入四分类训练池（messages.lance / train-val-test.jsonl）；
# mlm = 仅作 MLM/二分类语料（独立 lance，不进四分类训练池，与文档"使用定位"一致）；
# xlang = 跨语言/补充材料（label 全 -1，独立 lance，仅用于鲁棒性与泛化实验）
SOURCE_ROLE = {
    "tianchi5": "clf",
    "teleantifraud": "clf",
    "chifraud": "mlm",
    "spam80w": "mlm",
    "fgrc_scd": "clf",
    "anti_fraud": "clf",
    "llmgen_phishing": "clf",
    "fbs_sms": "clf",
    "phishmmf": "clf",
    "spamhunter": "xlang",
    "scamgen": "xlang",
    "zenodo_phishing": "xlang",
    "anti_fraud_sft": "xlang",
    "ccl2023": "clf",
    "spam10000": "clf",
}


def assign_ids(records: list[dict], source: str) -> None:
    """按 (source, subtype) 顺序编号，生成稳定唯一 ID。"""
    counters: dict[str, int] = {}
    for r in records:
        prefix = f"{source}-{r['subtype']}" if r["subtype"] else source
        counters[prefix] = counters.get(prefix, 0) + 1
        r["id"] = f"{prefix}-{counters[prefix]:06d}"


def stratified_split(records: list[dict]) -> None:
    """按 (label, subtype) 分层划分 train/val/test；已有特殊 split 的行（asr_review 等）保持不变。"""
    rng = random.Random(42)
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        if r["split"]:
            continue  # teleantifraud 的 asr_review / untranscribed 行不重新划分
        if r["label"] < 0:
            r["split"] = "train"  # 未映射行归入 train 侧（仅作 MLM/二分类材料，不进四分类）
            continue
        groups.setdefault((r["label"], r["subtype"]), []).append(r)
    for (label, subtype), rows in sorted(groups.items()):
        rng.shuffle(rows)
        n = len(rows)
        n_val = max(1, round(n * SPLIT_RATIO["val"]))
        n_test = max(1, round(n * SPLIT_RATIO["test"]))
        for r in rows[:n_val]:
            r["split"] = "val"
        for r in rows[n_val : n_val + n_test]:
            r["split"] = "test"
        for r in rows[n_val + n_test :]:
            r["split"] = "train"
        print(f"  划分 label={label}({LABEL_NAMES[label]}) subtype={subtype}: train={n - n_val - n_test} val={n_val} test={n_test}")


def write_lance(jsonl_paths: list[Path], out: Path) -> None:
    import lance
    import pyarrow as pa

    rows = []
    for p in jsonl_paths:
        rows.extend(json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip())
    columns = {
        "id": [r.get("id") or "" for r in rows],
        "text": [r.get("text") or "" for r in rows],
        "label": [int(r.get("label", -1)) for r in rows],
        "subtype": [r.get("subtype") or "" for r in rows],
        "source": [r.get("source") or "" for r in rows],
        "split": [r.get("split") or "" for r in rows],
        "meta": [json.dumps(r.get("meta") or {}, ensure_ascii=False) for r in rows],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    # lance>=12 移除了 write_table，改用 write_dataset
    lance.write_dataset(pa.table(columns), str(out), mode="overwrite")
    print(f"[lance] {out} ({len(rows)} 行)")


def main() -> None:
    parser = argparse.ArgumentParser(description="rawdata → 统一 JSONL + Lance")
    parser.add_argument("--only", default=None, help="逗号分隔：tianchi5,teleantifraud")
    parser.add_argument("--no-lance", action="store_true", help="只产出 JSONL，不写 Lance")
    parser.add_argument(
        "--promote-asr-review",
        action="store_true",
        help="人工审查通过后使用：把 teleantifraud 的 split=asr_review 行放行，并入分层划分",
    )
    args = parser.parse_args()

    names = list(PARSERS)
    if args.only:
        names = [n.strip() for n in args.only.split(",") if n.strip() in PARSERS]

    PROCESSED.mkdir(parents=True, exist_ok=True)
    per_source: dict[str, list[dict]] = {}
    for name in names:
        records = PARSERS[name]()
        if args.promote_asr_review:
            promoted = 0
            for r in records:
                if r["source"] == "teleantifraud" and r["split"] == "asr_review" and r["text"]:
                    r["split"] = ""  # 交回 stratified_split 分层划分
                    promoted += 1
            print(f"[promote] 已放行 {promoted} 条 asr_review 行进入分层划分")
        assign_ids(records, name)
        stratified_split(records)
        per_source[name] = records
        write_jsonl(PROCESSED / f"{name}.jsonl", records)

    # 训练划分：仅"clf 角色"来源中已映射(label>=0)的行，按 split 字段分流。
    # 始终从磁盘读取全部 clf 源的 JSONL（而非仅本次解析的源），避免 --only 局部重建清空全局划分。
    mapped: list[dict] = []
    for name, role in SOURCE_ROLE.items():
        if role != "clf":
            continue
        src = PROCESSED / f"{name}.jsonl"
        if not src.exists():
            continue
        for line in src.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("label", -1) >= 0:
                mapped.append(r)
    for split in ("train", "val", "test"):
        rows = [r for r in mapped if r["split"] == split]
        write_jsonl(PROCESSED / f"{split}.jsonl", rows)
        print(f"[split] {split}: {len(rows)} 条")

    if not args.no_lance:
        for name, records in per_source.items():
            if not records:
                print(f"[lance] {name} 无数据，跳过")
                continue
            src = PROCESSED / f"{name}.jsonl"
            write_lance([src], DATASET_DIR / f"{name}.lance")
        write_lance(
            [PROCESSED / f"{s}.jsonl" for s in ("train", "val", "test")],
            DATASET_DIR / "messages.lance",
        )

    # 汇总统计
    print("\n=== 汇总 ===")
    for name, records in per_source.items():
        by_label: dict[int, int] = {}
        for r in records:
            by_label[r["label"]] = by_label.get(r["label"], 0) + 1
        stat = {LABEL_NAMES[k] if k >= 0 else "未映射": v for k, v in sorted(by_label.items())}
        print(f"{name}: 共 {len(records)} 条，{stat}")


if __name__ == "__main__":
    main()
