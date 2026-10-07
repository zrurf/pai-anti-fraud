"""配置加载

configs/default.yaml 是全部超参的唯一来源
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
