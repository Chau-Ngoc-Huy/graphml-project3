"""Nạp cấu hình YAML của thí nghiệm + ghi đè từ dòng lệnh (--set a.b=1)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml


def load_yaml(path: Path | str) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_experiment_config(config_dir: Path, exp: str, config_file: str | None = None) -> dict:
    """configs/experiments/<exp>.yaml, có thể kế thừa file khác qua khóa `extends:` (đường dẫn tương đối config_dir)."""
    path = Path(config_file) if config_file else config_dir / "experiments" / f"{exp}.yaml"
    cfg = load_yaml(path)
    seen = {path.resolve()}
    while cfg.get("extends"):
        base_path = (path.parent / cfg.pop("extends")).resolve()
        if base_path in seen:
            raise ValueError(f"extends lặp vòng: {base_path}")
        seen.add(base_path)
        cfg = deep_merge(load_yaml(base_path), cfg)
    cfg.setdefault("exp", exp)
    return cfg


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def parse_value(text: str):
    """'1' → 1, 'true' → True, '[1,2]' → list, chuỗi khác giữ nguyên."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        lowered = text.lower()
        if lowered in ("true", "false"):
            return lowered == "true"
        if lowered in ("null", "none"):
            return None
        return text


def apply_overrides(cfg: dict, items: list[str]) -> dict:
    cfg = copy.deepcopy(cfg)
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--set cần dạng key=value, nhận: {item!r}")
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"--set {key}: '{part}' không phải dict")
        node[parts[-1]] = parse_value(value)
    return cfg
