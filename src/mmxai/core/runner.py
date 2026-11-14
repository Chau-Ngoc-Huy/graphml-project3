"""Vòng chạy chung: đọc cấu hình → run_id → seed → gọi experiments/<exp>.run(ctx) → metrics.json → results/."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import results as results_mod
from .config import apply_overrides, load_experiment_config
from .env import ROOT, environment_info, set_seed, user_slug
from .vendor import vendor_config

PKG = __name__.split(".")[0]


@dataclass
class Context:
    """Những gì một thí nghiệm nhận được. Adapter chỉ dùng ctx, không tự đọc argv."""

    exp: str
    cfg: dict
    run_dir: Path
    seed: int
    limit: int | None
    args: dict
    vendor_cfg: dict = field(default_factory=dict)
    root: Path = ROOT

    def log_path(self, name: str = "vendor.log") -> Path:
        return self.run_dir / name

    def save_json(self, name: str, obj) -> Path:
        path = self.run_dir / name
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
        return path

    def data_dir(self, key: str = "data") -> Path | None:
        """Đường dẫn dữ liệu từ cfg[key] (trên Kaggle đã được kernel đổi thành /kaggle/input/...)."""
        val = self.cfg.get(key)
        return Path(val) if val else None


def load_experiment(exp: str):
    try:
        return importlib.import_module(f"{PKG}.experiments.{exp}")
    except ModuleNotFoundError as e:
        if e.name and e.name.endswith(exp):
            sys.exit(f"Không có thí nghiệm '{exp}': tạo src/{PKG}/experiments/{exp}.py (xem _template.py) "
                     f"và configs/experiments/{exp}.yaml")
        raise


def make_run_id(exp: str, user: str, tag: str | None) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}_{user}_{exp}" + (f"-{tag}" if tag else "")


def peak_vram_gb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    except Exception:
        pass
    return None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp", required=True, help="tên thí nghiệm = experiments/<exp>.py + configs/experiments/<exp>.yaml")
    p.add_argument("--config", help="file YAML khác thay cho configs/experiments/<exp>.yaml")
    p.add_argument("--config-dir", default=str(ROOT / "configs"))
    p.add_argument("--set", action="append", default=[], metavar="KEY=VAL", help="ghi đè cấu hình, vd. --set model.name=x")
    p.add_argument("--limit", type=int, help="chỉ chạy N mẫu/bước đầu (lượt thử; luôn làm trước lượt đầy đủ)")
    p.add_argument("--seed", type=int, help="mặc định cfg.seed hoặc 42")
    p.add_argument("--tag", help="hậu tố run_id để nói rõ đang thử gì, vd. lr1e-4")
    p.add_argument("--runs-dir", default=str(ROOT / "runs"))
    p.add_argument("--run-id", help="ép run_id (ít dùng)")
    p.add_argument("--no-publish", action="store_true", help="không chép metrics sang results/ (Kaggle: pull sẽ chép)")
    p.add_argument("--dry-run", action="store_true", help="chỉ in cấu hình đã hợp nhất rồi thoát")
    return p


def main(argv: list[str] | None = None) -> Path | None:
    args = build_parser().parse_args(argv)
    cfg = apply_overrides(load_experiment_config(Path(args.config_dir), args.exp, args.config), args.set)
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    cfg["seed"] = seed
    if args.limit:
        cfg["limit"] = args.limit
    if args.dry_run:
        print(json.dumps(cfg, ensure_ascii=False, indent=1))
        return None

    module = load_experiment(args.exp)
    user = user_slug()
    run_id = args.run_id or make_run_id(args.exp, user, args.tag)
    run_dir = Path(args.runs_dir) / args.exp / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    vcfg = vendor_config()
    ctx = Context(exp=args.exp, cfg=cfg, run_dir=run_dir, seed=seed, limit=args.limit, args=vars(args), vendor_cfg=vcfg)
    set_seed(seed)
    print(f"[{PKG}] exp={args.exp} run_id={run_id} seed={seed} limit={args.limit}", flush=True)

    started = datetime.now(timezone.utc)
    t0 = time.time()
    error = None
    out: dict = {}
    try:
        out = module.run(ctx) or {}
    except Exception:
        error = traceback.format_exc()
        print(error, file=sys.stderr)
    wall = time.time() - t0
    n = out.get("n")
    metrics = {
        "metrics": out.get("metrics", {}),
        "n": n,
        "paper": cfg.get("paper"),
        "run": {
            "exp": args.exp, "run_id": run_id, "tag": args.tag, "user": user, "args": vars(args), "config": cfg,
            "started_at": started.isoformat(timespec="seconds"),
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "wall_time_s": round(wall, 2),
            "sec_per_sample": round(wall / n, 3) if n else None,
            "peak_vram_gb": peak_vram_gb(),
            "error": error,
            "extra": out.get("extra"),
        },
        "environment": environment_info(vcfg.get("track_packages"), ROOT / "vendor" / str(vcfg.get("name", ""))),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{PKG}] metrics: {json.dumps(metrics['metrics'], ensure_ascii=False)}")
    print(f"[{PKG}] run dir: {run_dir}")
    if error:
        sys.exit(1)
    if not args.no_publish:
        print(f"[{PKG}] results: {results_mod.publish(run_dir)}")
    return run_dir
