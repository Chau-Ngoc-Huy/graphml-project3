"""Kiểm tra harness: thí nghiệm khói chạy end-to-end, mọi config có module tương ứng, ghi đè --set hoạt động."""

import json
from pathlib import Path

import pytest

from mmxai.core.config import apply_overrides, load_experiment_config
from mmxai.core.runner import main
from mmxai.experiments import available

ROOT = Path(__file__).resolve().parents[1]


def test_smoke_end_to_end(tmp_path):
    run_dir = main(["--exp", "smoke", "--limit", "7", "--runs-dir", str(tmp_path), "--no-publish", "--tag", "t"])
    assert run_dir is not None and run_dir.name.endswith("_smoke-t")
    m = json.loads((run_dir / "metrics.json").read_text())
    assert m["n"] == 7 and 0 <= m["metrics"]["accuracy"] <= 1
    assert m["run"]["error"] is None and m["environment"]["python"]
    assert (run_dir / "config.json").exists() and (run_dir / "predictions.json").exists()


def test_smoke_is_deterministic(tmp_path):
    a = main(["--exp", "smoke", "--runs-dir", str(tmp_path / "a"), "--no-publish"])
    b = main(["--exp", "smoke", "--runs-dir", str(tmp_path / "b"), "--no-publish"])
    ma, mb = (json.loads((d / "metrics.json").read_text())["metrics"] for d in (a, b))
    assert ma == mb


@pytest.mark.parametrize("cfg", sorted((ROOT / "configs" / "experiments").glob("*.yaml")))
def test_every_config_has_experiment(cfg):
    assert cfg.stem in available(), f"configs/experiments/{cfg.name} không có experiments/{cfg.stem}.py"


def test_every_experiment_has_config():
    for exp in available():
        assert (ROOT / "configs" / "experiments" / f"{exp}.yaml").exists(), f"experiments/{exp}.py thiếu configs/experiments/{exp}.yaml"


def test_overrides():
    cfg = load_experiment_config(ROOT / "configs", "smoke")
    cfg = apply_overrides(cfg, ["n=5", "model.name=x", "flag=true", "lst=[1,2]"])
    assert cfg["n"] == 5 and cfg["model"]["name"] == "x" and cfg["flag"] is True and cfg["lst"] == [1, 2]


def test_unknown_experiment_exits(tmp_path):
    with pytest.raises(SystemExit):
        main(["--exp", "khong-co", "--runs-dir", str(tmp_path), "--no-publish"])
