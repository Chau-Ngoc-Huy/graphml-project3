#!/usr/bin/env python3
"""Sinh web/data.js từ thư mục kết quả của notebook Nhiệm vụ 1 (chỉ dùng thư viện chuẩn).

    python build_site.py ../kaggle_results          # thư mục có results/, split/, features_meta.json
    python build_site.py /kaggle/working/task1      # hoặc WORK của notebook (features/meta.json)

Sau đó mở web/index.html bằng trình duyệt (double-click là đủ, không cần server).
"""
import csv
import json
import re
import struct
import sys
import zipfile
from array import array
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "../kaggle_results").resolve()
OUT = Path(__file__).resolve().parent / "data.js"
RES = SRC / "results"
METRICS = ["R@1", "R@5", "R@10", "MRR", "nDCG@10"]


def load_npz(path):
    """Đọc .npz chứa mảng int64/float64 1 chiều mà không cần numpy."""
    out = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            raw = z.read(name)
            assert raw[:6] == b"\x93NUMPY"
            major = raw[6]
            hlen = struct.unpack("<H", raw[8:10])[0] if major == 1 else struct.unpack("<I", raw[8:12])[0]
            hstart = 10 if major == 1 else 12
            header = raw[hstart:hstart + hlen].decode("latin1")
            descr = re.search(r"'descr':\s*'([^']+)'", header).group(1)
            code = {"<i8": "q", "<i4": "i", "<f8": "d", "<f4": "f"}[descr]
            a = array(code)
            a.frombytes(raw[hstart + hlen:])
            out[name[:-4]] = list(a)
    return out


def rank_hist(ranks):
    bins = [(1, 1, "1"), (2, 2, "2"), (3, 5, "3–5"), (6, 10, "6–10"), (11, 20, "11–20"), (21, 50, "21–50"), (51, 100, "51–100"), (101, 10 ** 9, ">100")]
    n = len(ranks)
    return [dict(bin=lab, count=sum(1 for r in ranks if lo <= r <= hi), pct=round(100 * sum(1 for r in ranks if lo <= r <= hi) / n, 2)) for lo, hi, lab in bins]


def main():
    base = json.load(open(RES / "clip_baseline.json", encoding="utf-8"))
    ex = json.load(open(RES / "example_rerank_lexical.json", encoding="utf-8"))
    split = json.load(open(SRC / "split" / "split.json", encoding="utf-8"))
    fmeta_path = SRC / "features_meta.json" if (SRC / "features_meta.json").exists() else SRC / "features" / "meta.json"
    fmeta = json.load(open(fmeta_path, encoding="utf-8"))

    def rows_of(res, method, direction):
        rows = []
        for ps in res["per_seed"]:
            rows.append(dict(method=method, direction=direction, split=res["split"],
                             protocol="full" if ps["seed"] is None else f"seed{ps['seed']}", gallery=ps["gallery_size"], n=ps["n"],
                             MedR=ps["MedR"], **{m: round(ps[m], 2) for m in METRICS}))
        ms = res["mean_std"]
        rows.append(dict(method=method, direction=direction, split=res["split"], protocol="mean3", gallery=res.get("gallery_size", 1000), n=ms["n"], MedR=None, **{m: round(ms[m]["mean"], 2) for m in METRICS}))
        rows.append(dict(method=method, direction=direction, split=res["split"], protocol="std3", gallery=res.get("gallery_size", 1000), n=ms["n"], MedR=None, **{m: round(ms[m]["std"], 2) for m in METRICS}))
        return rows

    metrics = (rows_of(base["t2i_test"], "clip", "t2i") + rows_of(base["t2i_val"], "clip", "t2i") +
               rows_of(base["i2t_test"], "clip", "i2t") + rows_of(base["i2t_val"], "clip", "i2t") +
               rows_of(ex["result"], "clip_lex", "t2i"))

    def groups_of(res, method):
        out = []
        for g, a in res["by_group"].items():
            out.append(dict(method=method, split=res["split"], group=g, n=a["n"], **{m: round(a[m]["mean"], 2) for m in METRICS}, **{m + "_std": round(a[m]["std"], 2) for m in METRICS}))
        full = next(ps for ps in res["per_seed"] if ps["seed"] is None)
        for g, v in full.get("by_group", {}).items():
            out.append(dict(method=method, split=res["split"] + "_full", group=g, n=v["n"], **{m: round(v[m], 2) for m in METRICS}, **{m + "_std": 0.0 for m in METRICS}))
        return out

    by_group = groups_of(base["t2i_test"], "clip") + groups_of(base["t2i_val"], "clip") + groups_of(ex["result"], "clip_lex")

    ranks = load_npz(RES / "ranks_clip_t2i_test.npz")
    rank_dist = {k: rank_hist(v) for k, v in ranks.items()}

    examples = list(csv.DictReader(open(RES / "clip_success_failure.csv", encoding="utf-8")))
    html = (RES / "clip_success_failure.html").read_text(encoding="utf-8") if (RES / "clip_success_failure.html").exists() else ""
    urls = re.findall(r"<img src='([^']+)'", html)
    for i, r in enumerate(examples):
        r["rank"] = int(r["rank"])
        if len(urls) >= 2 * len(examples):
            r["gt_url"], r["top1_url"] = urls[2 * i], urls[2 * i + 1]

    data = dict(
        meta=dict(model=fmeta["model"], pretrained=fmeta["pretrained"], dim=fmeta["dim"], n_images=fmeta["n_images"], n_images_ok=fmeta["n_images_ok"],
                  n_captions=fmeta["n_captions"], split=dict(train=len(split["train"]), val=len(split["val"]), test=len(split["test"])),
                  pool=split["meta"]["pool_sizes"], seed_data=split["meta"]["seed_data"], seeds=split["meta"]["seeds"], gallery_size=split["meta"]["gallery_size"],
                  top_k=split["meta"]["top_k"], created=split["meta"]["created"], source=split["meta"]["source"],
                  alpha=ex["alpha"], bootstrap=ex["bootstrap_vs_clip"]),
        metrics=metrics, by_group=by_group, alpha_curve=[dict(alpha=c["alpha"], **{m: round(c[m], 2) for m in ["R@1", "R@5", "R@10", "MRR"]}) for c in ex["curve"]],
        rank_dist=rank_dist, examples=examples,
    )
    OUT.write_text("// Sinh tự động bởi build_site.py — không sửa tay\nwindow.T1DATA = " + json.dumps(data, ensure_ascii=False, indent=1) + ";\n", encoding="utf-8")
    print(f"đã ghi {OUT} ({OUT.stat().st_size / 1024:.1f} KB): {len(metrics)} dòng metric, {len(by_group)} dòng nhóm, {len(examples)} ví dụ")


if __name__ == "__main__":
    main()
