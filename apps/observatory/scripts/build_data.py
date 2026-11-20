#!/usr/bin/env python3
"""Build the Observatory data layer (public/data/*.json) from the team's REAL result files.

    python3 scripts/build_data.py --team ../.. --bundle ../../data/task1_bundle

--team    root of the graphml-project3 repo (reads results/, results/adamf, queries/, results/clip_baseline)
--bundle  unzipped Task 1 bundle (data/images.jsonl, data/captions.jsonl, data/object_vocab_train.json); only used for
          test-set captions, image objects/relations and the concept vocabulary (no features are read)

Standard library only. Every per-query rank file is cross-checked: MRR recomputed from ranks must equal the MRR the
owner reported, and group MRR from the CLIP ranks must match the group breakdown in protocol_coco_full.json (proves query order alignment).
"""
import argparse
import glob
import json
import re
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--team", default="../..")
ap.add_argument("--bundle", default="../../data/task1_bundle")
ap.add_argument("--out", default="public/data")
a = ap.parse_args()
TEAM, BUNDLE, OUT = Path(a.team).resolve(), Path(a.bundle).resolve(), Path(a.out)
OUT.mkdir(parents=True, exist_ok=True)
METRICS = ["R@1", "R@5", "R@10", "MRR", "nDCG@10"]


def rd(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def mrr(ranks):
    return 100.0 * sum(1.0 / r for r in ranks) / len(ranks)


def ms(d):  # {metric: {mean, std}} -> compact
    return {m: {"mean": round(d[m]["mean"], 2), "std": round(d[m]["std"], 2)} for m in METRICS if m in d}


def single(d):
    return {m: {"mean": round(d[m], 2), "std": None} for m in METRICS if m in d}


# --------------------------------------------------------------------------- test queries (same order as every rank file)
images = jl(BUNDLE / "data" / "images.jsonl")
caps = jl(BUNDLE / "data" / "captions.jsonl")
vocab = rd(BUNDLE / "data" / "object_vocab_train.json")
test_imgs = sorted(im["idx"] for im in images if im["split"] == "test")
caps_of = {}
for c in caps:
    caps_of.setdefault(c["img"], []).append(c)
img_pos = {idx: i for i, idx in enumerate(test_imgs)}
vid = {v: i for i, v in enumerate(vocab)}
GROUPS = ["direct", "indirect", "no_concept"]
q_text, q_img, q_group, q_conc = [], [], [], []
for idx in test_imgs:
    for c in caps_of[idx]:
        q_text.append(c["text"])
        q_img.append(img_pos[idx])
        q_group.append(GROUPS.index(c["group"]))
        q_conc.append([vid[x] for x in c.get("concepts", []) if x in vid])
assert len(q_text) == 10696, len(q_text)
img_out = []
for idx in test_imgs:
    im = images[idx]
    img_out.append({"vg": im["vg_id"], "coco": im["coco_id"], "url": im["url"], "objects": im["objects"][:30],
                    "relations": im["relations"][:40]})

# --------------------------------------------------------------------------- models with per-query ranks
KF = rd(TEAM / "results/adamf/protocol_coco_full.json")
KA = rd(TEAM / "results/adamf/protocol_coco_adapter.json")
HR = TEAM / "results/ranks"


def huy_official(exp):
    f = sorted(glob.glob(str(TEAM / f"results/{exp}/*kaggle-3seed*.json")))
    return rd(f[-1]) if f else None


def official_from_huy(exp):
    d = huy_official(exp)
    m, run = d["metrics"], d["run"]
    g = run.get("extra", {}).get("graph", {})
    return {
        "protocol": "3 seed (Kaggle T4)",
        "test1k": ms(m["test"]), "full": ms(m["test_full"]),
        "groups1k": {k: ms(v) for k, v in m["by_group"].items()},
        "alpha": round(m["alpha"]["mean"], 3),
        "cost": {"wall_time_s": run.get("wall_time_s"), "peak_vram_gb": run.get("peak_vram_gb"),
                 "sec_per_sample": run.get("sec_per_sample"), "device": "Kaggle Tesla T4", "scope": "3 seed huấn luyện + đánh giá"},
        "graph": {k: g.get(k) for k in ("n_label", "E_contains", "E_sg_rel", "E_concept", "variant")} if g else None,
        "config": {"gnn": run["config"]["gnn"].get("kind"), "variant": run["config"].get("graph", {}).get("variant"),
                   "extra": run["config"].get("extra")},
        "source": f"results/{exp}/" + Path(sorted(glob.glob(str(TEAM / f'results/{exp}/*kaggle-3seed*.json')))[-1]).name,
    }


def official_from_khanh(rec, src):
    out = {"protocol": "3 fold gallery, 1 seed huấn luyện" if "Adapter" in rec["name"] or "AdaMF" in rec["name"] else "3 fold gallery (tất định)",
           "test1k": ms(rec["mean_std"]), "full": single(rec["full"]),
           "groups1k": {k: ms(v) for k, v in rec.get("by_group", {}).items()}, "source": src}
    m = re.search(r"a=([0-9.]+)", rec["name"])
    out["alpha"] = float(m.group(1)) if m else None
    return out


MODELS = [
    # id, name, short, owner, family, flags, rank source, official
    dict(id="clip", name="CLIP (đóng băng)", short="CLIP", family="baseline",
         desc="OpenCLIP ViT-B/32 laion2b, cosine văn bản–ảnh, không học gì. Mốc cho mọi phương pháp.",
         flags=dict(adapter=False, graph=False, relations=False, conceptnet=False, learned=False, gnn=None),
         ranks=("khanh", KF["CLIP"]["full"]["ranks"]), official=official_from_khanh(KF["CLIP"], "results/adamf/protocol_coco_full.json")),
    dict(id="clip_lex", name="CLIP + presence matching", short="+Presence", family="lexical",
         desc="α·z(cos) + (1−α)·z(tỉ lệ concept của câu có trong danh sách object của ảnh). Không relation, không học.",
         flags=dict(adapter=False, graph=False, relations=False, conceptnet=False, learned=False, gnn=None),
         ranks=("khanh", KF["CLIP+presence (no relations)"]["full"]["ranks"]),
         official=official_from_khanh(KF["CLIP+presence (no relations)"], "results/adamf/protocol_coco_full.json")),
    dict(id="clip_adamf", name="CLIP + AdaMF-MAT", short="+AdaMF-MAT", family="kgc",
         desc="AdaMF-MAT (LREC-COLING 2024) học MMKG ảnh–concept; điểm = độ hợp lý của (ảnh, contains, concept).",
         flags=dict(adapter=False, graph=True, relations=True, conceptnet=False, learned=True, gnn=None),
         ranks=("khanh", KF["CLIP+AdaMF-MAT"]["full"]["ranks"]),
         official=official_from_khanh(KF["CLIP+AdaMF-MAT"], "results/adamf/protocol_coco_full.json")),
    dict(id="k_adapter", name="CLIP + Adapter-S (huấn luyện riêng)", short="Adapter-S", family="adapter",
         desc="Hai phép chiếu tuyến tính 512×512 học bằng InfoNCE trên caption/ảnh train.",
         flags=dict(adapter=True, graph=False, relations=False, conceptnet=False, learned=True, gnn=None),
         ranks=("khanh", KA["CLIP+Adapter"]["full"]["ranks"]), official=official_from_khanh(KA["CLIP+Adapter"], "results/adamf/protocol_coco_adapter.json")),
    dict(id="k_adapter_lex", name="Adapter-S + presence matching", short="Adapter-S+Presence", family="lexical",
         desc="Adapter-S (huấn luyện riêng) + presence matching (không relation).",
         flags=dict(adapter=True, graph=False, relations=False, conceptnet=False, learned=True, gnn=None),
         ranks=("khanh", KA["CLIP+Adapter+presence (no relations)"]["full"]["ranks"]),
         official=official_from_khanh(KA["CLIP+Adapter+presence (no relations)"], "results/adamf/protocol_coco_adapter.json")),
    dict(id="k_adapter_adamf", name="Adapter-S + AdaMF-MAT", short="Adapter-S+AdaMF", family="kgc",
         desc="Adapter-S (huấn luyện riêng) + điểm AdaMF-MAT, ghép bằng α trên z-score.",
         flags=dict(adapter=True, graph=True, relations=True, conceptnet=False, learned=True, gnn=None),
         ranks=("khanh", KA["CLIP+Adapter+AdaMF-MAT"]["full"]["ranks"]),
         official=official_from_khanh(KA["CLIP+Adapter+AdaMF-MAT"], "results/adamf/protocol_coco_adapter.json")),
]
HUY = [
    ("t1_qa_ablation_no_edge", "CLIP + Adapter-J (học chung)", "Adapter-J", "adapter", "Adapter học chung, graph bị tắt (no_edge ⇒ s_G = 0). Mốc đúng để đo đóng góp của graph.",
     dict(adapter=True, graph=False, relations=False, conceptnet=False, learned=True, gnn=None)),
    ("t1_qa", "QA-Graph · GAT · contains + sg_rel", "GAT", "graph", "Graph score phụ thuộc query: concept của câu ↔ label của ảnh qua GNN trên label graph (edge contains + relation scene graph).",
     dict(adapter=True, graph=True, relations=True, conceptnet=False, learned=True, gnn="GAT")),
    ("t1_qa_ablation_no_rel", "QA-Graph · chỉ contains", "GAT·contains", "ablation", "Bỏ toàn bộ edge label–label (sg_rel); chỉ còn ảnh–chứa–label.",
     dict(adapter=True, graph=True, relations=False, conceptnet=False, learned=True, gnn="GAT")),
    ("t1_qa_cn", "QA-Graph · GAT · + ConceptNet", "GAT+CN", "graph", "Thêm 1 000 concept và ~16k edge ConceptNet (IsA, AtLocation, Synonym…).",
     dict(adapter=True, graph=True, relations=True, conceptnet=True, learned=True, gnn="GAT")),
    ("t1_qa_cn_no_sg", "QA-Graph · GAT · contains + ConceptNet", "GAT·CN−sg", "graph", "ConceptNet nhưng bỏ edge sg_rel.",
     dict(adapter=True, graph=True, relations=True, conceptnet=True, learned=True, gnn="GAT")),
    ("t1_qa_cn_sage", "QA-Graph · SAGE · + ConceptNet", "SAGE+CN", "graph", "Như GAT+CN nhưng GNN = GraphSAGE.",
     dict(adapter=True, graph=True, relations=True, conceptnet=True, learned=True, gnn="SAGE")),
    ("t1_qa_cn_no_sg_sage", "QA-Graph · SAGE · contains + ConceptNet", "SAGE·CN−sg ★", "graph", "Cấu hình chính đề xuất (JOURNAL #9): cao nhất và đơn giản nhất.",
     dict(adapter=True, graph=True, relations=True, conceptnet=True, learned=True, gnn="SAGE")),
    ("t1_qa_cn_shuffle", "QA-Graph · edge shuffle", "Shuffle", "ablation", "Ablation: giữ số edge nhưng xáo đầu cuối. Kiểm tra model có dùng NỘI DUNG edge hay không.",
     dict(adapter=True, graph=True, relations=True, conceptnet=True, learned=True, gnn="GAT")),
]
for exp, name, short, fam, desc, flags in HUY:
    MODELS.append(dict(id=exp, name=name, short=short, family=fam, desc=desc, flags=flags,
                       ranks=("huy", rd(HR / f"{exp}.seed0.test_full.json")["ranks"]), official=official_from_huy(exp)))
OFFICIAL_ONLY = [
    ("t1_qa_cn_gcn", "QA-Graph · GCN · + ConceptNet", "GCN+CN", dict(adapter=True, graph=True, relations=True, conceptnet=True, learned=True, gnn="GCN")),
    ("t1_qa_cn_no_sg_gcn", "QA-Graph · GCN · contains + ConceptNet", "GCN·CN−sg", dict(adapter=True, graph=True, relations=True, conceptnet=True, learned=True, gnn="GCN")),
]
for exp, name, short, flags in OFFICIAL_ONLY:
    MODELS.append(dict(id=exp, name=name, short=short, family="graph", flags=flags,
                       desc="Biến thể GNN = GCN (chỉ có số chính thức 3 seed, chưa có rank từng query).",
                       ranks=None, official=official_from_huy(exp)))

# --------------------------------------------------------------------------- cross-checks (fail loudly)
ranks_out, checks = {}, []
for m in MODELS:
    if not m["ranks"]:
        m["hasRanks"] = False
        continue
    src, r = m["ranks"]
    assert len(r) == 10696, (m["id"], len(r))
    got = mrr(r)
    if src == "khanh":
        want = m["official"]["full"]["MRR"]["mean"]
    else:  # Huy: seed-0 full-pool MRR is the first value of test_full.MRR
        d = huy_official(m["id"])["metrics"]["test_full"]["MRR"]["values"][0]
        want = round(d, 2)
    assert abs(got - want) < 0.02, (m["id"], got, want)
    checks.append({"model": m["id"], "mrr_from_ranks": round(got, 2), "reported": want})
    m["seed0FullMRR"] = round(got, 2)
    m["hasRanks"] = True
    ranks_out[m["id"]] = r
    del m["ranks"]
for m in MODELS:
    m.pop("ranks", None)
kclip = KF["CLIP"]["full"]["by_group"]
for gi, g in enumerate(["direct", "indirect"]):
    rr = [r for r, gg in zip(ranks_out["clip"], q_group) if gg == gi]
    assert abs(mrr(rr) - kclip[g]["MRR"]) < 0.01 and len(rr) == kclip[g]["n"], (g, mrr(rr), kclip[g])
checks.append({"model": "clip", "check": "group alignment direct/indirect matches khanh by_group", "ok": True})

# --------------------------------------------------------------------------- other results
PAIRED_ID = {"CLIP (Thắng)": "clip", "CLIP+khớp tên (Thắng/Khánh)": "clip_lex", "CLIP+AdaMF-MAT (Khánh)": "clip_adamf"}
paired = {PAIRED_ID.get(k, k): v for k, v in rd(TEAM / "results/adamf/vs_huy_paired.json").items()}
kq = rd(TEAM / "queries/eval_queries.json")
lvl2 = []
md = (TEAM / "results/adamf/level2_vg_own_experiment.md").read_text(encoding="utf-8")
sec = md.split("## Results on the 24 eval queries", 1)[1].split("\n## ", 1)[0]
grp = None
for line in sec.splitlines():
    cells = [c.strip().strip("*") for c in line.strip().strip("|").split("|")]
    if len(cells) == 7 and cells[1] and not cells[1].startswith("---") and cells[1] != "Method":
        grp = cells[0] or grp
        lvl2.append({"group": grp, "method": cells[1], **{k: float(v) for k, v in zip(METRICS, cells[2:])}})
repro = []
rmd = (TEAM / "results/adamf/adamf_mat_repro.md").read_text(encoding="utf-8")
for line in rmd.split("## Comparison with the paper", 1)[1].splitlines():
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    if len(cells) == 6 and cells[0] in ("MKG-W", "MKG-Y") and cells[1] in ("paper", "ours"):
        repro.append({"dataset": cells[0], "who": cells[1], **{k: float(v) for k, v in zip(["MRR", "Hits@1", "Hits@3", "Hits@10"], cells[2:])}})
tb = rd(TEAM / "results/clip_baseline/results/clip_baseline.json")
i2t = {"test1k": ms(tb["i2t_test"]["mean_std"]), "full": single(tb["i2t_test"]["full"])}
examples = []
for line in (TEAM / "results/clip_baseline/results/clip_success_failure.csv").read_text(encoding="utf-8").splitlines()[1:]:
    pass  # qualitative CLIP examples are recomputed in the browser from ranks (same definition), kept file as provenance

split = rd(TEAM / "results/clip_baseline/split/split.json")["meta"]
fmeta = rd(TEAM / "results/clip_baseline/features_meta.json")
meta = {
    "title": "Truy vấn đa phương thức có giải thích bằng đồ thị",
    "dataset": {"name": "Visual Genome ∩ MS-COCO", "pool": split["pool_sizes"], "split": {"train": split["n_train"] or 15000,
                "val": split["pool_sizes"]["val"], "test": split["pool_sizes"]["test"]},
                "images_encoded": fmeta["n_images_ok"], "captions": fmeta["n_captions"], "clip": f"{fmeta['model']} / {fmeta['pretrained']}",
                "test_queries": len(q_text), "test_images": len(test_imgs), "vocab": len(vocab)},
    "protocol": {"gallery_size": split["gallery_size"], "top_k": split["top_k"], "seeds": split["seeds"], "seed_data": split["seed_data"]},
    "checks": checks,
    "sources": {"huy": "results/*/…kaggle-3seed.json, results/ranks/*.json, results/JOURNAL.md",
                "khanh": "results/adamf/protocol_coco_full.json, protocol_coco_adapter.json, vs_huy_paired.json, adamf_mat_repro.md, level2_vg_own_experiment.md, queries/eval_queries.json",
                "thang": "results/clip_baseline/results/clip_baseline.json, split/split.json"},
}
results = {"paired": paired, "level2": lvl2, "repro": repro, "i2t": i2t, "kqueries": kq}

def dump(name, obj):
    p = OUT / name
    p.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"  {p} {p.stat().st_size / 1e3:.0f} KB")

dump("meta.json", meta)
dump("models.json", MODELS)
dump("results.json", results)
dump("queries.json", {"text": q_text, "img": q_img, "group": q_group, "concepts": q_conc, "groups": GROUPS})
dump("images.json", img_out)
dump("vocab.json", vocab)
dump("ranks.json", ranks_out)
print("checks:", len(checks), "passed")
