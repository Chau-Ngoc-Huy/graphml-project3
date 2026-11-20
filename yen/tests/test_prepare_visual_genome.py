import json
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import torch
from PIL import Image

from yen.extract_visual_genome_features import extract
from yen.prepare_visual_genome import make_splits, prepare, repair_scene_graph


def test_repairs_missing_named_relationship_endpoint_for_1592647():
    record = {
        "image_id": 1592647,
        "objects": [{"object_id": 1, "names": ["person"]}],
        "relationships": [
            {
                "relationship_id": 4,
                "subject_id": 1,
                "object_id": 2,
                "predicate": "holding",
                "subject": {"object_id": 1, "names": ["person"]},
                "object": {"object_id": 2, "names": ["umbrella"]},
            },
            {
                "relationship_id": 5,
                "subject_id": 1,
                "object_id": 3,
                "predicate": "near",
            },
        ],
    }

    repaired, unresolved = repair_scene_graph(record)

    assert [obj["object_id"] for obj in repaired["objects"]["objects"]] == [1, 2]
    assert repaired["objects"]["objects"][1]["names"] == ["umbrella"]
    assert len(repaired["relationships"]["relationships"]) == 1
    assert repaired["relationships"]["relationships"][0]["object"]["names"] == ["umbrella"]
    assert unresolved[0]["object_id"] == 3
    assert unresolved[0]["missing_names"] == ["object"]


def test_fixed_split_maps_coco_ids_to_vg_ids_and_keeps_sets_disjoint(tmp_path):
    fixed = tmp_path / "split.json"
    fixed.write_text(json.dumps({
        "train": [10],
        "val": [20],
        "test": [30],
        "meta": {"seed_data": 7},
    }), encoding="utf-8")

    split, _ = make_splits(
        {10: [1001, 1002], 20: [2001], 30: [3001]},
        {10, 20, 30},
        fixed,
        seed=0,
        ratios=(0.8, 0.1, 0.1),
    )

    assert split["seed"] == 7
    assert split["train"] == [1001, 1002]
    assert split["val"] == [2001]
    assert split["test"] == [3001]
    assert not (set(split["train"]) & set(split["val"]) |
                set(split["train"]) & set(split["test"]) |
                set(split["val"]) & set(split["test"]))


def test_prepare_writes_mmxai_bundle_and_reports_repair(tmp_path):
    scene_graphs = tmp_path / "scene_graphs.json"
    image_data = tmp_path / "image_data.json"
    coco = tmp_path / "dataset_coco.json"
    fixed = tmp_path / "split.json"
    output = tmp_path / "bundle"

    scene_graphs.write_text(json.dumps([{
        "image_id": 1592647,
        "objects": [{"object_id": 1, "names": ["person"]}],
        "relationships": [{
            "relationship_id": 1,
            "subject_id": 1,
            "object_id": 2,
            "predicate": "holding",
            "subject": {"object_id": 1, "names": ["person"]},
            "object": {"object_id": 2, "names": ["umbrella"]},
        }],
    }]), encoding="utf-8")
    image_data.write_text(json.dumps([{"image_id": 1592647, "coco_id": 42}]), encoding="utf-8")
    coco.write_text(json.dumps({"images": [{
        "cocoid": 42,
        "sentences": [{"sentid": 3, "raw": "A person holds an umbrella."}],
    }]}), encoding="utf-8")
    fixed.write_text(json.dumps({"train": [42], "val": [], "test": []}), encoding="utf-8")

    result = prepare(scene_graphs, image_data, coco, output, fixed)

    assert result["images"] == 1
    assert json.loads((output / "splits.json").read_text(encoding="utf-8"))["train"] == [1592647]
    assert json.loads((output / "captions.json").read_text(encoding="utf-8"))[0]["image_id"] == 1592647
    report = json.loads((output / "preprocess_report.json").read_text(encoding="utf-8"))
    assert report["recovered_object_ids"] == {"1592647": [2]}


def test_feature_extractor_writes_aligned_normalized_features_and_train_vocab(tmp_path, monkeypatch):
    data = tmp_path / "bundle"
    image_dir = tmp_path / "images"
    (data / "vg").mkdir(parents=True)
    image_dir.mkdir()
    (data / "captions.json").write_text(json.dumps([
        {"caption_id": "c1", "image_id": 1, "text": "train caption"},
        {"caption_id": "c2", "image_id": 2, "text": "test caption"},
    ]), encoding="utf-8")
    (data / "splits.json").write_text(json.dumps({"train": [1], "val": [], "test": [2]}), encoding="utf-8")
    (data / "vg" / "objects.json").write_text(json.dumps([
        {"image_id": 1, "objects": [{"names": ["Train label"]}]},
        {"image_id": 2, "objects": [{"names": ["Test-only label"]}]},
    ]), encoding="utf-8")
    Image.new("RGB", (4, 4), (255, 0, 0)).save(image_dir / "1.jpg")
    Image.new("RGB", (4, 4), (0, 255, 0)).save(image_dir / "2.jpg")

    class FakeModel:
        def eval(self):
            return self

        def encode_image(self, batch):
            return batch.mean(dim=(2, 3))

        def encode_text(self, tokens):
            return tokens

    def preprocess(image):
        pixels = np.asarray(image, dtype=np.float32) / 255
        return torch.from_numpy(pixels).permute(2, 0, 1)

    def tokenizer(texts):
        return torch.tensor([[len(text), 1.0, 2.0] for text in texts], dtype=torch.float32)

    fake_open_clip = types.SimpleNamespace(
        create_model_and_transforms=Mock(return_value=(FakeModel(), None, preprocess)),
        get_tokenizer=Mock(return_value=tokenizer),
    )
    monkeypatch.setitem(sys.modules, "open_clip", fake_open_clip)

    summary = extract(data, [image_dir], batch_size=1)

    features = data / "features"
    image_emb = np.load(features / "image_emb.npy")
    caption_emb = np.load(features / "caption_emb.npy")
    label_emb = np.load(features / "label_emb.npy")
    assert summary["images"] == 2
    assert json.loads((features / "image_ids.json").read_text(encoding="utf-8")) == [1, 2]
    assert json.loads((features / "caption_ids.json").read_text(encoding="utf-8")) == ["c1", "c2"]
    assert json.loads((features / "labels.json").read_text(encoding="utf-8")) == ["train label"]
    assert image_emb.shape == (2, 3)
    assert caption_emb.shape == (2, 3)
    assert label_emb.shape == (1, 3)
    np.testing.assert_allclose(np.linalg.norm(image_emb, axis=1), 1, atol=1e-6)

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "huy" / "src"))
    from mmxai.data.contract import load

    bundle = load(data)
    assert bundle.image_ids == [1, 2]
    assert bundle.splits == {"train": [1], "val": [], "test": [2]}
    assert bundle.captions[1]["caption_id"] == "c2"
