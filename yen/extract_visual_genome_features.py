"""Extract contract-compatible, frozen OpenCLIP features for a prepared VG bundle."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image

if __package__:
    from .prepare_visual_genome import read_json, write_json
else:
    from prepare_visual_genome import read_json, write_json


def normalize_label(name: str, synonyms: dict[str, str]) -> str:
    normalized = re.sub(r"\s+", " ", name.strip().lower())
    return synonyms.get(normalized, normalized)


def find_image(image_id: int, image_dirs: list[Path]) -> Path | None:
    for image_dir in image_dirs:
        path = image_dir / f"{image_id}.jpg"
        if path.is_file():
            return path
    return None


def extract(
    data: Path,
    image_dirs: list[Path],
    model_name: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    prompt: str = "a photo of a {}",
    batch_size: int = 128,
) -> dict[str, int]:
    import open_clip

    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    captions = read_json(data / "captions.json")
    splits = read_json(data / "splits.json")
    objects = read_json(data / "vg" / "objects.json")
    synonyms_path = data / "synonyms.json"
    synonyms = read_json(synonyms_path) if synonyms_path.exists() else {}
    synonyms = {str(key).lower(): str(value).lower() for key, value in synonyms.items()}

    image_ids = sorted({int(record["image_id"]) for record in objects})
    missing = [image_id for image_id in image_ids if find_image(image_id, image_dirs) is None]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} prepared images were not found in --images-dir; "
            f"examples: {missing[:10]}"
        )

    train_ids = {int(image_id) for image_id in splits["train"]}
    labels = sorted({
        normalize_label(name, synonyms)
        for record in objects
        if int(record["image_id"]) in train_ids
        for obj in record.get("objects", [])
        for name in (obj.get("names") or [])[:1]
        if str(name).strip()
    })
    if not labels:
        raise ValueError("No object labels found in the training split")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    feature_dir = data / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    image_features = []
    with torch.inference_mode():
        for start in range(0, len(image_ids), batch_size):
            batch_ids = image_ids[start:start + batch_size]
            tensors = []
            for image_id in batch_ids:
                path = find_image(image_id, image_dirs)
                if path is None:
                    raise FileNotFoundError(f"Image {image_id}.jpg disappeared during feature extraction")
                with Image.open(path) as image:
                    tensors.append(preprocess(image.convert("RGB")))
            encoded = model.encode_image(torch.stack(tensors).to(device)).float()
            image_features.append(functional.normalize(encoded, dim=-1).cpu().numpy())
    np.save(feature_dir / "image_emb.npy", np.concatenate(image_features).astype(np.float32))
    write_json(feature_dir / "image_ids.json", image_ids)

    caption_features = []
    texts = [str(caption["text"]) for caption in captions]
    if not texts:
        raise ValueError("No captions found in prepared bundle")
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            tokens = tokenizer(texts[start:start + batch_size]).to(device)
            encoded = model.encode_text(tokens).float()
            caption_features.append(functional.normalize(encoded, dim=-1).cpu().numpy())
    np.save(feature_dir / "caption_emb.npy", np.concatenate(caption_features).astype(np.float32))
    write_json(feature_dir / "caption_ids.json", [caption["caption_id"] for caption in captions])

    label_features = []
    prompts = [prompt.format(label) for label in labels]
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            tokens = tokenizer(prompts[start:start + batch_size]).to(device)
            encoded = model.encode_text(tokens).float()
            label_features.append(functional.normalize(encoded, dim=-1).cpu().numpy())
    np.save(feature_dir / "label_emb.npy", np.concatenate(label_features).astype(np.float32))
    write_json(feature_dir / "labels.json", labels)
    write_json(feature_dir / "clip_model.json", {
        "model": model_name,
        "pretrained": pretrained,
        "prompt": prompt,
    })

    return {
        "images": len(image_ids),
        "captions": len(captions),
        "train_labels": len(labels),
        "dimension": int(image_features[0].shape[1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "data" / "visual_genome_mmxai")
    parser.add_argument("--images-dir", type=Path, nargs="+", required=True,
                        help="directories containing <Visual Genome image_id>.jpg")
    parser.add_argument("--model", default="ViT-B-32")
    parser.add_argument("--pretrained", default="laion2b_s34b_b79k")
    parser.add_argument("--prompt", default="a photo of a {}")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    result = extract(args.data, args.images_dir, args.model, args.pretrained, args.prompt, args.batch_size)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
