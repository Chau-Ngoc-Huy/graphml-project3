"""Frozen CLIP features (open_clip) + a fake encoder for CPU smoke tests.

Real:  Encoder("clip", img_dirs=[...])   Fake: Encoder("fake", images=images)
Both give .images(ids) -> (kept_ids, [n,d] L2-normed) and .texts(list) -> [m,d] L2-normed.
"""
import hashlib
import os

import numpy as np


def _unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


class Encoder:
    def __init__(self, mode="clip", img_dirs=(), images=None, model="ViT-B-32", pretrained="laion2b_s34b_b79k",
                 device=None, batch=256, workers=4):
        self.mode, self.img_dirs, self.images_meta = mode, list(img_dirs), images
        if mode == "clip":
            import open_clip
            import torch
            self.torch = torch
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model, _, self.pre = open_clip.create_model_and_transforms(model, pretrained=pretrained)
            self.model = self.model.to(self.device).eval()
            if self.device == "cuda":
                self.model = self.model.half()
            self.tok = open_clip.get_tokenizer(model)
            self.batch, self.workers = batch, workers

    # ---- fake: deterministic word vectors, images = mean of their object words + noise
    def _w(self, word, d=64):
        h = int(hashlib.md5(word.encode()).hexdigest()[:8], 16)
        return np.random.RandomState(h).randn(d)

    def _fake_text(self, t):
        return np.mean([self._w(w) for w in t.split()], axis=0)

    def texts(self, items):
        if self.mode == "fake":
            return _unit(np.stack([self._fake_text(t) for t in items]).astype("float32"))
        out = []
        with self.torch.no_grad():
            for i in range(0, len(items), 512):
                tk = self.tok(items[i:i + 512]).to(self.device)
                out.append(self.model.encode_text(tk).float().cpu().numpy())
        return _unit(np.concatenate(out))

    def _find(self, iid):
        for d in self.img_dirs:
            p = os.path.join(d, f"{iid}.jpg")
            if os.path.exists(p):
                return p
        return None

    def images(self, ids):
        if self.mode == "fake":
            rs = np.random.RandomState(0)
            f = [np.mean([self._w(w) for w in self.images_meta[i]["objs"]] or [np.zeros(64)], axis=0)
                 + 0.5 * rs.randn(64) for i in ids]
            return list(ids), _unit(np.stack(f).astype("float32"))
        from PIL import Image
        torch = self.torch
        paths = [(i, self._find(i)) for i in ids]
        paths = [(i, p) for i, p in paths if p]
        print(f"images found: {len(paths)}/{len(ids)}")

        class DS(torch.utils.data.Dataset):
            def __len__(s):
                return len(paths)

            def __getitem__(s, k):
                return self.pre(Image.open(paths[k][1]).convert("RGB"))

        dl = torch.utils.data.DataLoader(DS(), batch_size=self.batch, num_workers=self.workers)
        out = []
        with torch.no_grad():
            for x in dl:
                x = x.to(self.device)
                x = x.half() if self.device == "cuda" else x
                out.append(self.model.encode_image(x).float().cpu().numpy())
        return [i for i, _ in paths], _unit(np.concatenate(out))
