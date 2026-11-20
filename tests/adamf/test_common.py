import numpy as np

from mmxai.adamf.common import metrics, split_ids


def test_metrics_hand_example():
    m = metrics([[3, 1, 2], [9, 8, 7]], [{1}, {5}])
    assert m["R@1"] == 0 and m["R@5"] == 0.5 and abs(m["MRR"] - 0.25) < 1e-9


def test_split_is_image_level_and_deterministic():
    a, b = split_ids(range(100), 0), split_ids(range(100), 0)
    assert a == b and len(a["train"]) == 80
    assert not set(a["train"]) & set(a["test"]) and not set(a["train"]) & set(a["val"])
