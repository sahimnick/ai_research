"""`locate_template` must find a patch it was given, and reject a mismatch.

§9.21 measured the averaged prototype `_attention_heatmap` uses scoring 0.000
finding a tag in the tag's own frame -- worse than a deliberately wrong tag at
0.188. The spatial version scores 1.000 there. These are the two properties that
result rests on, asserted rather than assumed.
"""
import sys

import numpy as np

sys.path.insert(0, ".")
from neurobrain.vision.ventral import locate_template                # noqa: E402


def test_finds_its_own_patch():
    rng = np.random.default_rng(0)
    m = rng.random((6, 40, 40)).astype(np.float32)
    for (y, x) in ((12, 25), (0, 0), (32, 32)):
        t = m[:, y:y + 8, x:x + 8].copy()
        (r, c), score = locate_template(m, t)
        assert (r, c) == (y, x), f"expected ({y}, {x}), got ({r}, {c})"
        assert score.max() > 0.99, f"self-match should saturate, got {score.max()}"


def test_rejects_wrong_shape():
    m = np.zeros((4, 20, 20), np.float32)
    for bad in (np.zeros((3, 5, 5), np.float32),     # wrong channel count
                np.zeros((4, 25, 5), np.float32)):   # bigger than the map
        try:
            locate_template(m, bad)
        except ValueError:
            continue
        raise AssertionError(f"should have rejected template {bad.shape}")


def test_flat_map_does_not_crash():
    """A constant map has zero variance everywhere -- the denominator guard."""
    m = np.full((3, 12, 12), 0.7, np.float32)
    (r, c), score = locate_template(m, m[:, 2:6, 2:6].copy())
    assert np.isfinite(score).all(), "flat map produced a non-finite score"
    assert 0 <= r < 9 and 0 <= c < 9


if __name__ == "__main__":
    test_finds_its_own_patch()
    test_rejects_wrong_shape()
    test_flat_map_does_not_crash()
    print("all locate_template tests passed")
