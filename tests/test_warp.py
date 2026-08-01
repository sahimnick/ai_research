"""`warp_tile` is hand-written, so its identities are asserted rather than trusted.

scipy would have supplied a resampler already known to be correct. This project
is numpy-only, so the correctness has to come from somewhere else, and it comes
from here: the transformations that must be exact are checked to be exact, and
the ones that must merely be close are checked with a stated tolerance.

A silently wrong resampler is the worst possible failure for the experiment it
feeds -- it would change the invariance being measured while every arm still
ran, every number still printed, and every gate still passed.
"""
import sys

import numpy as np

sys.path.insert(0, ".")
from neurobrain.sensing.streams import MovingObject, MovingScene, warp_tile


def _tile(n=32, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.random((n, n)).astype(np.float32) * 255.0
    a[4:9, 6:20] = 255.0          # an asymmetric mark, so rotation is visible
    return a


def test_identity_is_exact():
    a = _tile()
    out = warp_tile(a, scale=1.0, theta=0.0)
    assert np.array_equal(out, a), "scale 1, angle 0 must be the input itself"
    print("ok  identity is exact")


def test_full_turn_returns_the_input():
    a = _tile()
    out = warp_tile(a, scale=1.0, theta=2 * np.pi)
    assert np.allclose(out, a, atol=1e-3), f"max diff {np.abs(out - a).max()}"
    print("ok  a full turn is the input")


def test_four_quarter_turns_compose_back():
    a = _tile()
    out = a.copy()
    for _ in range(4):
        out = warp_tile(out, theta=np.pi / 2)
    # four exact quarter turns about the centre; resampling makes this close
    # rather than exact, and the tolerance is stated instead of assumed
    err = float(np.abs(out - a).mean() / (a.mean() + 1e-9))
    assert err < 0.05, f"four quarter turns drifted by {err:.3f} of the mean"
    print(f"ok  four quarter turns compose back (drift {err:.4f})")


def test_rotation_actually_changes_the_image():
    """The guard against a warp that silently does nothing."""
    a = _tile()
    out = warp_tile(a, theta=np.pi / 4)
    d = float(np.abs(out - a).mean() / (a.mean() + 1e-9))
    assert d > 0.05, f"a 45-degree turn moved the image by only {d:.4f}"
    print(f"ok  rotation changes the image (by {d:.3f} of the mean)")


def test_scaling_changes_the_footprint():
    a = np.zeros((32, 32), np.float32)
    a[12:20, 12:20] = 255.0                     # a centred square
    small = warp_tile(a, scale=0.5)
    big = warp_tile(a, scale=1.5)
    n_a, n_s, n_b = [(x > 127).sum() for x in (a, small, big)]
    assert n_s < n_a < n_b, f"footprints did not order: {n_s} < {n_a} < {n_b}"
    print(f"ok  scaling changes the footprint ({n_s} < {n_a} < {n_b} px)")


def test_no_nans_and_range_is_preserved():
    a = _tile()
    for sc in (0.6, 1.0, 1.6):
        for th in (0.0, 0.7, 3.0):
            out = warp_tile(a, scale=sc, theta=th)
            assert np.isfinite(out).all(), f"non-finite at scale {sc} angle {th}"
            assert out.min() >= -1e-4, f"negative value at scale {sc}"
            assert out.max() <= a.max() + 1e-3, "interpolation overshot the input"
    print("ok  finite, non-negative, no overshoot across the range used")


def test_moving_scene_defaults_are_unchanged():
    """Every earlier measurement ran with spin=zoom=0; it must stay that way."""
    imgs = np.stack([_tile(seed=k) for k in range(6)])
    lab = np.arange(6)
    a = MovingScene(imgs, lab, size=96, n_objects=3, seed=0)
    b = MovingScene(imgs, lab, size=96, n_objects=3, spin=0.0, zoom=0.0, seed=0)
    assert np.array_equal(a.render(), b.render())
    for o in a.objects:
        assert o.theta == 0.0 and o.dtheta == 0.0 and o.dscale == 0.0
        assert np.array_equal(o.view(), o.image)
    print("ok  MovingScene defaults are the old world exactly")


def test_moving_scene_pose_actually_advances():
    imgs = np.stack([_tile(seed=k) for k in range(6)])
    a = MovingScene(imgs, np.arange(6), size=96, n_objects=3,
                    spin=0.2, zoom=0.03, seed=0)
    th0 = [o.theta for o in a.objects]
    sc0 = [o.scale for o in a.objects]
    for _ in range(5):
        a.step()
    assert any(abs(o.theta - t) > 1e-6 for o, t in zip(a.objects, th0)), \
        "spin was requested but no object turned"
    assert any(abs(o.scale - s) > 1e-6 for o, s in zip(a.objects, sc0)), \
        "zoom was requested but no object changed size"
    assert all(0.6 <= o.scale <= 1.6 for o in a.objects), "scale left its bounds"
    print("ok  pose advances when spin/zoom are on, and stays in bounds")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\n{len(fns)} passed")
