"""Regression lock: read-outs must not assume labels are ``0..n_class-1``.

`_nearest_prototype` built its prototypes with ``for c in range(n_class)``, so
it silently assumed the label set was ``0..n_class-1``. Handing it ESC-50's
*nature* classes -- whose labels are 10-19 -- with ``n_class=10`` built ten
all-zero prototypes, made every dot product zero, and returned index 0 for
every clip. The score was **exactly 0.000**.

That is the worst kind of bug: it does not raise, it returns a number, and the
number is the shape of a real scientific finding ("the auditory front end
collapses on environmental sound"). It was caught only because the *linear
probe* on the same features reached 0.55, and because the one subset that
worked -- animals, labels 0-9 -- was exactly the subset whose labels happened
to fall inside the assumed range.

These tests pin the behaviour on both sides: arbitrary label sets must decode,
and the legacy contract (labels already ``0..n_class-1``, including a class
missing from the training split) must be unchanged.

Run with pytest, or directly:  ``python3 tests/test_readout_labels.py``
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neurobrain.audition.auditorycortex import linear_probe
from neurobrain.vision.widev1 import _nearest_prototype


def _separable(labels, dim=32, noise=0.05, seed=0):
    """Cleanly separable codes for an arbitrary label set.

    Any correct read-out must score ~1.0 here, so a low score is the read-out's
    fault and never the data's."""
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    rng = np.random.default_rng(seed)
    P = rng.standard_normal((len(uniq), dim)).astype(np.float32)
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    row = {int(v): i for i, v in enumerate(uniq)}
    X = np.stack([P[row[int(v)]] for v in labels])
    X = X + noise * rng.standard_normal(X.shape).astype(np.float32)
    return (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)


# `np.tile`, not `np.repeat`: every class must appear in both halves, or the
# split is class-disjoint and *every* read-out scores 0 for a legitimate reason.
_LAB = np.tile(np.arange(10), 8)
_TR, _TE = slice(0, 40), slice(40, None)


def test_esc50_nature_labels_decode():
    """Labels 10-19 with ``n_class=10`` -- the case that returned 0.000."""
    y = _LAB + 10
    X = _separable(y)
    acc = float(np.mean(_nearest_prototype(X[_TR], y[_TR], X[_TE], 10) == y[_TE]))
    assert acc > 0.9, f"nature labels (10-19) decoded at {acc:.3f}"


def test_esc50_urban_labels_decode():
    """Labels 40-49 -- four times outside the assumed range, same failure."""
    y = _LAB + 40
    X = _separable(y)
    acc = float(np.mean(_nearest_prototype(X[_TR], y[_TR], X[_TE], 10) == y[_TE]))
    assert acc > 0.9, f"urban labels (40-49) decoded at {acc:.3f}"


def test_non_contiguous_labels_decode():
    """Gaps, not just an offset: nothing may depend on the labels being dense."""
    y = _LAB * 11                      # 0, 11, 22, ... 99
    X = _separable(y)
    acc = float(np.mean(_nearest_prototype(X[_TR], y[_TR], X[_TE]) == y[_TE]))
    assert acc > 0.9, f"sparse labels decoded at {acc:.3f}"


def test_predictions_are_label_values():
    """The return value is a LABEL, not a row index into the prototype stack."""
    y = _LAB + 40
    X = _separable(y)
    pred = _nearest_prototype(X[_TR], y[_TR], X[_TE], 10)
    assert set(np.unique(pred)).issubset(set(np.unique(y))), (
        f"returned {sorted(set(np.unique(pred)))[:5]}, not labels")


def test_legacy_contract_unchanged():
    """Labels already ``0..n_class-1`` must behave exactly as before."""
    X = _separable(_LAB)
    acc = float(np.mean(_nearest_prototype(X[_TR], _LAB[_TR], X[_TE], 10)
                        == _LAB[_TE]))
    assert acc > 0.9, f"legacy 0..9 decoded at {acc:.3f}"


def test_class_missing_from_training_split():
    """A class present in test but absent from train stays unpredictable.

    The old code allocated a zero prototype for it; the new code drops it from
    the class list. Either way it can never be predicted, so accuracy is capped
    near ``1 - 1/k`` and nothing else degrades."""
    X = _separable(_LAB)
    keep = _LAB[_TR] != 3
    acc = float(np.mean(_nearest_prototype(X[_TR][keep], _LAB[_TR][keep],
                                           X[_TE], 10) == _LAB[_TE]))
    assert 0.85 < acc < 0.95, f"expected ~0.9 with one class held out, got {acc:.3f}"


def test_linear_probe_already_handled_arbitrary_labels():
    """The probe used ``np.unique`` all along -- which is how the bug was found.

    Pinning it matters: the probe-versus-prototype disagreement is the signal
    that catches this class of mistake next time."""
    y = _LAB + 40
    X = _separable(y)
    acc = linear_probe(X[_TR], y[_TR], X[_TE], y[_TE])
    assert acc > 0.9, f"probe on labels 40-49 scored {acc:.3f}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
