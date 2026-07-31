"""The space imagined codes live in, and what a singleton concept can produce.

Two silent failures, both found by `composition.py` and both cheap to lock down.

**The space.** `Wv` rows, `imagine_vision`, `imagine_composite` and `prep_v` all
live in the mean-subtracted, variance-normalised space. The stored codes a
benchmark compares them against do not, unless it converts them. The same
photograph sits at cosine ~0.9 to itself across the two representations, which
is small enough to look like a plausible distance and large enough to change
every conclusion drawn from one. `imagination_shape.py` reported a fidelity of
0.886 that was really 0.979.

**The singleton.** Vigilance recruits by copying a pair into an uncommitted cell
verbatim, and `_grow_subspace` is only reached on the non-recruiting path, so a
cell that wins exactly once holds a stored photograph with no subspace at all.
For that cell `imagine_vision` returns the same vector at every temperature.
That is not a bug to patch -- one observation carries no variation -- but it is
a property any future change to the sampling has to keep in view, because a
third of test sounds wake such a cell and no temperature moves them.
"""
import numpy as np

from neurobrain.cognition.multimodal import AssociationArea


def _area(n_vis=64, n_aud=16, n_concept=8, seed=0):
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((12, n_vis)).astype(np.float32)
    A = rng.standard_normal((12, n_aud)).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    a = AssociationArea(n_vis=n_vis, n_aud=n_aud, n_concept=n_concept, seed=seed)
    a.set_stats(V, A)
    return a, V, A


def test_weights_live_in_prep_space_not_raw():
    """A recruited cell holds the *prepped* code, not the raw one."""
    a, V, A = _area()
    win = a.bind(V[0], A[0])
    assert np.allclose(a.Wv[win], a.prep_v(V[0]), atol=1e-5)
    raw = float(a.Wv[win] @ (V[0] / np.linalg.norm(V[0])))
    assert raw < 0.999, (
        f"raw and prepped space came out identical (cos {raw}); this test "
        "cannot detect the mix-up it exists to detect")


def test_singleton_concept_is_temperature_invariant():
    """A cell that won once has no subspace, so no temperature moves it."""
    a, V, A = _area()
    win = a.bind(V[0], A[0])
    assert a.wins[win] == 1
    assert a.mode_var[win].max() == 0.0
    base = a.imagine_vision(win, temperature=0.0)
    for T in (1.0, 8.0, 64.0):
        got = a.imagine_vision(win, temperature=T,
                               rng=np.random.default_rng(1))
        assert np.allclose(base, got, atol=1e-6), (
            f"a singleton moved at T={T}; if the mechanism now gives recruited "
            "cells a subspace, this test should be updated deliberately")


def test_composition_moves_a_singleton_off_its_memory():
    """What a single concept cannot do, several together can."""
    a, V, A = _area()
    cells = [a.bind(V[i], A[i]) for i in range(4)]
    assert len(set(cells)) > 1, "every pair landed on one cell; nothing to mix"
    solo = a.imagine_vision(cells[0], temperature=0.0)
    mixed = a.imagine_composite(cells[:3], temperature=0.0)
    assert float(solo @ mixed) < 0.99, (
        "composing several concepts returned the first one unchanged")
    stored = np.stack([a.prep_v(v) for v in V])
    assert float((stored @ mixed).max()) < 0.999, (
        "the composite is bit-identical to a stored code")


def test_anchor_weight_one_is_pure_perception():
    """The degeneracy `composition.py` excludes from its ranking, stated here.

    At ``anchor_weight=1`` the result is the anchor, so any score measured
    against real held-out data is satisfied by copying rather than imagining."""
    a, V, A = _area()
    win = a.bind(V[0], A[0])
    got = a.imagine_composite([win], temperature=0.0, anchor=V[5],
                              anchor_weight=1.0)
    assert np.allclose(got, a.prep_v(V[5]), atol=1e-5)


def test_composite_of_nothing_is_not_a_crash():
    a, _, _ = _area()
    out = a.imagine_composite([], temperature=1.0)
    assert out.shape == (a.Wv.shape[1],)
    assert not np.any(np.isnan(out))
