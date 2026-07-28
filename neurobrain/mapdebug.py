"""
mapdebug.py
===========

**Why did the map collapse?** A diagnostic bench for self-organizing layers.

v0.25 left an unsolved failure: the same development that grows oriented edge
detectors in vision produces a collapsed map on cochleagram patches -- 62-82
near-duplicate filter pairs, mean pairwise similarity 0.87, and accuracy stuck
at 50% against 70.8% for the designed bank. Three parameter settings were tried
and none of them helped, which is the signature of a problem that is not in the
parameters.

Guessing further would be cheap and useless. This module measures the five
things that can actually cause a Kohonen map to collapse, so the cause can be
*read off* rather than argued about:

1. **Input diversity.** If the patches themselves lie near a single direction,
   no learning rule can produce diverse filters -- there is nothing to be
   diverse about. Measured as the **participation ratio** of the patch
   covariance: the effective number of dimensions the data actually occupies.
   A map with 37 units needs input with far more than 37 effective dimensions
   to fill them, and if the input has 3, the collapse is the data's fault.

2. **Winner concentration.** If a handful of units win almost every patch, the
   rest never move from their initial noise and the map is dead on arrival.
   Measured as the entropy of the winner distribution and the dead-unit count.

3. **The neighbourhood.** A Kohonen neighbourhood that is still wide at the end
   of development glues neighbouring units together by construction. Measured
   by comparing the final map against the same development with the
   neighbourhood switched off.

4. **Dynamic range.** If normalization has flattened every patch to nearly the
   same vector length and shape, the competition has almost nothing to work
   with. Measured as the spread of patch norms and of their pairwise cosines.

5. **Quantization error.** How well the finished map explains its own input.
   A collapsed map has high residual error; a map that is fine but whose input
   was trivial has low error *and* low input diversity, which distinguishes the
   two cases.

Each number is reported with the vision case beside it, because the whole point
is that one modality works and the other does not, on the same code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .widev1 import WideV1, _unit


# ---------------------------------------------------------------------------
# The five measurements
# ---------------------------------------------------------------------------
def participation_ratio(X: np.ndarray) -> float:
    """Effective number of dimensions the data occupies.

    ``PR = (sum lambda)^2 / sum lambda^2`` over the covariance eigenvalues. For
    data spread evenly over d dimensions this is d; for data lying along one
    direction it is 1. This is the ceiling on how many genuinely different
    filters any unsupervised rule could possibly learn."""
    X = np.asarray(X, np.float32)
    if len(X) < 2:
        return 0.0
    Xc = X - X.mean(0, keepdims=True)
    # eigenvalues of the covariance, via the (cheaper) Gram matrix when wide
    if X.shape[1] > X.shape[0]:
        lam = np.linalg.eigvalsh(Xc @ Xc.T / max(len(X) - 1, 1))
    else:
        lam = np.linalg.eigvalsh(Xc.T @ Xc / max(len(X) - 1, 1))
    lam = np.maximum(lam, 0.0)
    s1, s2 = lam.sum(), (lam ** 2).sum()
    return float(s1 * s1 / s2) if s2 > 1e-20 else 0.0


def patch_statistics(patches: Sequence[np.ndarray]) -> Dict[str, float]:
    """What the input actually looks like, before any map is fitted."""
    X = np.asarray(patches, np.float32)
    X = X.reshape(len(X), -1)
    norms = np.linalg.norm(X, axis=1)
    keep = norms > 1e-6
    Xn = X[keep] / norms[keep][:, None]
    n = min(len(Xn), 400)
    S = Xn[:n] @ Xn[:n].T
    iu = np.triu_indices(n, 1)
    return {
        "n_patches": float(len(X)),
        "dimension": float(X.shape[1]),
        "participation_ratio": participation_ratio(Xn[:2000]),
        "mean_pairwise_cosine": float(S[iu].mean()) if n > 1 else 0.0,
        "sd_pairwise_cosine": float(S[iu].std()) if n > 1 else 0.0,
        "norm_cv": float(norms.std() / (norms.mean() + 1e-9)),
        "empty_fraction": float((~keep).mean()),
    }


def winner_statistics(layer: WideV1, patches: Sequence[np.ndarray],
                      position: int = 0) -> Dict[str, float]:
    """Which units win, and how often. A dead map shows up here immediately."""
    nf = layer.n_cells // layer.n_pos
    W = layer.Wt[np.arange(nf) * layer.n_pos + position]
    X = np.asarray(patches, np.float32).reshape(len(patches), -1)
    n = np.linalg.norm(X, axis=1)
    X = X[n > 1e-6] / n[n > 1e-6][:, None]
    if not len(X):
        return {"winner_entropy": 0.0, "dead_fraction": 1.0,
                "top1_share": 1.0, "quantization_error": 1.0}
    win = (X @ W.T).argmax(1)
    c = np.bincount(win, minlength=nf).astype(np.float64)
    p = c / c.sum()
    nz = p[p > 0]
    best = (X @ W.T).max(1)
    return {
        "winner_entropy": float(-(nz * np.log(nz)).sum() / np.log(max(nf, 2))),
        "dead_fraction": float((c == 0).mean()),
        "top1_share": float(p.max()),
        "quantization_error": float(1.0 - best.mean()),
    }


def map_diversity(layer: WideV1, position: int = 0,
                  thresh: float = 0.98) -> Dict[str, float]:
    """How distinct the finished filters are, at one cortical location."""
    nf = layer.n_cells // layer.n_pos
    F = layer.Wt[np.arange(nf) * layer.n_pos + position]
    F = F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-6)
    S = F @ F.T
    np.fill_diagonal(S, 0.0)
    iu = np.triu_indices(nf, 1)
    return {
        "n_filters": float(nf),
        "duplicate_pairs": float((S[iu] > thresh).sum()),
        "mean_similarity": float(S[iu].mean()),
        "effective_filters": participation_ratio(F),
    }


# ---------------------------------------------------------------------------
# The full diagnosis
# ---------------------------------------------------------------------------
@dataclass
class MapDiagnosis:
    """Everything measured about one modality's development."""

    name: str
    input: Dict[str, float] = field(default_factory=dict)
    winners: Dict[str, float] = field(default_factory=dict)
    diversity: Dict[str, float] = field(default_factory=dict)
    no_neighbourhood: Dict[str, float] = field(default_factory=dict)
    verdict: str = ""

    def lines(self) -> List[str]:
        i, w, d = self.input, self.winners, self.diversity
        return [
            f"  {self.name}",
            f"    input : {i['dimension']:.0f} dims, effective "
            f"{i['participation_ratio']:.1f}   cosine "
            f"{i['mean_pairwise_cosine']:.2f} +/- {i['sd_pairwise_cosine']:.2f}"
            f"   norm CV {i['norm_cv']:.2f}",
            f"    map   : {d['n_filters']:.0f} filters, effective "
            f"{d['effective_filters']:.1f}   duplicates "
            f"{d['duplicate_pairs']:.0f}   mean sim {d['mean_similarity']:.2f}",
            f"    winners: entropy {w['winner_entropy']:.2f}, dead "
            f"{w['dead_fraction']:.0%}, top-1 takes {w['top1_share']:.0%}, "
            f"residual {w['quantization_error']:.3f}",
            f"    without the Kohonen neighbourhood: duplicates "
            f"{self.no_neighbourhood.get('duplicate_pairs', 0):.0f}, "
            f"effective filters "
            f"{self.no_neighbourhood.get('effective_filters', 0):.1f}",
        ]


def diagnose(patches: Sequence[np.ndarray], shape: Tuple[int, int],
             name: str = "layer", n_cells: int = 1024, rf: int = 10,
             stride: int = 3, epochs: int = 3, seed: int = 0) -> MapDiagnosis:
    """Develop a map on these patches and measure why it came out as it did."""
    from .selforganize import randomise_filters, develop_v1

    d = MapDiagnosis(name)
    d.input = patch_statistics(patches)

    layer = randomise_filters(WideV1(image_shape=shape, n_cells=n_cells, rf=rf,
                                     stride=stride, seed=seed), seed=seed)
    develop_v1(layer, patches, epochs=epochs, seed=seed)
    d.diversity = map_diversity(layer)
    d.winners = winner_statistics(layer, _sample_patches(layer, patches))

    # control: the same development with the neighbourhood collapsed to a point
    flat = randomise_filters(WideV1(image_shape=shape, n_cells=n_cells, rf=rf,
                                    stride=stride, seed=seed), seed=seed)
    develop_v1(flat, patches, epochs=epochs, sigma0=0.4, sigma1=0.3, seed=seed)
    d.no_neighbourhood = map_diversity(flat)

    d.verdict = _verdict(d)
    return d


def _sample_patches(layer: WideV1, images: Sequence[np.ndarray],
                    n: int = 600) -> np.ndarray:
    out = []
    for im in images[:n]:
        P = layer.patches(im)
        e = np.linalg.norm(P, axis=1)
        live = np.flatnonzero(e > 0.15 * (e.max() + 1e-9))
        if len(live):
            out.append(P[live[0]])
    return np.asarray(out, np.float32) if out else np.zeros((1, layer.rf ** 2),
                                                            np.float32)


def _verdict(d: MapDiagnosis) -> str:
    """Read the causes off the numbers instead of arguing about them.

    Every cause that fires is reported, not just the first. The measurement
    that prompted this: on hearing, the input-limit test and the neighbourhood
    test BOTH fire, and reporting only the first would have hidden the one that
    turned out to be fixable."""
    nf = d.diversity["n_filters"]
    pr = d.input["participation_ratio"]
    eff = d.diversity["effective_filters"]
    found: List[str] = []
    if d.no_neighbourhood.get("effective_filters", 0) > 1.5 * eff:
        found.append(
            f"NEIGHBOURHOOD-LIMITED: with the Kohonen neighbourhood switched "
            f"off the map reaches {d.no_neighbourhood['effective_filters']:.1f} "
            f"effective filters against {eff:.1f} with it -- the smoothing is "
            f"gluing the units together.")
    if pr < 0.5 * nf:
        found.append(
            f"INPUT-LIMITED: the patches occupy only {pr:.1f} effective "
            f"dimensions but the map has {nf:.0f} units.")
    if d.winners["dead_fraction"] > 0.4:
        found.append(
            f"COMPETITION-LIMITED: {d.winners['dead_fraction']:.0%} of units "
            f"never win, so most of the map never moves.")
    if d.input["sd_pairwise_cosine"] < 0.10:
        found.append(
            f"CONTRAST-LIMITED: all patches sit at cosine "
            f"{d.input['mean_pairwise_cosine']:.2f} +/- "
            f"{d.input['sd_pairwise_cosine']:.2f} of each other.")
    return " | ".join(found) or "HEALTHY: no single cause dominates."


def diagnose_vision_vs_hearing(n_images: int = 300, n_sounds: int = 24,
                               verbose: bool = False
                               ) -> Dict[str, MapDiagnosis]:
    """The comparison that matters: one modality works, the other does not.

    Same development code, same measurements, side by side. Whatever differs
    between the two columns is the cause."""
    from .realworld import load_mnist
    from .audio import sound_dataset, Cochleagram
    from .streams import AuditoryBelt

    trx, _, _, _ = load_mnist(n_train=n_images, n_test=5)
    vis = diagnose(list(trx), (28, 28), "VISION (works)", n_cells=1024,
                   rf=10, stride=3)

    coch = Cochleagram()
    sigs, _, _ = sound_dataset(n_per_class=max(1, n_sounds // 8), seed=5)
    belt = AuditoryBelt(n_freq=coch.n_freq, seed=0)
    slices: List[np.ndarray] = []
    for s in sigs:
        slices.extend(belt.slices(belt.adapt(coch.forward(s)[0])))
    aud = diagnose(slices, (coch.n_freq, belt.slice_w), "HEARING (collapses)",
                   n_cells=1024, rf=10, stride=3)

    if verbose:
        for d in (vis, aud):
            for line in d.lines():
                print(line)
            print(f"    VERDICT: {d.verdict}\n")
    return {"vision": vis, "hearing": aud}
