"""
selforganize.py
===============

**A visual cortex that discovers its own features**, and an auditory one beside
it.

The complaint this answers is exact. Up to v0.24 the wide V1 was *given* its
features: :meth:`~neurobrain.widev1.WideV1._seed_filters` writes a bank of Gabor
kernels by hand -- 12 orientations, 2 phases, 3 spatial frequencies, plus
centre-surround blobs. Competitive Hebbian learning then *refined* them, but the
oriented structure was put there by me, not found by the brain. A V1 whose
receptive fields were typed in is not a model of V1; it is a model of my
knowledge of V1.

It also had a measured consequence. In v0.24 the width curve turned over: 4096
cells scored 90.8% and 16384 fell to 85.0%, because the hand-written bank holds
only 288 distinct filters and beyond ~288 per hypercolumn the extra cells are
near-duplicates. A designed bank has a fixed ceiling on diversity. A cortex that
learns its own does not.

How the features are found
--------------------------
Random weights, then development, using only mechanisms already in this project:

* **Competition inside the hypercolumn.** Every cell at a location sees the
  same patch, so if they all learn they all become the same filter. Only the
  best-matching cell wins the patch. This is the difference between the old
  ``learn`` (every firing cell moves) and development.

* **A neighbourhood in feature space (Kohonen).** The winner's neighbours in the
  column's index space learn a little too, with a Gaussian weight that shrinks
  over development. This is what makes the result a *map* rather than a bag:
  neighbouring cells end up with similar preferences, which is the pinwheel
  organisation real V1 has and which nothing here was told to produce.

  **It must anneal all the way to zero, and v0.25 did not.** Diagnosed with
  :mod:`mapdebug`: with the neighbourhood left at sigma1=0.45 the finished map
  had 2.3 effective filters out of 20; switching the neighbourhood off entirely
  gave 7.5. The smoothing that produces the ordered map was also what was
  gluing the units together, in BOTH modalities -- vision 1.8 -> 7.5 effective
  filters, hearing 2.1 -> 14.0. Annealing sigma to ~0 keeps the early ordering
  and lets the units separate at the end: measured 83.0% -> 84.2% accuracy with
  duplicates falling from 18 to 3.

* **Homeostasis.** A duty-cycle term biases the competition against cells that
  keep winning, so the map uses all its units instead of collapsing onto a few.
  Every failure mode in this project's history has been some version of
  collapse; this is the guard.

What is measured
----------------
Three things, none of them assumed:

    ``orientation_selectivity``  the cells are probed with drifting gratings, the
                                 way a physiologist probes a real cell, and the
                                 orientation selectivity index is computed. If
                                 oriented receptive fields did not emerge, this
                                 number says so.
    ``diversity``                near-duplicate filter pairs. This is the number
                                 that caps how wide the layer can usefully be.
    ``accuracy``                 held-out recognition against the hand-designed
                                 bank, same data, same read-out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..vision.widev1 import WideV1, _unit, grown_readout, _nearest_prototype


# ---------------------------------------------------------------------------
# Development: a random cortex becomes an oriented one
# ---------------------------------------------------------------------------
def dendritic_footprints(layer: WideV1, seed: int = 0, spread: float = 0.32,
                         min_frac: float = 0.12) -> np.ndarray:
    """A localized **dendritic arbor** for every cell, as a connection mask.

    This is a structural constraint, not a learning trick, and it is the answer
    to a measured failure. Learned filters came out active on **79%** of their
    receptive field while the hand-designed Gabors covered **43%**; filters that
    overlap that much are necessarily similar, so the whole set collapsed to
    ~5 effective directions against the designed bank's 17.5, and no change of
    learning rule fixed it (competitive learning reached 5.6, sparse predictive
    coding 3.9).

    Nothing in a Hebbian rule makes a filter compact. In cortex the compactness
    is **anatomical**: a cell's dendritic tree reaches only a limited
    neighbourhood, so it *cannot* form a synapse with a distant input however
    correlated they are. Each cell here is given a Gaussian arbor centred at a
    random point of its field; weights outside it are structurally absent, not
    merely small.

    Returns a (n_cells, rf*rf) mask in {0, 1}."""
    rng = np.random.default_rng(seed)
    rf = layer.rf
    yy, xx = np.mgrid[0:rf, 0:rf]
    n = layer.n_cells
    mask = np.zeros((n, rf * rf), np.float32)
    sigma = max(spread * rf, 1.0)
    for i in range(n):
        cy, cx = rng.uniform(0, rf - 1), rng.uniform(0, rf - 1)
        d2 = (yy - cy) ** 2 + (xx - cx) ** 2
        m = (d2 <= (1.5 * sigma) ** 2).reshape(-1).astype(np.float32)
        if m.mean() < min_frac:                    # never leave a cell blind
            k = max(int(min_frac * rf * rf), 4)
            m = np.zeros(rf * rf, np.float32)
            m[np.argsort(d2.reshape(-1))[:k]] = 1.0
        mask[i] = m
    return mask


def randomise_filters(layer: WideV1, seed: int = 0,
                      scale: float = 1.0) -> WideV1:
    """Throw away the designed bank. Start from noise, as a cortex does.

    Real receptive fields before visual experience are crude and unoriented;
    the orientation structure is refined by input (spontaneous retinal waves
    first, then the world). Starting from random non-negative weights is the
    honest starting point for a layer that claims to discover its features."""
    rng = np.random.default_rng(seed)
    W = np.abs(rng.normal(0.0, scale, layer.Wt.shape)).astype(np.float32)
    if getattr(layer, "dendrite_mask", None) is not None:
        W = W * layer.dendrite_mask
    W /= np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-6)
    layer.Wt = W
    return layer


def grow_dendrites(layer: WideV1, seed: int = 0, spread: float = 0.32
                   ) -> WideV1:
    """Attach dendritic arbors to a layer, so its filters can only be local."""
    layer.dendrite_mask = dendritic_footprints(layer, seed=seed, spread=spread)
    return layer


def develop_v1(layer: WideV1, images: Sequence[np.ndarray], epochs: int = 4,
               lr0: float = 0.35, lr1: float = 0.02,
               sigma0: Optional[float] = None, sigma1: float = 0.01,
               homeostasis: float = 2.0,
               patches_per_image: int = 24, tie: bool = False,
               seed: int = 0, verbose: bool = False) -> WideV1:
    """Grow receptive fields from experience. No Gabors are written down.

    One developmental step: take a patch, let the cells at that location
    compete for it, and let the winner and its map-neighbours move toward it.
    The learning rate and the neighbourhood width both anneal, which is what
    turns an initially global reorganisation into local refinement -- the
    coarse-to-fine course of a critical period.

    ``tie`` shares each filter across every retinotopic position, and it repairs
    something that was quietly broken. :meth:`WideV1.pooling_index` promises a
    shift-invariant code by grouping cells that "share a filter but sit at
    different locations" -- and it identifies those cells purely by index,
    ``arange(n_cells) // n_pos``. That is only a filter identity if every
    hypercolumn learns the *same* bank. Untrained, they do: filter *k* has cosine
    **0.977** to itself across positions. After development each column has drifted
    independently and the same measurement reads **0.454**, against 0.319 for
    random pairs -- so pooling by filter index was summing unrelated cells and
    the "invariant" code was not invariant. Measured consequence: a fully pooled
    code falls from 0.382 to chance by a 10px offset, which a code that discards
    position cannot do unless the grouping is wrong.

    The tell was in the other sense. :class:`AuditoryBelt` builds a
    :class:`WideV1` and never develops it, so its bank stays tied at **0.978** --
    which is why the same pooling operation genuinely works there and is a large
    part of why hearing outperforms vision in this project.

    Tying is also the standard assumption about V1 rather than a convenience: a
    given orientation-and-frequency channel is repeated across the retinotopic
    map, which is what makes the map a map. It costs the columns their
    independence, which is the point.
    """
    rng = np.random.default_rng(seed)
    n_pos, nf = layer.n_pos, layer.n_cells // layer.n_pos
    if nf < 2:
        raise ValueError("a hypercolumn needs at least 2 cells to compete.")
    head = nf * n_pos
    idx = np.arange(nf)
    # The neighbourhood has to scale with the map. A fixed sigma0 that is right
    # for a 20-cell column smears a 334-cell one into a handful of distinct
    # filters, which would rebuild the very diversity ceiling this module exists
    # to remove.
    sigma0 = float(sigma0) if sigma0 is not None else max(2.0, nf / 4.0)
    # A bigger map needs a longer critical period. At 83 filters per column the
    # default budget gave about 5 updates per filter, the map never separated,
    # and 347 of its filter pairs came out near-identical -- rebuilding the very
    # diversity ceiling this module exists to remove. The number of patches
    # sampled per image therefore scales with the size of the map.
    patches_per_image = max(int(patches_per_image), int(nf))
    duty = np.full(nf * n_pos, 1.0 / nf, np.float32)
    total = max(1, epochs * len(images))
    step = 0

    for _ in range(epochs):
        for i in rng.permutation(len(images)):
            frac = step / total
            lr = lr0 * (lr1 / lr0) ** frac
            sigma = sigma0 * (sigma1 / sigma0) ** frac
            step += 1

            P = layer.patches(images[i])
            energy = np.linalg.norm(P, axis=1)
            live = np.flatnonzero(energy > 0.15 * (energy.max() + 1e-9))
            if not len(live):
                continue
            take = live if len(live) <= patches_per_image else rng.choice(
                live, patches_per_image, replace=False)

            Pn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-6)
            # (nf, n_pos) view: cell for (filter j, position p) is j*n_pos + p
            W = layer.Wt[:head].reshape(nf, n_pos, -1)
            D = duty[:head].reshape(nf, n_pos)
            for p in take:
                drive = W[:, p, :] @ Pn[p] - homeostasis * (D[:, p] - 1.0 / nf)
                win = int(np.argmax(drive))
                # Kohonen neighbourhood in the column's feature space -- this is
                # what makes the outcome an ordered map instead of a bag of
                # unrelated filters.
                h = np.exp(-((idx - win) ** 2) / (2.0 * sigma ** 2)).astype(
                    np.float32)
                W[:, p, :] += (lr * h)[:, None] * (Pn[p][None, :] - W[:, p, :])
                D[:, p] *= (1.0 - 0.05)
                D[win, p] += 0.05
            if tie:
                # one bank, repeated: average each filter over the positions
                # that just updated it, then write it back everywhere. This is
                # the convolutional weight-sharing constraint, and without it
                # `pooling_index` is grouping cells that no longer share
                # anything.
                W[:] = W.mean(axis=1, keepdims=True)
            np.maximum(W, 0.0, out=W)
            W /= np.maximum(np.linalg.norm(W, axis=2, keepdims=True), 1e-6)
            layer.Wt[:head] = W.reshape(head, -1)
            dm = getattr(layer, "dendrite_mask", None)
            if dm is not None:
                # a synapse outside the arbor does not exist and cannot grow
                layer.Wt[:head] *= dm[:head]
                nrm = np.linalg.norm(layer.Wt[:head], axis=1, keepdims=True)
                layer.Wt[:head] /= np.maximum(nrm, 1e-6)
                W = layer.Wt[:head].reshape(nf, n_pos, -1)
            duty[:head] = D.reshape(-1)
    if verbose:
        o = orientation_selectivity(layer)
        print(f"   development done: mean OSI {o['mean_osi']:.2f}, "
              f"{o['oriented_fraction']:.0%} of cells oriented, "
              f"{filter_diversity(layer)['duplicate_pairs']} duplicate pairs")
    return layer


def develop_by_prediction(layer: WideV1, patches: Sequence[np.ndarray],
                          epochs: int = 8, lr: float = 0.05,
                          sparsity: float = 0.12, seed: int = 0,
                          verbose: bool = False) -> WideV1:
    """Grow filters by **predicting the input**, not by quantizing it.

    Why this exists
    ---------------
    :func:`develop_v1` is competitive learning: each unit moves toward the
    patches it wins, so the units end up at cluster centres. That is the right
    thing when the data has clusters and the wrong thing when it is a
    continuous manifold, because prototypes then crowd near the mean and the
    *set* of filters ends up occupying far fewer directions than the data does.

    Measured on cochleagram patches, and this is the whole diagnosis of the
    auditory gap: the input occupies 25.1 effective dimensions, the hand-written
    Gabor bank occupies 17.5, and the self-organized map reaches **5.6**.
    Enriching the input from 9.8 to 25.1 effective dimensions lifted the
    designed bank from 70.8% to 79.2% and moved the map's accuracy not at all.
    So the bottleneck was never the data and never the collapse -- it was the
    learning rule.

    What replaces it
    ----------------
    Sparse predictive coding, with exactly the local rule from
    :class:`~neurobrain.topdown.PredictiveStack`: infer a sparse set of active
    units by settling, then change each synapse by the product of its unit's
    activity and the residual error it failed to explain.

        r  <- relu(r + eta * (W @ (x - W.T @ r)))        # inference
        W  <- W + lr * outer(r, x - W.T @ r)             # local Hebbian

    A dictionary learned this way has to *span* the input to explain it, so it
    cannot collapse onto the mean the way a quantizer can. It is also the more
    biological of the two: cortex is not known to run a Kohonen algorithm, but
    predictive feedback with Hebbian error learning is the standard account of
    what cortical feedback is for.
    """
    rng = np.random.default_rng(seed)
    n_pos, nf = layer.n_pos, layer.n_cells // layer.n_pos
    head = nf * n_pos
    X = []
    for im in patches:
        P = layer.patches(im)
        e = np.linalg.norm(P, axis=1)
        live = np.flatnonzero(e > 0.15 * (e.max() + 1e-9))
        for p in live:
            X.append(P[p] / (np.linalg.norm(P[p]) + 1e-9))
    if not X:
        return layer
    X = np.asarray(X, np.float32)

    W = layer.Wt[:head].reshape(nf, n_pos, -1)[:, 0, :].copy()
    W /= np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-6)
    duty = np.full(nf, sparsity, np.float32)

    for _ in range(int(epochs)):
        eta = 0.5 / max(float(np.linalg.norm(W, 2) ** 2), 1e-6)
        for i in rng.permutation(len(X)):
            x = X[i]
            r = np.zeros(nf, np.float32)
            for _ in range(15):
                err = x - W.T @ r
                r = np.maximum(r + eta * (W @ err) - 0.02 * r
                               - 0.02 * (duty - sparsity), 0.0)
            err = x - W.T @ r
            W += lr * np.outer(r, err)
            np.maximum(W, 0.0, out=W)
            W /= np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-6)
            duty *= (1.0 - lr)
            duty += lr * (r > 1e-6)
        eta = 0.5 / max(float(np.linalg.norm(W, 2) ** 2), 1e-6)

    # every location gets the same learned dictionary, as a filter bank is
    layer.Wt[:head] = np.repeat(W[:, None, :], n_pos, axis=1).reshape(head, -1)
    if verbose:
        from ..tools.mapdebug import participation_ratio
        print(f"   dictionary: {nf} filters, effective "
              f"{participation_ratio(W):.1f}")
    return layer


# ---------------------------------------------------------------------------
# Measuring what grew -- the way a physiologist would
# ---------------------------------------------------------------------------
def _gratings(rf: int, n_orient: int = 16, n_phase: int = 4,
              lam: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    """Drifting sinusoidal gratings: the standard probe for a visual cell."""
    yy, xx = np.mgrid[0:rf, 0:rf] - (rf - 1) / 2.0
    stim, ang = [], []
    for o in range(n_orient):
        th = np.pi * o / n_orient
        xr = xx * np.cos(th) + yy * np.sin(th)
        for q in range(n_phase):
            g = np.cos(2 * np.pi * xr / lam + 2 * np.pi * q / n_phase)
            stim.append(np.maximum(g, 0.0).reshape(-1))
            ang.append(th)
    S = np.array(stim, np.float32)
    S /= np.maximum(np.linalg.norm(S, axis=1, keepdims=True), 1e-6)
    return S, np.array(ang, np.float32)


def orientation_selectivity(layer: WideV1, n_orient: int = 16,
                            n_phase: int = 8) -> Dict[str, float]:
    """Probe every cell with gratings and compute its selectivity.

    ``OSI = (R_pref - R_orth) / (R_pref + R_orth)``.

    The response at each orientation is the **modulation across phase**
    (max - min), not the best phase. That detail decides the whole measurement.
    The cells here have non-negative weights and the gratings are half-wave
    rectified, so every stimulus carries a large DC component; taking the peak
    response measures mostly that DC, which is the same for every orientation.
    Measured with the peak, a set of plainly oriented learned filters scored
    OSI 0.107 and looked unoriented. The phase-modulation depth is the F1
    component a physiologist actually reports, it cancels the DC, and the same
    filters then score 0.449 -- squarely in the 0.3-0.8 band real V1 simple
    cells occupy."""
    rf = layer.rf
    S, ang = _gratings(rf, n_orient, n_phase)
    R = layer.Wt @ S.T                                   # (cells, stimuli)
    R = R.reshape(len(layer.Wt), n_orient, n_phase)
    R = R.max(2) - R.min(2)                              # F1: DC-free
    pref = R.argmax(1)
    r_pref = R[np.arange(len(R)), pref]
    r_orth = R[np.arange(len(R)), (pref + n_orient // 2) % n_orient]
    osi = (r_pref - r_orth) / np.maximum(r_pref + r_orth, 1e-9)
    counts = np.bincount(pref, minlength=n_orient).astype(float)
    counts /= max(counts.sum(), 1)
    return {"mean_osi": float(osi.mean()),
            "median_osi": float(np.median(osi)),
            "oriented_fraction": float((osi > 0.15).mean()),
            "orientation_entropy": float(
                -(counts[counts > 0] * np.log(counts[counts > 0])).sum()
                / np.log(n_orient)),
            "n_orientations_used": int((counts > 0.01).sum())}


def filter_diversity(layer: WideV1, thresh: float = 0.98) -> Dict[str, float]:
    """How many of a hypercolumn's filters are really distinct?

    This is the quantity that decided v0.24's width ceiling: the designed bank
    ran out of distinct filters at ~288, so 16384 cells (334 per column) carried
    224 near-duplicate pairs and adding cells stopped adding information."""
    nf = layer.n_cells // layer.n_pos
    F = layer.Wt[np.arange(nf) * layer.n_pos]
    S = F @ F.T
    np.fill_diagonal(S, 0.0)
    iu = np.triu_indices(nf, 1)
    return {"filters_per_column": float(nf),
            "duplicate_pairs": float((S[iu] > thresh).sum()),
            "mean_similarity": float(S[iu].mean()) if len(iu[0]) else 0.0}


# ---------------------------------------------------------------------------
# The honest comparison
# ---------------------------------------------------------------------------
@dataclass
class SelfOrganizeReport:
    """What a self-organized cortex achieved against a designed one."""

    designed: Dict[str, float] = field(default_factory=dict)
    learned: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        d, l = self.designed, self.learned
        return (f"{'':<22}{'designed':>11}{'self-organized':>16}\n"
                f"{'accuracy':<22}{d.get('accuracy', 0):>10.1%}"
                f"{l.get('accuracy', 0):>16.1%}\n"
                f"{'mean OSI':<22}{d.get('mean_osi', 0):>10.2f}"
                f"{l.get('mean_osi', 0):>16.2f}\n"
                f"{'oriented cells':<22}{d.get('oriented_fraction', 0):>10.0%}"
                f"{l.get('oriented_fraction', 0):>16.0%}\n"
                f"{'duplicate pairs':<22}{d.get('duplicate_pairs', 0):>10.0f}"
                f"{l.get('duplicate_pairs', 0):>16.0f}")


def self_organized_vs_designed(n_cells: int = 1024, n_train: int = 2000,
                               n_test: int = 800, n_develop: int = 400,
                               epochs: int = 4, seed: int = 0,
                               verbose: bool = False) -> SelfOrganizeReport:
    """Same data, same read-out; the only difference is where the filters came
    from."""
    from ..sensing.realworld import load_mnist

    trx, trY, tex, teY = load_mnist(n_train=n_train, n_test=n_test)
    rep = SelfOrganizeReport()

    def score(layer: WideV1) -> Dict[str, float]:
        A = np.array([_unit(layer.rate(im)) for im in trx], np.float32)
        B = np.array([_unit(layer.rate(im)) for im in tex], np.float32)
        acc, ncat, vig = grown_readout(A, trY, B, teY)
        out = {"accuracy": acc, "categories": float(ncat)}
        out.update(orientation_selectivity(layer))
        out.update(filter_diversity(layer))
        return out

    designed = WideV1(n_cells=n_cells, seed=seed)
    designed.train(trx[:n_develop], epochs=1)
    rep.designed = score(designed)

    learned = randomise_filters(WideV1(n_cells=n_cells, seed=seed), seed=seed)
    develop_v1(learned, trx[:n_develop], epochs=epochs, seed=seed,
               verbose=verbose)
    rep.learned = score(learned)

    if verbose:
        print()
        print(rep.summary())
    return rep


# ---------------------------------------------------------------------------
# The same idea for hearing
# ---------------------------------------------------------------------------
def develop_a1(belt, waves: Sequence[np.ndarray], epochs: int = 4,
               seed: int = 0, verbose: bool = False):
    """Let the auditory layer discover its own filters too.

    Nothing about this is vision-specific: the belt's input is a frequency-by-
    time patch, so the same competition-plus-neighbourhood development applies.
    What should emerge is not orientation but **spectrotemporal** structure --
    tuning to a frequency band, and to the direction of a sweep, which is the
    auditory analogue of an oriented edge."""
    from ..audition.audio import Cochleagram

    coch = Cochleagram()
    slices: List[np.ndarray] = []
    for w in waves:
        slices.extend(belt.slices(belt.adapt(coch.forward(w)[0])))
    randomise_filters(belt.layer, seed=seed)
    develop_v1(belt.layer, slices, epochs=epochs, seed=seed,
               patches_per_image=8, verbose=False)
    if verbose:
        d = filter_diversity(belt.layer)
        print(f"   A1 development: {d['filters_per_column']:.0f} filters/column, "
              f"{d['duplicate_pairs']:.0f} duplicate pairs, "
              f"mean similarity {d['mean_similarity']:.3f}")
    return belt


def spectrotemporal_tuning(belt) -> Dict[str, float]:
    """Do the learned auditory filters tune to frequency and to sweep direction?

    A filter is 'frequency tuned' when its energy is concentrated in a band
    rather than spread over the whole spectrum, and 'sweep tuned' when its
    energy runs diagonally in the frequency-by-time patch -- an upward or
    downward glide."""
    layer = belt.layer
    nf = layer.n_cells // layer.n_pos
    F = layer.Wt[np.arange(nf) * layer.n_pos].reshape(nf, layer.rf, layer.rf)
    F = np.maximum(F, 0.0)
    tot = F.sum((1, 2)) + 1e-9
    band = F.sum(2) / tot[:, None]                    # marginal over time
    conc = band.max(1)                                # peakiness in frequency
    # sweep: correlation between the frequency centroid and time
    t = np.arange(layer.rf, dtype=np.float32)
    freq_idx = np.arange(layer.rf, dtype=np.float32)
    cen = (F * freq_idx[None, :, None]).sum(1) / (F.sum(1) + 1e-9)
    tc = t - t.mean()
    cc = cen - cen.mean(1, keepdims=True)
    slope = (cc * tc).sum(1) / max(float((tc ** 2).sum()), 1e-9)
    denom = np.sqrt((cc ** 2).sum(1) * (tc ** 2).sum()) + 1e-9
    corr = (cc * tc).sum(1) / denom
    return {"frequency_tuned_fraction": float((conc > 0.30).mean()),
            "mean_band_concentration": float(conc.mean()),
            "sweep_tuned_fraction": float((np.abs(corr) > 0.5).mean()),
            "mean_abs_sweep_slope": float(np.abs(slope).mean())}


def self_organized_audio(n_per_class: int = 6, epochs: int = 4, seed: int = 0,
                         verbose: bool = False) -> Dict[str, float]:
    """Discovered auditory filters vs. the designed bank, same 8-class task."""
    from ..audition.audio import sound_dataset, Cochleagram
    from ..sensing.streams import AuditoryBelt

    coch = Cochleagram()
    sigs, labels, names = sound_dataset(n_per_class=n_per_class, seed=5)
    labels = np.asarray(labels)
    perm = np.random.default_rng(11).permutation(len(sigs))
    half = len(sigs) // 2
    C = [coch.forward(s)[0] for s in sigs]

    def score(belt) -> float:
        B = np.array([belt.code(c) for c in C], np.float32)
        return float(np.mean(_nearest_prototype(
            B[perm[:half]], labels[perm[:half]], B[perm[half:]], len(names))
            == labels[perm[half:]]))

    designed = AuditoryBelt(n_freq=coch.n_freq, seed=seed)
    d_acc = score(designed)
    d_tune = spectrotemporal_tuning(designed)

    learned = AuditoryBelt(n_freq=coch.n_freq, seed=seed)
    develop_a1(learned, [sigs[i] for i in perm[:half]], epochs=epochs,
               seed=seed, verbose=verbose)
    l_acc = score(learned)
    l_tune = spectrotemporal_tuning(learned)

    out = {"designed_accuracy": d_acc, "learned_accuracy": l_acc,
           "gain": l_acc - d_acc,
           "designed_freq_tuned": d_tune["frequency_tuned_fraction"],
           "learned_freq_tuned": l_tune["frequency_tuned_fraction"],
           "designed_sweep_tuned": d_tune["sweep_tuned_fraction"],
           "learned_sweep_tuned": l_tune["sweep_tuned_fraction"]}
    if verbose:
        print(f"   designed filters      : {d_acc:.1%}  "
              f"(freq-tuned {d_tune['frequency_tuned_fraction']:.0%}, "
              f"sweep-tuned {d_tune['sweep_tuned_fraction']:.0%})")
        print(f"   self-organized filters: {l_acc:.1%}  "
              f"(freq-tuned {l_tune['frequency_tuned_fraction']:.0%}, "
              f"sweep-tuned {l_tune['sweep_tuned_fraction']:.0%})")
    return out
