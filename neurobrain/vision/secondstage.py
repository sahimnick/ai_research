"""
secondstage.py
==============

**A second cortical stage, retried where it might actually be needed.**

This project measured stacking as harmful. v0.21's V1->V2->V3 scored 39.2%
against 63.4% for a single wide layer, and `integrated.py` records that
"hierarchical depth has failed four separate times". That is a real result and
it is why :mod:`widev1` exists: width instead of depth.

But every one of those failures was measured on **MNIST digits with narrow
layers** -- 24 V1 cells feeding 160 feeding 300. A digit is one high-contrast
figure on an empty field, a single wide layer already reaches ~0.82 on it, and
a second stage built from 24 inputs has almost nothing to pass on. Under those
conditions "depth hurts" is nearly guaranteed and says little about depth.

Photographs are the opposite case, and `benchmarks/vision_ceiling.py` localised
the eye's failure there precisely. On CIFAR-10 the wide V1 reaches 1-NN 0.323
against 0.914 for the ear, and **nothing inside a single layer moves it**:

    width        1024 -> 4096 cells        0.323 -> 0.315   flat
    aperture     rf 7 / 10 / 14            0.323 / 0.315 / 0.272
    spike noise  spiking vs noiseless      0.322 vs 0.323   costs nothing
    integration  50 ms vs 200 ms           0.322 vs 0.323   nothing

Four levers, no movement. What is left is depth, so it is worth one honest test
under the conditions the earlier ones never had.

What this stage is
------------------
The classical account of V1 -> V2, and the one every model of the ventral
stream shares (Hubel & Wiesel's simple/complex distinction, Riesenhuber &
Poggio's HMAX, and the first stage of the unsupervised pipelines that work on
CIFAR):

1. **Complex-cell pooling.** Take the max of each V1 filter over a small
   neighbourhood of positions. The cell stops caring exactly *where* its feature
   was, which is the tolerance a photograph needs and a centred digit does not.
2. **A wider aperture.** Each V2 unit reads a ``span x span`` block of pooled
   positions across *all* V1 filters, so it sees combinations of features over
   an area several times a V1 receptive field.
3. **Competitive learning, no gradients.** Units compete for each patch and the
   winner moves toward it by the instar rule ``w += lr * (x - w)``, with a duty
   cycle keeping the map from collapsing onto a few units. The same rule
   :func:`~neurobrain.learning.selforganize.develop_v1` uses one stage down, and
   the same rule the category maps use. Nothing here differentiates anything.

The read-out is k-winners-take-all over the units, pooled across position -- so
a V2 code says *which combinations of features occurred*, with position largely
discarded, which is what an object identity code should say.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .widev1 import WideV1, _unit


class SecondStage:
    """V2 over a :class:`WideV1`: pool, widen, compete.

    Parameters
    ----------
    v1:       the first stage. Only its ``drive`` and grid shape are used.
    n_units:  how many V2 feature detectors.
    pool:     complex-cell neighbourhood, in V1 positions.
    span:     how many pooled positions across a V2 unit's aperture.
    sparsity: fraction of units allowed to respond to a patch.
    """

    def __init__(self, v1: WideV1, n_units: int = 512, pool: int = 2,
                 span: int = 2, sparsity: float = 0.10, seed: int = 0):
        self.v1 = v1
        self.pool = int(pool)
        self.span = int(span)
        self.sparsity = float(sparsity)
        self.n_filt = v1.n_cells // v1.n_pos
        self.n_units = int(n_units)
        rng = np.random.default_rng(seed)
        self.dim = self.n_filt * self.span * self.span
        self.W = np.abs(rng.standard_normal((self.n_units, self.dim))
                        ).astype(np.float32)
        self.W /= np.maximum(np.linalg.norm(self.W, axis=1, keepdims=True), 1e-6)
        self.duty = np.full(self.n_units, 1.0 / self.n_units, np.float32)
        self.rng = rng

    # -- the two geometric operations --------------------------------------
    def complex_cells(self, image: np.ndarray) -> np.ndarray:
        """V1 drive -> ``(n_filt, rows, cols)`` after max-pooling over ``pool``.

        The max, not the mean: a complex cell reports that its feature was
        present somewhere in its neighbourhood, and averaging would dilute a
        sharp local response into the empty positions around it."""
        d = self.v1.drive(image)
        g = d[:self.n_filt * self.v1.n_pos].reshape(
            self.n_filt, self.v1.n_rows, self.v1.n_cols)
        p = self.pool
        r, c = g.shape[1] // p, g.shape[2] // p
        if r < 1 or c < 1:
            return g
        g = g[:, :r * p, :c * p].reshape(self.n_filt, r, p, c, p)
        return g.max(axis=(2, 4))

    def patches(self, pooled: np.ndarray) -> np.ndarray:
        """Every ``span x span`` block of pooled positions, flattened."""
        s = self.span
        nf, r, c = pooled.shape
        if r < s or c < s:
            return pooled.reshape(1, -1)
        out = [pooled[:, i:i + s, j:j + s].reshape(-1)
               for i in range(r - s + 1) for j in range(c - s + 1)]
        return np.asarray(out, np.float32)

    # -- learning -----------------------------------------------------------
    def learn(self, images: Sequence[np.ndarray], epochs: int = 3,
              lr0: float = 0.30, lr1: float = 0.02, seed: int = 0
              ) -> List[float]:
        """Grow V2 features by competition. Returns mean winner-similarity.

        The rate anneals from a coarse reorganisation to a local refinement --
        the same critical-period schedule :func:`develop_v1` uses, for the same
        reason: an early large step lets units move across the space, and a late
        small one lets them settle without erasing what they found."""
        rng = np.random.default_rng(seed)
        pool = [self.patches(self.complex_cells(im)) for im in images]
        curve = []
        total = max(1, epochs * len(pool))
        step = 0
        for _ in range(int(epochs)):
            sims = []
            for k in rng.permutation(len(pool)):
                P = pool[k]
                for x in P[rng.permutation(len(P))]:
                    n = float(np.linalg.norm(x))
                    if n < 1e-6:
                        continue
                    xn = (x / n).astype(np.float32)
                    # duty-cycle bias, so a unit that keeps winning steps aside
                    drive = self.W @ xn - 2.0 * (self.duty - 1.0 / self.n_units)
                    w = int(np.argmax(drive))
                    lr = lr0 * (lr1 / lr0) ** (step / total)
                    self.W[w] = _unit(self.W[w] + lr * (xn - self.W[w]))
                    self.duty *= 0.999
                    self.duty[w] += 0.001
                    sims.append(float(self.W[w] @ xn))
                step += 1
            curve.append(float(np.mean(sims)) if sims else 0.0)
        return curve

    def learn_temporal(self, sequences: Sequence[Sequence[np.ndarray]],
                       epochs: int = 3, lr0: float = 0.30, lr1: float = 0.02,
                       seed: int = 0) -> List[float]:
        """Learn invariance from *time* instead of from competition alone.

        Why try this at all: the competitive version above failed, and so did
        every other single-layer lever. But the auditory front end that *works*
        in this project does not learn by competing over static snapshots -- it
        learns by predicting its own next input (:class:`PredictiveA1`,
        :class:`ContrastivePredictiveA1`). Vision never got the equivalent, and
        the eye is the modality that is failing. That asymmetry is at least
        suggestive about which stream got the better learning rule.

        The rule is Foldiak's (1991) **trace rule**, and it is one change: the
        winner is chosen once for a whole sequence of views of the same thing,
        and then moved toward *every* view. A unit that fires for an object at
        one fixation has its synapses strengthened onto the object at the next
        one, so it comes to respond to all of them. Invariance is not designed
        in; it is picked up from the fact that the world changes more slowly
        than the retina does (Wiskott & Sejnowski's slow feature analysis makes
        the same assumption).

        Nothing here differentiates anything -- the update is still
        ``w += lr * (x - w)`` with a duty cycle, exactly as above. The only
        difference is *which* patches one winner is updated on.

        ``sequences`` are views of one thing: small shifts stand in for
        fixational drift and micro-saccades, which is what actually arrives at a
        retina looking at a stationary object.
        """
        rng = np.random.default_rng(seed)
        pool = [[self.patches(self.complex_cells(v)) for v in seq]
                for seq in sequences]
        curve = []
        total = max(1, epochs * len(pool))
        step = 0
        for _ in range(int(epochs)):
            sims = []
            for k in rng.permutation(len(pool)):
                views = pool[k]
                if not views:
                    continue
                n_pos = min(len(v) for v in views)
                for j in rng.permutation(n_pos):
                    xs = [v[j] for v in views]
                    norms = [float(np.linalg.norm(x)) for x in xs]
                    if max(norms) < 1e-6:
                        continue
                    xs = [(x / n).astype(np.float32)
                          for x, n in zip(xs, norms) if n > 1e-6]
                    # ONE winner for the whole sequence, chosen on the view-
                    # averaged input -- this is the trace that carries across
                    # fixations
                    mean = _unit(np.mean(xs, axis=0))
                    drive = self.W @ mean - 2.0 * (self.duty - 1.0 / self.n_units)
                    w = int(np.argmax(drive))
                    lr = lr0 * (lr1 / lr0) ** (step / total)
                    for xn in xs:                    # ...and updated on all
                        self.W[w] = _unit(self.W[w] + lr * (xn - self.W[w]))
                    self.duty *= 0.999
                    self.duty[w] += 0.001
                    sims.append(float(self.W[w] @ mean))
                step += 1
            curve.append(float(np.mean(sims)) if sims else 0.0)
        return curve

    # -- the code a downstream area reads ----------------------------------
    def code(self, image: np.ndarray) -> np.ndarray:
        """k-winners per position, pooled over position, unit-normed.

        Mean **and** max over positions: what was present on average, and what
        was present at all. The second half is what keeps a small object from
        being averaged away by the background around it."""
        P = self.patches(self.complex_cells(image))
        n = np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-6)
        R = np.maximum(self.W @ (P / n).T, 0.0).T          # (positions, units)
        k = max(1, int(self.n_units * self.sparsity))
        if k < self.n_units:
            thresh = np.partition(R, -k, axis=1)[:, -k][:, None]
            R = R * (R >= thresh)
        return _unit(np.concatenate([R.mean(0), R.max(0)]))
