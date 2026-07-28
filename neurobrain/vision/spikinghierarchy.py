"""
spikinghierarchy.py
===================

A **spiking V1 -> V2 -> V3 hierarchy**, and **active causal discovery**.

Two upgrades to the two weakest measured results in the project.

**Headline result, measured: the hierarchy improves the REPRESENTATION but not
the score.** Separability rises properly up the stack (+0.144 -> +0.259 ->
+0.282), which is what a ventral stream is supposed to do, yet held-out accuracy
is **39.2%** -- worse than the single wide spiking layer's 63.4% and far below
the non-spiking 89.5%. Depth through narrow layers costs more than the added
abstraction returns here. That is the honest state of it, not a stepping-stone
framed as a win.

**1. A real spiking hierarchy.** :mod:`spikingvision` proved perception could be
done by actual Izhikevich neurons, but it was a *single* layer, and it cost 26
accuracy points against the vector version. One layer is also not what a ventral
stream is. Here three spiking populations are stacked, each learning from the
spikes of the one below:

  * **V1** -- oriented, Gabor-seeded simple cells on image patches. Local edges.
  * **V2** -- cells over V1's *spike counts*, pooled over space: combinations of
    orientations, i.e. corners and curve fragments.
  * **V3** -- cells over V2, pooled again: larger, more invariant configurations.

Every layer is a genuine :class:`~neurobrain.neuron.Population` that integrates
current and fires; the code passed upward is always a **spike-count vector**.
Learning is competitive Hebbian on what actually fired, with divisive
normalization (lateral inhibition) and homeostatic thresholds, because without
those a spiking rate code either saturates or collapses -- both failure modes
were hit and fixed on the way here.

**2. Active causal discovery.** In :mod:`grounding` the agent acted *at random*
and learned which visual features predict outcomes. Animals do not explore at
random: they act where they expect to *learn* something -- the epistemic drive
behind play. Here the agent picks the action whose outcome its model is most
uncertain about (predictions nearest 0.5, where an experiment is most
informative), and the effect on the learning curve is measured against random
acting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.neuron import Population


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v.astype(np.float32) if n < 1e-9 else (v / n).astype(np.float32)


class SpikingLayer:
    """One cortical stage: real neurons, learned weights, spike-count output."""

    def __init__(self, n_in: int, n_cells: int, window: int = 40,
                 k_frac: float = 0.12, seed: int = 0,
                 neuron_type: str = "regular_spiking"):
        rng = np.random.default_rng(seed)
        self.W = np.abs(rng.normal(0, 1.0, (n_cells, n_in))).astype(np.float32)
        self.W /= np.maximum(np.linalg.norm(self.W, axis=1, keepdims=True), 1e-6)
        self.n_cells, self.window = n_cells, window
        self.k = max(2, int(n_cells * k_frac))
        self.duty = np.full(n_cells, self.k / n_cells, np.float32)
        self.pop = Population(n_cells, neuron_type, rng=rng, jitter=0.02)
        self.i_floor, self.i_span = 8.0, 30.0

    def drive(self, x: np.ndarray) -> np.ndarray:
        """Divisive normalization into the band where rate actually grades."""
        raw = self.W @ _unit(x)
        z = (raw - raw.mean()) / (raw.std() + 1e-6)
        z = z - 2.0 * (self.duty - self.k / self.n_cells)
        return ((self.i_floor + self.i_span * np.clip(z, 0.0, 3.0) / 3.0)
                * (z > 0)).astype(np.float32)

    def spikes(self, x: np.ndarray) -> np.ndarray:
        d = self.drive(x)
        self.pop.reset()
        out = np.zeros(self.n_cells, np.float32)
        for _ in range(self.window):
            out += self.pop.step(d).astype(np.float32)
        return out

    def learn(self, x: np.ndarray, counts: np.ndarray, lr: float = 0.03) -> None:
        fired = counts > 0
        if not fired.any():
            return
        xu = _unit(x)
        w = counts[fired][:, None] / max(float(counts.max()), 1.0)
        self.W[fired] += lr * w * (xu[None, :] - self.W[fired])
        np.maximum(self.W[fired], 0.0, out=self.W[fired])
        self.W[fired] /= np.maximum(
            np.linalg.norm(self.W[fired], axis=1, keepdims=True), 1e-6)
        self.duty *= (1.0 - lr)
        self.duty[fired] += lr

    def seed_gabor(self, patch: int) -> "SpikingLayer":
        """Give V1 oriented receptive fields to start from, as genetics does."""
        yy, xx = np.mgrid[0:patch, 0:patch] - (patch - 1) / 2.0
        for i in range(self.n_cells):
            th = np.pi * (i % 8) / 8.0
            lam = 3.0 + 2.0 * ((i // 8) % 3)
            xr = xx * np.cos(th) + yy * np.sin(th)
            yr = -xx * np.sin(th) + yy * np.cos(th)
            g = np.exp(-(xr ** 2 + 0.6 * yr ** 2) / (2 * (patch / 4.0) ** 2))
            g = g * np.cos(2 * np.pi * xr / lam)
            self.W[i] = np.maximum(g.reshape(-1), 0.0)
        self.W /= np.maximum(np.linalg.norm(self.W, axis=1, keepdims=True), 1e-6)
        return self


def _patches(img: np.ndarray, patch: int, stride: int) -> np.ndarray:
    """Sliding windows -- V1 sees local pieces, not the whole retina."""
    H, W = img.shape
    out = []
    for y in range(0, H - patch + 1, stride):
        for x in range(0, W - patch + 1, stride):
            out.append(img[y:y + patch, x:x + patch].reshape(-1))
    return np.array(out, np.float32)


@dataclass
class SpikingVentralHierarchy:
    """V1 -> V2 -> V3, all spiking. Each stage reads the spikes below it."""

    V1: SpikingLayer
    V2: SpikingLayer
    V3: SpikingLayer
    patch: int = 7
    stride: int = 3
    grid: int = 7                      # spatial pooling of V1 before V2
    #
    # MEASURED, and it matters: pooling V1 into a 2x2 grid destroys the digit
    # (separability +0.009, against +0.142 for the raw pixels). Cortex pools
    # gently and keeps retinotopy; so must this. grid=7 keeps +0.088.
    accuracy: float = 0.0
    separability: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # -- the forward sweep, in spikes ---------------------------------------
    _batch: object = None      # a Population sized n_cells x n_patches

    def v1_map(self, image: np.ndarray) -> np.ndarray:
        """V1 spike counts for every patch, arranged as a coarse spatial map.

        All patches are run **simultaneously** through one big population --
        every patch has its own copy of the V1 cells, which is exactly the
        retinotopic arrangement of real V1, and it turns thousands of tiny
        simulations into one."""
        P = _patches(np.asarray(image, np.float32) / 255.0, self.patch, self.stride)
        side = int(np.sqrt(len(P)))
        n_p, n_c = len(P), self.V1.n_cells
        if self._batch is None or self._batch.n != n_p * n_c:
            self._batch = Population(n_p * n_c, "regular_spiking",
                                     rng=np.random.default_rng(7), jitter=0.02)
        drive = np.concatenate([self.V1.drive(P[i]) for i in range(n_p)])
        self._batch.reset()
        counts = np.zeros(n_p * n_c, np.float32)
        for _ in range(self.V1.window):
            counts += self._batch.step(drive).astype(np.float32)
        resp = counts.reshape(n_p, n_c)
        resp = resp.reshape(side, side, -1)
        # pool into grid x grid cells -> V2 sees *where* as well as *what*
        g = self.grid
        edges = np.linspace(0, side, g + 1).astype(int)
        pooled = [resp[edges[i]:edges[i + 1], edges[j]:edges[j + 1]].mean((0, 1))
                  for i in range(g) for j in range(g)]
        return np.concatenate(pooled).astype(np.float32)

    def code(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        c1 = self.v1_map(image)
        c2 = self.V2.spikes(c1)
        c3 = self.V3.spikes(c2)
        return {"V1": c1, "V2": c2, "V3": c3}

    def top_code(self, image: np.ndarray) -> np.ndarray:
        return _unit(self.code(image)["V3"])


def _separability(C: np.ndarray, y: np.ndarray, n: int = 120) -> float:
    """within-class minus between-class similarity of a code."""
    C = np.array([_unit(c) for c in C[:n]], np.float32)
    y = y[:n]
    same = [float(C[i] @ C[j]) for i in range(len(C)) for j in range(i + 1, len(C))
            if y[i] == y[j]]
    diff = [float(C[i] @ C[j]) for i in range(len(C)) for j in range(i + 1, len(C))
            if y[i] != y[j]]
    return float(np.mean(same) - np.mean(diff)) if same and diff else 0.0


def build_spiking_hierarchy(n_train: int = 4000, n_test: int = 1000,
                            n_v1: int = 24, n_v2: int = 160, n_v3: int = 300,
                            window: int = 40, vigilance: float = 0.94,
                            verbose: bool = False) -> SpikingVentralHierarchy:
    """Train V1->V2->V3 on REAL digits, all in spikes, and score it honestly."""
    from ..sensing.realworld import load_mnist
    from .spikingvision import SpikingCategoryMap

    def say(*a):
        if verbose:
            print(*a)

    trx, trY, tex, teY = load_mnist(n_train=n_train, n_test=n_test)
    patch, stride, grid = 7, 3, 7
    V1 = SpikingLayer(patch * patch, n_v1, window=window, seed=1).seed_gabor(patch)
    n_v1_out = n_v1 * grid * grid
    V2 = SpikingLayer(n_v1_out, n_v2, window=window, seed=2)
    V3 = SpikingLayer(n_v2, n_v3, window=window, seed=3)
    h = SpikingVentralHierarchy(V1, V2, V3, patch=patch, stride=stride, grid=grid)

    say("V1: refining Gabor-seeded spiking simple cells ...")
    for im in trx[:250]:
        for p in _patches(im.astype(np.float32) / 255.0, patch, stride)[::4]:
            V1.learn(p, V1.spikes(p), lr=0.02)

    say("V2: learning over V1 SPIKES (corners / curve fragments) ...")
    m1 = [h.v1_map(im) for im in trx[:700]]
    for c in m1:
        V2.learn(c, V2.spikes(c), lr=0.03)

    say("V3: learning over V2 spikes (larger configurations) ...")
    for c in m1:
        c2 = V2.spikes(c)
        V3.learn(c2, V3.spikes(c2), lr=0.03)

    say("encoding train/test through the whole spiking hierarchy ...")
    codes_tr = [h.code(im) for im in trx]
    Xtr = np.array([_unit(c["V3"]) for c in codes_tr], np.float32)
    Xte = np.array([_unit(h.code(im)["V3"]) for im in tex], np.float32)

    h.separability = (
        _separability(np.array([c["V1"] for c in codes_tr[:120]]), trY),
        _separability(np.array([c["V2"] for c in codes_tr[:120]]), trY),
        _separability(np.array([c["V3"] for c in codes_tr[:120]]), trY))

    cortex = SpikingCategoryMap(dim=Xtr.shape[1], vigilance=vigilance,
                                max_cells=3000)
    for x in Xtr:
        cortex.learn(x)
    cortex.name_cells(Xtr, trY)
    h.accuracy = float(np.mean(cortex.recognise(Xte) == teY))
    if verbose:
        print(f"   separability V1 {h.separability[0]:+.3f} -> "
              f"V2 {h.separability[1]:+.3f} -> V3 {h.separability[2]:+.3f}")
        print(f"   held-out accuracy (3-stage SPIKING): {h.accuracy:.1%} "
              f"({cortex.n_categories} categories)")
    return h


# -- active causal discovery -------------------------------------------------

def active_causal_discovery(n_episodes: int = 6000, seed: int = 0,
                            verbose: bool = False) -> Dict[str, float]:
    """Does **choosing** what to try beat acting at random?

    The agent picks the action whose outcome it is least sure about (predictions
    closest to 0.5 -- where an experiment carries the most information). This is
    the epistemic drive behind play, and it is compared head-to-head with random
    acting on identical machinery and the same number of episodes."""
    from ..world.grounding import (CausalFeatureLearner, render_object, _outcomes,
                            ACTIONS, ALL_PROPS, OUTCOMES)

    def run(active: bool) -> Tuple[float, float]:
        rng = np.random.default_rng(seed)
        L = CausalFeatureLearner(n_pixels=400, n_features=12, seed=seed)
        for _ in range(n_episodes):
            props = [p for p in ALL_PROPS if rng.random() < 0.4]
            img = render_object(props, size=20, rng=rng)
            if active:
                # pick the action whose predicted outcomes are most uncertain
                unc = {a: -np.mean([abs(v - 0.5) for v in
                                    L.predict_outcome(img, a).values()])
                       for a in ACTIONS}
                action = max(unc, key=unc.get)
            else:
                action = ACTIONS[rng.integers(len(ACTIONS))]
            L.learn(img, action, _outcomes(action, props))
        # score both on the SAME random held-out episodes
        rt = np.random.default_rng(seed + 999)
        per = 0.0
        n = 800
        for _ in range(n):
            props = [p for p in ALL_PROPS if rt.random() < 0.4]
            img = render_object(props, size=20, rng=rt)
            a = ACTIONS[rt.integers(len(ACTIONS))]
            truth = _outcomes(a, props)
            pr = L.predict_outcome(img, a)
            per += np.mean([(pr[o] >= 0.5) == bool(truth[o]) for o in OUTCOMES])
        return per / n, 0.0

    act, _ = run(True)
    rnd, _ = run(False)
    out = {"active": act, "random": rnd, "gain": act - rnd}
    if verbose:
        print(f"   acting at random     : {rnd:.1%} per-outcome")
        print(f"   acting to REDUCE doubt: {act:.1%} per-outcome  "
              f"({out['gain']:+.1%})")
    return out
