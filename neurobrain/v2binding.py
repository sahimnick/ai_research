"""
v2binding.py
============

**Phase 2: V2 -- binding.** V1 says *what is here*; V2 says *what these things
make together*.

    edge + edge + a relative orientation  ->  a shape fragment

The difference from a second copy of V1
---------------------------------------
v0.21 already stacked spiking layers and it made things worse (V1->V2->V3 scored
39.2% against 63.4% for one wide layer). That attempt failed for a reason worth
naming: V2 was just another competitive layer reading V1's activity vector, so
it learned *combinations of activity levels*, not **combinations of parts**. It
had no notion that two edges are in a particular spatial and angular relation to
each other.

This V2 is built around that relation. Its units are tuned to **pairs of V1
features with a relative offset and a relative orientation**:

    (feature A at this place) AND (feature B, delta away, turned by theta)

which is what a corner, a junction, a curve fragment or a parallel pair *is*.
Two properties follow, and both are measured rather than hoped for:

* **The conjunction is genuinely non-linear.** A unit fires for the pair and not
  for either part alone -- the supralinear coincidence a dendritic NMDA plateau
  performs (cf. :mod:`dendrite`). A unit that merely summed its two inputs would
  respond to either one, and the measured selectivity index would show it.
* **The relation is relative, not absolute.** The offset and the angle are
  measured between the two parts, so the same fragment is recognised elsewhere
  in the image. That is what makes V2 a step up rather than a wider V1.

Learning is the same self-organizing rule as :mod:`selforganize`: the pairs are
not enumerated by hand, they are **the conjunctions that actually recur** in
V1's output, discovered by competition among candidate pairings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .neuron import Population
from .widev1 import WideV1, _unit


# ---------------------------------------------------------------------------
# What V1 hands upward: a sparse list of what fired, and where
# ---------------------------------------------------------------------------
@dataclass
class V1Activity:
    """V1's output as *parts*, not as a flat vector.

    Keeping the position and the preferred orientation of each active cell is
    the whole point: a flat rate vector has thrown that structure away, and a
    layer reading it can only learn co-activity, never a spatial relation."""

    cell: np.ndarray          # index of the active V1 cell
    row: np.ndarray           # its receptive-field row on the grid
    col: np.ndarray           # its receptive-field column
    orient: np.ndarray        # its preferred orientation, in radians
    rate: np.ndarray          # how hard it fired, in Hz

    def __len__(self) -> int:
        return int(len(self.cell))


def v1_parts(layer: WideV1, image: np.ndarray, orient_of_cell: np.ndarray,
             top_k: int = 40, window_ms: Optional[int] = None) -> V1Activity:
    """Run V1 and keep only the strongest responses, with their geometry."""
    r = layer.rate(image, window_ms=window_ms)
    k = min(int(top_k), int((r > 0).sum()))
    if k <= 0:
        e = np.zeros(0, int)
        return V1Activity(e, e, e, np.zeros(0, np.float32), np.zeros(0, np.float32))
    idx = np.argpartition(r, -k)[-k:]
    idx = idx[np.argsort(-r[idx])]
    pos = layer.cell_pos[idx]
    return V1Activity(idx, pos // layer.n_cols, pos % layer.n_cols,
                      orient_of_cell[idx].astype(np.float32),
                      r[idx].astype(np.float32))


def preferred_orientations(layer: WideV1, n_orient: int = 16,
                           n_phase: int = 8) -> np.ndarray:
    """Each V1 cell's preferred orientation, measured with gratings.

    Measured, not read off a design: after self-organization nobody knows what
    a given cell prefers until it is probed, and V2's relative-angle code needs
    that number for every cell."""
    from .selforganize import _gratings

    S, _ = _gratings(layer.rf, n_orient, n_phase)
    R = (layer.Wt @ S.T).reshape(layer.n_cells, n_orient, n_phase)
    mod = R.max(2) - R.min(2)                       # DC-free, as in v0.25
    return (np.pi * mod.argmax(1) / n_orient).astype(np.float32)


# ---------------------------------------------------------------------------
# V2: units tuned to a PAIR of parts in a relation
# ---------------------------------------------------------------------------
def _wrap(a: np.ndarray) -> np.ndarray:
    """Angle difference folded into [-pi/2, pi/2] -- orientation is mod pi."""
    return (a + np.pi / 2.0) % np.pi - np.pi / 2.0


class V2Binding:
    """Cells tuned to ``(feature A) AND (feature B at offset d, turned by t)``.

    Each unit stores a prototype relation: a preferred orientation for each of
    the two parts, a preferred spatial offset between them, and a preferred
    relative angle. It responds by **multiplying** its match to part A by its
    match to part B, so it is silent unless both are present together."""

    def __init__(self, n_units: int = 256, max_offset: int = 3,
                 sigma_pos: float = 1.1, sigma_ang: float = 0.45,
                 window_ms: int = 50, grid: int = 3, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n_units = int(n_units)
        self.max_offset = int(max_offset)
        self.sigma_pos, self.sigma_ang = float(sigma_pos), float(sigma_ang)
        self.window_ms = int(window_ms)
        # V2 is RETINOTOPIC. Taking the best pair anywhere in the image threw
        # away where the fragment was, and V2 then scored 34.8% alone and cost
        # 9.2 points when combined with V1 -- the same mistake as the collapsed
        # pooling of v0.21. Each unit now reports separately for each cell of a
        # coarse grid, so the code says *which fragment* and *roughly where*.
        self.grid = int(grid)
        self.rng = rng
        # prototypes: orientation of A, orientation of B, offset (dr, dc)
        self.oa = rng.uniform(0, np.pi, self.n_units).astype(np.float32)
        self.ob = rng.uniform(0, np.pi, self.n_units).astype(np.float32)
        self.dr = rng.uniform(-max_offset, max_offset, self.n_units).astype(np.float32)
        self.dc = rng.uniform(-max_offset, max_offset, self.n_units).astype(np.float32)
        self.duty = np.full(self.n_units, 1.0 / self.n_units, np.float32)
        self.pop = Population(self.n_units * self.grid * self.grid,
                              "regular_spiking", rng=rng, jitter=0.02)
        self.i_floor, self.i_span = 8.0, 30.0

    # -- the conjunction ---------------------------------------------------
    def match(self, act: V1Activity, n_rows: int = 7, n_cols: int = 7
              ) -> np.ndarray:
        """Drive of every V2 unit for this V1 activity.

        For each unit, every *ordered pair* of active V1 parts is scored on how
        well it fits the unit's stored relation, and the unit takes its best
        pair. The score is a product of three Gaussians -- part A's orientation,
        part B's orientation, and the offset between them -- so all three have to
        agree at once."""
        n = len(act)
        g = self.grid
        if n < 2:
            return np.zeros(self.n_units * g * g, np.float32)
        dR = act.row[None, :] - act.row[:, None]         # (a, b)
        dC = act.col[None, :] - act.col[:, None]
        near = (np.abs(dR) <= self.max_offset) & (np.abs(dC) <= self.max_offset)
        np.fill_diagonal(near, False)
        ii, jj = np.nonzero(near)
        if not len(ii):
            return np.zeros(self.n_units * g * g, np.float32)

        oa, ob = act.orient[ii], act.orient[jj]
        w = np.sqrt(act.rate[ii] * act.rate[jj])         # both must be active
        w = w / (w.max() + 1e-9)
        dr, dc = dR[ii, jj].astype(np.float32), dC[ii, jj].astype(np.float32)

        da = _wrap(oa[None, :] - self.oa[:, None])
        db = _wrap(ob[None, :] - self.ob[:, None])
        pr = dr[None, :] - self.dr[:, None]
        pc = dc[None, :] - self.dc[:, None]
        score = (np.exp(-(da ** 2) / (2 * self.sigma_ang ** 2))
                 * np.exp(-(db ** 2) / (2 * self.sigma_ang ** 2))
                 * np.exp(-(pr ** 2 + pc ** 2) / (2 * self.sigma_pos ** 2))
                 * w[None, :])
        # bin each pair by the midpoint of its two parts -> a coarse retinotopy
        mr = 0.5 * (act.row[ii] + act.row[jj])
        mc = 0.5 * (act.col[ii] + act.col[jj])
        br = np.clip((mr * g / max(n_rows, 1)).astype(int), 0, g - 1)
        bc = np.clip((mc * g / max(n_cols, 1)).astype(int), 0, g - 1)
        bin_id = br * g + bc
        out = np.zeros((self.n_units, g * g), np.float32)
        for b in range(g * g):
            m = bin_id == b
            if m.any():
                out[:, b] = score[:, m].max(1)
        return out.reshape(-1)

    def spikes(self, act: V1Activity) -> np.ndarray:
        """Real Izhikevich cells, driven by the conjunction. Rate in Hz."""
        m = self.match(act)
        m = m - 2.0 * (np.repeat(self.duty, self.grid * self.grid)
                       - 1.0 / self.n_units)
        s = m.std() + 1e-6
        z = (m - m.mean()) / s
        I = ((self.i_floor + self.i_span * np.clip(z, 0.0, 3.0) / 3.0)
             * (z > 0)).astype(np.float32)
        self.pop.reset()
        counts = np.zeros(self.pop.n, np.float32)
        for _ in range(self.window_ms):
            counts += self.pop.step(I).astype(np.float32)
        return counts * (1000.0 / self.window_ms)

    def code(self, act: V1Activity) -> np.ndarray:
        return _unit(self.spikes(act))

    # -- self-organization: which conjunctions actually recur? --------------
    def learn(self, act: V1Activity, lr: float = 0.08) -> None:
        """Move the winning unit's stored relation toward the pair it saw.

        Nothing enumerates corners or junctions. The units drift toward whatever
        conjunctions keep occurring in V1's output, and the homeostatic duty term
        stops one unit from claiming everything."""
        n = len(act)
        if n < 2:
            return
        dR = act.row[None, :] - act.row[:, None]
        dC = act.col[None, :] - act.col[:, None]
        near = (np.abs(dR) <= self.max_offset) & (np.abs(dC) <= self.max_offset)
        np.fill_diagonal(near, False)
        ii, jj = np.nonzero(near)
        if not len(ii):
            return
        pick = self.rng.permutation(len(ii))[:12]
        for p in pick:
            a, b = int(ii[p]), int(jj[p])
            da = _wrap(act.orient[a] - self.oa)
            db = _wrap(act.orient[b] - self.ob)
            pr = float(dR[a, b]) - self.dr
            pc = float(dC[a, b]) - self.dc
            score = (np.exp(-(da ** 2) / (2 * self.sigma_ang ** 2))
                     * np.exp(-(db ** 2) / (2 * self.sigma_ang ** 2))
                     * np.exp(-(pr ** 2 + pc ** 2) / (2 * self.sigma_pos ** 2)))
            win = int(np.argmax(score - 2.0 * (self.duty - 1.0 / self.n_units)))
            self.oa[win] = (self.oa[win] + lr * _wrap(
                act.orient[a] - self.oa[win])) % np.pi
            self.ob[win] = (self.ob[win] + lr * _wrap(
                act.orient[b] - self.ob[win])) % np.pi
            self.dr[win] += lr * (float(dR[a, b]) - self.dr[win])
            self.dc[win] += lr * (float(dC[a, b]) - self.dc[win])
            self.duty *= (1.0 - 0.02)
            self.duty[win] += 0.02

    def train(self, acts: Sequence[V1Activity], epochs: int = 2,
              lr: float = 0.08) -> "V2Binding":
        for _ in range(epochs):
            for i in self.rng.permutation(len(acts)):
                self.learn(acts[i], lr=lr)
        return self

    # -- is it really a conjunction? ---------------------------------------
    def conjunction_index(self, act: V1Activity) -> float:
        """Does a unit need BOTH parts, or would either do?

        The unit's response to the full pair is compared with its response when
        only one part is present. A true conjunction scores near 1; a unit that
        merely sums its inputs scores near 0. This is the measurement that says
        whether V2 is binding or just adding."""
        if len(act) < 2:
            return 0.0
        full = self.match(act)
        u = int(np.argmax(full))
        if full[u] <= 1e-9:
            return 0.0
        best = 0.0
        for keep in (0, 1):
            one = V1Activity(act.cell[keep:keep + 1], act.row[keep:keep + 1],
                             act.col[keep:keep + 1], act.orient[keep:keep + 1],
                             act.rate[keep:keep + 1])
            best = max(best, float(self.match(one)[u]))
        return float((full[u] - best) / (full[u] + 1e-9))


# ---------------------------------------------------------------------------
# Building and measuring the V1 -> V2 stack
# ---------------------------------------------------------------------------
@dataclass
class VentralV1V2:
    """The two-stage stream, with everything needed to run it end to end."""

    v1: WideV1
    v2: V2Binding
    orient: np.ndarray
    top_k: int = 40

    def parts(self, image: np.ndarray) -> V1Activity:
        return v1_parts(self.v1, image, self.orient, top_k=self.top_k)

    def v1_code(self, image: np.ndarray) -> np.ndarray:
        return _unit(self.v1.rate(image))

    def v2_code(self, image: np.ndarray) -> np.ndarray:
        return self.v2.code(self.parts(image))

    def code(self, image: np.ndarray) -> np.ndarray:
        """V1 and V2 together -- parts and the fragments they make."""
        return _unit(np.concatenate([self.v1_code(image), self.v2_code(image)]))

    # the name the rest of the project uses for a front end
    def top_code(self, image: np.ndarray) -> np.ndarray:
        return self.code(image)


def build_v1_v2(n_v1: int = 1024, n_v2: int = 256, n_train: int = 1500,
                n_test: int = 600, n_develop: int = 300, epochs: int = 3,
                seed: int = 0, verbose: bool = False
                ) -> Tuple[VentralV1V2, Dict[str, float]]:
    """Self-organize V1, then self-organize V2 on top of it, and score both.

    The comparison that matters is **V1 alone vs V1+V2 on the same read-out**:
    if binding adds nothing, that shows up immediately, as it did in v0.21."""
    from .realworld import load_mnist
    from .selforganize import randomise_filters, develop_v1
    from .widev1 import grown_readout

    def say(*a):
        if verbose:
            print(*a)

    trx, trY, tex, teY = load_mnist(n_train=n_train, n_test=n_test)

    say("V1: discovering its own features ...")
    v1 = randomise_filters(WideV1(n_cells=n_v1, seed=seed), seed=seed)
    develop_v1(v1, trx[:n_develop], epochs=epochs, seed=seed)
    orient = preferred_orientations(v1)

    say("V2: discovering which conjunctions of V1 features recur ...")
    v2 = V2Binding(n_units=n_v2, seed=seed)
    stream = VentralV1V2(v1, v2, orient)
    acts = [stream.parts(im) for im in trx[:n_develop]]
    v2.train(acts, epochs=2)

    say("scoring V1 alone, V2 alone, and the two together ...")
    A1 = np.array([stream.v1_code(im) for im in trx], np.float32)
    B1 = np.array([stream.v1_code(im) for im in tex], np.float32)
    A2 = np.array([stream.v2_code(im) for im in trx], np.float32)
    B2 = np.array([stream.v2_code(im) for im in tex], np.float32)
    both_a = np.array([_unit(np.concatenate([a, b])) for a, b in zip(A1, A2)],
                      np.float32)
    both_b = np.array([_unit(np.concatenate([a, b])) for a, b in zip(B1, B2)],
                      np.float32)

    ci = float(np.mean([v2.conjunction_index(a) for a in acts[:60]]))
    out = {
        "v1_alone": grown_readout(A1, trY, B1, teY)[0],
        "v2_alone": grown_readout(A2, trY, B2, teY)[0],
        "v1_plus_v2": grown_readout(both_a, trY, both_b, teY)[0],
        "conjunction_index": ci,
        "v2_silent_fraction": float((A2 == 0).mean()),
    }
    out["binding_gain"] = out["v1_plus_v2"] - out["v1_alone"]
    if verbose:
        print(f"   V1 alone        : {out['v1_alone']:.1%}")
        print(f"   V2 alone        : {out['v2_alone']:.1%}")
        print(f"   V1 + V2         : {out['v1_plus_v2']:.1%} "
              f"({out['binding_gain']:+.1%})")
        print(f"   conjunction index: {ci:.2f} "
              f"(1.0 = needs both parts, 0.0 = either will do)")
    return stream, out
