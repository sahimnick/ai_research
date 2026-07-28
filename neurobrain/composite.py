"""
composite.py
============

**A task that actually needs binding**, and a V2 that can do it.

Why the old V2 could not fail informatively
-------------------------------------------
v0.25 built a V2 whose units were genuine conjunctions (measured index 1.00) and
it bought **nothing**: -0.4% +/- 1% over V1 alone. That is not evidence that
binding is useless -- it is evidence that the *task* did not need it. A centred
MNIST digit is identified perfectly well by which local features are present;
nothing about it requires knowing how those features are arranged relative to
each other, so a layer that encodes arrangement adds a cost and no information.

Two things are fixed here, both of them prerequisites for the question to mean
anything.

1. Overlapping composites
-------------------------
:func:`overlapping_pair` draws **two shapes on top of each other** in the same
region, so their strokes interleave and V1 cannot tell which edge belongs to
which object. The task is to say *which pair* is present. A bag of local
features is close to useless here: the same edges appear in many pairs, and only
their **relative arrangement** distinguishes one composite from another. This is
the binding problem in its classic form, and it is what V2 exists for.

The shapes are drawn as strokes rather than filled regions, so overlap really
does interleave contours instead of one shape simply occluding the other.

2. A topographic V2
-------------------
:class:`LocalV2` keeps the retinotopic map. The v0.25 V2 scored the best pair of
V1 features **anywhere in the image**; a coarse spatial binning was bolted on
afterwards, but the pairing itself was still global, so a feature at the top of
the image could bind with one at the bottom. Here:

    * a unit only ever pairs features inside its **own receptive neighbourhood**
      (a radius of a few hypercolumns), so a binding is by construction local;
    * every unit exists at **every position** of a topographic grid, exactly as
      V1's filters do, so the output is a map of *which fragment is where*
      rather than a list of fragments;
    * the pairing is still multiplicative, so the conjunction property that was
      measured at 1.00 is preserved.

Growth follows the same principle as everywhere else: the units start random and
drift toward the conjunctions that actually recur, with homeostasis to stop a
few units from claiming everything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .neuron import Population
from .widev1 import WideV1, _unit


# ---------------------------------------------------------------------------
# Stimuli that require binding
# ---------------------------------------------------------------------------
SHAPES = ("bar_h", "bar_v", "bar_d1", "bar_d2", "arc_l", "arc_r", "vee", "tee")


def _stroke(canvas: np.ndarray, pts: Sequence[Tuple[float, float]],
            value: float = 255.0) -> None:
    """Draw a polyline as a thin stroke -- contours, not filled regions."""
    for (y0, x0), (y1, x1) in zip(pts[:-1], pts[1:]):
        n = int(max(abs(y1 - y0), abs(x1 - x0)) * 2 + 2)
        for t in np.linspace(0.0, 1.0, n):
            y = int(round(y0 + t * (y1 - y0)))
            x = int(round(x0 + t * (x1 - x0)))
            if 0 <= y < canvas.shape[0] and 0 <= x < canvas.shape[1]:
                canvas[y, x] = value


def draw_shape(name: str, size: int = 28, cy: float = 14.0, cx: float = 14.0,
               scale: float = 8.0, rot: float = 0.0) -> np.ndarray:
    """One shape, drawn as a stroke on an empty canvas."""
    c = np.zeros((size, size), np.float32)
    s = scale

    def R(pts):
        out = []
        for (dy, dx) in pts:
            y = dy * np.cos(rot) - dx * np.sin(rot)
            x = dy * np.sin(rot) + dx * np.cos(rot)
            out.append((cy + y, cx + x))
        return out

    if name == "bar_h":
        _stroke(c, R([(0, -s), (0, s)]))
    elif name == "bar_v":
        _stroke(c, R([(-s, 0), (s, 0)]))
    elif name == "bar_d1":
        _stroke(c, R([(-s, -s), (s, s)]))
    elif name == "bar_d2":
        _stroke(c, R([(-s, s), (s, -s)]))
    elif name == "arc_l":
        th = np.linspace(np.pi / 2, 3 * np.pi / 2, 14)
        _stroke(c, R([(s * np.sin(t), s * np.cos(t)) for t in th]))
    elif name == "arc_r":
        th = np.linspace(-np.pi / 2, np.pi / 2, 14)
        _stroke(c, R([(s * np.sin(t), s * np.cos(t)) for t in th]))
    elif name == "vee":
        _stroke(c, R([(-s, -s), (s, 0), (-s, s)]))
    elif name == "tee":
        _stroke(c, R([(-s, 0), (s, 0)]))
        _stroke(c, R([(0, -s), (0, s)]))
    else:
        raise KeyError(name)
    return c


def overlapping_pair(a: str, b: str, size: int = 28, jitter: float = 2.0,
                     rot_jitter: float = 0.25,
                     rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Two shapes drawn **on top of each other**, contours interleaved.

    Both are centred on the same region (up to a small jitter), so there is no
    spatial gap that would let a bag-of-features read-out separate them. What
    distinguishes ``(bar_h, arc_l)`` from ``(bar_h, arc_r)`` is only how the
    parts are arranged with respect to one another."""
    rng = rng or np.random.default_rng()
    c = np.zeros((size, size), np.float32)
    for name in (a, b):
        cy = size / 2 + float(rng.uniform(-jitter, jitter))
        cx = size / 2 + float(rng.uniform(-jitter, jitter))
        rot = float(rng.uniform(-rot_jitter, rot_jitter))
        c = np.maximum(c, draw_shape(name, size, cy, cx,
                                     scale=size / 3.5, rot=rot))
    return c


def composite_dataset(n_per_class: int = 60, size: int = 28,
                      n_pairs: int = 12, seed: int = 0
                      ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """A classification set whose classes are **pairs** of overlapping shapes.

    The pairs are chosen so that every individual shape appears in several
    different classes. That is the whole design: a read-out that only knows
    *which parts are present* cannot separate the classes, because the parts are
    shared. Only the arrangement differs."""
    rng = np.random.default_rng(seed)
    pairs: List[Tuple[str, str]] = []
    for i in range(len(SHAPES)):
        for j in range(i + 1, len(SHAPES)):
            pairs.append((SHAPES[i], SHAPES[j]))
    rng.shuffle(pairs)
    pairs = pairs[:n_pairs]
    names = [f"{a}+{b}" for a, b in pairs]

    X, y = [], []
    for k, (a, b) in enumerate(pairs):
        for _ in range(n_per_class):
            X.append(overlapping_pair(a, b, size=size, rng=rng))
            y.append(k)
    idx = rng.permutation(len(X))
    return (np.asarray(X, np.float32)[idx], np.asarray(y)[idx], names)


def part_overlap_report(names: Sequence[str]) -> Dict[str, float]:
    """How much the classes share their parts -- the difficulty of the task.

    If every shape appeared in exactly one class the task would be solvable from
    parts alone and would prove nothing."""
    from collections import Counter

    c: Counter = Counter()
    for n in names:
        for part in n.split("+"):
            c[part] += 1
    shared = [v for v in c.values() if v > 1]
    return {"n_classes": float(len(names)),
            "n_distinct_parts": float(len(c)),
            "parts_in_more_than_one_class": float(len(shared)),
            "mean_classes_per_part": float(np.mean(list(c.values())))}


# ---------------------------------------------------------------------------
# A V2 that stays on the map
# ---------------------------------------------------------------------------
class LocalV2:
    """Conjunction units that live **at a place** and bind only nearby parts.

    The layer is a topographic grid of ``grid x grid`` sites. Each site holds
    ``n_units`` conjunction cells; the cell with index ``u`` at every site shares
    one prototype relation, exactly as one V1 filter is repeated across the
    retina. A cell binds two V1 features only if both fall inside its own
    ``radius`` of hypercolumns -- there is no global pairing anywhere in the
    layer."""

    def __init__(self, n_units: int = 64, grid: int = 4, radius: float = 2.0,
                 max_offset: int = 3, sigma_pos: float = 1.0,
                 sigma_ang: float = 0.45, window_ms: int = 50, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n_units, self.grid = int(n_units), int(grid)
        self.radius = float(radius)
        self.max_offset = int(max_offset)
        self.sigma_pos, self.sigma_ang = float(sigma_pos), float(sigma_ang)
        self.window_ms = int(window_ms)
        self.rng = rng
        self.oa = rng.uniform(0, np.pi, self.n_units).astype(np.float32)
        self.ob = rng.uniform(0, np.pi, self.n_units).astype(np.float32)
        self.dr = rng.uniform(-max_offset, max_offset, self.n_units).astype(np.float32)
        self.dc = rng.uniform(-max_offset, max_offset, self.n_units).astype(np.float32)
        self.duty = np.full(self.n_units, 1.0 / self.n_units, np.float32)
        self.n_out = self.n_units * self.grid * self.grid
        self.pop = Population(self.n_out, "regular_spiking", rng=rng,
                              jitter=0.02)
        self.i_floor, self.i_span = 8.0, 30.0

    # -- pairs, and where they sit -----------------------------------------
    def _pairs(self, act, n_rows: int, n_cols: int):
        n = len(act)
        if n < 2:
            return None
        dR = act.row[None, :] - act.row[:, None]
        dC = act.col[None, :] - act.col[:, None]
        near = (np.abs(dR) <= self.max_offset) & (np.abs(dC) <= self.max_offset)
        np.fill_diagonal(near, False)
        ii, jj = np.nonzero(near)
        if not len(ii):
            return None
        mr = 0.5 * (act.row[ii] + act.row[jj])
        mc = 0.5 * (act.col[ii] + act.col[jj])
        return ii, jj, dR[ii, jj].astype(np.float32), dC[ii, jj].astype(
            np.float32), mr.astype(np.float32), mc.astype(np.float32)

    def _site_centres(self, n_rows: int, n_cols: int) -> np.ndarray:
        g = self.grid
        r = (np.arange(g) + 0.5) * n_rows / g
        c = (np.arange(g) + 0.5) * n_cols / g
        return np.stack(np.meshgrid(r, c, indexing="ij"), -1).reshape(-1, 2)

    def match(self, act, n_rows: int = 7, n_cols: int = 7) -> np.ndarray:
        """Drive of every (unit, site) pair. Shape ``(n_units * grid * grid,)``.

        A pair contributes to a site only through a Gaussian window on the
        distance between the pair's midpoint and the site centre, so a feature
        far away simply cannot reach -- which is what "topographic" means."""
        got = self._pairs(act, n_rows, n_cols)
        if got is None:
            return np.zeros(self.n_out, np.float32)
        ii, jj, dr, dc, mr, mc = got
        w = np.sqrt(act.rate[ii] * act.rate[jj])
        w = w / (w.max() + 1e-9)

        from .v2binding import _wrap
        da = _wrap(act.orient[ii][None, :] - self.oa[:, None])
        db = _wrap(act.orient[jj][None, :] - self.ob[:, None])
        pr = dr[None, :] - self.dr[:, None]
        pc = dc[None, :] - self.dc[:, None]
        rel = (np.exp(-(da ** 2) / (2 * self.sigma_ang ** 2))
               * np.exp(-(db ** 2) / (2 * self.sigma_ang ** 2))
               * np.exp(-(pr ** 2 + pc ** 2) / (2 * self.sigma_pos ** 2))
               * w[None, :])                       # (units, pairs)

        sites = self._site_centres(n_rows, n_cols)  # (S, 2)
        d2 = ((mr[None, :] - sites[:, 0:1]) ** 2
              + (mc[None, :] - sites[:, 1:2]) ** 2)
        spatial = np.exp(-d2 / (2 * self.radius ** 2))       # (S, pairs)
        spatial = spatial * (d2 <= (2.5 * self.radius) ** 2)

        # The BEST pair per (unit, site), not the sum: a fragment is either
        # there or not, and summing would let many weak pairs impersonate one
        # good one.
        out = np.max(rel[:, None, :] * spatial[None, :, :], axis=2)
        return out.reshape(-1).astype(np.float32)

    def spikes(self, act, n_rows: int = 7, n_cols: int = 7) -> np.ndarray:
        m = self.match(act, n_rows, n_cols)
        bias = np.repeat(self.duty - 1.0 / self.n_units, self.grid * self.grid)
        m = m - 2.0 * bias
        z = (m - m.mean()) / (m.std() + 1e-6)
        I = ((self.i_floor + self.i_span * np.clip(z, 0.0, 3.0) / 3.0)
             * (z > 0)).astype(np.float32)
        self.pop.reset()
        counts = np.zeros(self.n_out, np.float32)
        for _ in range(self.window_ms):
            counts += self.pop.step(I).astype(np.float32)
        return counts * (1000.0 / self.window_ms)

    def code(self, act, n_rows: int = 7, n_cols: int = 7) -> np.ndarray:
        return _unit(self.spikes(act, n_rows, n_cols))

    # -- growth ------------------------------------------------------------
    def learn(self, act, n_rows: int = 7, n_cols: int = 7,
              lr: float = 0.08) -> None:
        got = self._pairs(act, n_rows, n_cols)
        if got is None:
            return
        ii, jj, dr, dc, mr, mc = got
        from .v2binding import _wrap
        for p in self.rng.permutation(len(ii))[:14]:
            oa, ob = float(act.orient[ii[p]]), float(act.orient[jj[p]])
            da, db = _wrap(oa - self.oa), _wrap(ob - self.ob)
            pr, pc = dr[p] - self.dr, dc[p] - self.dc
            score = (np.exp(-(da ** 2) / (2 * self.sigma_ang ** 2))
                     * np.exp(-(db ** 2) / (2 * self.sigma_ang ** 2))
                     * np.exp(-(pr ** 2 + pc ** 2) / (2 * self.sigma_pos ** 2)))
            win = int(np.argmax(score - 2.0 * (self.duty - 1.0 / self.n_units)))
            self.oa[win] = (self.oa[win] + lr * _wrap(oa - self.oa[win])) % np.pi
            self.ob[win] = (self.ob[win] + lr * _wrap(ob - self.ob[win])) % np.pi
            self.dr[win] += lr * (dr[p] - self.dr[win])
            self.dc[win] += lr * (dc[p] - self.dc[win])
            self.duty *= (1.0 - 0.02)
            self.duty[win] += 0.02

    def train(self, acts: Sequence, n_rows: int = 7, n_cols: int = 7,
              epochs: int = 2, lr: float = 0.08) -> "LocalV2":
        for _ in range(epochs):
            for i in self.rng.permutation(len(acts)):
                self.learn(acts[i], n_rows, n_cols, lr=lr)
        return self


# ---------------------------------------------------------------------------
# The experiment the old V2 could not run
# ---------------------------------------------------------------------------
@dataclass
class CompositeReport:
    """Does binding help when the task actually requires it?"""

    v1_alone: float = 0.0
    v1_plus_v2: float = 0.0
    v2_alone: float = 0.0
    chance: float = 0.0
    part_overlap: Dict[str, float] = field(default_factory=dict)
    n_train: int = 0

    def summary(self) -> str:
        return (f"chance                : {self.chance:.1%}\n"
                f"V1 alone (parts only) : {self.v1_alone:.1%}\n"
                f"V2 alone (relations)  : {self.v2_alone:.1%}\n"
                f"V1 + local V2         : {self.v1_plus_v2:.1%} "
                f"({self.v1_plus_v2 - self.v1_alone:+.1%})")


def composite_binding_experiment(n_v1: int = 1024, n_v2: int = 64,
                                 n_per_class: int = 90, n_pairs: int = 12,
                                 n_develop: int = 400, grid: int = 4,
                                 v2_weight: float = 0.6, seed: int = 0,
                                 verbose: bool = False) -> CompositeReport:
    """The honest test of binding: a task whose classes share their parts.

    V1 is developed on the composites themselves (self-organized, as in
    :mod:`selforganize`), then V2 grows on V1's output, and the two are scored
    with the same read-out used everywhere else in the project."""
    from .selforganize import randomise_filters, develop_v1
    from .v2binding import v1_parts, preferred_orientations
    from .widev1 import grown_readout

    def say(*a):
        if verbose:
            print(*a)

    X, y, names = composite_dataset(n_per_class=n_per_class, n_pairs=n_pairs,
                                    seed=seed)
    cut = int(0.7 * len(X))
    Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[cut:], y[cut:]

    rep = CompositeReport()
    rep.chance = float(np.bincount(yte).max() / len(yte))
    rep.part_overlap = part_overlap_report(names)
    rep.n_train = len(Xtr)
    say(f"   {len(names)} classes of overlapping pairs; every part appears in "
        f"{rep.part_overlap['mean_classes_per_part']:.1f} classes on average")

    say("V1 develops on the composites ...")
    v1 = randomise_filters(WideV1(n_cells=n_v1, seed=seed), seed=seed)
    develop_v1(v1, Xtr[:n_develop], epochs=3, seed=seed)
    orient = preferred_orientations(v1)

    say("local V2 grows on V1's output ...")
    v2 = LocalV2(n_units=n_v2, grid=grid, seed=seed)
    acts_tr = [v1_parts(v1, im, orient) for im in Xtr]
    acts_te = [v1_parts(v1, im, orient) for im in Xte]
    v2.train(acts_tr[:n_develop], v1.n_rows, v1.n_cols, epochs=2)

    say("scoring ...")
    A1 = np.array([_unit(v1.rate(im)) for im in Xtr], np.float32)
    B1 = np.array([_unit(v1.rate(im)) for im in Xte], np.float32)
    A2 = np.array([v2.code(a, v1.n_rows, v1.n_cols) for a in acts_tr], np.float32)
    B2 = np.array([v2.code(a, v1.n_rows, v1.n_cols) for a in acts_te], np.float32)
    both_a = np.array([_unit(np.concatenate([p, v2_weight * q]))
                       for p, q in zip(A1, A2)], np.float32)
    both_b = np.array([_unit(np.concatenate([p, v2_weight * q]))
                       for p, q in zip(B1, B2)], np.float32)

    rep.v1_alone = grown_readout(A1, ytr, B1, yte)[0]
    rep.v2_alone = grown_readout(A2, ytr, B2, yte)[0]
    rep.v1_plus_v2 = grown_readout(both_a, ytr, both_b, yte)[0]
    if verbose:
        print()
        print(rep.summary())
    return rep
