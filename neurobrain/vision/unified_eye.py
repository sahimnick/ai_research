"""
unified_eye.py
==============

One eye that **runs**: sees, tags, tracks, attends and reports properties, from
a single forward pass per frame.

Everything measured in EVALUATION.md §9.19–§9.33 was a benchmark -- a script
that answers one question and exits. This is the loop those measurements were
about, assembled as something that can be handed a video and left running:

    develop(frames)          grow the four stages on real imagery
    add_tag(name, frame, box)  store what to look for
    look(frame)  ->  Percept   ONE pass; every read-out comes out of it

Built only from mechanisms that measured as working, with the ones that did not
deliberately left out:

    locating a tag        spatial cross-correlation on the area's map, §9.21
                          (0.943), §9.23 (0.834 under real appearance change).
                          NOT the averaged prototype, which §9.21 measured at
                          0.000 -- worse than a deliberately wrong tag.
    selection             the same correlation used as a top-down gain, §9.28
                          (0.917 cue-following, 0.875 switch rate) and §9.29
                          (0.818 on video with the target really changing).
    angular size          a ridge probe on the deepest area, §9.31 (0.585
                          within scenes, above raw pixels' 0.360).
    relations             comparison of two located positions, §9.33 (0.906
                          against a 0.648 positional-prior control).

What this eye is known not to do
--------------------------------
Recorded here because a running system invites more trust than a benchmark
does, and these limits are measured rather than suspected:

* **It does not leave its development distribution.** §9.27: image information
  decodes at R² 0.39–0.47 within the scenes it grew on and **0.00** across
  unseen ones, at every breadth from 2 to 32 scenes and every decoder width
  from 4 to 128. Develop it on the footage you intend to run it on.
* **It localises coarsely, not precisely.** §9.30: removing tracker drift
  entirely lifts success AUC only 0.098 → 0.152, so the limit is per-frame
  precision. Good for *which of these*, not for *exactly where, continuously*.
* **Colour hurts it** (§9.18, −0.135) and **shadow is invisible to it**
  (§9.31: it responds to an illumination change exactly as to a new object).
* **The stages are not shown to specialise.** §9.33 found only one read-out
  with a separated winner. `area=` is exposed per call because the right choice
  is empirical, not because a hierarchy of roles has been established.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .ventral import build_ventral_stream_on, locate_template


@dataclass
class Tag:
    """What to look for: a spatial patch of an area's map, per area."""
    name: str
    templates: Dict[str, np.ndarray]
    box: Tuple[float, float, float, float]      # (y, x, h, w) it came from


@dataclass
class Percept:
    """Everything one look yields. All of it from a single forward pass."""
    where: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    confidence: Dict[str, float] = field(default_factory=dict)
    #: Angular size is a **frame-level** estimate, not per-tag: the probe reads
    #: the whole area map, so every tag in a frame would get the same number.
    #: Reported as one value rather than duplicated per name, which the first
    #: version did and which read as four independent measurements of one thing.
    frame_size: Optional[float] = None
    relations: Dict[Tuple[str, str], str] = field(default_factory=dict)
    attention: Dict[str, np.ndarray] = field(default_factory=dict)
    maps: Dict[str, np.ndarray] = field(default_factory=dict)

    def describe(self) -> str:
        bits = []
        for n, (y, x) in sorted(self.where.items()):
            bits.append(f"{n} at ({y:.0f},{x:.0f}) "
                        f"conf {self.confidence[n]:.2f}")
        if self.frame_size is not None:
            bits.append(f"frame log-size {self.frame_size:.2f}")
        for (a, b), r in sorted(self.relations.items()):
            bits.append(f"{a} is {r} of {b}")
        return "; ".join(bits) if bits else "(nothing tagged)"


class UnifiedEye:
    """The four-stage ventral stream, wired as a loop that can be run.

    ``areas`` are read out of every pass. The default set is the three the
    benchmarks used; ``size_area`` defaults to the deepest because §9.31
    measured angular size best there, and ``where_area`` is left as a parameter
    because §9.33 could not separate the stages at locating.
    """

    def __init__(self, size: int = 224,
                 areas: Sequence[str] = ("V2", "pool", "V4"),
                 where_area: str = "pool", size_area: str = "V4"):
        self.size = int(size)
        self.areas = tuple(areas)
        self.where_area = where_area
        self.size_area = size_area
        self.stream = None
        self.tags: Dict[str, Tag] = {}
        self._size_probe = None

    # ---------------------------------------------------------------- develop
    def develop(self, frames: Sequence[np.ndarray], verbose: bool = False):
        """Grow the stages on real imagery.

        §9.27 is the reason this takes *your* frames: what the eye learns here
        is what it can read later, and it does not generalise past it.
        """
        F = np.stack([self._prep(f) for f in frames])
        self.stream = build_ventral_stream_on(F, size=self.size,
                                              verbose=verbose)
        return self

    def _prep(self, frame: np.ndarray) -> np.ndarray:
        a = np.asarray(frame, np.float32)
        if a.ndim == 3:
            a = a.mean(0) if a.shape[0] in (1, 3) else a.mean(2)
        if a.max() > 1.5:
            a = a / 255.0
        h, w = a.shape
        if (h, w) != (self.size, self.size):
            y = max(0, (h - self.size) // 2)
            x = max(0, (w - self.size) // 2)
            a = a[y:y + self.size, x:x + self.size]
        return np.ascontiguousarray(a, np.float32)

    # ------------------------------------------------------------------ pass
    def _pass(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        if self.stream is None:
            raise RuntimeError("develop() the eye before looking through it")
        h = self.stream.hierarchy
        names = [getattr(l, "name", type(l).__name__) for l in h.layers]
        h.forward(self._prep(frame)[None])
        return {a: np.asarray(h.layers[names.index(a)].log["output"],
                              np.float32) for a in self.areas}

    def _crop(self, m, y, x, bh, bw):
        _, gh, gw = m.shape
        r0 = int(np.clip(y * gh / self.size, 0, gh - 2))
        c0 = int(np.clip(x * gw / self.size, 0, gw - 2))
        r1 = int(np.clip(np.ceil((y + bh) * gh / self.size), r0 + 1, gh))
        c1 = int(np.clip(np.ceil((x + bw) * gw / self.size), c0 + 1, gw))
        return m[:, r0:r1, c0:c1]

    # ------------------------------------------------------------------ tags
    def add_tag(self, name: str, frame: np.ndarray,
                box: Tuple[float, float, float, float]) -> Tag:
        """Store what to look for: the region ``box = (y, x, h, w)``.

        Kept as a **spatial patch** per area. §9.21 measured what averaging it
        to one channel vector costs: 0.000 at finding the object in the very
        frame the tag came from, against 0.943 keeping the layout.
        """
        M = self._pass(frame)
        y, x, bh, bw = box
        t = Tag(name, {a: self._crop(M[a], y, x, bh, bw) for a in self.areas},
                (float(y), float(x), float(bh), float(bw)))
        self.tags[name] = t
        return t

    # ------------------------------------------------------------------ look
    def look(self, frame: np.ndarray, cue: Optional[str] = None,
             relations: bool = True) -> Percept:
        """One pass; every read-out below comes out of it.

        ``cue`` restricts the attention map to one tag -- the top-down half of
        §9.28. Without it, an attention map is returned per tag.
        """
        M = self._pass(frame)
        p = Percept(maps=M)
        want = [cue] if cue is not None else list(self.tags)
        for n in want:
            t = self.tags.get(n)
            if t is None:
                continue
            m = M[self.where_area]
            tm = t.templates[self.where_area]
            if any(a > b for a, b in zip(tm.shape[1:], m.shape[1:])):
                continue
            (r, c), sc = locate_template(m, tm)
            th, tw = tm.shape[1:]
            _, gh, gw = m.shape
            p.where[n] = ((r + (th - 1) / 2.0) * self.size / gh,
                          (c + (tw - 1) / 2.0) * self.size / gw)
            p.confidence[n] = float(sc.max())
            g = np.maximum(sc, 0.0)
            pk = g.max()
            p.attention[n] = g / pk if pk > 1e-9 else g

        if self._size_probe is not None:
            mu, V, W = self._size_probe
            z = (M[self.size_area].ravel() - mu) @ V
            p.frame_size = float(np.ravel(np.append(z, 1.0) @ W)[0])
        if relations:
            for a in p.where:
                for b in p.where:
                    if a >= b:
                        continue
                    (ay, ax), (by, bx) = p.where[a], p.where[b]
                    p.relations[(a, b)] = (
                        "left" if ax < bx else "right") if abs(ax - bx) >= abs(
                        ay - by) else ("above" if ay < by else "below")
        return p

    # ------------------------------------------------------- angular size
    def fit_size(self, frames: Sequence[np.ndarray],
                 log_areas: Sequence[float], dim: int = 48, lam: float = 1.0):
        """Fit the angular-size read-out. §9.31: 0.585 within scenes.

        Ridge on PCA of the deepest area, closed form -- no gradients.
        """
        X = np.stack([self._pass(f)[self.size_area].ravel() for f in frames])
        y = np.asarray(log_areas, np.float32).reshape(-1, 1)
        mu = X.mean(0)
        A = X - mu
        w, U = np.linalg.eigh(A @ A.T)
        idx = np.argsort(w)[::-1][:min(dim, len(X) - 1)]
        V = (A.T @ U[:, idx]) / np.sqrt(np.maximum(w[idx], 1e-9))
        V = np.asarray(V, np.float32)
        Z = np.hstack([A @ V, np.ones((len(X), 1), np.float32)])
        P = np.eye(Z.shape[1], dtype=np.float32)
        P[-1, -1] = 0.0
        W = np.linalg.solve(Z.T @ Z + lam * P, Z.T @ y)
        self._size_probe = (mu, V, W)
        return self

    # ------------------------------------------------------------------ run
    def run(self, frames: Sequence[np.ndarray],
            cue: Optional[str] = None) -> List[Percept]:
        """The loop: look at every frame, keeping the tags fixed."""
        return [self.look(f, cue=cue) for f in frames]
