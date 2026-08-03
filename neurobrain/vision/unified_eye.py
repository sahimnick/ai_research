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

from .objectness import feature_report, propose, segment
from .ventral import build_ventral_stream_on, locate_template


@dataclass
class Tag:
    """What to look for: a spatial patch of an area's map, per area."""
    name: str
    templates: Dict[str, np.ndarray]
    box: Tuple[float, float, float, float]      # (y, x, h, w) it came from
    #: Where the object's centre sits inside its own template, in cells, per
    #: area. The crop floors one edge and ceils the other, so the template's
    #: geometric middle is NOT the object's centre -- reporting the middle
    #: biases every position by up to half a cell in a fixed direction, which
    #: is a systematic error rather than noise and shows up in a render as a
    #: box that is consistently off to one side.
    centre: Dict[str, Tuple[float, float]] = field(default_factory=dict)


@dataclass
class Percept:
    """Everything one look yields. All of it from a single forward pass."""
    where: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    confidence: Dict[str, float] = field(default_factory=dict)
    #: Tags the eye looked for and was **not confident enough** to report. A
    #: tracker that always answers turns every weak frame into a false
    #: detection; §9.35 measured the alternative -- on held-out sequences,
    #: abstaining below 0.35 lifts precision from 0.848 to 0.910 at half the
    #: frames answered.
    lost: Dict[str, float] = field(default_factory=dict)
    #: Angular size is a **frame-level** estimate, not per-tag: the probe reads
    #: the whole area map, so every tag in a frame would get the same number.
    #: Reported as one value rather than duplicated per name, which the first
    #: version did and which read as four independent measurements of one thing.
    frame_size: Optional[float] = None
    #: Distinctive regions the eye found **whether or not they were tagged** --
    #: objectness, not detection. No labels: naming one needs a classifier
    #: trained on labelled objects and this project has none.
    proposals: List[dict] = field(default_factory=list)
    #: A labelling of the area map by what each cell responds to. Feature
    #: segmentation, closer to superpixels than to YOLO's instance masks.
    segments: Optional[np.ndarray] = None
    relations: Dict[Tuple[str, str], str] = field(default_factory=dict)
    attention: Dict[str, np.ndarray] = field(default_factory=dict)
    maps: Dict[str, np.ndarray] = field(default_factory=dict)

    def describe(self) -> str:
        bits = []
        for n, (y, x) in sorted(self.where.items()):
            bits.append(f"{n} at ({y:.0f},{x:.0f}) "
                        f"conf {self.confidence[n]:.2f}")
        for n, c in sorted(self.lost.items()):
            bits.append(f"{n} LOST ({c:.2f})")
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
                 where_area: str = "pool", size_area: str = "V4",
                 min_confidence: float = 0.35, search_radius: float = 40.0):
        self.size = int(size)
        self.areas = tuple(areas)
        self.where_area = where_area
        self.size_area = size_area
        #: Below this correlation peak the eye reports the tag as **lost**
        #: rather than guessing. §9.35's held-out curve, with no threshold
        #: chosen from the data it is reported on: 0.25 -> 0.902 at 70%
        #: coverage, 0.30 -> 0.900 at 61%, **0.35 -> 0.910 at 51%**. Answering
        #: every frame instead scores 0.848. Above 0.40 precision falls again
        #: (0.812 at 0.50) as the surviving frames get few and noisy.
        self.min_confidence = float(min_confidence)
        #: Search this many pixels around where the tag was last seen. §9.35:
        #: 0.814 with a local window against 0.773 searching the whole frame,
        #: on consecutive frames. Set to None to always search everything.
        self.search_radius = search_radius
        self.stream = None
        self.tags: Dict[str, Tag] = {}
        self.last: Dict[str, Tuple[float, float]] = {}
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
        tmpl, ctr = {}, {}
        for a in self.areas:
            tmpl[a] = self._crop(M[a], y, x, bh, bw)
            _, gh, gw = M[a].shape
            r0 = int(np.clip(y * gh / self.size, 0, gh - 2))
            c0 = int(np.clip(x * gw / self.size, 0, gw - 2))
            # the object's true centre, in this template's own cell frame
            ctr[a] = ((y + bh / 2) * gh / self.size - r0,
                      (x + bw / 2) * gw / self.size - c0)
        t = Tag(name, tmpl, (float(y), float(x), float(bh), float(bw)), ctr)
        self.tags[name] = t
        self.last[name] = (float(y) + float(bh) / 2, float(x) + float(bw) / 2)
        return t

    # ------------------------------------------------------------------ look
    def features(self, frame: Optional[np.ndarray] = None) -> Dict[str, dict]:
        """What each area is tuned to, **probed** with gratings, as a list.

        See :func:`~neurobrain.vision.objectness.feature_report`. Pass a frame
        to also get which channels that picture drives hardest.
        """
        if self.stream is None:
            raise RuntimeError("develop() the eye first")
        return feature_report(self.stream, self.areas, img_size=self.size,
                              frame=self._prep(frame)
                              if frame is not None else None)

    def look(self, frame: np.ndarray, cue: Optional[str] = None,
             relations: bool = True, n_proposals: int = 0,
             n_segments: int = 0) -> Percept:
        """One pass; every read-out below comes out of it.

        ``cue`` restricts the attention map to one tag -- the top-down half of
        §9.28. Without it, an attention map is returned per tag.

        Two behaviours set in the constructor and measured in §9.35, both of
        which matter far more than anything else about how this performs:

        * the search is restricted to ``search_radius`` px around where the tag
          was last seen, since a thing does not usually teleport;
        * a tag whose best match falls below ``min_confidence`` is reported in
          ``percept.lost`` rather than ``percept.where``, because a box drawn on
          a 0.19 match is a false detection, not a weak one.

        The dominant variable is how far the target moves between calls: 0.814
        at ~16 px/step and 0.194 at ~63 px/step. Call this often.
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
            th, tw = tm.shape[1:]
            _, gh, gw = m.shape
            r0 = c0 = 0
            sub = m
            prev = self.last.get(n)
            if self.search_radius is not None and prev is not None:
                rad = self.search_radius
                r0 = int(np.clip((prev[0] - rad) * gh / self.size, 0,
                                 max(0, gh - th)))
                c0 = int(np.clip((prev[1] - rad) * gw / self.size, 0,
                                 max(0, gw - tw)))
                r1 = int(np.clip((prev[0] + rad) * gh / self.size, r0 + th, gh))
                c1 = int(np.clip((prev[1] + rad) * gw / self.size, c0 + tw, gw))
                sub = m[:, r0:r1, c0:c1]
                if sub.shape[1] < th or sub.shape[2] < tw:
                    r0 = c0 = 0
                    sub = m
            (r, c), sc = locate_template(sub, tm)
            conf = float(sc.max())
            # the object's centre, not the template's middle -- see Tag.centre
            oy, ox = t.centre.get(self.where_area,
                                  ((th - 1) / 2.0, (tw - 1) / 2.0))
            pos = ((r0 + r + oy) * self.size / gh,
                   (c0 + c + ox) * self.size / gw)
            g = np.maximum(sc, 0.0)
            pk = g.max()
            g = g / pk if pk > 1e-9 else g
            # Put the correlation back where in the FULL map it was computed.
            # Without this a restricted search returns a sub-window-sized map
            # that any viewer stretches across the whole frame, so the heat
            # lands somewhere the eye never looked -- the box and the heatmap
            # disagreed on screen, and the heatmap was the one lying.
            full = np.zeros((gh, gw), np.float32)
            full[r0:r0 + g.shape[0], c0:c0 + g.shape[1]] = g
            p.attention[n] = full
            if conf < self.min_confidence:
                p.lost[n] = conf          # say nothing rather than guess
                continue
            p.where[n] = pos
            p.confidence[n] = conf
            self.last[n] = pos

        if n_proposals:
            p.proposals = propose(M[self.where_area], self.size,
                                  k=int(n_proposals))
        if n_segments:
            p.segments = segment(M[self.where_area], k=int(n_segments))
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
