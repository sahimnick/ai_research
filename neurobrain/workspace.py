"""
workspace.py
============

A **shared cortical code** -- the thing that turns a collection of modules into
one brain.

Until now the faculties talked to each other in Python strings: vision returned
``"3"``, the causal graph took ``"cup"``, the self model took ``"lift"``. That is
integration at the level of an API, not at the level of representation. No real
brain passes symbols between areas; areas share a **common neural code**, and
the *name* of something is a late, optional read-out of that code -- not its
substance.

This module provides that common code, in the form of a **global workspace**
(Baars; Dehaene): a single high-dimensional sparse space that every modality
projects into, whose contents are broadcast to all the other faculties.

    ws = GlobalWorkspace(dim=512)
    code = ws.perceive("vision", image_features)   # -> a sparse cortical code
    ws.name(code)                                  # -> the nearest known concept

Design commitments, each with a biological reason:

  * **Codes are primary, names are derived.** :meth:`name` is a nearest-attractor
    read-out. If nothing is close enough, the answer is ``None`` -- the brain's
    honest "I have no word for this", not a forced label.

  * **Every modality lands in the same space.** Each gets a learned random
    projection (a plausible model of long-range cortico-cortical fan-in), then
    the same sparsification. This is what makes a visual code and an auditory
    code *comparable* -- the basis of cross-modal binding.

  * **Sparse and distributed.** k-winners-take-all, as inhibitory microcircuits
    enforce. Similar inputs share active cells, so similarity is meaningful.

  * **Broadcast.** Whatever wins the workspace is made available to every
    subscribed faculty -- the "ignition" of global availability that
    distinguishes conscious access from local processing in Dehaene's account.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v.astype(np.float32) if n < 1e-9 else (v / n).astype(np.float32)


class GlobalWorkspace:
    """One sparse code space shared by every faculty.

    Parameters
    ----------
    dim:        size of the shared cortical code.
    sparsity:   fraction of cells active (k-winners-take-all).
    vigilance:  how close a code must be to a known concept to earn its name.
    """

    def __init__(self, dim: int = 512, sparsity: float = 0.16,
                 vigilance: float = 0.35, seed: int = 0,
                 adapt: bool = True, adapt_rate: float = 0.02):
        # sparsity and adapt both moved after measurement. The shared code used
        # to cost ~9 accuracy points against reading the perceptual code
        # directly, and the obvious suspect -- the *random* projection -- turned
        # out to be innocent: learning it with competitive Hebbian rules made
        # things much WORSE (77.4% -> 49.5%), because that collapses the
        # projection rows onto cluster centroids and destroys the
        # distance-preserving property a random projection has for free.
        #
        # The real causes were the common-mode component (fixed by `adapt`) and
        # a code that was simply too sparse to carry the information. Measured
        # against a ceiling of 82.5%:
        #     sparsity 0.04 -> 69.2%,  0.08 -> 77.4%,  0.16 -> 82.1%,
        #     0.25 -> 75.6%   (so it is an optimum, not "more is better")
        # At dim 1024 and sparsity 0.16 the cost is -0.1 points.
        self.dim = int(dim)
        # Sensory adaptation: subtract a slowly-tracked running mean of each
        # modality's input before projecting. Without it the *common* component
        # of the input -- the part every stimulus shares -- dominates the
        # projection and the same cells win every time. Measured on wide-V1
        # codes: 34.6% of the active workspace cells were shared between
        # unrelated images, and the console read out a single label for
        # everything. With adaptation that falls to 7.8%.
        #
        # This is what adaptation is for in real sensory cortex: cells encode
        # departures from the recent average, not the average itself. It is off
        # by default so existing measurements stay comparable.
        self.adapt = bool(adapt)
        self.adapt_rate = float(adapt_rate)
        self._mean: Dict[str, np.ndarray] = {}
        self.k = max(2, int(dim * sparsity))
        self.vigilance = float(vigilance)
        self.rng = np.random.default_rng(seed)
        self._proj: Dict[str, np.ndarray] = {}      # modality -> projection
        self.concepts: Dict[str, np.ndarray] = {}   # name -> prototype code
        self.counts: Dict[str, int] = {}
        self.content: Optional[np.ndarray] = None   # what is currently "in mind"
        self.source: Optional[str] = None
        self.subscribers: List[Callable[[np.ndarray, Optional[str]], None]] = []

    # -- projecting any modality into the one shared space -------------------
    def _projection(self, modality: str, n_in: int) -> np.ndarray:
        key = f"{modality}:{n_in}"
        if key not in self._proj:
            # sparse random fan-in: a standard model of long-range cortical
            # projections, and it preserves similarity (Johnson-Lindenstrauss)
            P = self.rng.normal(0, 1.0 / np.sqrt(n_in),
                                (self.dim, n_in)).astype(np.float32)
            self._proj[key] = P
        return self._proj[key]

    def learn_projection(self, modality: str, X: np.ndarray, epochs: int = 2,
                         lr: float = 0.05, seed: int = 0) -> "GlobalWorkspace":
        """Shape the cortico-cortical fan-in by experience instead of chance.

        The projection starts random, which is a fair model of the initial
        wiring but a poor model of an adult cortex: long-range projections are
        pruned and strengthened by what actually arrives through them. Here the
        same competitive Hebbian rule used everywhere else in the project is
        applied to the projection itself -- the cells that win for an input move
        their fan-in toward it, and a homeostatic term stops a few cells from
        winning everything and the rest from going dead.

        Nothing here uses labels; it is driven purely by the statistics of what
        the modality sends."""
        rng = np.random.default_rng(seed)
        X = np.asarray(X, np.float32)
        P = self._projection(modality, X.shape[1])
        duty = np.full(self.dim, self.k / self.dim, np.float32)
        target = self.k / self.dim
        for _ in range(int(epochs)):
            for i in rng.permutation(len(X)):
                x = _unit(X[i])
                drive = P @ x - 2.0 * (duty - target)
                idx = np.argpartition(drive, -self.k)[-self.k:]
                P[idx] += lr * (x[None, :] - P[idx])
                n = np.linalg.norm(P[idx], axis=1, keepdims=True)
                P[idx] /= np.maximum(n, 1e-6)
                P[idx] /= np.sqrt(X.shape[1])      # keep the original scale
                duty *= (1.0 - lr)
                duty[idx] += lr
        self._proj[f"{modality}:{X.shape[1]}"] = P
        return self

    def encode(self, modality: str, x: np.ndarray) -> np.ndarray:
        """Project a modality-specific feature vector into the shared code."""
        x = np.asarray(x, np.float32).reshape(-1)
        if self.adapt:
            key = f"{modality}:{x.size}"
            m = self._mean.get(key)
            if m is None:
                m = np.zeros_like(x)
            self._mean[key] = ((1.0 - self.adapt_rate) * m
                               + self.adapt_rate * x).astype(np.float32)
            x = x - self._mean[key]
        drive = self._projection(modality, x.size) @ _unit(x)
        code = np.zeros(self.dim, np.float32)
        idx = np.argpartition(drive, -self.k)[-self.k:]
        code[idx] = np.maximum(drive[idx], 0.0)
        return _unit(code)

    # -- the workspace proper: broadcast ------------------------------------
    def broadcast(self, code: np.ndarray, source: Optional[str] = None
                  ) -> np.ndarray:
        """Put a code in the workspace and make it available to every faculty."""
        self.content = _unit(np.asarray(code, np.float32))
        self.source = source
        for fn in self.subscribers:
            fn(self.content, source)
        return self.content

    def perceive(self, modality: str, x: np.ndarray) -> np.ndarray:
        """Encode and broadcast in one step -- the normal path for a percept."""
        return self.broadcast(self.encode(modality, x), source=modality)

    def subscribe(self, fn: Callable[[np.ndarray, Optional[str]], None]) -> None:
        self.subscribers.append(fn)

    # -- naming is a READ-OUT of the code, not its substance ----------------
    def learn_concept(self, name: str, code: np.ndarray, lr: float = 0.25
                      ) -> np.ndarray:
        """Attach a name to a region of the code space (or refine it)."""
        c = _unit(np.asarray(code, np.float32))
        if name in self.concepts:
            self.concepts[name] = _unit(self.concepts[name] + lr * (c - self.concepts[name]))
            self.counts[name] += 1
        else:
            self.concepts[name] = c
            self.counts[name] = 1
        return self.concepts[name]

    def match(self, code: np.ndarray) -> Tuple[Optional[str], float]:
        """Nearest known concept and its similarity."""
        if not self.concepts:
            return None, 0.0
        c = _unit(np.asarray(code, np.float32))
        best, score = None, -np.inf
        for name, proto in self.concepts.items():
            s = float(c @ proto)
            if s > score:
                best, score = name, s
        return best, float(score)

    def name(self, code: Optional[np.ndarray] = None) -> Optional[str]:
        """The word for this code -- or ``None`` when nothing fits well enough.

        Returning ``None`` matters: it is the system declining to force a label
        onto something it does not recognise (the same honesty as the OOD
        'don't know' in :mod:`realworld`)."""
        c = self.content if code is None else code
        if c is None:
            return None
        best, score = self.match(c)
        return best if score >= self.vigilance else None

    # -- consolidation: collapse redundant concepts -------------------------
    def consolidate(self, similarity: float = 0.75) -> int:
        """Merge concepts whose codes have become indistinguishable.

        A vigilance-based system that meets noisy input keeps spawning new
        concepts for what is really one thing -- over-segmentation. The brain's
        answer is **consolidation**: offline, redundant traces for the same
        thing are collapsed into a single, better-estimated invariant (this is
        the cortical half of hippocampal replay). Returns how many merges
        happened."""
        merged = 0
        changed = True
        while changed:
            changed = False
            names = list(self.concepts)
            for i, a in enumerate(names):
                if a not in self.concepts:
                    continue
                for b in names[i + 1:]:
                    if b not in self.concepts or a not in self.concepts:
                        continue
                    if float(self.concepts[a] @ self.concepts[b]) >= similarity:
                        ca, cb = self.counts[a], self.counts[b]
                        self.concepts[a] = _unit(ca * self.concepts[a]
                                                 + cb * self.concepts[b])
                        self.counts[a] = ca + cb
                        del self.concepts[b]
                        del self.counts[b]
                        merged += 1
                        changed = True
        return merged

    # -- cross-modal binding lives naturally in a shared space --------------
    def bind(self, name: str, **modal_features: np.ndarray) -> np.ndarray:
        """Bind several modalities' views of the same thing into one concept
        code: the superposition of their projections."""
        codes = [self.encode(m, v) for m, v in modal_features.items()]
        joint = _unit(np.sum(codes, axis=0))
        return self.learn_concept(name, joint)

    def translate(self, from_modality: str, x: np.ndarray) -> Optional[str]:
        """Perceive in one modality and recognise the (possibly cross-modally
        learned) concept -- hearing something and knowing what it looks like."""
        return self.name(self.encode(from_modality, x))

    # -- diagnostics ---------------------------------------------------------
    @property
    def n_concepts(self) -> int:
        return len(self.concepts)

    def separation(self) -> float:
        """Mean pairwise dissimilarity of the learned concept codes (1 = all
        orthogonal). A workspace whose concepts collapse together is useless."""
        if len(self.concepts) < 2:
            return 1.0
        M = np.array([c for c in self.concepts.values()], np.float32)
        S = M @ M.T
        off = S[~np.eye(len(M), dtype=bool)]
        return float(1.0 - off.mean())


@dataclass
class GroundedConcept:
    """A concept as the brain holds it: a **code** first, with the name and the
    modality-specific evidence hanging off it."""

    code: np.ndarray
    name: Optional[str] = None
    modalities: Dict[str, np.ndarray] = field(default_factory=dict)
    properties: Dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        n = self.name if self.name else "<unnamed>"
        return f"GroundedConcept({n}, {len(self.modalities)} modalities)"


def workspace_binding_demo(n_concepts: int = 12, dim: int = 512,
                           noise: float = 0.6, seed: int = 0,
                           verbose: bool = False) -> Dict[str, float]:
    """Measure the two things a shared code must actually deliver:

    1. **cross-modal binding** -- learn a concept from vision AND sound together,
       then recognise it from *either* modality alone;
    2. **honest unfamiliarity** -- refuse to name something never learned.
    """
    rng = np.random.default_rng(seed)
    ws = GlobalWorkspace(dim=dim, seed=seed)
    vdim, adim = 64, 48
    vis = rng.normal(0, 1, (n_concepts, vdim)).astype(np.float32)
    aud = rng.normal(0, 1, (n_concepts, adim)).astype(np.float32)
    names = [f"c{i}" for i in range(n_concepts)]

    for i, nm in enumerate(names):                     # bind the two views
        for _ in range(12):
            ws.bind(nm,
                    vision=vis[i] + rng.normal(0, noise * 0.5, vdim).astype(np.float32),
                    sound=aud[i] + rng.normal(0, noise * 0.5, adim).astype(np.float32))

    # recognise from ONE modality (the other is missing)
    vok = aok = 0
    trials = 300
    for _ in range(trials):
        i = int(rng.integers(n_concepts))
        v = vis[i] + rng.normal(0, noise, vdim).astype(np.float32)
        a = aud[i] + rng.normal(0, noise, adim).astype(np.float32)
        vok += int(ws.name(ws.encode("vision", v)) == names[i])
        aok += int(ws.name(ws.encode("sound", a)) == names[i])

    # something never learned should get NO name
    unknown = sum(ws.name(ws.encode("vision",
                                    rng.normal(0, 1, vdim).astype(np.float32))) is None
                  for _ in range(200))

    out = {"vision_only_recognition": vok / trials,
           "sound_only_recognition": aok / trials,
           "unknown_rejected": unknown / 200,
           "concept_separation": ws.separation(),
           "n_concepts": float(ws.n_concepts)}
    if verbose:
        print(f"  bound {n_concepts} concepts across vision+sound")
        print(f"  recognise from VISION alone : {out['vision_only_recognition']:.0%}")
        print(f"  recognise from SOUND alone  : {out['sound_only_recognition']:.0%}")
        print(f"  never-seen thing -> no name : {out['unknown_rejected']:.0%}")
        print(f"  concept code separation     : {out['concept_separation']:.2f}")
    return out
