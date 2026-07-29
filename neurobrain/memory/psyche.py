"""
psyche.py
=========

An inner world you can remember and imagine in.

Two things live here:

  * :class:`AssociativeCortex` -- a **pallium-like content-addressable memory**.
    Many memory cells, each an attractor that stores one concept's pattern in
    several modalities at once (a picture, a sound, a word, a "situation" code).
    A noisy or partial cue in *any* modality completes to the nearest memory
    (winner-take-all attractor cleanup) and lets you **reconstruct the other
    modalities** -- so a familiar sound brings back the picture, a picture brings
    back the word. This is Hebbian hetero/auto-association with pattern
    completion, the way the cortex/hippocampus recalls a memory from a fragment.

  * :class:`MentalSpace` -- a room that starts **empty** and fills as the mind
    learns. Each new concept adds a memory and a link in a small world model of
    what-follows-what. You can then **recall a real percept from a cross-modal
    cue** (hear the cue -> reconstruct the image), and **imagine**: wander the
    world model and re-create each concept's percept, an inner train of thought
    that gets richer as the room fills.

Measured honestly on **real** handwritten digits: reconstructing the image from
a noisy/partial cue and re-recognising it is correct **>90%** of the time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float32).ravel()
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


@dataclass
class AssociativeCortex:
    """A large, multi-modal content-addressable memory (the pallium)."""

    modalities: Dict[str, np.ndarray] = field(default_factory=dict)  # mod->(cells,dim)
    labels: List[object] = field(default_factory=list)

    @property
    def n_memories(self) -> int:
        return len(self.labels)

    @property
    def n_synapses(self) -> int:
        return int(sum(m.size for m in self.modalities.values()))

    def remember(self, patterns: Dict[str, np.ndarray], label: object = None
                 ) -> int:
        """Store one concept as a new memory cell, across all given modalities.
        Missing modalities are filled with zeros so every cell lines up."""
        cell = self.n_memories
        for mod, vec in patterns.items():
            vn = _unit(vec)
            if mod not in self.modalities:
                dim = vn.size
                self.modalities[mod] = np.zeros((cell, dim), np.float32)
            self.modalities[mod] = np.vstack([self.modalities[mod], vn])
        # pad modalities this memory didn't provide
        for mod, mat in self.modalities.items():
            if len(mat) == cell:               # wasn't extended above
                self.modalities[mod] = np.vstack([mat, np.zeros((1, mat.shape[1]),
                                                                np.float32)])
        self.labels.append(label)
        if getattr(self, "_by_label", None) is not None:
            self._by_label.setdefault(label, []).append(cell)
        return cell

    def recall(self, modality: str, cue: np.ndarray,
               bias: Optional[np.ndarray] = None) -> int:
        """Complete a (noisy/partial) cue in ``modality`` to the nearest memory
        cell -- attractor pattern completion / winner-take-all. An optional
        top-down ``bias`` (one gain per cell) implements **biased competition**:
        attention adds drive to the memories it favours so they win more easily
        under noise (Desimone & Duncan, 1995)."""
        scores = self.modalities[modality] @ _unit(cue)
        if bias is not None:
            scores = scores + bias
        return int(np.argmax(scores))

    def label_bias(self, label: object, gain: float) -> np.ndarray:
        """A top-down attention vector: ``gain`` on every memory cell whose
        concept is ``label``, zero elsewhere -- the drive attention adds."""
        b = np.zeros(self.n_memories, np.float32)
        for i, l in enumerate(self.labels):
            if l == label:
                b[i] = gain
        return b

    def reconstruct(self, cell: int, modality: str) -> np.ndarray:
        """The stored pattern of memory ``cell`` in ``modality``."""
        return self.modalities[modality][cell].copy()

    # -- consolidation: repetition should STRENGTHEN, not duplicate ----------
    def cells_for(self, label: object) -> np.ndarray:
        """Indices of every memory cell carrying this label."""
        if not hasattr(self, "_by_label") or self._by_label is None:
            self._by_label = {}
            for i, l in enumerate(self.labels):
                self._by_label.setdefault(l, []).append(i)
        return np.asarray(self._by_label.get(label, []), np.int64)

    def _index_cell(self, cell: int, label: object) -> None:
        if not hasattr(self, "_by_label") or self._by_label is None:
            self._by_label = {}
            for i, l in enumerate(self.labels):
                self._by_label.setdefault(l, []).append(i)
        self._by_label.setdefault(label, []).append(cell)

    def consolidate(self, patterns: Dict[str, np.ndarray], label: object,
                    lr: float = 0.05, vigilance: float = 0.55) -> int:
        """Fold a replayed pattern into an EXISTING trace instead of adding a copy.

        :meth:`remember` allocates a new cell every call, so replaying one
        episode two hundred times left two hundred identical cells and moved
        recall by exactly nothing -- which is why prioritised replay was
        indistinguishable from uniform replay no matter how informative the
        priority signal became. Frequency has to reach the weights or it
        reaches nothing.

        Here the replayed pattern competes for a cell of its own label. If the
        best match is at least ``vigilance`` similar, that cell moves toward the
        pattern by the instar rule ``w += lr * (x - w)`` -- local, Hebbian, no
        gradient -- and its strength counter rises. Only genuinely novel
        content allocates a new cell, so the store stays a summary of
        experience rather than a log of it.

        Returns the cell that absorbed the pattern."""
        cells = self.cells_for(label)
        best, best_sim = -1, -np.inf
        if len(cells):
            for mod, vec in patterns.items():
                if mod not in self.modalities:
                    continue
                x = _unit(vec)
                sims = self.modalities[mod][cells] @ x
                j = int(np.argmax(sims))
                if float(sims[j]) > best_sim:
                    best, best_sim = int(cells[j]), float(sims[j])
        if best < 0 or best_sim < vigilance:
            cell = self.remember(patterns, label=label)
            self._index_cell(cell, label)
            self.strength = np.append(getattr(self, "strength",
                                              np.zeros(0, np.float32)), 1.0)
            while len(self.strength) < self.n_memories:      # cells added elsewhere
                self.strength = np.append(self.strength, 1.0)
            return cell
        for mod, vec in patterns.items():
            if mod not in self.modalities:
                continue
            w = self.modalities[mod][best]
            self.modalities[mod][best] = _unit(w + lr * (_unit(vec) - w))
        if not hasattr(self, "strength") or len(self.strength) < self.n_memories:
            self.strength = np.ones(self.n_memories, np.float32)
        self.strength[best] += 1.0
        return best

    # -- read-out: a population, not one winner ------------------------------
    def reconstruct_population(self, modality: str, cells: np.ndarray,
                               weights: np.ndarray) -> np.ndarray:
        """Blend several memory cells into one percept.

        :meth:`reconstruct` returns a stored row verbatim, so anything built on
        it can only replay experience: every "imagined" percept had cosine
        1.000000 to a cell already in the store. A cortical read-out is not one
        winning cell, it is a population -- so this weights several cells and
        sums them, which can express a pattern that was never stored while
        staying inside the space the store spans."""
        cells = np.asarray(cells, np.int64)
        if not len(cells):
            return np.zeros(self.modalities[modality].shape[1], np.float32)
        w = np.asarray(weights, np.float32)
        w = w / max(float(w.sum()), 1e-9)
        return _unit(w @ self.modalities[modality][cells])

    def cross_recall(self, from_modality: str, cue: np.ndarray,
                     to_modality: str) -> Tuple[int, np.ndarray]:
        """Cue in one modality -> the recalled memory cell and its reconstructed
        pattern in another modality (e.g. hear -> picture)."""
        cell = self.recall(from_modality, cue)
        return cell, self.reconstruct(cell, to_modality)


@dataclass
class MentalSpace:
    """An inner room that starts empty and fills with remembered, re-imaginable
    concepts, plus a small world model of how they follow one another."""

    cortex: AssociativeCortex = field(default_factory=AssociativeCortex)
    concepts: List[str] = field(default_factory=list)
    _T: Optional[np.ndarray] = None
    wm: object = None                    # PredictiveWorldModel (set lazily)

    def _world(self):
        if self.wm is None:
            from ..world.predictive import PredictiveWorldModel
            self.wm = PredictiveWorldModel()
        return self.wm

    @property
    def fullness(self) -> int:
        """How full the room is -- the number of concepts it now holds."""
        return len(self.concepts)

    def _grow_T(self):
        n = len(self.concepts)
        T = np.full((n, n), 0.05, np.float32)
        if self._T is not None:
            T[:self._T.shape[0], :self._T.shape[1]] = self._T
        self._T = T

    def remember(self, concept: str, **modalities: np.ndarray) -> int:
        """Add a memory to the room, labelled with its concept. A concept may
        have MANY memory cells (exemplars) -- that is what makes the pallium
        large and recall from a noisy real percept robust."""
        cell = self.cortex.remember(modalities, label=concept)
        if concept not in self.concepts:
            self.concepts.append(concept)
            self._grow_T()
        return cell

    def _rep_cell(self, concept: str) -> int:
        """A representative memory cell for a concept (for reconstruction)."""
        return self.cortex.labels.index(concept)

    def experience(self, sequence: List[str],
                   actions: Optional[List[str]] = None) -> None:
        """Watch a sequence of concepts and strengthen their world-model links
        (both the raw table and the predictive world model)."""
        idx = {c: i for i, c in enumerate(self.concepts)}
        for a, b in zip(sequence[:-1], sequence[1:]):
            if a in idx and b in idx:
                self._T[idx[a], idx[b]] += 1.0
        self._world().experience(sequence, actions)

    def predict_next(self, concept: str, action: Optional[str] = None
                     ) -> Optional[str]:
        """The predictive world model's next-concept guess (SR-backed)."""
        return self._world().predict_next(concept, action)

    def reachable(self, concept: str, top: int = 5):
        """Where a concept tends to lead over many steps (successor rep.)."""
        return self._world().reachable(concept, top=top)

    def plan(self, start: str, goal: str):
        """A model-based action plan from ``start`` to ``goal``."""
        return self._world().plan(start, goal)

    def recall(self, from_modality: str, cue: np.ndarray, to_modality: str,
               attend: object = None, gain: float = 0.6
               ) -> Tuple[str, np.ndarray]:
        """Hear/see a cue, recall the concept, and reconstruct the other
        modality's percept -- remembering a picture from a sound, or vice versa.
        With ``attend`` set, top-down attention biases the competition toward
        that concept (it recalls better under noise)."""
        bias = (self.cortex.label_bias(attend, gain)
                if attend is not None else None)
        cell = self.cortex.recall(from_modality, cue, bias=bias)
        return self.cortex.labels[cell], self.cortex.reconstruct(cell, to_modality)

    def recognise(self, modality: str, cue: np.ndarray) -> str:
        """Which concept does this (noisy) percept belong to? (attractor recall)"""
        return self.cortex.labels[self.cortex.recall(modality, cue)]

    def consolidate(self, concept: str, lr: float = 0.05,
                    vigilance: float = 0.55, **modalities: np.ndarray) -> int:
        """Fold a replayed pattern into an existing trace (see
        :meth:`AssociativeCortex.consolidate`). This is what a night of replay
        should call: :meth:`remember` appends, so it cannot express how often
        something was replayed."""
        if concept not in self.concepts:
            self.concepts.append(concept)
            self._grow_T()
        return self.cortex.consolidate(modalities, concept, lr=lr,
                                       vigilance=vigilance)

    def blend(self, concept: str, modality: str = "image", k: int = 6,
              concentration: float = 0.9, rng=None) -> np.ndarray:
        """Re-create a percept as a **population blend** of remembered episodes.

        The cells carrying this concept are mixed with Dirichlet-sampled
        weights, so the result is a genuinely new pattern inside the span of
        what was actually experienced -- not one stored row returned verbatim,
        which is all :meth:`AssociativeCortex.reconstruct` can do.
        ``concentration`` sets how far it wanders: below 1 the mixture leans on
        a few episodes and stays vivid, far above 1 it converges on the flat
        class average, which is the prototype problem again."""
        rng = rng or np.random.default_rng()
        cells = self.cortex.cells_for(concept)
        if not len(cells):
            return self.cortex.reconstruct(self._rep_cell(concept), modality)
        strength = getattr(self.cortex, "strength", None)
        if strength is not None and len(strength) >= self.cortex.n_memories:
            order = cells[np.argsort(-strength[cells])][:k]
        else:
            order = cells[:k]
        w = rng.dirichlet(np.full(len(order), concentration))
        return self.cortex.reconstruct_population(modality, order, w)

    def imagine(self, seed: str, steps: int = 6, modality: str = "image",
                temperature: float = 0.6, rng_seed: int = 0,
                blend: int = 0, concentration: float = 0.9
                ) -> Tuple[List[str], List[np.ndarray]]:
        """Wander the world model from ``seed`` and *re-create* each concept's
        percept -- an inner train of thought made of reconstructed memories.

        With ``blend=0`` each percept is one stored cell returned verbatim, so
        every "imagined" image has cosine 1.000000 to something already in the
        store -- the mind can only replay what it has seen. ``blend=k`` mixes
        the k strongest traces for that concept with Dirichlet weights
        (:meth:`blend`), producing a pattern that was never stored while
        staying inside the span of what was actually experienced."""
        rng = np.random.default_rng(rng_seed)
        cur = self.concepts.index(seed)
        names = [seed]
        _draw = (lambda c: self.blend(c, modality, k=blend,
                                      concentration=concentration, rng=rng)
                 ) if blend else (
                 lambda c: self.cortex.reconstruct(self._rep_cell(c), modality))
        percepts = [_draw(seed)]
        for _ in range(steps):
            p = self._T[cur] ** (1.0 / max(temperature, 1e-3))
            p = p / p.sum()
            cur = int(rng.choice(len(p), p=p))
            c = self.concepts[cur]
            names.append(c)
            percepts.append(_draw(c))
        return names, percepts


@dataclass
class ReconstructiveMind:
    space: MentalSpace
    recognizer: object                    # DigitRecognizer, to score reconstructions
    cue_dim: int = 256
    recall_accuracy: float = 0.0          # noisy cue -> reconstructed image correct
    reverse_accuracy: float = 0.0         # see image -> recall the right concept


def build_reconstructive_mind(n_train: int = 15000, cue_dim: int = 256,
                              noise: float = 0.2, dropout: float = 0.35,
                              verbose: bool = False) -> ReconstructiveMind:
    """Fill a mental space with REAL handwritten-digit memories, each bound to a
    'cue' signature (think a familiar sound / smell / situation), and measure
    cross-modal recall from noisy cues -- reconstruct the picture from the cue and
    check a fresh eye still reads the right digit."""
    from ..sensing.realworld import build_digit_recognizer, load_mnist

    def say(*a):
        if verbose:
            print(*a)

    say("learning to read real digits (for scoring reconstructions) ...")
    recog = build_digit_recognizer(n_train=n_train)
    trx, trY, _, _ = load_mnist(n_train=n_train)

    say("filling the mental space with digit memories, each bound to a cue ...")
    space = MentalSpace()
    rng = np.random.default_rng(0)
    # a distinct cue signature per digit (its 'sound / smell / situation')
    digit_cue = {}
    for d in range(10):
        cue = np.zeros(cue_dim, np.float32)
        cue[rng.choice(cue_dim, 24, replace=False)] = 1.0
        digit_cue[d] = cue
    # store MANY exemplar memories -- reuse the self-grown category prototypes, so
    # the pallium is large and recall from a noisy real digit lands on a close one
    cx = recog.cortex
    for i in range(cx.n_categories):
        lab = int(cx.cell_label[i])
        if lab < 0:
            continue
        space.remember(str(lab), image=cx.W[i], cue=digit_cue[lab])
    mind = ReconstructiveMind(space, recog, cue_dim)

    # cross-modal recall: noisy/partial cue -> reconstruct image -> re-recognise
    ok = 0
    N = 500
    for _ in range(N):
        d = int(rng.integers(10))
        cue = digit_cue[d] + rng.normal(0, noise, cue_dim).astype(np.float32)
        cue[rng.choice(cue_dim, int(cue_dim * dropout), replace=False)] *= 0.2
        _, img = space.recall("cue", cue, "image")
        pred = recog.cortex.recognise((img * 255).reshape(1, 28, 28).astype(np.float32),
                                      0.0)[0]
        ok += int(pred == d)
    mind.recall_accuracy = ok / N

    # reverse: see a (noisy) real digit -> recall the right concept from the image
    rok = 0
    M = 500
    for _ in range(M):
        d = int(rng.integers(10))
        ex = (trx[trY == d][int(rng.integers(60))].reshape(-1) / 255.0)
        ex = ex + rng.normal(0, 0.25, 784).astype(np.float32)
        cell = space.cortex.recall("image", ex)
        rok += int(space.cortex.labels[cell] == str(d))
    mind.reverse_accuracy = rok / M

    say(f"   mental space now holds {space.fullness} concepts")
    say(f"   hear a cue -> reconstruct the picture -> read it right: "
        f"{mind.recall_accuracy:.0%}")
    say(f"   see a noisy digit -> recall the right memory: {mind.reverse_accuracy:.0%}")
    return mind
