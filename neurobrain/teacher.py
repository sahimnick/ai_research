"""
teacher.py
==========

Teaching the brain by **induction**, not by the usual method.

The user asked for the first lessons to be given *القایی* (by induction /
imprinting) rather than by ordinary supervised training (backpropagation and a
loss function). The biological way to do that is simple and is what this module
implements:

    Show the stimulus **and, at the same time, make the concept's neurons fire.**
    Because pre-synaptic (sensory) and post-synaptic (concept) cells are active
    together, STDP strengthens the synapses between them -- "cells that fire
    together wire together". After a few repetitions the stimulus *alone* is
    enough to wake the concept. Nothing was back-propagated; the association was
    *induced*.

The same trick, applied to a *sequence* of concepts, teaches the world model
what tends to follow what -- giving the brain its first predictive structure and
bringing it to an initial threshold of general understanding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .brain import Brain


@dataclass
class Lesson:
    """One taught association: an input shown while a concept is induced."""

    input_region: str
    stimulus: np.ndarray
    concept: str


class Teacher:
    """Induces concepts and associations into a :class:`Brain` by co-activation.

    Parameters
    ----------
    brain:
        The brain to teach (must be finalized -- ``brain.world`` exists).
    concept_region:
        Name of the region whose neurons hold concept assemblies.
    clamp_current:
        How hard to drive a concept's neurons while inducing it. Strong enough
        to force them to spike alongside the stimulus.
    assembly_size:
        Default number of neurons in a freshly-allocated concept assembly.
    """

    def __init__(
        self,
        brain: Brain,
        concept_region: str = "concepts",
        clamp_current: float = 22.0,
        assembly_size: int = 12,
    ):
        if brain.world is None:
            brain.finalize()
        self.brain = brain
        self.concept_region = concept_region
        self.clamp = float(clamp_current)
        self.assembly_size = int(assembly_size)

        # concept name -> (region_name, local indices)
        self.concepts: Dict[str, Tuple[str, np.ndarray]] = {}
        self._used: Dict[str, set] = {}
        self.rng = brain.rng

    # -- defining concepts ------------------------------------------------
    def allocate_concept(
        self,
        name: str,
        size: Optional[int] = None,
        region: Optional[str] = None,
        word: Optional[str] = None,
        kind: str = "word",
    ) -> np.ndarray:
        """Reserve a fresh, non-overlapping assembly for a new concept.

        Returns the global neuron ids of the assembly.
        """
        region = region or self.concept_region
        size = size or self.assembly_size
        r = self.brain.region(region)
        used = self._used.setdefault(region, set())
        available = [i for i in range(r.size) if i not in used]
        if len(available) < size:
            # The region is full -- grow it (neurogenesis) rather than fail, so
            # the brain can keep learning new concepts, like a real brain making
            # room for new memories.
            grow_by = max(size - len(available), self.assembly_size * 4)
            self.brain.grow_region(region, grow_by)
            r = self.brain.region(region)
            available = [i for i in range(r.size) if i not in used]
        idx = self.rng.choice(available, size=size, replace=False)
        used.update(idx.tolist())
        return self.define_concept(name, idx, region=region, word=word,
                                   kind=kind)

    def define_concept(
        self,
        name: str,
        indices: Sequence[int],
        region: Optional[str] = None,
        word: Optional[str] = None,
        kind: str = "word",
    ) -> np.ndarray:
        """Register a concept over explicit neuron indices in a region."""
        region = region or self.concept_region
        indices = np.asarray(indices, dtype=np.int64)
        gids = self.brain.gids(region, indices)
        self.brain.world.add_concept(name, gids, word=word, kind=kind)
        self.concepts[name] = (region, indices)
        return gids

    # -- the induction mechanism -----------------------------------------
    def _clamp_sequence(self, region_name: str, indices: np.ndarray,
                        length: int) -> np.ndarray:
        """A (length, region_size) current array that drives ``indices`` hard."""
        r = self.brain.region(region_name)
        seq = np.zeros((length, r.size))
        seq[:, indices] = self.clamp
        return seq

    def induce(
        self,
        input_region: str,
        stimulus: np.ndarray,
        concept: str,
        repeats: int = 5,
        settle: int = 12,
    ) -> None:
        """Imprint: show ``stimulus`` while forcing ``concept`` to fire.

        Repeats the pairing so STDP consolidates the input -> concept synapses.
        """
        if concept not in self.concepts:
            self.allocate_concept(concept)
        region, indices = self.concepts[concept]
        stimulus = np.atleast_2d(stimulus)
        T = len(stimulus)

        for _ in range(repeats):
            self.brain.reset_state(keep_weights=True)
            self.brain.stimulate(input_region, stimulus)
            self.brain.stimulate(region, self._clamp_sequence(region, indices, T))
            self.brain.run(T + settle, record=False)

    def teach(self, lessons: Sequence[Lesson], repeats: int = 5) -> None:
        """Induce a batch of input -> concept associations."""
        for lesson in lessons:
            self.induce(lesson.input_region, lesson.stimulus,
                        lesson.concept, repeats=repeats)

    def teach_sequence(
        self,
        concepts: Sequence[str],
        repeats: int = 6,
        window: int = 14,
        gap: int = 4,
    ) -> None:
        """Teach temporal order A -> B -> C by inducing the concepts in turn.

        This strengthens forward links inside the concept region and, more
        importantly, lets the world model learn the transition statistics that
        make prediction possible.
        """
        for name in concepts:
            if name not in self.concepts:
                self.allocate_concept(name)

        # Teach the expectation A -> B -> C directly and cleanly.
        for a, b in zip(concepts, concepts[1:]):
            self.brain.world.learn_transition(a, b, count=float(repeats))

        # And strengthen the *neural* forward links by inducing the assemblies
        # in order, so the wiring itself carries the sequence too.
        for _ in range(repeats):
            self.brain.reset_state(keep_weights=True)
            for name in concepts:
                region, indices = self.concepts[name]
                clamp = self._clamp_sequence(region, indices, window)
                self.brain.stimulate(region, clamp)
                self.brain.run(window + gap, record=False)

    # -- checking what was learned ---------------------------------------
    def recall(
        self,
        input_region: str,
        stimulus: np.ndarray,
        settle: int = 25,
        record: bool = True,
        threshold: float = 0.12,
    ) -> Tuple[Optional[str], List]:
        """Present a stimulus *without* clamping and see which concept wakes.

        Returns ``(recognised_word_or_name, frames)``. This is the test of
        whether induction worked: the input alone should now evoke the concept.
        The winner is the concept whose assembly reaches the highest activation
        at any moment during the presentation (a transient evoked response).
        """
        self.brain.reset_state(keep_weights=True)
        was_plastic = self.brain.plastic
        self.brain.plastic = False
        try:
            frames = self.brain.present(input_region, stimulus, settle=settle,
                                        record=record)
        finally:
            self.brain.plastic = was_plastic
        scores = self._peak_scores(frames)
        # Recognise words, not category assemblies, so a member ("dark") comes
        # back as itself rather than its class ("brightness").
        scores = {n: s for n, s in scores.items()
                  if self.brain.world.concepts[n].kind == "word"}
        if not scores:
            return None, frames
        best = max(scores, key=scores.get)
        if scores[best] < threshold:
            return None, frames
        concept = self.brain.world.concepts[best]
        return (concept.word or concept.name), frames

    def concept_scores(
        self, input_region: str, stimulus: np.ndarray, settle: int = 25
    ) -> Dict[str, float]:
        """Like :meth:`recall`, but return every concept's peak activation."""
        self.brain.reset_state(keep_weights=True)
        was_plastic = self.brain.plastic
        self.brain.plastic = False
        try:
            frames = self.brain.present(input_region, stimulus, settle=settle,
                                        record=False)
        finally:
            self.brain.plastic = was_plastic
        return self._peak_scores(frames)

    def _peak_scores(self, frames: List) -> Dict[str, float]:
        """Peak (over time) mean-assembly activation for each concept."""
        if not frames:
            return {}
        stack = np.array([f.activation for f in frames])   # (T, n_neurons)
        scores: Dict[str, float] = {}
        for name, concept in self.brain.world.concepts.items():
            if len(concept.gids):
                scores[name] = float(stack[:, concept.gids].mean(axis=1).max())
            else:
                scores[name] = 0.0
        return scores
