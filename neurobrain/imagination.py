"""
imagination.py
==============

**Continuous imagination** in the world model (تخیل دائم).

Left alone, the brain does not sit blank -- it *imagines*. This module runs a
never-ending generative walk over the world model: from the concept it is
currently imagining it samples the next one (using the transitions it has
learned), gently activates that concept's assembly from the top down, and moves
on. The result is a self-sustaining stream of imagined concepts -- day-dreaming,
replay, free association -- that plays continuously inside the main loop.

It is generative, not deterministic:
    * it follows *learned* transitions, so the imagination is structured
      (one -> two -> three, day -> light, ...);
    * a **temperature** controls how randomly it samples (creativity);
    * a small **novelty** chance makes it jump to an unrelated concept (a fresh
      association out of nowhere);
    * **attention** can bias which concepts it tends to imagine.

The imagined concept is injected as a weak top-down current into its assembly,
so it colours the ongoing activity and becomes part of the current thought --
without forcing the neurons the way teaching does.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional

import numpy as np

from .brain import Brain


class Imagination:
    """A perpetual generative walk over the world model.

    Parameters
    ----------
    brain:        the brain whose world model + concept region to use.
    strength:     top-down current injected into the imagined assembly.
    temperature:  sampling randomness (0 = greedy, 1 = as learned, >1 = wild).
    dwell:        how many frames one imagined concept is held before moving on.
    novelty:      probability per transition of a random free-association jump.
    region:       region holding the concept assemblies.
    """

    def __init__(self, brain: Brain, strength: float = 9.0,
                 temperature: float = 0.75, dwell: int = 8,
                 novelty: float = 0.15, region: str = "memory",
                 seed: int = 0):
        self.brain = brain
        self.world = brain.world
        self.strength = float(strength)
        self.temperature = float(temperature)
        self.dwell = int(dwell)
        self.novelty = float(novelty)
        self.region = region
        self.enabled = True
        self.rng = np.random.default_rng(seed)

        self.current: Optional[str] = None
        self._age = 0
        self.stream: Deque[str] = deque(maxlen=64)

    # -- the generative walk ---------------------------------------------
    def _word_concepts(self) -> List[str]:
        return [n for n, c in self.world.concepts.items()
                if c.kind == "word" and len(c.gids)]

    def _sample_next(self, current: Optional[str],
                     bias: Optional[Dict[str, float]] = None) -> Optional[str]:
        probs = self.world.transition_probabilities(current) if current else {}
        take_learned = probs and self.rng.random() > self.novelty
        if take_learned:
            names = list(probs)
            p = np.array([probs[n] for n in names], dtype=np.float64)
            # temperature: sharpen (<1) or flatten (>1) the distribution
            t = max(1e-3, self.temperature)
            p = np.power(p, 1.0 / t)
        else:
            names = self._word_concepts()
            if not names:
                return None
            p = np.ones(len(names), dtype=np.float64)
        if bias:                       # attention makes some concepts likelier
            p = p * np.array([bias.get(n, 1.0) for n in names])
        s = p.sum()
        if s <= 0:
            return self.rng.choice(names) if names else None
        return str(self.rng.choice(names, p=p / s))

    def step(self, bias: Optional[Dict[str, float]] = None) -> Optional[str]:
        """Advance the day-dream by one frame; return the imagined concept."""
        if not self.enabled:
            self.current, self._age = None, 0
            return None
        if self.current is None or self._age >= self.dwell:
            nxt = self._sample_next(self.current, bias)
            self.current, self._age = nxt, 0
            if nxt:
                self.stream.append(nxt)
        self._age += 1
        return self.current

    # -- top-down injection ----------------------------------------------
    def top_down_current(self) -> np.ndarray:
        """A current over the concept region that lights the imagined assembly."""
        r = self.brain.region(self.region)
        vec = np.zeros(r.size, dtype=np.float32)
        if self.current and self.current in self.world.concepts:
            gids = self.world.concepts[self.current].gids
            local = gids - r.gid_start
            local = local[(local >= 0) & (local < r.size)]
            vec[local] = np.float32(self.strength)
        return vec

    def imagined_word(self) -> Optional[str]:
        if self.current and self.current in self.world.concepts:
            c = self.world.concepts[self.current]
            return c.word or c.name
        return self.current
