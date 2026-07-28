"""
attention.py
============

**Multi-faceted attention** (توجه چند وجهی).

Attention is a spotlight the brain can point along several facets at once. Here
it works on two axes simultaneously:

1. **Modality attention** -- turn a sense up and the others down (attend to what
   you *hear* over what you *see*). Implemented as a gain on each sensory
   region's driving current.

2. **Concept / semantic attention** -- hold one or more concepts (or a whole
   category's members) in mind, biasing them from the top down so they are
   easier to evoke and more likely to be imagined ("think about numbers").

Both can be set at the same time, so the brain can, say, attend to the *text*
sense **and** to the concept *animal* together -- that is what "multi-faceted"
means. Attention also feeds a bias into :mod:`imagination`, so what you attend to
is what the brain tends to day-dream about.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np

from ..core.brain import Brain


class Attention:
    """A spotlight over modalities and concepts.

    Parameters
    ----------
    brain:          the brain to modulate.
    sensory:        the modality (sensory) region names.
    concept_boost:  top-down current added to an attended concept's assembly.
    bias_strength:  how strongly attention skews imagination toward its focus.
    region:         region holding the concept assemblies.
    """

    def __init__(self, brain: Brain,
                 sensory: Iterable[str] = ("vision", "sound", "text"),
                 concept_boost: float = 11.0, bias_strength: float = 4.0,
                 region: str = "memory"):
        self.brain = brain
        self.world = brain.world
        self.sensory = list(sensory)
        self.concept_boost = float(concept_boost)
        self.bias_strength = float(bias_strength)
        self.region = region

        #: per-modality gain (1.0 = neutral). Facet 1.
        self.modality_gain: Dict[str, float] = {}
        #: concept name -> focus weight (0..1). Facet 2.
        self.concept_focus: Dict[str, float] = {}

    # -- setting the spotlight -------------------------------------------
    def attend_modality(self, gains: Dict[str, float]) -> None:
        """Set gains for sensory regions, e.g. ``{'sound': 1.8, 'vision': 0.4}``."""
        self.modality_gain.update({k: float(v) for k, v in gains.items()})

    def attend_concepts(self, names: Iterable[str], weight: float = 1.0) -> None:
        """Hold these concepts (by name or word) in mind."""
        self.concept_focus = {}
        for n in names:
            concept = self.world.concept_for_word(n)
            name = concept.name if concept else n
            if name in self.world.concepts:
                self.concept_focus[name] = float(weight)

    def clear(self) -> None:
        self.modality_gain = {}
        self.concept_focus = {}
        for r in self.sensory:
            self.brain.region_gain[r] = 1.0

    # -- applying it each frame ------------------------------------------
    def apply_gains(self) -> None:
        """Push the modality gains onto the brain (facet 1)."""
        for r in self.sensory:
            self.brain.region_gain[r] = self.modality_gain.get(r, 1.0)

    def top_down_current(self) -> np.ndarray:
        """Top-down current holding the attended concepts in mind (facet 2)."""
        r = self.brain.region(self.region)
        vec = np.zeros(r.size, dtype=np.float32)
        for name, w in self.concept_focus.items():
            gids = self.world.concepts[name].gids
            local = gids - r.gid_start
            local = local[(local >= 0) & (local < r.size)]
            vec[local] += np.float32(self.concept_boost * w)
        return vec

    def imagination_bias(self) -> Optional[Dict[str, float]]:
        """A per-concept multiplier that skews imagination toward the focus."""
        if not self.concept_focus:
            return None
        return {n: 1.0 + self.bias_strength * w
                for n, w in self.concept_focus.items()}

    # -- introspection ----------------------------------------------------
    def summary(self) -> dict:
        return {"modality_gain": dict(self.modality_gain),
                "concept_focus": sorted(self.concept_focus)}
