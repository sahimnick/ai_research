"""
world_model.py
==============

The **world model** (مدل جهان / ورد مدل).

A brain is not just reactive wiring; it builds an internal model of its world so
it can *recognise*, *name*, and *anticipate*. This module gives the simulated
brain three abilities on top of the raw spiking network:

1. **Concept assemblies** -- a concept ("light", "sound", the word "cat") is
   represented by a specific *group of neurons* that tend to fire together.
   This is the cell-assembly idea (Hebb, 1949): meaning lives in populations,
   not single cells.

2. **Grounding words <-> concepts** -- because the user wrote "ورد مدل", the
   model deliberately doubles as a *word* model: every concept can carry a text
   label, so the brain can turn activity into a word and a word into activity.

3. **Prediction of the world** -- by watching which assemblies follow which, the
   model learns a transition table and can predict the next concept. That is a
   minimal *world model*: an internal expectation of what happens next.

The world model reads the brain's global activation vector each step; it does
not change neuron dynamics itself (learning of the wiring is done by STDP and by
the Teacher). It is the brain's *interpreter*.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Concept:
    """A named cell assembly: the neurons whose joint firing *means* something.

    Attributes
    ----------
    name:    concept id, e.g. ``"light"`` or the word ``"cat"``.
    gids:    global neuron ids that make up the assembly.
    word:    optional human-language label (the "word model" side).
    """

    name: str
    gids: np.ndarray
    word: Optional[str] = None
    #: "word" for an ordinary grounded concept, "category" for a class that
    #: groups other concepts. Word recognition ignores categories, so showing
    #: "dark" is recognised as the word "dark", not its category "brightness".
    kind: str = "word"

    def __post_init__(self) -> None:
        self.gids = np.asarray(self.gids, dtype=np.int64)


class WorldModel:
    """Interprets brain activity as concepts and predicts what comes next.

    Parameters
    ----------
    n_neurons:
        Total number of neurons in the brain (size of the global activation
        vector this model consumes).
    activation_threshold:
        Mean assembly activation above which a concept is considered *active*.
    """

    def __init__(self, n_neurons: int, activation_threshold: float = 0.25):
        self.n_neurons = int(n_neurons)
        self.threshold = float(activation_threshold)

        self.concepts: Dict[str, Concept] = {}
        self.word_index: Dict[str, str] = {}   # word -> concept name

        # Transition counts: how often concept A was followed by concept B.
        self._transitions: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._history: List[str] = []          # sequence of active concepts
        self._last_active: List[str] = []

    # -- concept management ----------------------------------------------
    def add_concept(
        self, name: str, gids, word: Optional[str] = None, kind: str = "word"
    ) -> Concept:
        """Register (or overwrite) a concept assembly."""
        concept = Concept(name=name, gids=gids, word=word, kind=kind)
        self.concepts[name] = concept
        if word:
            self.word_index[word.lower()] = name
        return concept

    def word_concepts(self) -> Dict[str, "Concept"]:
        """The concepts that are grounded words (excludes categories)."""
        return {n: c for n, c in self.concepts.items() if c.kind == "word"}

    def concept_for_word(self, word: str) -> Optional[Concept]:
        name = self.word_index.get(word.lower())
        return self.concepts.get(name) if name else None

    # -- reading the brain ------------------------------------------------
    def assembly_activation(self, activation: np.ndarray) -> Dict[str, float]:
        """Mean activation of each concept assembly given a global vector."""
        out: Dict[str, float] = {}
        for name, c in self.concepts.items():
            if len(c.gids):
                out[name] = float(activation[c.gids].mean())
            else:
                out[name] = 0.0
        return out

    def active_concepts(
        self, activation: np.ndarray
    ) -> List[Tuple[str, float]]:
        """Concepts currently 'on', sorted strongest first."""
        scores = self.assembly_activation(activation)
        active = [(n, s) for n, s in scores.items() if s >= self.threshold]
        active.sort(key=lambda kv: kv[1], reverse=True)
        return active

    def observe(self, activation: np.ndarray, learn: bool = True) -> List[str]:
        """Update the world model from one step of brain activity.

        Records which concepts are active and (when ``learn`` is True) learns the
        transition from the previously-active concept to the newly-active one.
        During recall/inference ``learn`` is set False so testing does not
        rewrite the world model with the order things happened to be probed in.

        Returns the list of concept names active on this step.
        """
        active = [name for name, _ in self.active_concepts(activation)]

        # Learn transitions from the strongest previous concept to each new one.
        if learn and self._last_active and active:
            prev = self._last_active[0]
            for cur in active:
                if cur != prev:
                    self._transitions[prev][cur] += 1.0

        if active:
            top = active[0]
            if not self._history or self._history[-1] != top:
                self._history.append(top)
            self._last_active = active
        return active

    def learn_transition(self, a: str, b: str, count: float = 1.0) -> None:
        """Directly strengthen the expectation that ``a`` is followed by ``b``.

        Used by the teacher when a temporal lesson (a -> b) is given explicitly,
        so the world model's expectations stay clean and are not diluted by the
        incidental order in which concepts happened to be probed.
        """
        if a != b:
            self._transitions[a][b] += count

    # -- the "world model" proper: prediction ----------------------------
    def predict_next(self, concept: Optional[str] = None) -> Optional[str]:
        """Predict the concept most likely to follow ``concept``.

        With no argument, predicts what follows the most recent concept. This is
        the brain saying "given where I am, I expect *this* next".
        """
        if concept is None:
            concept = self._history[-1] if self._history else None
        if concept is None:
            return None
        following = self._transitions.get(concept)
        if not following:
            return None
        return max(following.items(), key=lambda kv: kv[1])[0]

    def transition_probabilities(self, concept: str) -> Dict[str, float]:
        """Normalised P(next | concept) learned so far."""
        following = self._transitions.get(concept, {})
        total = sum(following.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in following.items()}

    def recognise(self, activation: np.ndarray) -> Optional[str]:
        """Name (as a word if available) the strongest active *word* concept.

        Category assemblies are skipped so a member is recognised as itself.
        """
        active = [(n, s) for n, s in self.active_concepts(activation)
                  if self.concepts[n].kind == "word"]
        if not active:
            return None
        concept = self.concepts[active[0][0]]
        return concept.word or concept.name

    # -- introspection ----------------------------------------------------
    def summary(self) -> str:  # pragma: no cover - cosmetic
        lines = [f"WorldModel: {len(self.concepts)} concepts, "
                 f"{sum(len(v) for v in self._transitions.values())} "
                 f"learned transitions"]
        for name, c in self.concepts.items():
            nxt = self.predict_next(name)
            w = f" [{c.word}]" if c.word else ""
            arrow = f"  -> expects: {nxt}" if nxt else ""
            lines.append(f"  - {name}{w}: {len(c.gids)} neurons{arrow}")
        return "\n".join(lines)

    def reset_history(self) -> None:
        self._history.clear()
        self._last_active = []
