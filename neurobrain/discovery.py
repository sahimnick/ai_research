"""
discovery.py
============

**Unsupervised concept discovery** (کشف بی‌نظارت مفاهیم از داده).

Until now every concept was *defined by us* and forced on the brain by clamping.
That is not how a brain forms concepts. A real brain watches an unlabelled
stream of experience and lets stable patterns *organise themselves* into
assemblies -- no teacher, no labels.

This module implements exactly that with a real neural learning rule:
**competitive Hebbian learning** with a k-winners-take-all step (the winner
selection is what the concept region's inhibitory interneurons do in the wire).

    for each input x:
        drive   = W · x               # how strongly each unit matches the input
        winners = top-k by drive       # lateral inhibition: only a few fire
        W[winners] += lr · (x - W[winners])   # Hebbian: move toward the input
        renormalise the winners        # conserved-resource competition

Over many inputs, different units specialise for different recurring patterns.
Groups of units that consistently co-win for the same kind of input become a
**discovered concept assembly** -- learned from the data alone.

The module can then *instantiate* the discovered assemblies into a
:class:`Brain`'s world model, so the self-organised concepts live in the same
spiking system that imagines and reasons.

Everything here is measurable: :func:`make_clustered_data` generates a stream
with a *hidden* ground truth, and :meth:`ConceptDiscovery.purity` /
:meth:`match_to_truth` report, honestly, how well the unsupervised discovery
recovered the categories it was never told about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data with a hidden ground truth (so discovery can be measured honestly)
# ---------------------------------------------------------------------------
def make_clustered_data(
    n_categories: int = 5,
    dim: int = 200,
    active: int = 20,
    samples_per_category: int = 60,
    noise: float = 0.25,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate an unlabelled stream that secretly has ``n_categories`` clusters.

    Each category has a prototype sparse pattern; samples are noisy copies of a
    prototype (some active bits dropped, some random bits added). Returns
    ``(X, labels, prototypes)`` -- ``labels`` is the hidden truth, used only for
    scoring, never for training.
    """
    rng = np.random.default_rng(seed)
    prototypes = np.zeros((n_categories, dim), dtype=np.float32)
    for c in range(n_categories):
        bits = rng.choice(dim, size=active, replace=False)
        prototypes[c, bits] = 1.0

    X, labels = [], []
    for c in range(n_categories):
        proto_bits = np.nonzero(prototypes[c])[0]
        for _ in range(samples_per_category):
            x = prototypes[c].copy()
            # drop some active bits
            drop = rng.random(active) < noise
            x[proto_bits[drop]] = 0.0
            # add some spurious bits
            n_add = int(noise * active)
            if n_add:
                x[rng.choice(dim, size=n_add, replace=False)] = 1.0
            X.append(x)
            labels.append(c)

    X = np.array(X, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    order = rng.permutation(len(X))          # shuffle: the stream is unordered
    return X[order], labels[order], prototypes


# ---------------------------------------------------------------------------
# Competitive Hebbian concept discovery
# ---------------------------------------------------------------------------
@dataclass
class ConceptDiscovery:
    """Discovers concept assemblies from unlabelled input by competition.

    Parameters
    ----------
    input_dim:  size of the input pattern (e.g. a sensory region).
    n_units:    number of concept units (make it comfortably > expected clusters).
    k_winners:  how many units fire per input (the assembly size / sparsity).
    lr:         learning rate for the competitive update.
    seed:       RNG seed.
    """

    input_dim: int
    n_units: int = 64
    k_winners: int = 6
    lr: float = 0.08
    seed: int = 0

    W: np.ndarray = field(init=False)
    _wins: np.ndarray = field(init=False)          # lifetime win count per unit

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        # start from small random prototypes, each normalised
        self.W = rng.random((self.n_units, self.input_dim)).astype(np.float32)
        self._normalise()
        self._wins = np.zeros(self.n_units, dtype=np.int64)
        self.rng = rng

    def _normalise(self) -> None:
        norm = np.linalg.norm(self.W, axis=1, keepdims=True)
        norm[norm < 1e-6] = 1.0
        self.W /= norm

    # -- learning ---------------------------------------------------------
    def winners(self, x: np.ndarray) -> np.ndarray:
        """The units that fire for input ``x`` (top-k by match)."""
        drive = self.W @ x
        return np.argpartition(drive, -self.k_winners)[-self.k_winners:]

    def observe(self, x: np.ndarray, lr: Optional[float] = None) -> np.ndarray:
        """Learn from one unlabelled input; return the winning units."""
        lr = self.lr if lr is None else lr
        win = self.winners(x)
        # competitive Hebbian move: winners drift toward the input pattern
        self.W[win] += lr * (x - self.W[win])
        np.clip(self.W[win], 0.0, None, out=self.W[win])
        # renormalise only the winners (conserved-resource competition)
        norm = np.linalg.norm(self.W[win], axis=1, keepdims=True)
        norm[norm < 1e-6] = 1.0
        self.W[win] /= norm
        self._wins[win] += 1
        return win

    def learn(self, X: np.ndarray, epochs: int = 3,
              anneal: bool = True) -> "ConceptDiscovery":
        """Watch an unlabelled dataset for several epochs (lr anneals down)."""
        for e in range(epochs):
            lr = self.lr * (1.0 - 0.6 * e / max(1, epochs - 1)) if anneal \
                else self.lr
            for i in self.rng.permutation(len(X)):
                self.observe(X[i], lr=lr)
        return self

    # -- reading out the discovered concepts -----------------------------
    def best_unit(self, x: np.ndarray) -> int:
        return int(np.argmax(self.W @ x))

    def assembly(self, x: np.ndarray) -> np.ndarray:
        """The discovered assembly (winning units) for an input."""
        return np.sort(self.winners(x))

    def active_units(self, min_wins: int = 1) -> List[int]:
        """Units that actually specialised (won at least ``min_wins`` times)."""
        return [i for i in range(self.n_units) if self._wins[i] >= min_wins]

    # -- honest scoring ---------------------------------------------------
    def purity(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Clustering purity of the discovered units against hidden labels.

        Each sample is assigned to its best unit; purity is the fraction of
        samples whose unit's majority label matches. 1.0 = perfect discovery,
        1/n_categories = chance.
        """
        assign = np.array([self.best_unit(x) for x in X])
        total = 0
        for u in np.unique(assign):
            lbl = labels[assign == u]
            if len(lbl):
                total += np.bincount(lbl).max()
        return total / len(X)

    def match_to_truth(self, prototypes: np.ndarray
                       ) -> Dict[int, int]:
        """Map each true category's prototype to the unit that best represents it."""
        return {c: self.best_unit(prototypes[c]) for c in range(len(prototypes))}

    def prototype_bits(self, unit: int, active: int = 20) -> np.ndarray:
        """The input bits that most define a discovered unit (its 'meaning')."""
        return np.sort(np.argsort(self.W[unit])[-active:])

    def instantiate_in_brain(self, brain, teacher, region: str = "text",
                             max_concepts: int = 10, min_wins: int = 8,
                             active: int = 20, amplitude: float = 18.0,
                             window: int = 24, repeats: int = 3) -> List[str]:
        """Write the *self-discovered* concepts into the brain as assemblies.

        The data decided *which* patterns are concepts (unsupervised); this just
        imprints each discovered prototype into a fresh assembly (auto-named
        ``c<u>``) via the normal grounding path, so the brain can then recognise,
        reason over, and imagine the concepts it found for itself. Returns the
        names created.
        """
        units = self.active_units(min_wins=min_wins)
        units.sort(key=lambda u: -self._wins[u])
        units = units[:max_concepts]
        size = brain.region(region).size
        names: List[str] = []
        for u in units:
            current = np.zeros(size, dtype=np.float32)
            bits = self.prototype_bits(u, active)
            current[bits[bits < size]] = amplitude
            stim = np.tile(current, (window, 1))
            name = f"c{u}"
            teacher.allocate_concept(name, word=name)
            teacher.induce(region, stim, name, repeats=repeats)
            names.append(name)
        return names
