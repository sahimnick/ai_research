"""
analogy.py
==========

Analogy, and relations *between* relations.

Every symbol -- a thing (``dog``, ``paris``) or a relation (``bigger``,
``opposite``) -- gets a random, near-orthogonal high-dimensional vector (a sparse
distributed code, the way the cortex represents things). A relation is stored as
a **Hebbian associative map** between those vectors: seeing ``a --rel--> b`` adds
the outer product ``v[b] (x) v[a]`` to the relation's synaptic matrix, so later
``R[rel] @ v[a]`` recalls ``v[b]``. No back-prop, no table -- the relation lives
in the synapses.

Because relations are symbols too, the *same* machinery does **second-order**
reasoning: a meta-relation like ``bigger --inverse--> smaller`` is just another
Hebbian map, this time between relation-vectors. So the model can answer both
``dog : mammal :: cat : ?`` (first order) and ``bigger : smaller :: hotter : ?``
(analogy over relations -- both pairs are `inverse`s), and say what a relation's
own relations are.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


class RelationalMind:
    """Distributed concepts + relations-as-synaptic-maps, for analogy and
    relations-between-relations."""

    def __init__(self, dim: int = 2048, seed: int = 0):
        self.dim = dim
        self._rng = np.random.default_rng(seed)
        self.vec: Dict[str, np.ndarray] = {}
        self.R: Dict[str, np.ndarray] = {}                 # relation -> map
        self.facts: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.symmetric = set()
        self.relation_symbols = set()      # names that act as relations (1st/2nd order)

    def _v(self, name: str) -> np.ndarray:
        if name not in self.vec:
            v = self._rng.standard_normal(self.dim).astype(np.float32)
            self.vec[name] = v / np.linalg.norm(v)
        return self.vec[name]

    def relate(self, a: str, rel: str, b: str, symmetric: bool = False) -> None:
        """Store the fact ``a --rel--> b`` in the relation's synaptic map."""
        va, vb = self._v(a), self._v(b)
        self._v(rel)                                       # a relation is a symbol too
        self.relation_symbols.add(rel)
        if rel not in self.R:
            self.R[rel] = np.zeros((self.dim, self.dim), np.float32)
        self.R[rel] += np.outer(vb, va)
        self.facts[rel].append((a, b))
        if symmetric:
            self.R[rel] += np.outer(va, vb)
            self.facts[rel].append((b, a))
            self.symmetric.add(rel)

    def _nearest(self, v: np.ndarray, among: Optional[List[str]] = None) -> str:
        names = among if among is not None else list(self.vec)
        M = np.stack([self.vec[n] for n in names])
        return names[int(np.argmax(M @ (v / (np.linalg.norm(v) + 1e-9))))]

    def apply(self, rel: str, a: str, among: Optional[List[str]] = None) -> str:
        """Follow a relation: ``a --rel--> ?`` (recall through the synaptic map)."""
        return self._nearest(self.R[rel] @ self._v(a), among)

    def _which_relation(self, a: str, b: str) -> Optional[str]:
        """Which stored relation best takes ``a`` to ``b``?"""
        vb = self._v(b)
        best, score = None, -1e9
        for rel, M in self.R.items():
            s = float(vb @ (M @ self._v(a)))
            if s > score:
                best, score = rel, s
        return best

    def analogy(self, a: str, b: str, c: str,
                among: Optional[List[str]] = None) -> Optional[str]:
        """``a : b :: c : ?`` -- find the relation that links a->b, apply it to c."""
        rel = self._which_relation(a, b)
        if rel is None:
            return None
        pool = among or [x for x in self.vec if x not in (a, b, c)]
        return self.apply(rel, c, among=pool)

    # -- second order: relations between relations --------------------------
    def meta_relate(self, rel1: str, meta: str, rel2: str,
                    symmetric: bool = True) -> None:
        """A relation between two relations, e.g. ``bigger --inverse--> smaller``.
        Uses the very same associative machinery, on the relation-vectors."""
        self.relate(rel1, meta, rel2, symmetric=symmetric)   # 'meta' is the relation
        self.relation_symbols.update((rel1, rel2))           # rel1/rel2 are relations too

    def relations_of(self, rel: str, meta: str,
                     among: Optional[List[str]] = None) -> str:
        """What does relation ``rel`` map to under meta-relation ``meta``?"""
        pool = among or sorted(self.relation_symbols)
        return self.apply(meta, rel, among=pool)

    def relation_analogy(self, r1: str, r2: str, r3: str) -> Optional[str]:
        """Analogy over relations: ``r1 : r2 :: r3 : ?`` (e.g. bigger:smaller ::
        hotter:colder, both being `inverse` pairs)."""
        pool = [r for r in sorted(self.relation_symbols) if r not in (r1, r2, r3)]
        return self.analogy(r1, r2, r3, among=pool)
