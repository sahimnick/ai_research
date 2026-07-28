"""
reasoning.py
============

A **multi-relational world model** and **compositional reasoning**
(ورد مدل واقعی + استدلال ترکیبی).

The earlier "world model" was a single transition table -- one flat dimension
("what tends to come next"). A real world model has *many* dimensions: things
are *kinds of* other things, have *properties*, are *parts of* wholes, are
*opposites*, and *cause* one another. This module stores those typed relations
as a **semantic graph** over the concepts, and reasons over it:

* **transitive inference** -- if dog *is-a* mammal and mammal *is-a* animal,
  then dog *is-a* animal (a conclusion never stored, only derived);
* **analogy** -- hot : cold :: big : ?  →  small  (find the shared relation,
  then apply it to the new term);
* **composition** -- bind two concepts ("big" + "cat") into a new composite
  assembly, and decompose one back into its parts;
* **explanation** -- return the chain of relations that connects two concepts.

Concepts are the same cell assemblies the rest of the brain uses, so a composed
or inferred concept can be pushed back into the network and *imagined*. Relations
can also be *learned from the discovered concepts* (see discovery.py), so the
reasoning is not hand-wired symbol pushing bolted on the side -- it operates on
whatever concepts the brain actually holds.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# Relations that chain (a r b, b r c  =>  a r c). Others (opposite, property)
# do not compose transitively.
TRANSITIVE = {"is_a", "part_of", "causes", "before"}
SYMMETRIC = {"opposite", "sibling", "related"}


@dataclass
class SemanticNetwork:
    """A typed relation graph over concepts, with reasoning over it.

    ``world`` (optional) is a :class:`WorldModel`; when present, concept names
    map to assemblies so composed/inferred concepts can drive the brain.
    """

    world: object = None
    _fwd: Dict[str, Dict[str, Set[str]]] = field(default_factory=dict)  # rel->a->{b}
    _rev: Dict[str, Dict[str, Set[str]]] = field(default_factory=dict)  # rel->b->{a}
    composites: Dict[str, Tuple[str, ...]] = field(default_factory=dict)

    # -- building the graph ----------------------------------------------
    def add(self, a: str, rel: str, b: str) -> None:
        """Add ``a --rel--> b`` (and the reverse for symmetric relations)."""
        self._fwd.setdefault(rel, defaultdict(set))[a].add(b)
        self._rev.setdefault(rel, defaultdict(set))[b].add(a)
        if rel in SYMMETRIC:
            self._fwd[rel][b].add(a)
            self._rev[rel][a].add(b)

    def add_many(self, triples) -> None:
        for a, rel, b in triples:
            self.add(a, rel, b)

    def relations(self) -> List[str]:
        return sorted(self._fwd)

    def neighbors(self, a: str, rel: str) -> Set[str]:
        return set(self._fwd.get(rel, {}).get(a, set()))

    # -- reasoning --------------------------------------------------------
    def infer(self, a: str, rel: str, max_depth: int = 6) -> List[str]:
        """All ``b`` reachable by following ``rel`` (transitive closure).

        For transitive relations this derives conclusions never stored directly.
        """
        seen: Set[str] = set()
        frontier = deque((n, 1) for n in self.neighbors(a, rel))
        while frontier:
            node, d = frontier.popleft()
            if node in seen:
                continue
            seen.add(node)
            if rel in TRANSITIVE and d < max_depth:
                for nxt in self.neighbors(node, rel):
                    if nxt not in seen:
                        frontier.append((nxt, d + 1))
        return sorted(seen)

    def holds(self, a: str, rel: str, b: str) -> bool:
        """Does ``a rel b`` hold, directly or by inference?"""
        return b in self.neighbors(a, rel) or b in self.infer(a, rel)

    def which_relation(self, a: str, b: str) -> Optional[str]:
        """The (a direct) relation linking a to b, if any."""
        for rel in self._fwd:
            if b in self.neighbors(a, rel):
                return rel
        return None

    def analogy(self, a: str, b: str, c: str) -> Optional[str]:
        """a : b :: c : ?  -- find the relation a→b, then apply it to c."""
        rel = self.which_relation(a, b)
        if rel is None:
            return None
        candidates = self.neighbors(c, rel)
        if not candidates:
            return None
        # prefer a candidate that is not c itself
        candidates = [x for x in candidates if x != c] or list(candidates)
        return sorted(candidates)[0]

    def explain(self, a: str, b: str, max_depth: int = 6
                ) -> Optional[List[Tuple[str, str]]]:
        """A shortest chain of relations connecting ``a`` to ``b``.

        Returns a list of ``(relation, node)`` steps, or None if unconnected.
        """
        if a == b:
            return []
        frontier: deque = deque([(a, [])])
        seen = {a}
        while frontier:
            node, path = frontier.popleft()
            for rel in self._fwd:
                for nxt in self.neighbors(node, rel):
                    if nxt == b:
                        return path + [(rel, nxt)]
                    if nxt not in seen and len(path) < max_depth:
                        seen.add(nxt)
                        frontier.append((nxt, path + [(rel, nxt)]))
        return None

    # -- composition (binding concepts) ----------------------------------
    def compose(self, *parts: str, name: Optional[str] = None) -> str:
        """Bind concepts into a new composite concept ("big" + "cat").

        The composite's assembly is the union of its parts' assemblies, so it
        overlaps every part and can be recognised as, and decomposed into, them.
        Registered in the world model if one is attached.
        """
        name = name or "+".join(parts)
        self.composites[name] = tuple(parts)
        if self.world is not None:
            gids = []
            for p in parts:
                c = self.world.concepts.get(p)
                if c is not None:
                    gids.append(c.gids)
            if gids:
                union = np.unique(np.concatenate(gids))
                self.world.add_concept(name, union, word=name, kind="composite")
        return name

    def decompose(self, name: str) -> List[str]:
        """The parts a composite was built from."""
        return list(self.composites.get(name, ()))

    def parts_present(self, assembly_gids: np.ndarray,
                      overlap: float = 0.4) -> List[str]:
        """Which known concepts overlap an assembly (decompose by overlap).

        Lets the brain read a composite (or any activity) back into the concepts
        that make it up -- the inverse of :meth:`compose`.
        """
        if self.world is None:
            return []
        s = set(int(g) for g in assembly_gids)
        out = []
        for name, c in self.world.concepts.items():
            if c.kind == "composite" or not len(c.gids):
                continue
            inter = sum(1 for g in c.gids if int(g) in s)
            if inter / len(c.gids) >= overlap:
                out.append(name)
        return out

    # -- learning relations from the world model -------------------------
    def import_transitions(self, rel: str = "before",
                           min_count: float = 1.0) -> int:
        """Turn the world model's learned transitions into 'before' relations."""
        n = 0
        if self.world is None:
            return 0
        for a, tos in getattr(self.world, "_transitions", {}).items():
            for b, c in tos.items():
                if c >= min_count:
                    self.add(a, rel, b)
                    n += 1
        return n
