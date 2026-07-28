"""
relational.py
=============

**Relations stored in synapses, not in a table** (رابطه در سیناپس، نه جدول).

Earlier, "light is the opposite of dark" lived in a Python dictionary -- a
symbolic shortcut, not how a brain stores anything. Here a relation is stored
the way memory is really stored: as **synaptic weights between the concept
assemblies**, with a *separate synaptic pathway per relation type* (the relation
context routing activity through the right projection, as a gating interneuron
would).

    store("light", "opposite", "dark")
        -> Hebbian strengthening of the synapses FROM the 'light' assembly
           TO the 'dark' assembly, in the 'opposite' weight matrix W_opposite.

    recall("light", "opposite")
        -> drive = W_opposite @ (light assembly active)
           the assembly that lights up is 'dark' -- read out neurally.

Nothing is looked up in a table: recall is **activity propagating through the
learned synapses**, and multi-step inference (dog -> mammal -> animal) is that
propagation *chained*, so the conclusion is produced by the network, not by a
graph search.

Each ``W_rel`` is literally a neuron-to-neuron weight matrix over the concept
region -- the synaptic store. (For big regions one would keep it sparse; here it
is dense for clarity and small concept layers.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


SYMMETRIC = {"opposite", "sibling", "related", "same_as"}
TRANSITIVE = {"is_a", "part_of", "causes", "before"}


class RelationalMemory:
    """Stores typed relations as synaptic weight matrices between assemblies.

    Parameters
    ----------
    brain:  a finalized brain (uses ``brain.world`` concept assemblies).
    concept_region:  region holding the concept assemblies.
    """

    def __init__(self, brain, concept_region: str = "memory"):
        self.brain = brain
        self.world = brain.world
        self.region = concept_region
        r = brain.region(concept_region)
        self.gid0 = r.gid_start
        self.size = r.size
        # one synaptic weight matrix (pathway) per relation type
        self.W: Dict[str, np.ndarray] = {}

    # -- helpers ----------------------------------------------------------
    def _local(self, name: str) -> Optional[np.ndarray]:
        c = self.world.concepts.get(name)
        if c is None:
            c = self.world.concept_for_word(name)
        if c is None:
            return None
        loc = np.asarray(c.gids, dtype=np.int64) - self.gid0
        return loc[(loc >= 0) & (loc < self.size)]

    def _matrix(self, rel: str) -> np.ndarray:
        if rel not in self.W:
            self.W[rel] = np.zeros((self.size, self.size), dtype=np.float32)
        return self.W[rel]

    def _resize(self) -> None:
        r = self.brain.region(self.region)
        if r.size != self.size:                 # the concept region grew
            new = r.size
            for rel, w in self.W.items():
                bigger = np.zeros((new, new), dtype=np.float32)
                bigger[:self.size, :self.size] = w
                self.W[rel] = bigger
            self.size = new

    # -- storing a relation (Hebbian, synaptic) --------------------------
    def store(self, a: str, rel: str, b: str, strength: float = 1.0) -> bool:
        """Strengthen the synapses a-assembly -> b-assembly in pathway ``rel``."""
        self._resize()
        la, lb = self._local(a), self._local(b)
        if la is None or lb is None or not len(la) or not len(lb):
            return False
        W = self._matrix(rel)
        # Hebbian outer product: every a-neuron -> every b-neuron gets stronger
        W[np.ix_(lb, la)] += strength / len(la)
        if rel in SYMMETRIC:
            W[np.ix_(la, lb)] += strength / len(lb)
        return True

    def store_many(self, triples) -> int:
        return sum(1 for a, r, b in triples if self.store(a, r, b))

    # -- recall (propagate activity through the learned synapses) --------
    def _drive(self, a: str, rel: str) -> Optional[np.ndarray]:
        la = self._local(a)
        if la is None or rel not in self.W:
            return None
        x = np.zeros(self.size, dtype=np.float32)
        x[la] = 1.0
        return self.W[rel] @ x          # synaptic propagation

    def recall(self, a: str, rel: str, threshold: float = 0.3,
               exclude=()) -> Optional[str]:
        """The concept the ``rel`` pathway drives from ``a`` (read out neurally)."""
        drive = self._drive(a, rel)
        if drive is None or drive.max() <= 0:
            return None
        best, best_s = None, 0.0
        for name, c in self.world.concepts.items():
            if name == a or name in exclude or c.kind == "composite":
                continue
            loc = self._local(name)
            if loc is None or not len(loc):
                continue
            s = float(drive[loc].mean())
            if s > best_s:
                best, best_s = name, s
        if best is None or best_s < threshold:
            return None
        c = self.world.concepts[best]
        return c.word or c.name

    def holds(self, a: str, rel: str, b: str) -> bool:
        return b in self.infer(a, rel) or self.recall(a, rel) == b

    # -- inference: chain the propagation through the synapses -----------
    def infer(self, a: str, rel: str, max_steps: int = 6) -> List[str]:
        """Follow the ``rel`` pathway repeatedly -- neural transitive inference."""
        out: List[str] = []
        seen = {a}
        cur = a
        for _ in range(max_steps):
            nxt = self.recall(cur, rel, exclude=seen)
            if not nxt or nxt in seen:
                break
            out.append(nxt)
            seen.add(nxt)
            if rel not in TRANSITIVE:
                break
            cur = nxt
        return out

    def analogy(self, a: str, b: str, c: str) -> Optional[str]:
        """a:b :: c:?  -- find the pathway giving b from a, apply it to c."""
        for rel in self.W:
            if self.recall(a, rel) == b or b in self.infer(a, rel):
                d = self.recall(c, rel)
                if d and d != c:
                    return d
        return None

    # -- introspection ----------------------------------------------------
    def relations(self) -> List[str]:
        return sorted(self.W)

    def synapse_count(self, rel: Optional[str] = None) -> int:
        rels = [rel] if rel else list(self.W)
        return int(sum((self.W[r] > 0).sum() for r in rels if r in self.W))
