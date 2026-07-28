"""
pallium.py
==========

The pallium as a **distributed predictive relational architecture**.

Calling the pallium "a big associative memory" (as :mod:`psyche` does) sells it
short. What makes pallial tissue -- mammalian cortex, and equally the avian
DVR/nidopallium, which supports crow- and parrot-grade cognition without any
six-layer neocortex -- capable of flexible intelligence is four properties
working together:

  1. **Hierarchical representation.** A stack of populations, each recoding the
     one below into something sparser and more invariant.

  2. **Recurrent loops.** Within a level, dense recurrent excitation with
     inhibitory competition, so activity *settles* into an attractor: partial
     input completes, noise is cleaned up, and the state persists.

  3. **Predictive top-down feedback.** Every level sends a prediction down and
     receives an error up (Rao & Ballard). Representation is what survives that
     negotiation, not a feed-forward readout.

  4. **Action prediction and relational/spatial reasoning.** The same substrate
     that holds a state also holds *operators* on states -- an action, or a
     relation -- so it can say what an action will cause and compose relations
     it was never taught (A left-of B, B left-of C => A left-of C).

Everything is local and Hebbian/delta-rule; there is no backprop through the
hierarchy. Each level learns only from signals available at that level: what
came in from below, and what was predicted from above.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v.astype(np.float32) if n < 1e-9 else (v / n).astype(np.float32)


def _kwta(x: np.ndarray, k: int) -> np.ndarray:
    """k-winners-take-all: the inhibitory microcircuit that keeps codes sparse."""
    if k >= x.size:
        return np.maximum(x, 0.0)
    out = np.zeros_like(x)
    idx = np.argpartition(x, -k)[-k:]
    out[idx] = np.maximum(x[idx], 0.0)
    return out


@dataclass
class PalliumLayer:
    """One pallial level: feed-forward recoding + recurrent settling + top-down.

    ``W_up``   (n, n_below) feed-forward weights (learned Hebbian)
    ``W_rec``  (n, n)       recurrent weights (attractor / pattern completion)
    ``W_down`` (n_below, n) top-down generative weights (predicts the level below)
    """

    n: int
    n_below: int
    k: int                                    # sparsity (active cells)
    W_up: np.ndarray = field(default=None)
    W_rec: np.ndarray = field(default=None)
    W_down: np.ndarray = field(default=None)
    state: np.ndarray = field(default=None)
    duty: np.ndarray = field(default=None)    # running firing rate per cell
    boost_beta: float = 2.0                   # strength of homeostatic control

    def __post_init__(self):
        rng = np.random.default_rng(abs(hash((self.n, self.n_below))) % (2**31))
        if self.W_up is None:
            self.W_up = rng.normal(0, 1.0, (self.n, self.n_below)).astype(np.float32)
            self._scale_rows()
        if self.W_rec is None:
            self.W_rec = np.zeros((self.n, self.n), np.float32)
        if self.W_down is None:
            self.W_down = np.zeros((self.n_below, self.n), np.float32)
        self.state = np.zeros(self.n, np.float32)
        self.duty = np.full(self.n, self.k / self.n, np.float32)

    def _scale_rows(self) -> None:
        """Synaptic scaling: keep each cell's total input drive comparable, so
        no cell can win by sheer weight growth (Turrigiano homeostasis)."""
        nrm = np.linalg.norm(self.W_up, axis=1, keepdims=True)
        self.W_up /= np.maximum(nrm, 1e-6)

    @property
    def _boost(self) -> np.ndarray:
        """Homeostatic intrinsic plasticity: a cell that has been firing more
        than its fair share raises its own threshold. This is what stops one
        assembly from swallowing every input (runaway Hebbian collapse)."""
        return self.boost_beta * (self.duty - self.k / self.n)

    # -- 1+2: recode from below, then settle in the recurrent loop ----------
    def encode(self, below: np.ndarray, settle_steps: int = 4,
               rec_gain: float = 0.25) -> np.ndarray:
        """Feed-forward drive (homeostatically balanced), then **recurrent
        settling** into an attractor."""
        drive = self.W_up @ _unit(below) - self._boost
        x = _kwta(drive, self.k)
        for _ in range(settle_steps):
            x = _kwta(drive + rec_gain * (self.W_rec @ _unit(x)), self.k)
        self.state = _unit(x)
        return self.state

    # -- 3: top-down prediction of the level below --------------------------
    def predict_below(self, state: Optional[np.ndarray] = None) -> np.ndarray:
        s = self.state if state is None else state
        return self.W_down @ s

    # -- local learning: Hebbian up/down + recurrent auto-association -------
    def learn(self, below: np.ndarray, lr: float = 0.05) -> float:
        """Learn from this sample and return the (predictive) reconstruction
        error -- the only signal that crosses levels."""
        b = _unit(below)
        s = self.state
        active = s > 0
        pred = self.predict_below(s)
        err = b - _unit(pred)
        # generative (top-down) weights: reduce the prediction error
        self.W_down += lr * np.outer(err, s)
        # recognition (bottom-up): competitive Hebbian -- the winners move their
        # receptive fields toward the input they won on (Kohonen/Foldiak style)
        self.W_up[active] += lr * np.outer(s[active], b)
        self._scale_rows()
        # recurrent auto-association: the state becomes its own attractor
        self.W_rec += lr * np.outer(s, s)
        np.fill_diagonal(self.W_rec, 0.0)
        # homeostasis: track how often each cell fires
        self.duty *= (1.0 - lr)
        self.duty += lr * active.astype(np.float32)
        return float(np.linalg.norm(err))


class PredictivePallium:
    """A hierarchy of :class:`PalliumLayer` with action and relation operators.

    Beyond storing states it holds **operators**: an action or a relation is a
    matrix that maps a pallial state to the state it leads to. Because operators
    compose, relations chain transitively without ever being taught the chain."""

    def __init__(self, sizes: Sequence[int] = (256, 128, 64),
                 sparsity: float = 0.1, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.sizes = list(sizes)
        self.layers: List[PalliumLayer] = []
        for i in range(1, len(sizes)):
            self.layers.append(PalliumLayer(n=sizes[i], n_below=sizes[i - 1],
                                            k=max(2, int(sizes[i] * sparsity))))
        self.dim = sizes[-1]
        # operators over the TOP-level code
        self.action_op: Dict[str, np.ndarray] = {}
        self.relation_op: Dict[str, np.ndarray] = {}
        self.symbols: Dict[str, np.ndarray] = {}     # named top-level states

    # -- hierarchy ----------------------------------------------------------
    def encode(self, x: np.ndarray) -> np.ndarray:
        """Bottom-up through the hierarchy, each level settling recurrently."""
        cur = _unit(np.asarray(x, np.float32))
        for layer in self.layers:
            cur = layer.encode(cur)
        return cur

    def reconstruct(self, top: np.ndarray) -> np.ndarray:
        """Top-down generative sweep: an abstract code re-creates the input."""
        cur = top
        for layer in reversed(self.layers):
            cur = _unit(layer.predict_below(cur))
        return cur

    def learn(self, x: np.ndarray, lr: float = 0.05) -> float:
        """One perceive-and-learn pass; returns total predictive error."""
        cur = _unit(np.asarray(x, np.float32))
        inputs = []
        for layer in self.layers:
            inputs.append(cur)
            cur = layer.encode(cur)
        err = 0.0
        for layer, below in zip(self.layers, inputs):
            err += layer.learn(below, lr=lr)
        return err

    def train(self, X: np.ndarray, epochs: int = 8, lr: float = 0.05
              ) -> List[float]:
        """Unsupervised: learn the hierarchy from data. Returns error per epoch."""
        hist = []
        for _ in range(epochs):
            e = 0.0
            order = self.rng.permutation(len(X))
            for i in order:
                e += self.learn(X[i], lr=lr)
            hist.append(e / len(X))
        return hist

    # -- symbols and operators ----------------------------------------------
    def bind(self, name: str, x: np.ndarray) -> np.ndarray:
        """Name a top-level pallial state (a concept token)."""
        s = self.encode(x)
        self.symbols[name] = s
        return s

    def symbol(self, name: str) -> np.ndarray:
        if name not in self.symbols:      # a fresh random sparse code
            v = np.zeros(self.dim, np.float32)
            k = max(2, int(self.dim * 0.1))
            v[self.rng.choice(self.dim, k, replace=False)] = 1.0
            self.symbols[name] = _unit(v)
        return self.symbols[name]

    def _learn_op(self, store: Dict[str, np.ndarray], op: str,
                  a: np.ndarray, b: np.ndarray, lr: float = 1.0) -> None:
        if op not in store:
            store[op] = np.zeros((self.dim, self.dim), np.float32)
        store[op] += lr * np.outer(b, a)          # Hebbian outer product

    # -- 4a: action prediction ----------------------------------------------
    def learn_action(self, state: str, action: str, next_state: str) -> None:
        """"Doing ``action`` in ``state`` leads to ``next_state``."""
        self._learn_op(self.action_op, action,
                       self.symbol(state), self.symbol(next_state))

    def predict_action(self, state: str, action: str) -> Optional[str]:
        """What does this action lead to? (operator applied, nearest symbol)"""
        if action not in self.action_op:
            return None
        v = self.action_op[action] @ self.symbol(state)
        return self._nearest(v, exclude=state)

    # -- 4b: relational + spatial reasoning ---------------------------------
    def learn_relation(self, a: str, relation: str, b: str,
                       inverse: Optional[str] = None) -> None:
        """``a relation b``; if an inverse name is given, learn it too."""
        self._learn_op(self.relation_op, relation, self.symbol(a), self.symbol(b))
        if inverse:
            self._learn_op(self.relation_op, inverse,
                           self.symbol(b), self.symbol(a))

    def apply_relation(self, a: str, relation: str) -> Optional[str]:
        if relation not in self.relation_op:
            return None
        return self._nearest(self.relation_op[relation] @ self.symbol(a),
                             exclude=a)

    def transitive(self, a: str, relation: str, depth: int = 2) -> List[str]:
        """Compose an operator with itself: A left-of B, B left-of C  =>  the
        chain from A. Never taught -- it falls out of operator composition."""
        out, cur = [], a
        for _ in range(depth):
            nxt = self.apply_relation(cur, relation)
            if nxt is None or nxt in out or nxt == cur:
                break
            out.append(nxt)
            cur = nxt
        return out

    def spatial_infer(self, a: str, c: str, relation: str,
                      chain: Sequence[str]) -> bool:
        """Is ``a relation c`` true, given only neighbouring links along
        ``chain``? (transitive spatial inference over the learned operator)"""
        if a not in chain or c not in chain:
            return False
        return chain.index(a) < chain.index(c) and \
            c in self.transitive(a, relation, depth=len(chain))

    def _nearest(self, v: np.ndarray, exclude: Optional[str] = None
                 ) -> Optional[str]:
        v = _unit(v)
        best, score = None, -np.inf
        for name, s in self.symbols.items():
            if name == exclude:
                continue
            sc = float(v @ s)
            if sc > score:
                best, score = name, sc
        return best

    # -- diagnostics --------------------------------------------------------
    @property
    def n_synapses(self) -> int:
        s = sum(l.W_up.size + l.W_rec.size + l.W_down.size for l in self.layers)
        s += sum(m.size for m in self.action_op.values())
        s += sum(m.size for m in self.relation_op.values())
        return int(s)

    def abstraction_profile(self, X: np.ndarray, groups: np.ndarray
                            ) -> List[float]:
        """How **invariant** each level is: within-group similarity minus
        between-group similarity of its codes. Higher levels should abstract
        more -- that is what a hierarchy is *for*."""
        cur = np.array([_unit(x) for x in X], np.float32)
        out = []
        for layer in self.layers:
            cur = np.array([layer.encode(c) for c in cur], np.float32)
            S = cur @ cur.T
            same = groups[:, None] == groups[None, :]
            eye = np.eye(len(X), dtype=bool)
            within = S[same & ~eye].mean() if (same & ~eye).any() else 0.0
            between = S[~same].mean() if (~same).any() else 0.0
            out.append(float(within - between))
        return out


def build_predictive_pallium(sizes=(256, 128, 64), seed: int = 0,
                             verbose: bool = False) -> PredictivePallium:
    """A pallium trained on structured patterns, with action and relational
    operators taught on top -- the four properties in one object."""
    rng = np.random.default_rng(seed)
    p = PredictivePallium(sizes=sizes, seed=seed)

    # structured sensory data: 8 prototype "objects", noisy instances of each
    protos = rng.normal(0, 1, (8, sizes[0])).astype(np.float32)
    X, groups = [], []
    for g in range(8):
        for _ in range(40):
            X.append(protos[g] + rng.normal(0, 0.6, sizes[0]).astype(np.float32))
            groups.append(g)
    X = np.array(X, np.float32)
    groups = np.array(groups)
    hist = p.train(X, epochs=8)
    if verbose:
        print(f"  predictive error {hist[0]:.3f} -> {hist[-1]:.3f}")
    return p
