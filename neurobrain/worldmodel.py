"""
worldmodel.py
=============

A world model that **predicts**, not merely remembers.

The earlier "world model" was a count matrix ``_T`` of which concept follows
which -- a first-order Markov table. That is the weakest possible world model:
it cannot look several steps ahead, cannot say what an *action* would cause, and
has no notion of surprise. This module upgrades it, stage by stage, with
mechanisms that are grounded in neuroscience rather than borrowed from classical
ML:

  * **Stage A -- predictive transitions.** Learn ``P(next | state)`` online, and
    expose the *surprise* (prediction error) each observation causes. Prediction
    error is the brain's universal teaching signal (predictive coding; dopamine
    as reward-prediction error).

  * **Stage B -- successor representation (SR).** Learn ``M ~= sum_k gamma^k T^k``
    by a temporal-difference rule. The SR is *where a state tends to lead over
    many steps* -- a multi-step look-ahead that composes. It is not an ML trick:
    Stachenfeld, Botvinick & Gershman (2017) showed hippocampal **place and grid
    cells are the SR of the animal's world** (Dayan, 1993).

  * **Stage C -- action-conditioned transitions.** Learn ``T[a]``: "in state s,
    doing action a leads to s'." This is what turns a passive predictor into a
    model you can **plan** with (model-based control).

  * **Stage D -- latent belief (partial).** Keep a state estimate that survives
    a *missing* observation by rolling forward through the model -- a miniature
    of object permanence / working memory.

Honest deferral (**Stage E**): grounding real space, physics and causality from
continuous sensory streams (video) is **not** here. That is the open frontier;
see ``docs/18_worldmodel_plan_fa.md``.

Everything below is online, local and interpretable -- no backprop, no offline
training loop.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np


class PredictiveWorldModel:
    """A staged predictive model of "what follows what" (and "what an action
    causes"), learned online from experience.

    Parameters
    ----------
    gamma:  discount for the successor representation (how far ahead it looks).
    lr:     temporal-difference learning rate for the SR.
    """

    def __init__(self, gamma: float = 0.9, lr: float = 0.2, seed: int = 0):
        self.gamma = float(gamma)
        self.lr = float(lr)
        self.states: List[str] = []
        self.index: Dict[str, int] = {}
        self.T = np.zeros((0, 0), np.float32)          # transition counts
        self.M = np.zeros((0, 0), np.float32)          # successor representation
        self.Ta: Dict[str, np.ndarray] = {}            # action-conditioned counts
        self.belief: Optional[int] = None              # latent state estimate
        self._rng = np.random.default_rng(seed)

    # -- growth: the model's state space grows as the world reveals itself ---
    def _ensure(self, name: str) -> int:
        if name in self.index:
            return self.index[name]
        i = len(self.states)
        self.index[name] = i
        self.states.append(name)
        n = i + 1
        T = np.zeros((n, n), np.float32)
        if self.T.size:
            T[:i, :i] = self.T
        self.T = T
        M = np.eye(n, dtype=np.float32)                # SR prior: you are where you are
        if self.M.size:
            M[:i, :i] = self.M
        self.M = M
        for a in list(self.Ta):
            A = np.zeros((n, n), np.float32)
            A[:i, :i] = self.Ta[a]
            self.Ta[a] = A
        return i

    @property
    def n_states(self) -> int:
        return len(self.states)

    # -- Stage A + B + C: learn from one observed transition -----------------
    def observe(self, prev: str, cur: str, action: Optional[str] = None) -> float:
        """Learn ``prev -> cur`` (optionally under ``action``) and return the
        **surprise** it caused -- prediction error, the teaching signal."""
        surprise = self.surprise(prev, cur)
        i = self._ensure(prev)
        j = self._ensure(cur)
        self.T[i, j] += 1.0
        if action is not None:
            if action not in self.Ta:
                self.Ta[action] = np.zeros_like(self.T)
            self.Ta[action][i, j] += 1.0
        # SR by TD(0):  M[i] <- M[i] + lr ( e_i + gamma * M[j] - M[i] )
        e = np.zeros(self.n_states, np.float32)
        e[i] = 1.0
        self.M[i] += self.lr * (e + self.gamma * self.M[j] - self.M[i])
        self.belief = j
        return float(surprise)

    def experience(self, seq: List[str],
                   actions: Optional[List[str]] = None) -> None:
        """Watch a whole trajectory (optionally with the actions that drove it)."""
        for k in range(len(seq) - 1):
            a = None if actions is None else actions[k]
            self.observe(seq[k], seq[k + 1], a)

    # -- Stage A: prediction + surprise --------------------------------------
    def _row(self, state: str, action: Optional[str]) -> Optional[np.ndarray]:
        i = self.index.get(state)
        if i is None:
            return None
        if action is not None and action in self.Ta:
            return self.Ta[action][i]
        return self.T[i]

    def transition_probs(self, state: str,
                         action: Optional[str] = None) -> Dict[str, float]:
        row = self._row(state, action)
        if row is None or row.sum() <= 0:
            return {}
        row = row / row.sum()
        return {self.states[j]: float(row[j])
                for j in range(self.n_states) if row[j] > 0}

    def predict_next(self, state: str,
                     action: Optional[str] = None) -> Optional[str]:
        """The single most likely next state (optionally under an action)."""
        row = self._row(state, action)
        if row is None or row.sum() <= 0:
            return None
        return self.states[int(row.argmax())]

    def surprise(self, prev: str, cur: str) -> float:
        """1 - P(cur | prev): how unexpected this transition was (in [0, 1])."""
        return 1.0 - self.transition_probs(prev).get(cur, 0.0)

    # -- Stage B: successor representation / multi-step look-ahead -----------
    def successor(self, state: str) -> Dict[str, float]:
        """Expected discounted future occupancy from ``state`` (the SR row)."""
        i = self.index.get(state)
        if i is None:
            return {}
        row = self.M[i]
        return {self.states[j]: float(row[j]) for j in range(self.n_states)}

    def reachable(self, state: str, top: int = 5) -> List[Tuple[str, float]]:
        """Where does this state tend to lead? (ranked by the SR, self excluded)"""
        sr = [(k, v) for k, v in self.successor(state).items() if k != state]
        return sorted(sr, key=lambda kv: -kv[1])[:top]

    # -- Stage C: model-based planning ---------------------------------------
    def plan(self, start: str, goal: str, max_depth: int = 20
             ) -> Optional[List[str]]:
        """A sequence of **actions** the learned action-model expects to lead
        from ``start`` to ``goal`` (breadth-first over expected outcomes)."""
        if not self.Ta or start not in self.index or goal not in self.index:
            return None
        q = deque([(start, [])])
        seen = {start}
        while q:
            s, path = q.popleft()
            if s == goal:
                return path
            if len(path) >= max_depth:
                continue
            for a, A in self.Ta.items():
                i = self.index[s]
                if A[i].sum() <= 0:
                    continue
                nxt = self.states[int(A[i].argmax())]
                if nxt not in seen:
                    seen.add(nxt)
                    q.append((nxt, path + [a]))
        return None

    def rollout(self, start: str, steps: int = 6, temperature: float = 0.5
                ) -> List[str]:
        """Imagine a trajectory by sampling the predictive transitions."""
        cur = start
        out = [start]
        for _ in range(steps):
            probs = self.transition_probs(cur)
            if not probs:
                break
            names = list(probs)
            p = np.array([probs[n] for n in names], np.float64)
            p = p ** (1.0 / max(temperature, 1e-3))
            p /= p.sum()
            cur = names[int(self._rng.choice(len(names), p=p))]
            out.append(cur)
        return out

    # -- Stage D: latent belief (partial object permanence) ------------------
    def update_belief(self, obs: Optional[str] = None,
                      action: Optional[str] = None) -> Optional[str]:
        """Maintain a state estimate. With an observation, adopt it; **without
        one, roll the belief forward through the model** -- the state persists
        and evolves even while unobserved."""
        if obs is not None:
            self.belief = self._ensure(obs)
        elif self.belief is not None:
            nxt = self.predict_next(self.states[self.belief], action)
            if nxt is not None:
                self.belief = self.index[nxt]
        return None if self.belief is None else self.states[self.belief]


# -- a structured environment to MEASURE the model on ------------------------

class GridWorld:
    """A tiny deterministic-ish grid environment, used to *measure* the world
    model honestly: states are cells, actions move between them (with optional
    slip noise). It is the ground truth the model must learn to predict/plan."""

    ACTIONS = ("up", "down", "left", "right")

    def __init__(self, w: int = 5, h: int = 5, slip: float = 0.0, seed: int = 0):
        self.w, self.h, self.slip = w, h, float(slip)
        self._rng = np.random.default_rng(seed)

    def name(self, x: int, y: int) -> str:
        return f"c{x},{y}"

    def step(self, x: int, y: int, a: str) -> Tuple[int, int]:
        if self.slip and self._rng.random() < self.slip:
            a = self.ACTIONS[self._rng.integers(4)]     # a slip: acted-otherwise
        if a == "up":
            y = min(self.h - 1, y + 1)
        elif a == "down":
            y = max(0, y - 1)
        elif a == "left":
            x = max(0, x - 1)
        elif a == "right":
            x = min(self.w - 1, x + 1)
        return x, y

    def random_walk(self, steps: int, seed: int = 0
                    ) -> Tuple[List[str], List[str]]:
        rng = np.random.default_rng(seed)
        x, y = int(rng.integers(self.w)), int(rng.integers(self.h))
        states, actions = [self.name(x, y)], []
        for _ in range(steps):
            a = self.ACTIONS[rng.integers(4)]
            x, y = self.step(x, y, a)
            actions.append(a)
            states.append(self.name(x, y))
        return states, actions

    def true_dist(self, s: str, t: str) -> int:
        (ax, ay), (bx, by) = self._coord(s), self._coord(t)
        return abs(ax - bx) + abs(ay - by)

    @staticmethod
    def _coord(name: str) -> Tuple[int, int]:
        x, y = name[1:].split(",")
        return int(x), int(y)
