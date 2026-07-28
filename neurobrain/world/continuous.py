"""
continuous.py
=============

A **continuous predictive model** to replace the discrete loop.

The measured failure this fixes was blunt: in a continuous arena the
concept-naming loop of :mod:`loop` spawned 1364 "concepts" for a space that has
no discrete states, and its spatial error (1.63) was *worse* than the trivial
"assume nothing moved" baseline (0.96). Carving a continuum into named states is
the wrong move, and no amount of tuning rescues it.

What replaces it is what a brain actually does with continuous space: keep a
**population code** and learn how each action **transforms** that code. There
are no states to enumerate. The model is one matrix per action, ``A[a]``,
predicting the next place-cell population vector from the current one, trained
purely by its own prediction error (delta rule, local, online).

Two things make it work where the discrete loop failed:

  * **Predict the change, not the state.** The model learns ``delta = next -
    now`` and adds it to the current code. Since most of the next observation is
    the current one, this hands the model the stay-baseline for free and makes
    it spend its capacity on the part that actually moves -- which is the part
    an action causes.
  * **Score in the world's own units.** The predicted code is decoded back to a
    position with a population vector read-out, and compared against both chance
    and the stay baseline. Beating chance is meaningless here; beating *stay* is
    the real bar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ContinuousPredictor:
    """Learns, per action, how the population code is transformed.

    ``A[action]`` maps the current place-cell code to the **change** it will
    undergo. Learning is the delta rule on the model's own error -- no labels,
    no backprop through time, nothing global."""

    dim: int
    lr: float = 0.05
    A: Dict[str, np.ndarray] = field(default_factory=dict)
    bias: Dict[str, np.ndarray] = field(default_factory=dict)
    errors: List[float] = field(default_factory=list)

    def _op(self, action: str) -> np.ndarray:
        if action not in self.A:
            self.A[action] = np.zeros((self.dim, self.dim), np.float32)
            self.bias[action] = np.zeros(self.dim, np.float32)
        return self.A[action]

    def predict(self, obs: np.ndarray, action: str) -> np.ndarray:
        """Next observation = now + the change this action causes."""
        o = np.asarray(obs, np.float32)
        return o + self._op(action) @ o + self.bias[action]

    def learn(self, obs: np.ndarray, action: str, nxt: np.ndarray) -> float:
        """One step of the delta rule on the prediction error."""
        o = np.asarray(obs, np.float32)
        target = np.asarray(nxt, np.float32) - o          # the CHANGE
        pred = self._op(action) @ o + self.bias[action]
        err = target - pred
        self.A[action] += self.lr * np.outer(err, o) / (float(o @ o) + 1e-6)
        self.bias[action] += self.lr * 0.1 * err
        e = float(np.linalg.norm(err))
        self.errors.append(e)
        return e

    def live(self, world, steps: int = 4000, rng: Optional[np.random.Generator] = None
             ) -> List[float]:
        """Wander the world, predicting before every move and correcting after."""
        rng = rng or np.random.default_rng(0)
        obs = world.observe()
        for _ in range(steps):
            a = str(rng.choice(world.ACTIONS))
            nxt = world.act(a)
            self.learn(obs, a, nxt)
            obs = nxt
        return self.errors


def decode_position(world, code: np.ndarray) -> np.ndarray:
    """Population-vector read-out on the torus: place-cell centres weighted by
    activity, averaged as circular means."""
    w = np.maximum(np.asarray(code, np.float32), 0.0)
    if w.sum() <= 1e-9:
        return world.centres.mean(0)
    ang = world.centres / world.size * 2 * np.pi
    sx = (w[:, None] * np.sin(ang)).sum(0)
    cx = (w[:, None] * np.cos(ang)).sum(0)
    return (np.arctan2(sx, cx) % (2 * np.pi)) / (2 * np.pi) * world.size


def continuous_prediction_experiment(steps: int = 6000, trials: int = 400,
                                     size: float = 10.0, n_place_cells: int = 120,
                                     seed: int = 0, verbose: bool = False
                                     ) -> Dict[str, float]:
    """Does the continuous model beat the baseline the discrete loop lost to?

    Scored in metres of the arena: the model's error, the "nothing moved"
    baseline, and random chance."""
    from .loop import ContinuousWorld
    world = ContinuousWorld(size=size, n_place_cells=n_place_cells, seed=seed)
    model = ContinuousPredictor(dim=n_place_cells, lr=0.05)
    rng = np.random.default_rng(seed)
    model.live(world, steps=steps, rng=rng)

    errs, stay, chance = [], [], []
    for _ in range(trials):
        a = str(rng.choice(world.ACTIONS))
        obs = world.observe()
        pred = model.predict(obs, a)
        here = decode_position(world, obs)
        world.act(a)
        truth = world.true_position()
        errs.append(world.true_distance(decode_position(world, pred), truth))
        stay.append(world.true_distance(here, truth))
        chance.append(world.true_distance(
            rng.uniform(0, world.size, 2).astype(np.float32), truth))

    n = max(1, len(model.errors) // 5)
    out = {"error": float(np.mean(errs)),
           "stay_baseline": float(np.mean(stay)),
           "chance_error": float(np.mean(chance)),
           "beats_stay": float(np.mean(stay) - np.mean(errs)),
           "learning_early": float(np.mean(model.errors[:n])),
           "learning_late": float(np.mean(model.errors[-n:]))}
    if verbose:
        print(f"   prediction error while living: "
              f"{out['learning_early']:.3f} -> {out['learning_late']:.3f}")
        print(f"   random chance          : {out['chance_error']:.2f}")
        print(f"   'nothing moved' baseline: {out['stay_baseline']:.2f}  "
              f"<- the real bar")
        print(f"   CONTINUOUS MODEL        : {out['error']:.2f}  "
              f"({out['beats_stay']:+.2f} vs the baseline)")
    return out
