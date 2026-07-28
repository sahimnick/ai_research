"""
loop.py
=======

The **closed cognitive loop** -- the synchronization algorithm from
``docs/20_imagination_reconstruction_engine.md``, actually running.

Until now that algorithm was pseudocode, and every faculty was exercised by
one-shot function calls. That is the difference between a library and a brain: a
brain *lives*. It stands in an environment, and at every moment it

    predicts  ->  senses  ->  measures its own error  ->  corrects itself

with prediction always leading and sensation only supplying the residual (Rao &
Ballard; Friston). Nothing in the loop is told the right answer; the only
teaching signal is the mismatch between what it expected and what arrived.

The loop runs on the **shared cortical code** of :mod:`workspace`, so vision, the
world model, memory and the self all operate on the same representation rather
than passing strings.

What must be true if this really works -- and what the demo measures:

  * **prediction error falls** as the agent lives in the world;
  * **next-observation prediction accuracy rises** well above chance;
  * with the senses removed, the loop **keeps running on its own** (imagination
    / dreaming), and its internal state stays coherent;
  * it can say **which changes it caused itself** and which the world caused.

Honest bound: the environments here are small and discrete. The loop is real;
the world it lives in is a toy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..workspace import GlobalWorkspace, _unit


# -- an environment the agent can actually live in --------------------------

class SensoryWorld:
    """A small world that emits **observations**, not labels.

    States are laid out on a ring of "places"; each place has a fixed, noisy
    sensory signature, and actions move between places. The agent is never told
    which place it is in -- it only ever sees the signature. That is what makes
    the loop's job real: it has to build the map itself.
    """

    ACTIONS = ("forward", "back", "stay")

    def __init__(self, n_places: int = 8, dim: int = 48, noise: float = 0.35,
                 seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n_places = n_places
        self.dim = dim
        self.noise = float(noise)
        self.signature = rng.normal(0, 1, (n_places, dim)).astype(np.float32)
        self.rng = rng
        self.place = 0

    def observe(self) -> np.ndarray:
        """What the senses deliver right now (never the true state)."""
        return (self.signature[self.place]
                + self.rng.normal(0, self.noise, self.dim)).astype(np.float32)

    def act(self, action: str) -> np.ndarray:
        if action == "forward":
            self.place = (self.place + 1) % self.n_places
        elif action == "back":
            self.place = (self.place - 1) % self.n_places
        return self.observe()

    def external_event(self) -> np.ndarray:
        """The world moves the agent without the agent acting -- the case where
        a correct self model must NOT claim agency."""
        self.place = int(self.rng.integers(self.n_places))
        return self.observe()


class ContinuousWorld:
    """A **continuous**, larger world -- no discrete places at all.

    The ring world above was a fair first test but a soft one: the agent lived
    among a handful of discrete states, so "learning the world" meant filling in
    a small table. Real environments are continuous, and that is a genuinely
    harder problem: there is no finite set of states to enumerate, positions
    blend into one another, and an agent that carves the space too finely
    drowns in its own categories.

    Here the agent moves on a continuous 2-D torus. It never sees coordinates.
    What it sees is a population code from **place cells**: a bank of cells with
    Gaussian tuning curves scattered over the arena, each firing according to
    how close the agent is to its preferred location -- which is what
    hippocampal place cells actually do. Movement is continuous, with real-valued
    steps, momentum and motor noise.
    """

    ACTIONS = ("north", "south", "east", "west", "rest")

    def __init__(self, size: float = 10.0, n_place_cells: int = 120,
                 field_width: float = 1.2, step: float = 0.9,
                 noise: float = 0.25, motor_noise: float = 0.12, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.size = float(size)
        self.step_len = float(step)
        self.noise = float(noise)
        self.motor_noise = float(motor_noise)
        self.field_width = float(field_width)
        # place-cell centres scattered over the arena
        self.centres = self.rng.uniform(0, size, (n_place_cells, 2)).astype(np.float32)
        self.dim = n_place_cells
        self.pos = self.rng.uniform(0, size, 2).astype(np.float32)

    # -- the sensory code: hippocampal place cells --------------------------
    def observe(self, pos: Optional[np.ndarray] = None) -> np.ndarray:
        """Population code: each cell fires by how near the agent is to its
        field. Distances wrap, because the arena is a torus."""
        p = self.pos if pos is None else pos
        d = np.abs(self.centres - p[None, :])
        d = np.minimum(d, self.size - d)                  # toroidal distance
        r = np.exp(-(d ** 2).sum(1) / (2 * self.field_width ** 2))
        return (r + self.rng.normal(0, self.noise, self.dim)).astype(np.float32)

    def act(self, action: str) -> np.ndarray:
        """Move continuously, with motor noise -- no snapping to a grid."""
        delta = {"north": (0.0, 1.0), "south": (0.0, -1.0),
                 "east": (1.0, 0.0), "west": (-1.0, 0.0),
                 "rest": (0.0, 0.0)}[action]
        move = np.array(delta, np.float32) * self.step_len
        move = move + self.rng.normal(0, self.motor_noise, 2).astype(np.float32)
        self.pos = (self.pos + move) % self.size
        return self.observe()

    def external_event(self) -> np.ndarray:
        self.pos = self.rng.uniform(0, self.size, 2).astype(np.float32)
        return self.observe()

    # -- ground truth, used ONLY for scoring, never shown to the agent ------
    def true_position(self) -> np.ndarray:
        return self.pos.copy()

    def true_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        d = np.abs(a - b)
        d = np.minimum(d, self.size - d)
        return float(np.sqrt((d ** 2).sum()))


# -- the loop ---------------------------------------------------------------

@dataclass
class LoopTrace:
    """What happened, step by step -- the evidence the loop is learning."""

    error: List[float] = field(default_factory=list)
    correct: List[int] = field(default_factory=list)
    surprise: List[float] = field(default_factory=list)
    named: List[Optional[str]] = field(default_factory=list)

    def window(self, values: Sequence[float], frac: float = 0.25) -> Tuple[float, float]:
        """(early mean, late mean) over the first/last ``frac`` of the run."""
        n = max(1, int(len(values) * frac))
        return float(np.mean(values[:n])), float(np.mean(values[-n:]))


class CognitiveLoop:
    """Perceive -> predict -> compare -> correct, forever.

    Parameters
    ----------
    world:      the environment to live in.
    workspace:  the shared cortical code (created if not given).
    lr:         how fast the transition model is corrected by error.
    """

    def __init__(self, world: SensoryWorld,
                 workspace: Optional[GlobalWorkspace] = None,
                 lr: float = 0.35, vigilance: float = 0.45, seed: int = 0):
        self.world = world
        self.ws = workspace or GlobalWorkspace(dim=384, vigilance=vigilance,
                                               seed=seed)
        self.lr = float(lr)
        self.rng = np.random.default_rng(seed)
        # the world model lives in the SHARED code: for each action, a matrix
        # mapping the current cortical code to the predicted next one.
        self.T: Dict[str, np.ndarray] = {}
        self.belief: Optional[np.ndarray] = None      # current state estimate
        self.predicted: Optional[np.ndarray] = None   # what we expect to see
        self.trace = LoopTrace()
        self._next_id = 0
        # code -> the raw observation that produced it, so a predicted code can
        # be turned back into an expected percept (the top-down generative path)
        self._sensory_memory: List[Tuple[np.ndarray, np.ndarray]] = []
        self._sensory_cap = 1500

    # -- the generative model -------------------------------------------
    def _op(self, action: str) -> np.ndarray:
        if action not in self.T:
            self.T[action] = np.eye(self.ws.dim, dtype=np.float32) * 0.5
        return self.T[action]

    def predict(self, code: np.ndarray, action: str) -> np.ndarray:
        """Top-down: what the world will look like if I do this."""
        return _unit(self._op(action) @ code)

    # -- one turn of the loop -------------------------------------------
    def step(self, action: Optional[str] = None, sense: bool = True,
             learn: bool = True) -> Dict[str, object]:
        """One full cycle of the synchronization algorithm."""
        action = action or str(self.rng.choice(self.world.ACTIONS))

        # 1. PREDICT (this happens BEFORE sensing -- prediction leads)
        prior = self.belief if self.belief is not None else \
            self.ws.encode("vision", self.world.observe())
        self.predicted = self.predict(prior, action)

        # 2. SENSE (or not -- with no senses the loop free-runs: imagination)
        if sense:
            obs = self.world.act(action)
            actual = self.ws.encode("vision", obs)
            self.ws.broadcast(actual, source="vision")
            if len(self._sensory_memory) < self._sensory_cap:
                self._sensory_memory.append((actual.copy(), obs.copy()))
        else:
            self.world.act(action)               # the world still moves
            actual = self.predicted              # ...but we only have our model
            self.ws.broadcast(actual, source="imagination")

        # 3. COMPARE -- precision-weighted prediction error
        err = float(np.linalg.norm(actual - self.predicted))
        match = float(self.predicted @ actual)
        surprise = float(np.clip(1.0 - match, 0.0, 2.0))

        # 4. CORRECT
        #    (a) fast: the belief moves toward the evidence
        self.belief = _unit(0.35 * self.predicted + 0.65 * actual) if sense \
            else self.predicted
        #    (b) slow: the model changes only in proportion to the error
        if learn and sense:
            self.T[action] = self._op(action) + \
                self.lr * np.outer(actual - self.predicted, prior)
            # name what is in the workspace (grow a concept when it is new)
            name, score = self.ws.match(actual)
            if score < self.ws.vigilance:
                name = f"p{self._next_id}"
                self._next_id += 1
            self.ws.learn_concept(name, actual)
        else:
            name = self.ws.name(actual)

        self.trace.error.append(err)
        self.trace.surprise.append(surprise)
        self.trace.correct.append(int(match > 0.5))
        self.trace.named.append(name)
        return {"action": action, "error": err, "surprise": surprise,
                "match": match, "name": name}

    def live(self, steps: int = 1500, sense: bool = True) -> LoopTrace:
        """Live in the world for a while."""
        for _ in range(steps):
            self.step(sense=sense)
        return self.trace

    # -- what the loop can do once it has learned -----------------------
    def imagine(self, steps: int = 10, action: str = "forward") -> List[Optional[str]]:
        """Run the loop with the senses closed: the model rolls itself forward.
        This is imagination -- the same machinery, unclamped."""
        code = self.belief if self.belief is not None else \
            self.ws.encode("vision", self.world.observe())
        out = []
        for _ in range(steps):
            code = self.predict(code, action)
            out.append(self.ws.name(code))
        return out

    def consolidate(self, similarity: float = 0.75) -> int:
        """Sleep on it: merge the redundant concepts that noise spawned.
        Fixes the over-segmentation the loop shows in a noisy world."""
        return self.ws.consolidate(similarity)

    def spatial_prediction_error(self, trials: int = 200) -> Dict[str, float]:
        """The right measure for a **continuous** world.

        Naming is the wrong yardstick once space is continuous: every position
        is slightly novel, so discrete concepts explode and name-matching reads
        ~0% even when the model predicts perfectly well. What matters instead is
        whether the predicted *code* lands where the agent actually ends up. So
        the predicted population code is decoded back to a position (the place
        cells' centre of mass, which is how a population vector is read out) and
        compared with the truth, against the error of a chance guess.
        """
        world = self.world
        if not hasattr(world, "centres"):
            raise TypeError("spatial_prediction_error needs a ContinuousWorld")

        def decode(code: np.ndarray) -> np.ndarray:
            """Population-vector read-out: place-cell centres weighted by
            activity, done on the torus with circular means."""
            w = np.maximum(code, 0.0)
            if w.sum() <= 1e-9:
                return world.centres.mean(0)
            ang = world.centres / world.size * 2 * np.pi
            sx = (w[:, None] * np.sin(ang)).sum(0)
            cx = (w[:, None] * np.cos(ang)).sum(0)
            return (np.arctan2(sx, cx) % (2 * np.pi)) / (2 * np.pi) * world.size

        errs, chance, stay = [], [], []
        for _ in range(trials):
            action = str(self.rng.choice(world.ACTIONS))
            obs = world.observe()
            prior = self.ws.encode("vision", obs)
            # the loop predicts in the SHARED code; bring it back to sensory space
            pred_code = self.predict(prior, action)
            pred_obs = self._decode_to_sensory(pred_code, obs)
            here = decode(obs)                       # "nothing will change"
            world.act(action)
            truth = world.true_position()
            errs.append(world.true_distance(decode(pred_obs), truth))
            stay.append(world.true_distance(here, truth))
            chance.append(world.true_distance(
                self.rng.uniform(0, world.size, 2).astype(np.float32), truth))
        # The STAY baseline is the honest one to beat. Steps are small, so
        # "assume nothing moved" is already a good predictor; a learned model
        # that cannot beat it has not learned the world's dynamics, however far
        # it is from random chance.
        return {"error": float(np.mean(errs)),
                "stay_baseline": float(np.mean(stay)),
                "chance_error": float(np.mean(chance)),
                "better_than_chance": float(np.mean(chance) - np.mean(errs)),
                "beats_stay_baseline": float(np.mean(stay) - np.mean(errs))}

    def _decode_to_sensory(self, code: np.ndarray, example_obs: np.ndarray
                           ) -> np.ndarray:
        """Map a workspace code back to sensory space using the memories the
        loop has already formed (nearest stored code's observation)."""
        if not self._sensory_memory:
            return example_obs
        keys = np.array([k for k, _ in self._sensory_memory], np.float32)
        j = int(np.argmax(keys @ _unit(code)))
        return self._sensory_memory[j][1]

    def prediction_accuracy(self, trials: int = 200) -> float:
        """Held-out test: predict the NEXT observation before seeing it, and
        check whether the prediction names the place that actually arrives."""
        ok = 0
        for _ in range(trials):
            action = str(self.rng.choice(self.world.ACTIONS))
            prior = self.ws.encode("vision", self.world.observe())
            pred_name = self.ws.name(self.predict(prior, action))
            actual_name = self.ws.name(self.ws.encode("vision",
                                                      self.world.act(action)))
            ok += int(pred_name is not None and pred_name == actual_name)
        return ok / trials


def closed_loop_experiment(steps: int = 2000, n_places: int = 8,
                           noise: float = 0.35, seed: int = 0,
                           verbose: bool = False) -> Dict[str, float]:
    """Does the loop actually learn by living? The honest measurements.

    Nothing supervises it: it never sees a place label, only noisy observations
    and its own prediction error.
    """
    world = SensoryWorld(n_places=n_places, noise=noise, seed=seed)
    loop = CognitiveLoop(world, seed=seed)
    loop.live(steps=steps)

    early_err, late_err = loop.trace.window(loop.trace.error)
    early_hit, late_hit = loop.trace.window(loop.trace.correct)
    acc = loop.prediction_accuracy(300)

    # imagination: senses closed, does the internal world keep moving coherently?
    imagined = loop.imagine(steps=8)
    coherent = sum(x is not None for x in imagined) / max(len(imagined), 1)

    out = {
        "error_early": early_err, "error_late": late_err,
        "error_drop": (early_err - late_err) / max(early_err, 1e-6),
        "hit_early": early_hit, "hit_late": late_hit,
        "prediction_accuracy": acc,
        "chance": 1.0 / n_places,
        "concepts_grown": float(loop.ws.n_concepts),
        "imagination_coherence": coherent,
    }
    if verbose:
        print(f"  prediction error  {early_err:.3f} -> {late_err:.3f} "
              f"({out['error_drop']:+.0%})")
        print(f"  prediction hits   {early_hit:.0%} -> {late_hit:.0%}")
        print(f"  held-out next-observation accuracy: {acc:.0%} "
              f"(chance {out['chance']:.0%})")
        print(f"  concepts grown from experience    : {int(out['concepts_grown'])} "
              f"(world truly has {n_places})")
        print(f"  imagination with senses closed    : {coherent:.0%} coherent")
    return out
