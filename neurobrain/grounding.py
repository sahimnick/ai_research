"""
grounding.py
============

**vision -> objects**: discovering visual properties from **causal prediction
error**, so the causal world graph stops being fed hand-typed symbols.

:mod:`objects` learns real causal rules, but its inputs are property *words*
("fragile", "contains_liquid") that a human wrote down. That is the symbol-
grounding gap: the causal reasoning is genuine, the symbols are borrowed.

The obvious way to close it is a trap, and it is worth naming: train a classifier
to read "fragile" off the image and hand that to the graph. That grounds nothing
-- it only moves the human annotation one layer down. The features are still
defined by the labels a person chose.

The way out is to define a property by **what it does**, which is also what a
property *is* to a brain: a dimension of appearance that changes what happens
when you act. Gibson's affordances are exactly this. So:

    a visual feature is useful precisely insofar as knowing it
    reduces the error of predicting the outcome of an action

Concretely, this module learns a small bank of visual feature detectors whose
*only* training signal is causal prediction error: a feature's weights move to
make the outcome of ``action(object)`` more predictable. No property names are
ever shown. Afterwards we can *check* -- but never train on -- how well the
discovered features line up with the physical properties the environment
actually used, and feed them to :class:`~neurobrain.objects.CausalWorldGraph`.

Honest framing: the environment here is a rendered 2-D micro-world whose physics
this repo does write. What is *not* given to the learner is which visual
dimensions matter -- that is what it has to discover, and that is the part being
tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v.astype(np.float32) if n < 1e-9 else (v / n).astype(np.float32)


# -- a visual micro-world whose LOOK is correlated with its PHYSICS -----------

#: hidden physical properties, and the visual signature each one produces.
#: The learner never sees this table -- it only sees pixels and outcomes.
_APPEARANCE = {
    "fragile":         dict(texture="glassy",  edge=0.85),   # thin bright outline
    "elastic":         dict(texture="matte",   bounce=0.9),  # rounded, saturated
    "contains_liquid": dict(texture="filled",  fill=0.7),    # partly filled body
    "heavy":           dict(texture="dense",   dark=0.8),    # dark, solid
    "soft":            dict(texture="fuzzy",   blur=0.8),    # blurred boundary
}


def render_object(props: Sequence[str], size: int = 20,
                  rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Draw an object whose APPEARANCE reflects its hidden physical properties.

    The mapping is deliberately indirect and noisy -- fragility shows up as a
    thin bright rim, liquid as a partly filled body, softness as a blurred
    edge -- so the learner has to find the informative dimensions, not read a
    label off a pixel."""
    rng = rng or np.random.default_rng()
    img = np.zeros((size, size), np.float32)
    cy, cx = size / 2 + rng.uniform(-1, 1), size / 2 + rng.uniform(-1, 1)
    r = size * rng.uniform(0.26, 0.34)
    yy, xx = np.mgrid[0:size, 0:size]
    d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    body = 0.35 + 0.1 * rng.normal()
    img[d <= r] = body
    if "heavy" in props:                       # dense, dark body
        img[d <= r] = 0.85 + 0.05 * rng.normal()
    if "contains_liquid" in props:             # lower part filled
        img[(d <= r) & (yy > cy)] = 0.65
    rim = np.abs(d - r) < 1.2
    img[rim] = 0.9 if "fragile" in props else 0.55   # fragile -> bright thin rim
    if "elastic" in props:                     # rounded highlight
        img[(d < r * 0.55)] = np.maximum(img[(d < r * 0.55)], 0.75)
    if "soft" in props:                        # blurred boundary
        k = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 16.0
        pad = np.pad(img, 1, mode="edge")
        img = sum(k[i, j] * pad[i:i + size, j:j + size]
                  for i in range(3) for j in range(3)).astype(np.float32)
    img += rng.normal(0, 0.05, (size, size)).astype(np.float32)
    return np.clip(img, 0, 1).astype(np.float32)


def _outcomes(action: str, props: Sequence[str]) -> Dict[str, int]:
    """The world's physics. The learner NEVER sees this -- it only ever observes
    which outcomes happened after it acted."""
    p = set(props)
    out = {"fall": 0, "break": 0, "spill": 0, "bounce": 0, "deform": 0,
           "slide": 0}
    if action == "drop":
        out["fall"] = 1
        out["break"] = int("fragile" in p)
        out["spill"] = int("contains_liquid" in p)
        out["bounce"] = int("elastic" in p)
        out["deform"] = int("soft" in p)
    elif action == "squeeze":
        out["deform"] = int("soft" in p or "elastic" in p)
        out["break"] = int("fragile" in p)
        out["spill"] = int("contains_liquid" in p)
    elif action == "push":
        out["slide"] = int("heavy" not in p)
        out["bounce"] = int("elastic" in p)
    return out


ACTIONS = ("drop", "squeeze", "push")
OUTCOMES = ("fall", "break", "spill", "bounce", "deform", "slide")
ALL_PROPS = tuple(_APPEARANCE)


def sample_episode(rng: np.random.Generator, size: int = 20
                   ) -> Tuple[np.ndarray, str, Dict[str, int], List[str]]:
    """One experience: see an object, act on it, observe what happened."""
    props = [p for p in ALL_PROPS if rng.random() < 0.4]
    img = render_object(props, size=size, rng=rng)
    action = ACTIONS[rng.integers(len(ACTIONS))]
    return img, action, _outcomes(action, props), props


# -- the learner: features defined by their causal usefulness ---------------

class CausalFeatureLearner:
    """Learns visual features whose ONLY reason to exist is causal prediction.

    ``F`` holds ``n_features`` linear filters over the image. A feature's
    activation is ``sigmoid(F @ pixels)``. From those activations, one logistic
    read-out per (action, outcome) predicts whether the outcome will happen.
    **Both** the read-outs and the filters are trained by the same delta rule on
    the outcome error -- so a filter only survives by making outcomes
    predictable. Nothing in the objective mentions a property name.
    """

    def __init__(self, n_pixels: int, n_features: int = 12, lr: float = 0.08,
                 seed: int = 0):
        rng = np.random.default_rng(seed)
        self.F = rng.normal(0, 0.05, (n_features, n_pixels)).astype(np.float32)
        self.b = np.zeros(n_features, np.float32)
        self.n_features = n_features
        self.lr = float(lr)
        # read-out weights per (action, outcome)
        self.W: Dict[Tuple[str, str], np.ndarray] = {}
        self.c: Dict[Tuple[str, str], float] = {}

    def features(self, image: np.ndarray) -> np.ndarray:
        x = np.asarray(image, np.float32).reshape(-1)
        return _sigmoid(self.F @ x + self.b).astype(np.float32)

    def _head(self, key: Tuple[str, str]) -> np.ndarray:
        if key not in self.W:
            self.W[key] = np.zeros(self.n_features, np.float32)
            self.c[key] = 0.0
        return self.W[key]

    def predict_outcome(self, image: np.ndarray, action: str) -> Dict[str, float]:
        f = self.features(image)
        return {o: float(_sigmoid(self._head((action, o)) @ f + self.c[(action, o)]))
                for o in OUTCOMES}

    def learn(self, image: np.ndarray, action: str,
              outcomes: Dict[str, int]) -> float:
        """One episode. The error of predicting what happened trains BOTH the
        read-out and the visual filters -- this is the whole idea."""
        x = np.asarray(image, np.float32).reshape(-1)
        f = self.features(image)
        dfeat = np.zeros(self.n_features, np.float32)
        total = 0.0
        for o in OUTCOMES:
            key = (action, o)
            w = self._head(key)
            p = float(_sigmoid(w @ f + self.c[key]))
            err = float(outcomes.get(o, 0)) - p
            total += abs(err)
            w += self.lr * err * f                    # read-out
            self.c[key] += self.lr * err
            dfeat += err * w                          # error flowing to features
        # the filters move to make the outcome more predictable
        g = dfeat * f * (1.0 - f)                     # through the sigmoid
        self.F += self.lr * 0.5 * np.outer(g, x)
        self.b += self.lr * 0.5 * g
        return total / len(OUTCOMES)

    def train(self, n_episodes: int = 6000, size: int = 20, seed: int = 0
              ) -> List[float]:
        """Live through episodes; return the error curve (should fall)."""
        rng = np.random.default_rng(seed)
        hist, run = [], []
        for i in range(n_episodes):
            img, action, out, _ = sample_episode(rng, size=size)
            run.append(self.learn(img, action, out))
            if (i + 1) % 500 == 0:
                hist.append(float(np.mean(run[-500:])))
        return hist


@dataclass
class GroundedCausalMind:
    """Vision -> discovered features -> causal graph, with nothing hand-typed."""

    learner: CausalFeatureLearner
    size: int = 20
    outcome_accuracy: float = 0.0        # exact outcome-SET match, unseen objects
    per_outcome_accuracy: float = 0.0    # per-outcome correctness (the fair read)
    property_alignment: float = 0.0      # do features track real properties?
    n_grounded_symbols: int = 0

    def perceive_properties(self, image: np.ndarray, thresh: float = 0.5
                            ) -> List[str]:
        """The object's *discovered* property symbols -- named f0, f1, ... because
        the system found them itself and nobody told it what they mean."""
        f = self.learner.features(image)
        return [f"f{i}" for i in range(len(f)) if f[i] > thresh]

    def expect(self, image: np.ndarray, action: str, thresh: float = 0.5
               ) -> List[str]:
        """What will this action do to this object -- straight from pixels."""
        p = self.learner.predict_outcome(image, action)
        return [o for o, v in p.items() if v >= thresh]


def build_grounded_causal_mind(n_episodes: int = 8000, size: int = 20,
                               seed: int = 0, verbose: bool = False
                               ) -> GroundedCausalMind:
    """Learn visual features from causal error alone, then measure honestly.

    Two very different questions get answered:
      * can it predict the consequences of acting on **objects it has never
        seen**, straight from pixels? (the thing that matters)
      * do the features it invented line up with the physical properties the
        world actually used? (a diagnostic -- never a training signal)
    """
    rng = np.random.default_rng(seed)
    learner = CausalFeatureLearner(n_pixels=size * size, n_features=12, seed=seed)

    def say(*a):
        if verbose:
            print(*a)

    say(f"living through {n_episodes} act-and-see episodes "
        f"(no property names, ever) ...")
    hist = learner.train(n_episodes=n_episodes, size=size, seed=seed)
    say(f"   outcome-prediction error: {hist[0]:.3f} -> {hist[-1]:.3f}")

    mind = GroundedCausalMind(learner, size=size)

    # -- does it predict consequences for objects it has never seen? --------
    ok = tot = 0
    per = np.zeros(len(OUTCOMES))
    rng_t = np.random.default_rng(seed + 999)
    for _ in range(1200):
        img, action, out, _ = sample_episode(rng_t, size=size)
        probs = learner.predict_outcome(img, action)
        pred = {o for o, v in probs.items() if v >= 0.5}
        true = {o for o, v in out.items() if v}
        ok += int(pred == true)
        for j, o in enumerate(OUTCOMES):
            per[j] += int((probs[o] >= 0.5) == bool(out[o]))
        tot += 1
    mind.outcome_accuracy = ok / tot
    mind.per_outcome_accuracy = float(per.mean() / tot)

    # -- diagnostic only: do the invented features track real properties? ---
    rng_d = np.random.default_rng(seed + 555)
    F, P = [], []
    for _ in range(900):
        props = [p for p in ALL_PROPS if rng_d.random() < 0.4]
        img = render_object(props, size=size, rng=rng_d)
        F.append(learner.features(img))
        P.append([int(p in props) for p in ALL_PROPS])
    F, P = np.array(F), np.array(P, float)
    aligns = []
    for j in range(P.shape[1]):
        if P[:, j].std() < 1e-6:
            continue
        cors = [abs(np.corrcoef(F[:, i], P[:, j])[0, 1]) for i in range(F.shape[1])]
        aligns.append(float(np.nanmax(cors)))
    mind.property_alignment = float(np.mean(aligns)) if aligns else 0.0
    mind.n_grounded_symbols = learner.n_features

    if verbose:
        print(f"   predict consequences for UNSEEN objects: "
              f"{mind.per_outcome_accuracy:.1%} per outcome, "
              f"{mind.outcome_accuracy:.1%} exact-set")
        print(f"   discovered features vs. real properties : "
              f"{mind.property_alignment:.2f} correlation")
        print(f"   symbols the system made for itself      : "
              f"{mind.n_grounded_symbols} (f0..f{mind.n_grounded_symbols - 1})")
    return mind


def grounded_vs_handtyped(n_episodes: int = 8000, verbose: bool = False
                          ) -> Dict[str, float]:
    """The comparison that matters: pixels-with-discovered-features versus the
    hand-typed property symbols :mod:`objects` was given."""
    from .objects import build_causal_world, causal_generalisation_score
    grounded = build_grounded_causal_mind(n_episodes=n_episodes, verbose=verbose)
    handtyped = causal_generalisation_score(build_causal_world())
    out = {"grounded_from_pixels": grounded.per_outcome_accuracy,
           "grounded_exact_set": grounded.outcome_accuracy,
           "hand_typed_symbols": float(handtyped),
           "property_alignment": grounded.property_alignment}
    if verbose:
        print(f"\n   from PIXELS, features discovered by causal error : "
              f"{out['grounded_from_pixels']:.1%}")
        print(f"   from HAND-TYPED property symbols                 : "
              f"{out['hand_typed_symbols']:.1%}")
    return out
