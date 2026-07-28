"""
development.py
==============

**Developmental learning** and **dream consolidation** -- the two temporal
processes that turn experience into knowledge.

A brain is not trained once on a shuffled dataset. It *develops*:

  * **Critical periods.** Plasticity is high early and declines, and it declines
    **per circuit**, gated by how competent that circuit already is. Ocular
    dominance plasticity closes when V1 has learned to see; language plasticity
    closes later. Here, plasticity decays as a function of the circuit's own
    mastery, so a circuit that has not yet learned stays plastic.

  * **Staged curricula.** Development gates *what can be learned next*: the
    system does not advance to the next stage until it has mastered the current
    one. This is not curriculum engineering; it is the reason a child learns
    objects before verbs and verbs before syntax.

  * **Sleep / dream consolidation.** During slow-wave sleep the hippocampus
    **replays** compressed sequences of the day's episodes and the neocortex
    slowly extracts the invariants. Replay is **prioritised**: surprising and
    rewarded episodes are replayed more (Foster & Wilson 2006; O'Neill 2010).
    This is the mechanism behind "sleep on it": generalisation improves
    overnight *without any new data*.

The honest claim this module supports is measurable: **the same experience,
consolidated by prioritised replay, generalises better than the same experience
without sleep.** No new information is added -- only re-processed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# -- episodic buffer (hippocampus) ------------------------------------------

@dataclass
class Episode:
    """One experienced event, tagged with the surprise it caused."""

    pattern: np.ndarray
    label: object
    surprise: float = 0.0
    replays: int = 0


class EpisodicBuffer:
    """A hippocampal store of the day's episodes, prioritised by surprise.

    Region: hippocampus (CA3/CA1) + subiculum.
    Function: rapid one-shot storage; the source of night-time replay.
    """

    def __init__(self, capacity: int = 2000):
        self.capacity = capacity
        self.episodes: List[Episode] = []

    def store(self, pattern: np.ndarray, label: object,
              surprise: float = 0.0) -> None:
        self.episodes.append(Episode(np.asarray(pattern, np.float32), label,
                                     float(surprise)))
        if len(self.episodes) > self.capacity:      # forget the least surprising
            self.episodes.sort(key=lambda e: -e.surprise)
            self.episodes = self.episodes[:self.capacity]

    def __len__(self) -> int:
        return len(self.episodes)

    def priorities(self, temperature: float = 1.0) -> np.ndarray:
        """Replay probability -- surprising episodes are replayed more."""
        s = np.array([e.surprise for e in self.episodes], np.float64)
        s = np.clip(s, 1e-3, None) ** (1.0 / max(temperature, 1e-3))
        return s / s.sum()

    def sample_replay(self, n: int, rng: np.random.Generator,
                      prioritised: bool = True) -> List[Episode]:
        if not self.episodes:
            return []
        n = min(n, len(self.episodes) * 4)
        p = self.priorities() if prioritised else None
        idx = rng.choice(len(self.episodes), size=n, replace=True, p=p)
        out = []
        for i in idx:
            self.episodes[i].replays += 1
            out.append(self.episodes[i])
        return out


# -- critical periods --------------------------------------------------------

class CriticalPeriod:
    """Competence-gated plasticity: a circuit stays plastic until it is good.

    Region: any cortical circuit (PV-interneuron maturation / perineuronal nets
    close the window in real cortex).
    Rule: ``plasticity = floor + (1 - floor) * (1 - mastery)^sharpness``.
    """

    def __init__(self, floor: float = 0.05, sharpness: float = 2.0):
        self.floor = float(floor)
        self.sharpness = float(sharpness)
        self.mastery: Dict[str, float] = {}

    def plasticity(self, circuit: str) -> float:
        m = float(np.clip(self.mastery.get(circuit, 0.0), 0.0, 1.0))
        return float(self.floor + (1.0 - self.floor) * (1.0 - m) ** self.sharpness)

    def update_mastery(self, circuit: str, competence: float,
                       lr: float = 0.3) -> float:
        m = self.mastery.get(circuit, 0.0)
        m += lr * (float(np.clip(competence, 0.0, 1.0)) - m)
        self.mastery[circuit] = float(m)
        return self.plasticity(circuit)

    def is_open(self, circuit: str, thresh: float = 0.25) -> bool:
        return self.plasticity(circuit) > thresh


# -- staged development ------------------------------------------------------

@dataclass
class Stage:
    """One developmental stage: you do not advance until you have mastered it."""

    name: str
    circuit: str
    train: Callable[[float], float]       # (plasticity) -> competence in [0,1]
    threshold: float = 0.9
    max_epochs: int = 40


class DevelopmentalProgram:
    """Run stages in order, advancing only on mastery, with plasticity that
    closes as each circuit matures."""

    def __init__(self, stages: Sequence[Stage], floor: float = 0.05):
        self.stages = list(stages)
        self.periods = CriticalPeriod(floor=floor)
        self.log: List[Dict[str, object]] = []

    def run(self, verbose: bool = False) -> List[Dict[str, object]]:
        for stage in self.stages:
            epochs = 0
            competence = 0.0
            while epochs < stage.max_epochs and competence < stage.threshold:
                p = self.periods.plasticity(stage.circuit)
                competence = float(stage.train(p))
                self.periods.update_mastery(stage.circuit, competence)
                epochs += 1
            entry = {"stage": stage.name, "circuit": stage.circuit,
                     "epochs": epochs, "competence": round(competence, 3),
                     "mastered": competence >= stage.threshold,
                     "plasticity_left": round(self.periods.plasticity(stage.circuit), 3)}
            self.log.append(entry)
            if verbose:
                print(f"  stage {stage.name:<12} competence {competence:.2f} "
                      f"in {epochs} epochs; plasticity now "
                      f"{entry['plasticity_left']:.2f}")
        return self.log


# -- sleep: replay-driven consolidation -------------------------------------

class SleepConsolidator:
    """Offline transfer of episodic memories into a semantic (cortical) model.

    Region: hippocampus -> neocortex, during slow-wave sleep.
    Mechanism: prioritised replay of stored episodes; the cortical model learns
    slowly from the replayed stream, extracting what is common across episodes.
    """

    def __init__(self, buffer: EpisodicBuffer, seed: int = 0):
        self.buffer = buffer
        self.rng = np.random.default_rng(seed)

    def sleep(self, learner: Callable[[np.ndarray, object, float], None],
              cycles: int = 3, replays_per_cycle: int = 400,
              prioritised: bool = True, lr: float = 0.05) -> int:
        """Run ``cycles`` of slow-wave replay. ``learner(pattern, label, lr)``
        is the cortical update. Returns how many replays happened."""
        total = 0
        for c in range(cycles):
            # later cycles consolidate more gently (deep -> light sleep)
            cycle_lr = lr * (0.6 ** c)
            for ep in self.buffer.sample_replay(replays_per_cycle, self.rng,
                                                prioritised=prioritised):
                learner(ep.pattern, ep.label, cycle_lr)
                total += 1
        return total


# -- a semantic (cortical) prototype model consolidation acts on -------------

class SemanticCortex:
    """Slow-learning cortical store: one prototype per category, refined by
    replay. Deliberately simple so the *effect of sleep* is unambiguous."""

    def __init__(self, dim: int):
        self.dim = dim
        self.proto: Dict[object, np.ndarray] = {}

    def learn(self, pattern: np.ndarray, label: object, lr: float = 0.05) -> None:
        p = self.proto.get(label)
        x = np.asarray(pattern, np.float32)
        self.proto[label] = x.copy() if p is None else (p + lr * (x - p))

    def classify(self, x: np.ndarray) -> object:
        best, score = None, -np.inf
        xn = x / max(float(np.linalg.norm(x)), 1e-6)
        for label, p in self.proto.items():
            pn = p / max(float(np.linalg.norm(p)), 1e-6)
            s = float(xn @ pn)
            if s > score:
                best, score = label, s
        return best

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean([self.classify(X[i]) == y[i] for i in range(len(X))]))


# -- the honest experiment: does sleep actually help? -----------------------

def sleep_benefit_experiment(n_classes: int = 12, dim: int = 48,
                             n_day: int = 260, noise: float = 2.2,
                             seed: int = 0, verbose: bool = False
                             ) -> Dict[str, float]:
    """Same experience, two brains: one sleeps, one does not.

    Both see exactly the *same* day of episodes and learn online. Then one is
    allowed **prioritised replay** (sleep) and the other is not. Generalisation
    is measured on held-out data neither has seen. No new information enters
    during sleep -- only re-processing of what was already stored.

    The day is deliberately **imbalanced** (a few things happen often, most
    happen rarely), because that is what real experience is like -- and it is
    exactly where replay matters: online learning leaves the rare categories
    badly estimated, and prioritised replay spends the night fixing them.
    """
    rng = np.random.default_rng(seed)
    protos = rng.normal(0, 1, (n_classes, dim)).astype(np.float32)

    # Zipf-like frequencies: a couple of common categories, a long rare tail
    freq = 1.0 / np.arange(1, n_classes + 1) ** 1.4
    freq = freq / freq.sum()

    def make(n, r, balanced=False):
        y = r.integers(0, n_classes, n) if balanced else \
            r.choice(n_classes, size=n, p=freq)
        X = protos[y] + r.normal(0, noise, (n, dim)).astype(np.float32)
        return X.astype(np.float32), y

    Xday, yday = make(n_day, rng)                              # imbalanced day
    Xtest, ytest = make(900, np.random.default_rng(seed + 99), # fair test
                        balanced=True)

    awake = SemanticCortex(dim)          # experiences the day, never sleeps
    sleeper = SemanticCortex(dim)        # same day, then sleeps
    buf = EpisodicBuffer()

    def novelty(cortex: SemanticCortex, x: np.ndarray) -> float:
        """Hippocampal novelty: how unlike anything already known this is.
        (Novelty, not classification error, is what drives replay in vivo.)"""
        if not cortex.proto:
            return 1.0
        xn = x / max(float(np.linalg.norm(x)), 1e-6)
        best = max(float(xn @ (p / max(float(np.linalg.norm(p)), 1e-6)))
                   for p in cortex.proto.values())
        return float(np.clip(1.0 - best, 0.01, 2.0))

    for i in range(n_day):
        x, lab = Xday[i], int(yday[i])
        surprise = novelty(sleeper, x)
        awake.learn(x, lab, lr=0.05)
        sleeper.learn(x, lab, lr=0.05)
        buf.store(x, lab, surprise)

    before = sleeper.accuracy(Xtest, ytest)

    # --- SLEEP: prioritised replay of the SAME episodes, no new data --------
    cons = SleepConsolidator(buf, seed=seed)
    n_replay = cons.sleep(lambda p, l, lr: sleeper.learn(p, l, lr),
                          cycles=3, replays_per_cycle=500, prioritised=True)

    # --- control: an equal amount of UNPRIORITISED replay -------------------
    flat = SemanticCortex(dim)
    for i in range(n_day):
        flat.learn(Xday[i], int(yday[i]), lr=0.05)
    cons2 = SleepConsolidator(buf, seed=seed + 1)
    cons2.sleep(lambda p, l, lr: flat.learn(p, l, lr), cycles=3,
                replays_per_cycle=500, prioritised=False)

    res = {
        "awake_only": float(awake.accuracy(Xtest, ytest)),
        "before_sleep": float(before),
        "after_sleep_prioritised": float(sleeper.accuracy(Xtest, ytest)),
        "after_sleep_uniform": float(flat.accuracy(Xtest, ytest)),
        "replays": float(n_replay),
    }
    res["sleep_gain"] = res["after_sleep_prioritised"] - res["awake_only"]
    if verbose:
        print(f"  awake only (no sleep)        : {res['awake_only']:.1%}")
        print(f"  after prioritised replay     : {res['after_sleep_prioritised']:.1%}"
              f"   (gain {res['sleep_gain']:+.1%})")
        print(f"  after uniform replay         : {res['after_sleep_uniform']:.1%}")
        print(f"  replays performed            : {int(res['replays'])} "
              f"(no new data)")
    return res


def build_developmental_program(verbose: bool = False
                                ) -> Tuple[DevelopmentalProgram, Dict[str, float]]:
    """A three-stage development where each stage's circuit closes its critical
    period as it masters its job, and later stages build on earlier ones."""
    rng = np.random.default_rng(0)
    state = {"features": 0.0, "objects": 0.0, "relations": 0.0}

    def learn_features(p: float) -> float:
        state["features"] = min(1.0, state["features"] + 0.30 * p)
        return state["features"]

    def learn_objects(p: float) -> float:
        # objects can only be learned as well as the features they rest on
        ceiling = state["features"]
        state["objects"] = min(ceiling, state["objects"] + 0.30 * p)
        return state["objects"]

    def learn_relations(p: float) -> float:
        ceiling = state["objects"]
        state["relations"] = min(ceiling, state["relations"] + 0.30 * p)
        return state["relations"]

    prog = DevelopmentalProgram([
        Stage("features", "v1", learn_features, threshold=0.9),
        Stage("objects", "it", learn_objects, threshold=0.9),
        Stage("relations", "pallium", learn_relations, threshold=0.9),
    ])
    log = prog.run(verbose=verbose)
    scores = {e["stage"]: float(e["competence"]) for e in log}
    scores["all_mastered"] = float(all(e["mastered"] for e in log))
    return prog, scores
