"""
selfmodel.py
============

The **self model** -- the brain does not only model the world, it models the
body that acts in it.

This was the largest single gap in the system. Without a self model there is no
answer to "can I do this?", no distinction between *I moved* and *it moved*, and
no way for bodily state to shape cognition. A biological agent has four
interlocking self-representations:

  * **Body model** -- a somatotopic/proprioceptive map of the body's structure
    and its *physical limits* (reach, joint range, maximum force). Primary
    somatosensory + posterior parietal cortex, and the cerebellar body schema.

  * **Motor capability model** -- learned knowledge of what this body can
    actually achieve, acquired by *trying and failing*. It is not read off the
    anatomy; it is learned from outcomes (M1/premotor + cerebellum + basal
    ganglia).

  * **Interoception** -- the state of the body itself: energy, fatigue, pain.
    Insular cortex; it continuously *modulates* capability, so the same action
    is possible when rested and impossible when exhausted.

  * **Agency prediction** -- when a motor command is issued, an **efference
    copy** goes to a forward model that predicts the sensory consequence. If the
    actual sensation matches, the event is tagged *self-caused*; if it does not,
    *externally caused*. This comparator (cerebellum + parietal) is the
    computational basis of the sense of agency, and its failure is a classic
    account of delusions of control.

The headline behaviour the design calls for::

    self.can("lift", load_kg=500)
    # -> False, because the BODY MODEL predicts failure, and it can say why

Nothing here is a lookup table: the capability model is learned from attempted
actions and their outcomes by the delta rule (dopaminergic prediction error).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


def _sigma(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0))))


# -- 1. Body model ----------------------------------------------------------

@dataclass
class BodyModel:
    """Structure and hard physical limits of the body (the body schema).

    Region: S1 / posterior parietal + cerebellar body schema.
    Cells:  proprioceptive relay cells, somatotopic pyramidal maps.
    """

    max_force_n: float = 700.0        # newtons the body can exert (~70 kg lift)
    reach_m: float = 0.75
    height_m: float = 1.7
    mass_kg: float = 70.0
    effectors: Tuple[str, ...] = ("left_hand", "right_hand", "mouth")

    @property
    def max_lift_kg(self) -> float:
        """Heaviest liftable mass, from force limit (F = m g)."""
        return self.max_force_n / 9.81

    def required_force(self, load_kg: float) -> float:
        return load_kg * 9.81

    def within_reach(self, distance_m: float) -> bool:
        return distance_m <= self.reach_m

    def physical_margin(self, action: str, load_kg: float = 0.0,
                        distance_m: float = 0.0) -> float:
        """How much headroom the body has for this demand, in [-1, 1].
        Positive = comfortably possible, negative = beyond the body's limits."""
        if action in ("lift", "carry", "push"):
            need = self.required_force(load_kg)
            return float((self.max_force_n - need) / max(self.max_force_n, 1e-6))
        if action == "reach":
            return float((self.reach_m - distance_m) / max(self.reach_m, 1e-6))
        return 0.0


# -- 2. Interoception -------------------------------------------------------

@dataclass
class Interoception:
    """The body's internal state, which *modulates* what the body can do.

    Region: insular cortex (+ hypothalamic/brainstem afferents).
    Signals: energy, fatigue, pain -- all in [0, 1].
    """

    energy: float = 1.0
    fatigue: float = 0.0
    pain: float = 0.0

    @property
    def capability_scale(self) -> float:
        """Multiplier on available force. Exhaustion really does shrink it."""
        return float(np.clip(0.25 + 0.75 * self.energy
                             - 0.5 * self.fatigue - 0.3 * self.pain, 0.05, 1.0))

    def exert(self, effort: float) -> None:
        """Doing work costs energy and builds fatigue."""
        effort = float(np.clip(effort, 0.0, 1.0))
        self.energy = float(np.clip(self.energy - 0.05 * effort, 0.0, 1.0))
        self.fatigue = float(np.clip(self.fatigue + 0.08 * effort, 0.0, 1.0))

    def rest(self, amount: float = 1.0) -> None:
        self.energy = float(np.clip(self.energy + 0.15 * amount, 0.0, 1.0))
        self.fatigue = float(np.clip(self.fatigue - 0.20 * amount, 0.0, 1.0))

    def state(self) -> Dict[str, float]:
        return {"energy": round(self.energy, 3), "fatigue": round(self.fatigue, 3),
                "pain": round(self.pain, 3),
                "capability_scale": round(self.capability_scale, 3)}


# -- 3. Motor capability model ---------------------------------------------

class MotorCapability:
    """Learned belief about what this body can achieve -- acquired by trying.

    Region: M1/premotor + cerebellum + basal ganglia.
    Learning: delta rule on (predicted success - actual outcome), i.e. the
    dopaminergic prediction error. The *features* are the body's physical
    margin and the interoceptive scale, so the learned model generalises to
    loads never attempted."""

    def __init__(self, lr: float = 0.2):
        self.lr = float(lr)
        # weights over [physical_margin, interoceptive_scale, bias], per action
        self.w: Dict[str, np.ndarray] = {}
        self.attempts: Dict[str, int] = {}

    def _features(self, margin: float, scale: float) -> np.ndarray:
        return np.array([margin, scale, 1.0], np.float32)

    def confidence(self, action: str, margin: float, scale: float) -> float:
        """P(success) this body assigns to the attempt."""
        w = self.w.get(action)
        if w is None:
            return 0.5                     # never tried: genuinely uncertain
        return _sigma(float(w @ self._features(margin, scale)))

    def learn(self, action: str, margin: float, scale: float,
              success: bool) -> float:
        """Delta rule / reward-prediction error on the attempt's outcome."""
        if action not in self.w:
            self.w[action] = np.zeros(3, np.float32)
            self.attempts[action] = 0
        f = self._features(margin, scale)
        p = _sigma(float(self.w[action] @ f))
        err = (1.0 if success else 0.0) - p
        self.w[action] += self.lr * err * f
        self.attempts[action] += 1
        return float(err)


# -- 4. Agency (efference copy + forward model comparator) ------------------

class AgencyModel:
    """Am *I* the cause of what just happened?

    A motor command emits an **efference copy**; a forward model predicts the
    sensory consequence; the comparator comes down to whether the observed
    sensation matches the prediction. Match => self-caused. Mismatch => the
    world (or someone else) did it.

    Region: cerebellum (forward model) + posterior parietal / TPJ (comparator).
    """

    def __init__(self, dim: int = 32, lr: float = 0.3, tol: float = 0.6,
                 seed: int = 0):
        self.dim = dim
        self.lr = float(lr)
        self.tol = float(tol)                 # similarity needed to claim agency
        self.rng = np.random.default_rng(seed)
        self.forward: Dict[str, np.ndarray] = {}   # action -> predicted sensation

    def _cmd(self, action: str) -> np.ndarray:
        if action not in self.forward:
            self.forward[action] = np.zeros(self.dim, np.float32)
        return self.forward[action]

    def predict_sensation(self, action: str) -> np.ndarray:
        """Efference copy -> forward model -> expected sensory consequence."""
        return self._cmd(action).copy()

    def learn(self, action: str, sensation: np.ndarray) -> None:
        """Learn the sensory consequence of one's own action (cerebellar
        supervised learning: climbing-fibre error)."""
        p = self._cmd(action)
        self.forward[action] = p + self.lr * (sensation.astype(np.float32) - p)

    def judge(self, action: Optional[str], sensation: np.ndarray
              ) -> Tuple[bool, float]:
        """Return ``(self_caused, match)``. With no action issued, nothing the
        agent did can explain the sensation -- so it is external."""
        if action is None:
            return False, 0.0
        pred = self.predict_sensation(action)
        n1, n2 = np.linalg.norm(pred), np.linalg.norm(sensation)
        if n1 < 1e-6 or n2 < 1e-6:
            return False, 0.0
        match = float((pred @ sensation) / (n1 * n2))
        return bool(match >= self.tol), match


# -- the whole self ---------------------------------------------------------

@dataclass
class SelfModel:
    """Body + capability + interoception + agency: a model of the actor."""

    body: BodyModel = field(default_factory=BodyModel)
    intero: Interoception = field(default_factory=Interoception)
    motor: MotorCapability = field(default_factory=MotorCapability)
    agency: AgencyModel = field(default_factory=AgencyModel)

    # -- "can I?" -----------------------------------------------------------
    def confidence(self, action: str, load_kg: float = 0.0,
                   distance_m: float = 0.0) -> float:
        margin = self.body.physical_margin(action, load_kg, distance_m)
        return self.motor.confidence(action, margin, self.intero.capability_scale)

    def can(self, action: str, load_kg: float = 0.0, distance_m: float = 0.0,
            threshold: float = 0.5) -> bool:
        """Does this body believe it can do this? ("I cannot lift 500 kg.")"""
        return self.confidence(action, load_kg, distance_m) >= threshold

    def explain(self, action: str, load_kg: float = 0.0,
                distance_m: float = 0.0) -> str:
        """Why -- grounded in the body model, not a canned string."""
        margin = self.body.physical_margin(action, load_kg, distance_m)
        conf = self.confidence(action, load_kg, distance_m)
        scale = self.intero.capability_scale
        if action in ("lift", "carry", "push"):
            need = self.body.required_force(load_kg)
            have = self.body.max_force_n * scale
            verdict = "possible" if conf >= 0.5 else "beyond this body"
            return (f"{action}({load_kg:g}kg): needs {need:.0f}N, this body can "
                    f"produce ~{have:.0f}N (limit {self.body.max_force_n:.0f}N x "
                    f"interoceptive {scale:.2f}) -> {verdict} "
                    f"[confidence {conf:.0%}, margin {margin:+.2f}]")
        if action == "reach":
            return (f"reach({distance_m:g}m): arm reaches {self.body.reach_m:g}m "
                    f"-> {'possible' if conf >= 0.5 else 'too far'} "
                    f"[confidence {conf:.0%}]")
        return f"{action}: confidence {conf:.0%}"

    # -- acting: attempt, observe outcome, learn ---------------------------
    def attempt(self, action: str, load_kg: float = 0.0, distance_m: float = 0.0,
                success: Optional[bool] = None,
                sensation: Optional[np.ndarray] = None) -> Dict[str, object]:
        """Try an action. If ``success`` is not supplied, physics decides it.
        The capability model learns from the outcome; the body pays the cost."""
        margin = self.body.physical_margin(action, load_kg, distance_m)
        scale = self.intero.capability_scale
        predicted = self.motor.confidence(action, margin, scale)
        if success is None:
            success = self._physics_outcome(action, load_kg, distance_m, scale)
        err = self.motor.learn(action, margin, scale, success)
        effort = float(np.clip(1.0 - margin, 0.0, 1.0))
        self.intero.exert(effort)
        if sensation is not None:
            self.agency.learn(action, sensation)
        return {"action": action, "success": bool(success),
                "predicted": round(predicted, 3),
                "prediction_error": round(err, 3),
                "margin": round(margin, 3)}

    def _physics_outcome(self, action: str, load_kg: float, distance_m: float,
                         scale: float) -> bool:
        """Ground truth of the world -- the self model never sees this rule, it
        only sees whether attempts succeeded."""
        if action in ("lift", "carry", "push"):
            return self.body.required_force(load_kg) <= self.body.max_force_n * scale
        if action == "reach":
            return distance_m <= self.body.reach_m
        return True

    # -- who did that? ------------------------------------------------------
    def attribute(self, sensation: np.ndarray, action: Optional[str] = None
                  ) -> Dict[str, object]:
        """Self-caused or externally caused? (efference-copy comparator)"""
        mine, match = self.agency.judge(action, sensation)
        return {"self_caused": mine, "match": round(match, 3),
                "agent": "self" if mine else "external"}


def build_self_model(seed: int = 0, verbose: bool = False
                     ) -> Tuple[SelfModel, Dict[str, float]]:
    """Grow a self model by *acting*: attempt many lifts/reaches across a range
    of loads, learn from the outcomes, then measure how well the learned body
    knowledge generalises to loads and distances never attempted."""
    rng = np.random.default_rng(seed)
    s = SelfModel()

    # -- developmental motor babbling: try things, succeed and fail ---------
    # Over development the body attempts actions in MANY different internal
    # states -- rested, tired, hungry. Sampling that space is what teaches the
    # capability model that possibility depends on body limit *and* body state.
    def _random_body_state():
        s.intero.energy = float(rng.uniform(0.2, 1.0))
        s.intero.fatigue = float(rng.uniform(0.0, 0.7))
        s.intero.pain = float(rng.uniform(0.0, 0.2))

    for _ in range(1200):
        _random_body_state()
        s.attempt("lift", load_kg=float(rng.uniform(0, 160)))
    for _ in range(400):
        _random_body_state()
        s.attempt("reach", distance_m=float(rng.uniform(0, 1.5)))
    s.intero.energy, s.intero.fatigue, s.intero.pain = 1.0, 0.0, 0.0

    # -- learn the sensory consequences of one's own actions ---------------
    base = {a: rng.normal(0, 1, s.agency.dim).astype(np.float32)
            for a in ("lift", "reach", "push")}
    for _ in range(80):
        for a, v in base.items():
            s.agency.learn(a, v + rng.normal(0, 0.15, s.agency.dim).astype(np.float32))

    # -- measure: capability judgement on UNSEEN loads/distances -----------
    ok = 0
    trials = 400
    for _ in range(trials):
        # unseen loads AND unseen body states -- a fair generalisation test
        s.intero.energy = float(rng.uniform(0.2, 1.0))
        s.intero.fatigue = float(rng.uniform(0.0, 0.7))
        s.intero.pain = float(rng.uniform(0.0, 0.2))
        load = float(rng.uniform(0, 200))
        truth = s.body.required_force(load) <= s.body.max_force_n * \
            s.intero.capability_scale
        ok += int(s.can("lift", load_kg=load) == truth)
    capability_accuracy = ok / trials
    s.intero.energy, s.intero.fatigue, s.intero.pain = 1.0, 0.0, 0.0

    rok = 0
    for _ in range(200):
        d = float(rng.uniform(0, 1.6))
        rok += int(s.can("reach", distance_m=d) == (d <= s.body.reach_m))
    reach_accuracy = rok / 200

    # -- measure: agency attribution (self vs external) --------------------
    aok = 0
    for _ in range(400):
        if rng.random() < 0.5:                     # I acted: sensation matches
            a = "lift"
            sens = base[a] + rng.normal(0, 0.2, s.agency.dim).astype(np.float32)
            aok += int(s.attribute(sens, action=a)["self_caused"] is True)
        else:                                      # the world acted
            sens = rng.normal(0, 1, s.agency.dim).astype(np.float32)
            act = "lift" if rng.random() < 0.5 else None
            aok += int(s.attribute(sens, action=act)["self_caused"] is False)
    agency_accuracy = aok / 400

    scores = {"capability_accuracy": capability_accuracy,
              "reach_accuracy": reach_accuracy,
              "agency_accuracy": agency_accuracy}
    if verbose:
        print(f"  capability (unseen loads) : {capability_accuracy:.0%}")
        print(f"  reach judgement           : {reach_accuracy:.0%}")
        print(f"  agency (self vs external) : {agency_accuracy:.0%}")
        print("  " + s.explain("lift", load_kg=500))
        print("  " + s.explain("lift", load_kg=20))
    return s, scores
