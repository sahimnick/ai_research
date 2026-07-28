"""
dopamine.py
===========

**The distal reward problem, solved the way Izhikevich (2007) solved it.**

The problem
-----------
A synapse that helped cause a good outcome fired half a second before the
outcome arrived. By the time reward exists, the spikes that earned it are long
over, and the synapse has no way to know it was involved -- millions of other
synapses fired in that half second too. Hebbian plasticity cannot bridge the
gap, because it only sees *now*.

The answer
----------
The synapse does not change when it fires. It leaves a **tag**: a decaying
memory of its own pre-before-post correlation, the eligibility trace. The trace
does nothing on its own. When a global neuromodulatory signal arrives -- dopamine
from the midbrain, broadcast to everything at once -- the weight change is the
product of the two:

    e(t+1) = e(t) * exp(-1/tau_e) + pre(t) * post(t)
    w      = w + lr * DA(t) * e(t)

That is the whole rule. It is local in space (a synapse sees only its own two
cells plus a chemical concentration in the extracellular space) and it is
**not** a gradient: dopamine says *that was good*, not *here is your derivative*.
Nothing in this module computes an error signal for any synapse, and nothing
propagates anything backwards.

The one thing the time constant costs you
-----------------------------------------
With ``tau_e = 200 ms`` and a reward 500 ms after the action, only
``exp(-500/200) = 8.2%`` of the trace is left when dopamine arrives. Learning
still works -- that surviving 8% is the only thing correlated with the action,
while every other synapse's trace is noise averaging to zero -- but it is slower
than it would be with the ~1 s constant Izhikevich used. The 200 ms figure is
kept because it was specified; the arithmetic is stated so the cost is visible
rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# The tag
# ---------------------------------------------------------------------------
class EligibilityTrace:
    """A decaying memory of pre-before-post correlation, per synapse.

    Shape is ``(n_post, n_pre)`` -- one scalar per synapse, exactly as the
    weight matrix. Decay is exponential and per-millisecond, not a fixed
    subtraction: a fixed step would make the trace's lifetime depend on how
    often you happened to call it.
    """

    def __init__(self, shape: Tuple[int, int], tau: float = 200.0):
        self.e = np.zeros(shape, np.float32)
        self.tau = float(tau)
        # pre-synaptic and post-synaptic spike traces, so that "pre before
        # post" is what gets tagged rather than "pre exactly with post" --
        # simultaneity at 1 ms resolution would almost never happen.
        self.x_pre = np.zeros(shape[1], np.float32)
        self.x_post = np.zeros(shape[0], np.float32)
        self.tau_pre, self.tau_post = 20.0, 20.0

    def step(self, pre: np.ndarray, post: np.ndarray, dt: float = 1.0) -> None:
        """One millisecond: decay, then tag whatever correlation just happened."""
        self.e *= float(np.exp(-dt / self.tau))
        self.x_pre *= float(np.exp(-dt / self.tau_pre))
        self.x_post *= float(np.exp(-dt / self.tau_post))
        pre = np.asarray(pre, bool)
        post = np.asarray(post, bool)
        # a post spike tags every recently-active input onto that cell
        if post.any():
            self.e[post] += self.x_pre[None, :]
        # and a pre spike tags a recently-active target, more weakly: the
        # anti-causal order, which is what keeps the rule from tagging
        # everything symmetrically
        if pre.any():
            self.e[:, pre] -= 0.25 * self.x_post[:, None]
        self.x_pre[pre] += 1.0
        self.x_post[post] += 1.0

    def reset(self) -> None:
        self.e[:] = 0.0
        self.x_pre[:] = 0.0
        self.x_post[:] = 0.0


# ---------------------------------------------------------------------------
# The chemical
# ---------------------------------------------------------------------------
class DopaminergicModulator:
    """A global scalar that turns tags into weight changes.

    Parameters
    ----------
    tau_e:     eligibility decay, ms.
    tau_da:    how fast dopamine is cleared from the extracellular space, ms.
               Dopamine is not an instantaneous delta -- it is released, it
               diffuses, and it is taken back up, so a pulse acts over tens to
               hundreds of milliseconds.
    lr:        the only learning rate in the module.
    w_bounds:  hard bounds on the weight, since a synapse has a maximum
               conductance.
    """

    def __init__(self, shape: Tuple[int, int], tau_e: float = 200.0,
                 tau_da: float = 150.0, lr: float = 0.02,
                 w_bounds: Tuple[float, float] = (0.0, 1.0),
                 homeostasis: Optional[float] = None):
        # ``homeostasis``: total incoming weight each post-synaptic cell is
        # allowed to keep. Without it the rule has no way to express a
        # PREFERENCE, only an absolute strength, and on a stochastic task
        # everything drifts to the ceiling: measured on the 3-armed bandit,
        # 90-96% of every arm's synapses ended pinned at w=1.0, the WORST arm
        # ended highest, and the frozen network picked it 30 times out of 30 --
        # which is exactly the zero retention that was left unexplained.
        # Conserving a cell's synaptic budget makes potentiating one input cost
        # the others, which is what synaptic scaling does in real cortex.
        self.homeostasis = homeostasis
        self.trace = EligibilityTrace(shape, tau=tau_e)
        self.tau_da = float(tau_da)
        self.lr = float(lr)
        self.w_lo, self.w_hi = float(w_bounds[0]), float(w_bounds[1])
        self.da = 0.0
        self.history: List[Tuple[float, float]] = []   # (time, da)
        self.t = 0.0

    # -- the reward signal --------------------------------------------------
    def deliver_reward(self, reward_signal: float) -> None:
        """A dopamine burst (positive) or dip (negative). Clamped to [-1, 1]."""
        self.da = float(np.clip(self.da + float(reward_signal), -1.0, 1.0))

    # -- one millisecond of the world --------------------------------------
    def step(self, pre: np.ndarray, post: np.ndarray, weights: np.ndarray,
             dt: float = 1.0) -> np.ndarray:
        """Decay everything, tag the new correlation, apply ``lr * DA * e``."""
        self.trace.step(pre, post, dt=dt)
        if abs(self.da) > 1e-6:
            weights += self.lr * self.da * self.trace.e
            np.clip(weights, self.w_lo, self.w_hi, out=weights)
            if self.homeostasis:
                tot = weights.sum(axis=1, keepdims=True)
                np.divide(weights * float(self.homeostasis),
                          np.maximum(tot, 1e-6), out=weights)
                np.clip(weights, self.w_lo, self.w_hi, out=weights)
        self.da *= float(np.exp(-dt / self.tau_da))
        self.history.append((self.t, self.da))
        self.t += dt
        return weights

    def reset_trial(self) -> None:
        self.trace.reset()
        self.da = 0.0


# ---------------------------------------------------------------------------
# Task 8.4: predicting the reward instead of just receiving it
# ---------------------------------------------------------------------------
class RewardPredictionError:
    """``actual - expected``, which is what a VTA cell actually reports.

    A dopamine cell does not fire for reward. It fires for reward that was **not
    predicted**, and it goes silent when a predicted reward fails to arrive.
    Keeping a running expectation per state and broadcasting the difference is
    the minimal version of that, and it is what turns the module from a
    reinforcement device into a prediction device -- the thing active inference
    is built on.
    """

    def __init__(self, n_states: int, alpha: float = 0.15):
        self.expected = np.zeros(int(n_states), np.float32)
        self.alpha = float(alpha)

    def __call__(self, state: int, actual: float) -> float:
        s = int(state)
        delta = float(actual) - float(self.expected[s])
        self.expected[s] += self.alpha * delta
        return delta


# ---------------------------------------------------------------------------
# Task 8.2: the delayed-reward environment, in spiking tissue
# ---------------------------------------------------------------------------
class DelayedRewardTask:
    """Two stimuli, two actions, and a reward that arrives 500 ms too late.

    The whole point is the delay. The action is chosen and finished long before
    any reward exists; nothing about the outcome is available at the moment the
    synapses fire. If accuracy rises anyway, the eligibility trace is doing the
    work, because there is no other route.

    Nothing here optimises anything. There is no loss, no gradient, no
    optimiser, and no backward pass. The only weight update in the file is
    ``w += lr * DA * e``.
    """

    #: the four conditions the comparison needs. Only ``three_factor`` is the
    #: thing being tested; the other three exist to make its number mean
    #: something.
    RULES = ("stdp_only", "random_da", "three_factor", "oracle")

    def __init__(self, n_in: int = 40, n_act: int = 40, delay_ms: int = 500,
                 stim_ms: int = 120, lr: float = 0.02, tau_e: float = 200.0,
                 noise: float = 4.0, use_rpe: bool = False,
                 rule: str = "three_factor", homeostasis: Optional[float] = None,
                 seed: int = 0):
        from .neuron import Population
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.n_in, self.n_act = int(n_in), int(n_act)
        self.delay_ms, self.stim_ms = int(delay_ms), int(stim_ms)
        self.noise = float(noise)
        self.act = Population(self.n_act, "regular_spiking", rng=rng, jitter=0.02)
        # two stimuli over disjoint halves of the input, two action groups over
        # disjoint halves of the output
        self.stimuli = [np.zeros(self.n_in, bool), np.zeros(self.n_in, bool)]
        self.stimuli[0][: self.n_in // 2] = True
        self.stimuli[1][self.n_in // 2:] = True
        self.groups = [np.zeros(self.n_act, bool), np.zeros(self.n_act, bool)]
        self.groups[0][: self.n_act // 2] = True
        self.groups[1][self.n_act // 2:] = True
        # start deliberately symmetric: no action is preferred for any stimulus
        self.W = (0.5 + 0.02 * rng.standard_normal((self.n_act, self.n_in))
                  ).astype(np.float32)
        # Mutual inhibition between the action groups. Without it there is no
        # action to credit: measured, the two groups fired 59 spikes each on
        # every trial, so the eligibility trace was symmetric across them and
        # dopamine strengthened both equally, whichever one "won". A choice has
        # to be a competition that one side loses -- that is what the basal
        # ganglia and cortical winner-take-all circuits are for -- and only then
        # does the trace carry which action was actually taken.
        self.w_cross = 9.0
        self._grp_act = np.zeros(len(self.groups), np.float32)
        self.tau_grp = 25.0
        self.da = DopaminergicModulator(self.W.shape, tau_e=tau_e, lr=lr,
                                        w_bounds=(0.0, 1.0),
                                        homeostasis=homeostasis)
        self.rpe = RewardPredictionError(2) if use_rpe else None
        self.correct = [0, 1]              # stimulus 0 -> action 0, 1 -> 1
        if rule not in self.RULES:
            raise ValueError(f"rule must be one of {self.RULES}, got {rule!r}")
        self.rule = str(rule)

    # -- one trial ----------------------------------------------------------
    def trial(self, which: int, learn: bool = True) -> Tuple[int, float]:
        pattern = self.stimuli[which]
        counts = np.zeros(self.n_act, np.int32)
        # 1. the stimulus is present, the action population competes
        self._grp_act[:] = 0.0
        for _ in range(self.stim_ms):
            pre = (self.rng.random(self.n_in) < 0.35) & pattern   # Poisson-ish
            drive = (self.W @ pre.astype(np.float32)) * 2.2
            drive += self.noise * self.rng.standard_normal(self.n_act)
            # each group is inhibited by how loudly the OTHERS are speaking.
            # This was hardcoded to two groups, so on a three-armed bandit the
            # third arm received no competition at all and every condition --
            # including the oracle -- scored at or below the 33% chance line.
            # A broken benchmark reports failure for whatever you test on it.
            total = float(self._grp_act.sum())
            for g in range(len(self.groups)):
                drive[self.groups[g]] -= self.w_cross * (total - self._grp_act[g])
            fired = self.act.step(drive.astype(np.float32))
            self._grp_act *= float(np.exp(-1.0 / self.tau_grp))
            for g in range(len(self.groups)):
                self._grp_act[g] += float(fired[self.groups[g]].sum()) / \
                    max(1.0, float(self.groups[g].sum()))
            counts += fired
            if learn:
                self.da.step(pre, fired, self.W)
        # 2. the choice: whichever action group spoke louder
        scores = [float(counts[g].sum()) for g in self.groups]
        choice = int(np.argmax(scores)) if scores[0] != scores[1] else \
            int(self.rng.integers(2))
        # 3. nothing happens for half a second. The trace decays; the world is
        #    silent; the synapses that acted have no idea yet.
        for _ in range(self.delay_ms):
            pre = self.rng.random(self.n_in) < 0.01
            drive = (self.W @ pre.astype(np.float32)) * 2.2
            drive += self.noise * self.rng.standard_normal(self.n_act)
            fired = self.act.step(drive.astype(np.float32))
            if learn:
                self.da.step(pre, fired, self.W)
        # 4. now the reward arrives
        reward = self.outcome(which, choice)
        if learn:
            self._apply(which, choice, reward)
        return choice, reward

    def outcome(self, which: int, choice: int) -> float:
        """Deterministic here; the bandit overrides it with a stochastic one."""
        return 1.0 if choice == self.correct[which] else -1.0

    def _apply(self, which: int, choice: int, reward: float) -> None:
        """The one place the four conditions differ."""
        if self.rule == "stdp_only":
            # A: the tag becomes weight change with no gating at all. This is
            # plain correlational learning -- the network still sees the same
            # spikes, so if it learns the task here, reward was never needed.
            self.W += self.da.lr * self.da.trace.e
            np.clip(self.W, 0.0, 1.0, out=self.W)
            self.da.trace.reset()
            return
        if self.rule == "random_da":
            # B: dopamine exists, is the same size, and means nothing
            signal = float(self.rng.choice([-1.0, 1.0]))
        elif self.rule == "oracle":
            # D: the upper bound. Dopamine is delivered with NO delay, and it
            # reports what the correct action WOULD have earned rather than
            # what this action did -- a teacher, not a reinforcer. Nothing in
            # the brain has access to this signal; it is here to say how much
            # of the remaining gap is the delay and how much is the rule.
            signal = 1.0 if choice == self.correct[which] else -1.0
            self.da.trace.e[:] = self._oracle_trace(which)
        else:
            signal = self.rpe(which, reward) if self.rpe is not None else reward
        self.da.deliver_reward(float(np.clip(signal, -1.0, 1.0)))
        for _ in range(200):               # dopamine acts over its own lifetime
            self.da.step(np.zeros(self.n_in, bool),
                         np.zeros(self.n_act, bool), self.W)

    def _oracle_trace(self, which: int) -> np.ndarray:
        """A tag that has not decayed and points exactly at the right synapses."""
        e = np.zeros_like(self.W)
        e[np.ix_(self.groups[self.correct[which]], self.stimuli[which])] = 1.0
        return e

    # -- the measurement ----------------------------------------------------
    def run(self, trials: int = 200, report_every: int = 50) -> Dict[str, float]:
        hits: List[int] = []
        curve: Dict[int, float] = {}
        for i in range(int(trials)):
            which = int(self.rng.integers(2))
            choice, reward = self.trial(which, learn=True)
            hits.append(1 if reward > 0 else 0)
            if (i + 1) % report_every == 0:
                curve[i + 1] = float(np.mean(hits[-report_every:]))
        # the weight evidence: is the winning path stronger than the losing one?
        ratios = []
        for s, a in enumerate(self.correct):
            win = self.W[np.ix_(self.groups[a], self.stimuli[s])].mean()
            lose = self.W[np.ix_(self.groups[1 - a], self.stimuli[s])].mean()
            ratios.append(float(win / max(lose, 1e-6)))
        return {"final_accuracy": float(np.mean(hits[-max(20, trials // 4):])),
                "first_quarter": float(np.mean(hits[: max(20, trials // 4)])),
                "weight_ratio": float(np.mean(ratios)),
                "curve": curve}


# ---------------------------------------------------------------------------
# Task 8.3: dopamine wired into the visual hierarchy
# ---------------------------------------------------------------------------
class DopaminergicActionLoop:
    """pixels -> V1 -> V2 -> action, learning only by ``Δw = DA * e``.

    The visual layers are the ones already in the project: a self-organized V1
    and the topographic :class:`~neurobrain.composite.LocalV2`. Neither of them
    is retrained here. What learns is the V2 -> action projection, and it learns
    by exactly one rule, applied 500 ms after the action was over.

    This is deliberately the weakest possible claim to make about the loop: the
    only thing being credited is the decision, so if accuracy rises the credit
    survived the delay through the hierarchy. No error is propagated into V1 or
    V2 -- dopamine is a concentration in the tissue, not a message addressed to
    a layer.
    """

    def __init__(self, v1, v2=None, n_act: int = 40, n_choices: int = 2,
                 delay_ms: int = 500, stim_ms: int = 80, lr: float = 0.05,
                 tau_e: float = 200.0, noise: float = 4.0, w_cross: float = 14.0,
                 use_rpe: bool = False, sparsity: Optional[float] = 0.05,
                 accumulate: bool = True, commit_threshold: float = 60.0,
                 tau_evidence: float = 400.0, seed: int = 0):
        from .neuron import Population
        rng = np.random.default_rng(seed)
        self.rng, self.v1, self.v2 = rng, v1, v2
        self.sparsity = sparsity
        self.accumulate = bool(accumulate)
        self.commit_threshold = float(commit_threshold)
        self.tau_evidence = float(tau_evidence)
        self.n_choices = int(n_choices)
        self.delay_ms, self.stim_ms = int(delay_ms), int(stim_ms)
        self.noise, self.w_cross = float(noise), float(w_cross)
        self.n_act = int(n_act)
        self.act = Population(self.n_act, "regular_spiking", rng=rng, jitter=0.02)
        per = self.n_act // self.n_choices
        self.groups = []
        for g in range(self.n_choices):
            m = np.zeros(self.n_act, bool)
            m[g * per:(g + 1) * per] = True
            self.groups.append(m)
        self.W: Optional[np.ndarray] = None      # built on the first image
        self.da: Optional[DopaminergicModulator] = None
        self.lr, self.tau_e = float(lr), float(tau_e)
        self.rpe = RewardPredictionError(16) if use_rpe else None
        self._grp = np.zeros(self.n_choices, np.float32)

    # -- the sensory front end ---------------------------------------------
    def features(self, image: np.ndarray) -> np.ndarray:
        """Whatever the visual hierarchy makes of this image, as a rate vector.

        ``sparsity`` keeps only the strongest fraction of cells. It is not a
        tidying step -- it is the fix for what made this loop fail. The dense V1
        code has 40% of cells active and the two classes sit at cosine 0.82, so
        the synapses tagged as eligible for one class are largely the SAME
        synapses tagged for the other, and each reward overwrites the previous
        association. A sparse code gives the two classes nearly disjoint sets of
        active afferents, so the credit for one does not land on the other.
        This is the same 5%-sparsity result that mattered in the auditory work,
        arriving here for a different reason.
        """
        code = self.v1.rate_over([image]) if hasattr(self.v1, "rate_over") else \
            np.asarray(self.v1(image), np.float32).reshape(-1)
        code = np.asarray(code, np.float32).reshape(-1)
        if self.v2 is not None:
            code = np.concatenate([code, np.asarray(self.v2.code(image),
                                                    np.float32).reshape(-1)])
        if self.sparsity is not None and 0.0 < self.sparsity < 1.0:
            k = max(1, int(len(code) * float(self.sparsity)))
            keep = np.argpartition(code, -k)[-k:]
            sparse = np.zeros_like(code)
            sparse[keep] = code[keep]
            code = sparse
        m = float(np.abs(code).max()) or 1.0
        return (code / m).astype(np.float32)

    def _ensure(self, n_in: int) -> None:
        if self.W is None:
            self.W = (0.5 + 0.02 * self.rng.standard_normal((self.n_act, n_in))
                      ).astype(np.float32)
            self.da = DopaminergicModulator(self.W.shape, tau_e=self.tau_e,
                                            lr=self.lr, w_bounds=(0.0, 1.0))

    # -- one trial ----------------------------------------------------------
    def trial(self, image: np.ndarray, correct: int, state: int = 0,
              learn: bool = True) -> Tuple[int, float]:
        f = self.features(image)
        self._ensure(len(f))
        counts = np.zeros(self.n_act, np.int32)
        self._grp[:] = 0.0
        # Evidence accumulation with a commitment threshold, as in LIP and the
        # basal-ganglia decision models -- and as the diagnosis demanded. The
        # fast winner-take-all race commits on whichever group happens to fire
        # first, and at a measured signal-to-noise of 0.34 at the action cells
        # that is a coin flip: the drive difference between the two classes is
        # 1.36 against a per-millisecond noise of 4.0. An accumulator integrates
        # that difference over the whole stimulus, where the noise averages down
        # as 1/sqrt(t) and the evidence does not, and only then commits.
        evidence = np.zeros(self.n_choices, np.float32)
        committed = -1
        for _ in range(self.stim_ms):
            pre = self.rng.random(len(f)) < (0.6 * f)     # rate-coded afferents
            drive = (self.W @ pre.astype(np.float32)) * 2.2
            drive += self.noise * self.rng.standard_normal(self.n_act)
            if self.accumulate:
                # competition stays weak until the threshold is crossed, so
                # evidence can build; after commitment the winner suppresses
                # the rest, which is what makes a decision a decision
                if committed >= 0:
                    for g in range(self.n_choices):
                        if g != committed:
                            drive[self.groups[g]] -= self.w_cross
                else:
                    drive -= 0.15 * self.w_cross * float(self._grp.sum())
            else:
                total = float(self._grp.sum())
                for g in range(self.n_choices):
                    drive[self.groups[g]] -= self.w_cross * (total - self._grp[g])
            fired = self.act.step(drive.astype(np.float32))
            self._grp *= float(np.exp(-1.0 / 25.0))
            for g in range(self.n_choices):
                self._grp[g] += float(fired[self.groups[g]].sum()) / \
                    max(1.0, float(self.groups[g].sum()))
            counts += fired
            if self.accumulate and committed < 0:
                evidence *= float(np.exp(-1.0 / self.tau_evidence))
                for g in range(self.n_choices):
                    evidence[g] += float(fired[self.groups[g]].sum())
                lead = float(evidence.max() - np.median(evidence))
                if lead >= self.commit_threshold:
                    committed = int(np.argmax(evidence))
            if learn:
                self.da.step(pre, fired, self.W)
        scores = [float(counts[g].sum()) for g in self.groups]
        choice = committed if (self.accumulate and committed >= 0) \
            else int(np.argmax(scores))
        if not self.accumulate and len(set(scores)) == 1:
            choice = int(self.rng.integers(self.n_choices))
        for _ in range(self.delay_ms):                     # the silent gap
            pre = self.rng.random(len(f)) < 0.01
            drive = (self.W @ pre.astype(np.float32)) * 2.2
            drive += self.noise * self.rng.standard_normal(self.n_act)
            fired = self.act.step(drive.astype(np.float32))
            if learn:
                self.da.step(pre, fired, self.W)
        reward = 1.0 if choice == int(correct) else -1.0
        sig = self.rpe(state, reward) if self.rpe is not None else reward
        if learn:
            self.da.deliver_reward(float(np.clip(sig, -1.0, 1.0)))
            z_in = np.zeros(len(f), bool)
            z_out = np.zeros(self.n_act, bool)
            for _ in range(200):
                self.da.step(z_in, z_out, self.W)
        return choice, reward

    def run(self, images: Sequence[np.ndarray], labels: Sequence[int],
            trials: int = 150, learn: bool = True) -> Dict[str, float]:
        hits: List[int] = []
        for i in range(int(trials)):
            j = int(self.rng.integers(len(images)))
            _, r = self.trial(images[j], int(labels[j]), state=int(labels[j]),
                              learn=learn)
            hits.append(1 if r > 0 else 0)
        q = max(20, int(trials) // 4)
        return {"final_accuracy": float(np.mean(hits[-q:])),
                "first_quarter": float(np.mean(hits[:q]))}


class BanditTask(DelayedRewardTask):
    """A stochastic k-armed bandit -- one context, several actions, noisy reward.

    The delayed-association task has a right answer that is always rewarded.
    A bandit does not: the best arm pays off only most of the time, so a single
    punishment after a correct choice is normal and the rule has to average over
    trials rather than believe each one. That is a different and harder demand
    on a learning rule, and it is where a reward signal that means nothing
    (control B) is easiest to mistake for one that does.
    """

    def __init__(self, probs: Sequence[float] = (0.8, 0.3, 0.2), **kw):
        kw.setdefault("n_act", 60)
        super().__init__(**kw)
        self.probs = [float(p) for p in probs]
        k = len(self.probs)
        per = self.n_act // k
        self.groups = []
        for g in range(k):
            m = np.zeros(self.n_act, bool)
            m[g * per:(g + 1) * per] = True
            self.groups.append(m)
        self._grp_act = np.zeros(k, np.float32)   # resized for k arms
        # one context: every trial presents the same stimulus
        self.stimuli = [np.ones(self.n_in, bool)]
        self.correct = [int(np.argmax(self.probs))]
        self.rpe = RewardPredictionError(1) if self.rpe is not None else None

    def outcome(self, which: int, choice: int) -> float:
        return 1.0 if self.rng.random() < self.probs[choice] else -1.0

    def _oracle_trace(self, which: int) -> np.ndarray:
        e = np.zeros_like(self.W)
        e[np.ix_(self.groups[self.correct[0]], self.stimuli[0])] = 1.0
        return e

    def run(self, trials: int = 200, report_every: int = 50) -> Dict[str, float]:
        best = self.correct[0]
        picks: List[int] = []
        for _ in range(int(trials)):
            choice, _ = self.trial(0, learn=True)
            picks.append(1 if choice == best else 0)
        q = max(20, int(trials) // 4)
        return {"final_accuracy": float(np.mean(picks[-q:])),
                "first_quarter": float(np.mean(picks[:q])),
                "picks": picks}


# ---------------------------------------------------------------------------
# The comparison the whole module exists to support
# ---------------------------------------------------------------------------
def _trials_to_criterion(hits: Sequence[int], window: int = 20,
                         level: float = 0.75) -> float:
    """First trial at which a moving average holds above ``level``. NaN if never."""
    h = np.asarray(hits, float)
    if len(h) < window:
        return float("nan")
    ma = np.convolve(h, np.ones(window) / window, mode="valid")
    idx = np.flatnonzero(ma >= level)
    return float(idx[0] + window) if len(idx) else float("nan")


def compare_rules(task: str = "association", trials: int = 200,
                  delay_ms: int = 500, lr: float = 0.05, seed: int = 0
                  ) -> Dict[str, Dict[str, float]]:
    """Run all four conditions on the same task with the same seed.

    Returns, per rule: speed (trials to a 75% moving average), final accuracy,
    stability (standard deviation of the moving average over the last half --
    a rule that keeps rewriting a solved association is not stable), and
    retention (accuracy with plasticity switched off afterwards, which asks
    whether the learning is in the weights or in the ongoing dopamine).
    """
    out: Dict[str, Dict[str, float]] = {}
    for rule in DelayedRewardTask.RULES:
        if task == "bandit":
            t = BanditTask(rule=rule, lr=lr, delay_ms=delay_ms, seed=seed)
            r = t.run(trials=trials)
            hits = r["picks"]
        else:
            t = DelayedRewardTask(rule=rule, lr=lr, delay_ms=delay_ms, seed=seed)
            t.w_cross = 14.0
            hits = []
            for _ in range(int(trials)):
                w = int(t.rng.integers(len(t.stimuli)))
                _, rew = t.trial(w, learn=True)
                hits.append(1 if rew > 0 else 0)
        half = len(hits) // 2
        ma = np.convolve(np.asarray(hits, float), np.ones(20) / 20, mode="valid")
        # retention: freeze the weights and test
        frozen = []
        for _ in range(40):
            w = int(t.rng.integers(len(t.stimuli)))
            c, rew = t.trial(w, learn=False)
            frozen.append(1 if (c == t.correct[w if task != "bandit" else 0])
                          else 0)
        out[rule] = {
            "speed": _trials_to_criterion(hits),
            "final": float(np.mean(hits[-50:])),
            "stability": float(np.std(ma[half:])) if len(ma) > half else float("nan"),
            "retention": float(np.mean(frozen)),
        }
    return out
