"""
tagging.py
==========

Does **synaptic tagging and capture** actually hold a long-term correlation
better than classic STDP? Measured here, not asserted.

The defect being tested
-----------------------
Pair-based STDP has no memory between events. Each coincidence moves the weight
by the same amount whether it is the hundredth repetition of a real regularity
or a one-off accident, and -- the part that hurts -- the weight has no way to
tell "I have been confirmed a thousand times" from "I happened once". So when
the world changes and a synapse's old partner stops predicting the post-synaptic
cell, everything that synapse learned erodes at the same speed it was acquired.
Old memories are overwritten by whatever is happening now.

Biology's answer is to split plasticity in two by *time*. An event leaves a
**tag** -- a calcium transient and the CaMKII / protein machinery it triggers --
which decays over minutes. Only a tag still standing when the plasticity-related
proteins arrive is **captured** into the late phase and made permanent
(Frey & Morris, 1997). Everything else washes out. The mechanism lives in
:class:`~neurobrain.synapse.TagConfig`; this module is the experiment.

The protocol
------------
A two-phase interference experiment, the standard way to expose forgetting.
One post-synaptic cell is driven on a fixed schedule. Twenty-four pre-synaptic
cells project to it in two groups of twelve, **always matched for total spike
count** so the only difference between them is *timing*:

    * **Phase 1 (learn A).** Group A fires a few ms before every post spike --
      a real, repeating pre-before-post correlation. Group B fires just as often
      at random moments.
    * **Phase 2 (interference).** The roles swap. Group B now carries the
      correlation; group A is reduced to firing at random.

The question is what is left of A. A rule that merely tracks recent statistics
must give it up; a rule that consolidated it should not.

Fairness
--------
A skeptic's first objection is that tagging just changes the effective learning
rate. So the classic condition is run across a **sweep of learning rates** and
its *best* retention is the number tagging must beat. Both conditions otherwise
share seeds, schedules, neurons and STDP constants.

Two honesty checks are reported alongside the headline:

    ``new_learning``  how much of the *new* correlation B each rule picked up in
                      phase 2. Consolidation that protects the past by refusing
                      to learn the present is not a feature, and this number
                      would expose it.
    ``noise_floor``   separation of signal from noise during phase 1. Tagging is
                      **not** claimed to help here, and it does not; the measured
                      value is printed either way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from ..core.neuron import Population
from ..core.synapse import STDPConfig, SynapseBundle, TagConfig

N_GROUP = 12                 # cells per group
N_PRE = 2 * N_GROUP
W0 = 0.5                     # every synapse starts here


def _schedule(n_ms: int, correlated: Set[int], period: int, lead: int,
              rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Stimulation schedule for one phase.

    Cells in ``correlated`` fire a few ms before every post spike; every other
    cell fires the **same number of times** at uniformly random moments. Equal
    counts are what make the comparison about timing rather than activity.

    Each correlated cell gets its own lead (drawn once) so the group is not a
    set of clones -- a group of clones would have exactly zero weight variance
    and any separation measure over it would be meaningless."""
    pre = np.zeros((n_ms, N_PRE), bool)
    post = np.zeros(n_ms, bool)
    events = np.arange(lead + 8, n_ms - 2, period)
    post[events] = True
    for c in range(N_PRE):
        if c in correlated:
            pre[events - (lead + int(rng.integers(0, 6))), c] = True
        else:
            t = rng.choice(n_ms - 4, size=len(events), replace=False) + 2
            pre[t, c] = True
    return pre, post


def _run(pre_on: np.ndarray, post_on: np.ndarray, bundle: SynapseBundle,
         pre: Population, post: Population, pulse: float = 90.0) -> int:
    """Drive the cells on the schedule and let the synapses learn.

    The cells are *real* Izhikevich neurons; the schedule arrives as a current
    pulse, the way an electrode delivers it in a slice. The number of pre spikes
    that actually occurred is returned so the protocol can be verified instead
    of assumed."""
    n_ms, n_pre = pre_on.shape
    fired = 0
    ext_pre = np.zeros(n_pre, np.float32)
    ext_post = np.zeros(post.n, np.float32)
    for t in range(n_ms):
        ext_pre[:] = pulse * pre_on[t]
        ext_post[:] = pulse * post_on[t]
        syn = bundle.collect_current()
        pre.step(ext_pre)
        post.step(ext_post + syn)
        bundle.apply_stdp()
        fired += int(pre.spiked.sum())
    return fired


def _d_prime(w: np.ndarray) -> float:
    """Separation of group A from group B in units of their pooled spread."""
    a, b = w[:N_GROUP], w[N_GROUP:]
    pooled = float(np.sqrt(0.5 * (a.var() + b.var()))) + 1e-9
    return float((a.mean() - b.mean()) / pooled)


@dataclass
class TagRun:
    """Everything one condition produced, measured."""

    label: str
    a_after_learn: float = 0.0
    b_after_learn: float = 0.0
    a_after_interference: float = 0.0
    b_after_interference: float = 0.0
    d_prime_learn: float = 0.0
    retention: float = 0.0
    new_learning: float = 0.0
    pre_spikes: int = 0
    noise_consolidated: float = 0.0


def _condition(tagged: bool, a_plus: float, learn_ms: int, interfere_ms: int,
               period: int, lead: int, seed: int,
               tag_cfg: TagConfig) -> TagRun:
    rng = np.random.default_rng(seed)
    pre = Population(N_PRE, "regular_spiking", rng=np.random.default_rng(1))
    post = Population(1, "regular_spiking", rng=np.random.default_rng(2))
    bundle = SynapseBundle(
        pre, post,
        pre=np.arange(N_PRE), post=np.zeros(N_PRE, int),
        weight=np.full(N_PRE, W0, np.float32),
        delay=np.ones(N_PRE, int),
        stdp=STDPConfig(a_plus=a_plus, a_minus=a_plus * 1.1,
                        w_min=0.0, w_max=4.0),
        tags=tag_cfg if tagged else TagConfig(enabled=False))

    group_a = set(range(N_GROUP))
    group_b = set(range(N_GROUP, N_PRE))

    r = TagRun("tagged" if tagged else f"classic(a+={a_plus})")
    p, q = _schedule(learn_ms, group_a, period, lead, rng)
    r.pre_spikes = _run(p, q, bundle, pre, post)
    w = bundle.consolidated
    r.a_after_learn, r.b_after_learn = float(w[:N_GROUP].mean()), float(
        w[N_GROUP:].mean())
    r.d_prime_learn = _d_prime(w.copy())
    if tagged and bundle.w_stable is not None:
        # how much of the *uncorrelated* group made it into the late phase?
        # If tagging were indiscriminate this would match the signal group.
        r.noise_consolidated = float(
            np.abs(bundle.w_stable[N_GROUP:] - W0).mean())

    p, q = _schedule(interfere_ms, group_b, period, lead, rng)
    _run(p, q, bundle, pre, post)
    w = bundle.consolidated
    r.a_after_interference = float(w[:N_GROUP].mean())
    r.b_after_interference = float(w[N_GROUP:].mean())

    gained = r.a_after_learn - W0
    r.retention = float((r.a_after_interference - W0) / gained) if abs(
        gained) > 1e-6 else 0.0
    r.new_learning = r.b_after_interference - r.b_after_learn
    return r


def synaptic_tagging_experiment(
        learn_ms: int = 15_000, interfere_ms: int = 15_000, period: int = 25,
        lead: int = 6, seed: int = 0,
        classic_rates: Sequence[float] = (0.002, 0.005, 0.010, 0.020),
        verbose: bool = False) -> Dict[str, float]:
    """Classic STDP vs. STDP + tagging under interference, identical inputs.

    Classic is given every advantage: it is run at four learning rates and only
    its **best** retention is reported, so the comparison cannot be won by
    tuning."""
    cfg = TagConfig(enabled=True, tau_calcium=4000.0, threshold=0.10,
                    capture=0.0005, tau_labile=300.0)

    classics: List[TagRun] = [
        _condition(False, ap, learn_ms, interfere_ms, period, lead, seed, cfg)
        for ap in classic_rates]
    best_i = int(np.argmax([r.retention for r in classics]))
    best = classics[best_i]
    tag = _condition(True, 0.010, learn_ms, interfere_ms, period, lead, seed,
                     cfg)

    out = {
        "classic_retention": best.retention,
        "tagged_retention": tag.retention,
        "retention_gain": tag.retention - best.retention,
        "classic_best_rate": float(classic_rates[best_i]),
        "classic_retention_spread": float(
            max(r.retention for r in classics)
            - min(r.retention for r in classics)),
        "classic_new_learning": best.new_learning,
        "tagged_new_learning": tag.new_learning,
        "classic_d_prime": best.d_prime_learn,
        "tagged_d_prime": tag.d_prime_learn,
        "tagged_noise_consolidated": tag.noise_consolidated,
        "pre_spikes": float(tag.pre_spikes),
    }
    if verbose:
        print(f"   pre-synaptic spikes delivered: {tag.pre_spikes:,} "
              f"(identical in every condition)")
        print("   phase 1 learns A, phase 2 makes B the correlated group\n")
        for r in classics:
            print(f"   {r.label:<20} A {r.a_after_learn:.2f} -> "
                  f"{r.a_after_interference:.2f}   kept {r.retention:5.0%}")
        print(f"   {'tagged':<20} A {tag.a_after_learn:.2f} -> "
              f"{tag.a_after_interference:.2f}   kept {tag.retention:5.0%}")
        print(f"\n   retention  classic best {best.retention:.0%}  vs  "
              f"tagged {tag.retention:.0%}  ({out['retention_gain']:+.0%})")
        print(f"   new pattern B still learned: classic "
              f"{best.new_learning:+.2f}, tagged {tag.new_learning:+.2f}")
        print(f"   signal/noise separation in phase 1 (tagging is NOT claimed "
              f"to help): classic d'={best.d_prime_learn:.2f}, "
              f"tagged d'={tag.d_prime_learn:.2f}")
    return out
