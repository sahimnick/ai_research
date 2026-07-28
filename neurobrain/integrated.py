"""
integrated.py
=============

**One brain: every spiking module allocated from a single neuron budget, run in
one loop, and measured against the same modules standing alone.**

The question this module exists to answer
-----------------------------------------
Not "can we build 40 million neurons" -- that was answered, they build in
seconds and they spike. The question is whether wiring the modules into one
circuit **helps**, and the honest prior from this project is that it does not:
when dopamine learning was attached to real V1 features it scored at chance,
and hierarchical depth has failed four separate times. So this module is built
so that the comparison is unavoidable. Every module reports its own task score
twice -- alone, and inside the integrated brain -- and the difference is the
only number that matters.

The budget
----------
Cortex is not uniform, and neither is this. The allocation follows roughly the
proportions of a mammalian brain rather than dividing equally: most of the
tissue is sensory, a large associative pool sits above it, and the
neuromodulatory and action circuits are tiny by comparison. What each pool
*does* is what it did before; what is new is that they share a clock, and
activity from one reaches the next.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .neuron import Population
from .synapse import STDPConfig, SynapseBundle


# fractions of the total budget, in the spirit of cortical proportions
ALLOCATION = {
    "V1_visual": 0.30,
    "A1_auditory": 0.20,
    "association": 0.22,     # where vision and hearing meet
    "episodic": 0.13,        # hippocampal-like assemblies
    "imagination": 0.10,     # the generative loop
    "action": 0.03,
    "neuromodulatory": 0.02,
}


@dataclass
class AreaReport:
    name: str
    n_exc: int
    n_inh: int
    synapses: int
    spikes: int = 0
    rate_hz: float = 0.0


class IntegratedBrain:
    """Every area on one substrate, one millisecond at a time.

    Parameters
    ----------
    n_total:
        Total neuron budget, split by :data:`ALLOCATION`.
    fan_out:
        Synapses per neuron *within* an area. The between-area projections are
        sized separately and are much sparser, as long-range cortical
        connections are.
    """

    def __init__(self, n_total: int = 1_000_000, fan_out: int = 8,
                 inter_fan: int = 2, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.n_total = int(n_total)
        self.areas: Dict[str, Dict] = {}
        self.reports: Dict[str, AreaReport] = {}
        for name, frac in ALLOCATION.items():
            n = max(200, int(self.n_total * frac))
            n_exc, n_inh = int(n * 0.8), max(50, int(n * 0.2))
            exc = Population(n_exc, "regular_spiking", rng=rng, jitter=0.02)
            inh = Population(n_inh, "fast_spiking", rng=rng, jitter=0.02)
            rec = SynapseBundle.random(
                exc, exc, n_edges=n_exc * fan_out, w_scale=0.6,
                delay_range=(1, 4), rng=rng,
                stdp=STDPConfig(a_plus=0.02, a_minus=0.004, w_max=6.0,
                                normalize=True, target_in=float(fan_out) * 1.2))
            ei = SynapseBundle.random(exc, inh, n_edges=n_inh * fan_out,
                                      w_scale=3.0, delay_range=(1, 2), rng=rng,
                                      stdp=STDPConfig(enabled=False))
            ie = SynapseBundle.random(inh, exc, n_edges=n_exc * 2,
                                      w_scale=4.0, delay_range=(1, 2), rng=rng,
                                      stdp=STDPConfig(enabled=False))
            self.areas[name] = {"exc": exc, "inh": inh, "rec": rec,
                                "ei": ei, "ie": ie, "out": []}
            self.reports[name] = AreaReport(name, n_exc, n_inh,
                                            len(rec) + len(ei) + len(ie))
        # long-range projections: the wiring diagram of the whole thing
        self.tracts: List[Tuple[str, str, SynapseBundle]] = []
        for src, dst in [("V1_visual", "association"),
                         ("A1_auditory", "association"),
                         ("association", "episodic"),
                         ("episodic", "imagination"),
                         ("imagination", "association"),   # top-down
                         ("association", "action"),
                         ("action", "neuromodulatory"),
                         ("neuromodulatory", "association")]:
            a, b = self.areas[src], self.areas[dst]
            bundle = SynapseBundle.random(
                a["exc"], b["exc"], n_edges=b["exc"].n * inter_fan,
                w_scale=1.2, delay_range=(3, 12), rng=rng,
                stdp=STDPConfig(a_plus=0.02, a_minus=0.004, w_max=6.0))
            self.tracts.append((src, dst, bundle))
            self.reports[dst].synapses += len(bundle)
        self.t_ms = 0.0

    # -- census -------------------------------------------------------------
    @property
    def n_neurons(self) -> int:
        return sum(r.n_exc + r.n_inh for r in self.reports.values())

    @property
    def n_synapses(self) -> int:
        return sum(r.synapses for r in self.reports.values())

    # -- one millisecond of the whole brain --------------------------------
    def step(self, drive: Optional[Dict[str, np.ndarray]] = None,
             noise: float = 3.0, learn: bool = False) -> Dict[str, np.ndarray]:
        """Advance every area once. Returns each area's spike vector."""
        incoming = {name: np.zeros(a["exc"].n, np.float32)
                    for name, a in self.areas.items()}
        for src, dst, bundle in self.tracts:
            incoming[dst] += bundle.collect_current()
        fired: Dict[str, np.ndarray] = {}
        for name, a in self.areas.items():
            exc, inh = a["exc"], a["inh"]
            i_e = (a["rec"].collect_current() + a["ie"].collect_current()
                   + incoming[name]
                   + noise * self.rng.standard_normal(exc.n).astype(np.float32))
            if drive and name in drive and drive[name] is not None:
                d = np.asarray(drive[name], np.float32)
                i_e[:len(d)] += d[:exc.n]
            i_i = (a["ei"].collect_current()
                   + noise * self.rng.standard_normal(inh.n).astype(np.float32))
            f = exc.step(i_e.astype(np.float32))
            inh.step(i_i.astype(np.float32))
            fired[name] = f
            self.reports[name].spikes += int(f.sum())
            if learn:
                a["rec"].apply_stdp()
        if learn:
            for _, _, bundle in self.tracts:
                bundle.apply_stdp()
        self.t_ms += 1.0
        return fired

    def run(self, ms: int = 100, drive=None, noise: float = 3.0,
            learn: bool = False) -> Dict[str, np.ndarray]:
        """Run for ``ms`` and return per-area spike counts."""
        counts = {n: np.zeros(a["exc"].n, np.int32)
                  for n, a in self.areas.items()}
        for k in range(int(ms)):
            d = drive(k) if callable(drive) else drive
            f = self.step(drive=d, noise=noise, learn=learn)
            for n in counts:
                counts[n] += f[n]
        for n, r in self.reports.items():
            r.rate_hz = float(counts[n].sum()) / max(1.0, r.n_exc * ms / 1000.0)
        return counts

    # -- is it one brain, or several networks in one process? --------------
    def connectivity_report(self) -> Dict[str, float]:
        """Does activity in one area actually change another?

        A brain is not a list of areas that happen to be in the same object. The
        test is causal: drive V1 alone, and see whether the areas downstream of
        it fire more than they do with no drive at all. Anything that does not
        move is not connected in any sense that matters.
        """
        base = self.run(ms=60, drive=None, noise=3.0)
        base_rate = {n: float(c.sum()) for n, c in base.items()}
        v1 = self.areas["V1_visual"]["exc"].n
        d = {"V1_visual": np.full(v1, 12.0, np.float32)}
        driven = self.run(ms=60, drive=d, noise=3.0)
        out: Dict[str, float] = {}
        for n, c in driven.items():
            b = base_rate[n]
            out[n] = (float(c.sum()) - b) / max(b, 1.0)
        return out


def integration_experiment(n_total: int = 1_000_000, ms: int = 60,
                           seed: int = 0) -> Dict[str, object]:
    """Build the whole brain, run it, and report what actually happened."""
    t0 = time.time()
    brain = IntegratedBrain(n_total=n_total, seed=seed)
    build = time.time() - t0
    t0 = time.time()
    brain.run(ms=ms, noise=3.0)
    run_s = time.time() - t0
    prop = brain.connectivity_report()
    return {"neurons": brain.n_neurons, "synapses": brain.n_synapses,
            "build_s": build, "ms_per_ms": run_s / ms * 1000,
            "areas": {n: (r.n_exc + r.n_inh, r.rate_hz)
                      for n, r in brain.reports.items()},
            "propagation": prop}


# ---------------------------------------------------------------------------
# The sensory bridge: what makes 40M neurons fire ABOUT something
# ---------------------------------------------------------------------------
class SensoryBridge:
    """Feeds the working sensory front ends into the spiking areas.

    Until now the two halves of this project never touched. The front ends that
    actually work -- WideV1 at 89.5% on digits, the contrastive A1 at 87.6% on
    sounds -- are numpy, and the 40M-neuron circuit spiked at a uniform 48 Hz
    because every area was being driven by hand. Uniform activity is not
    perception; remove the drive and it goes quiet.

    This class removes the hand. An image goes through the real V1 filter bank,
    a sound through the real cochlea and A1, and the resulting rate vectors
    become the injected current of the corresponding cortical area. Each feature
    cell drives a contiguous block of area cells, which is what divergence in a
    real projection looks like, and the whole vector is sparsified first,
    because a dense drive makes every class look alike -- that was measured
    directly when dopamine learning was attached to dense V1 features and scored
    at chance.
    """

    def __init__(self, brain: "IntegratedBrain", v1=None, a1=None, coch=None,
                 gain: float = 26.0, sparsity: float = 0.05):
        self.brain, self.v1, self.a1, self.coch = brain, v1, a1, coch
        self.gain, self.sparsity = float(gain), float(sparsity)

    @staticmethod
    def _sparse(code: np.ndarray, frac: float) -> np.ndarray:
        c = np.asarray(code, np.float32).reshape(-1)
        if 0.0 < frac < 1.0:
            k = max(1, int(len(c) * frac))
            keep = np.argpartition(c, -k)[-k:]
            out = np.zeros_like(c)
            out[keep] = c[keep]
            c = out
        m = float(np.abs(c).max()) or 1.0
        return c / m

    def _spread(self, code: np.ndarray, area: str) -> np.ndarray:
        """One feature cell -> a block of cortical cells (divergence)."""
        n = self.brain.areas[area]["exc"].n
        c = np.asarray(code, np.float32).reshape(-1)
        reps = int(np.ceil(n / len(c)))
        return (np.repeat(c, reps)[:n] * self.gain).astype(np.float32)

    def image_drive(self, image: np.ndarray) -> np.ndarray:
        code = self.v1.rate_over([np.asarray(image, np.float32)])
        return self._spread(self._sparse(code, self.sparsity), "V1_visual")

    def audio_drive(self, wav: np.ndarray) -> np.ndarray:
        c = self.coch.forward(np.asarray(wav, np.float32))[0]
        code = self.a1.code(c, step=6)
        return self._spread(self._sparse(code, self.sparsity), "A1_auditory")

    def perceive(self, image=None, wav=None, ms: int = 60,
                 noise: float = 2.0) -> Dict[str, np.ndarray]:
        """Show the brain something and return every area's spike counts."""
        drive: Dict[str, np.ndarray] = {}
        if image is not None:
            drive["V1_visual"] = self.image_drive(image)
        if wav is not None:
            drive["A1_auditory"] = self.audio_drive(wav)
        return self.brain.run(ms=ms, drive=drive, noise=noise)
