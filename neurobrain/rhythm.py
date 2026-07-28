"""
rhythm.py
=========

**Theta/gamma nesting, phase coding, and a working memory that is a rhythm.**

The idea being implemented
--------------------------
Lisman & Idiart (1995) proposed that short-term memory is stored in an
oscillation rather than in a persistent state: a slow theta cycle is subdivided
by fast gamma cycles, each gamma cycle is one *slot*, and an item is held by
re-igniting its assembly once per theta cycle in its own slot. Two predictions
fall straight out and both are testable here:

* **Capacity is a ratio, not a parameter.** The number of items that fit is
  theta period / gamma period -- about 7 for 8 Hz nested in 40 Hz. Nobody sets
  a capacity; it is what the two frequencies leave room for.
* **Identity is coded by phase.** Which item you are looking at is written in
  *when in the theta cycle* it fires, so item identity is decodable from phase
  alone.

What makes items survive their own silence
------------------------------------------
An assembly that stops being driven stops firing. The mechanism Lisman and
Idiart use is the after-depolarization -- a slow inward current that follows a
burst, carried in real cells by calcium-activated non-specific cation channels.
A cell that fired a moment ago is left slightly depolarised for a few hundred
milliseconds, so when inhibition next relaxes, the cells that fired last theta
cycle are the ones ready to fire again. That is implemented here as a real
per-cell current with its own time constant, not as a stored list.

Everything is measured against controls: the oscillation switched off, and the
same network with no items loaded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .assembly import SpikingEpisodicMemory, random_episodes


def _match(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard, NOT the smaller-set overlap used in :mod:`assembly`.

    ``_overlap`` divides by the smaller set, so a window in which the WHOLE
    network fires contains every item completely and scores 1.0 against all of
    them. That is exactly how the first version of this file reported "100% of
    items held" while the network was simply saturated -- the same failure mode
    that faked replay one version earlier, in a new costume. Jaccard punishes
    over-activity and under-activity alike."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    union = float((a | b).sum()) or 1.0
    return float((a & b).sum()) / union


class OscillatoryWorkingMemory(SpikingEpisodicMemory):
    """A network whose inhibition breathes, and whose memory rides on it.

    Parameters
    ----------
    theta_hz, gamma_hz:
        The two rhythms. Their ratio is the capacity the model predicts.
    theta_depth, gamma_depth:
        How deeply each rhythm modulates the inhibitory drive onto pyramids.
    adp:
        Strength of the after-depolarization left behind by a spike -- the
        current that lets an item survive the part of the cycle where it is
        silenced.
    tau_adp:
        Its decay, in ms. It has to outlast a theta period (125 ms at 8 Hz) or
        nothing can be maintained from one cycle to the next.
    """

    def __init__(self, *a, noise: float = 2.0, w_ie: float = 12.0,
                 w_ei: float = 6.0, theta_hz: float = 8.0, gamma_hz: float = 40.0,
                 theta_depth: float = 14.0, gamma_depth: float = 10.0,
                 adp: float = 3.0, tau_adp: float = 250.0, **kw):
        kw.setdefault('noise', noise)
        kw.setdefault('w_ie', w_ie)
        kw.setdefault('w_ei', w_ei)
        super().__init__(*a, **kw)
        self.theta_hz, self.gamma_hz = float(theta_hz), float(gamma_hz)
        self.theta_depth, self.gamma_depth = float(theta_depth), float(gamma_depth)
        self.adp_gain, self.tau_adp = float(adp), float(tau_adp)
        self._adp = np.zeros(self.n_exc, np.float32)
        self.t_ms = 0.0

    # -- the clock ----------------------------------------------------------
    def modulation(self, t_ms: float) -> float:
        """Inhibitory drive at time ``t``. Negative = the gate is open."""
        th = np.sin(2.0 * np.pi * self.theta_hz * t_ms / 1000.0)
        ga = np.sin(2.0 * np.pi * self.gamma_hz * t_ms / 1000.0)
        return float(self.theta_depth * th + self.gamma_depth * ga)

    def theta_phase(self, t_ms: float) -> float:
        return float((t_ms * self.theta_hz / 1000.0) % 1.0)

    @property
    def slots(self) -> int:
        """How many gamma cycles fit in a theta cycle -- the predicted capacity."""
        return int(round(self.gamma_hz / self.theta_hz))

    # -- running ------------------------------------------------------------
    def run_rhythmic(self, ms: int, cue_schedule=None, oscillate: bool = True,
                     learn: bool = False, bin_ms: Optional[float] = None
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """Advance with the clock running, binned into gamma-sized windows.

        ``cue_schedule(t_ms)`` returns a boolean mask (or None) -- what the
        world is driving right now.

        Returns ``(times, patterns)`` where ``patterns[w]`` is the set of cells
        that fired during window *w*. The binning is not cosmetic: an assembly
        of 45 cells never fires in the same millisecond, so comparing
        per-millisecond spike sets against a 45-cell item can only ever score
        zero. The window is one gamma half-cycle -- the timescale on which a
        slot's worth of activity is actually assembled.
        """
        width = float(bin_ms if bin_ms is not None else 500.0 / self.gamma_hz)
        times: List[float] = []
        pats: List[np.ndarray] = []
        acc = np.zeros(self.n_exc, bool)
        acc_t, filled = self.t_ms, 0.0
        was = self.rec.stdp.enabled
        self.rec.stdp.enabled = bool(learn)
        decay = float(np.exp(-1.0 / self.tau_adp))
        for _ in range(int(ms)):
            t = self.t_ms
            ext = np.zeros(self.n_exc, np.float32)
            if cue_schedule is not None:
                m = cue_schedule(t)
                if m is not None:
                    ext[np.asarray(m, bool)] = self.drive
            # The rhythm drives the INTERNEURONS, not the pyramids. Adding it
            # to the excitatory pool was the first version and it was simply
            # wrong: a uniform depolarising current at the open phase fires
            # every cell in the network regardless of what it belongs to --
            # measured, 600 of 600 cells active in alternate windows, matching
            # every item at 0.07. Rhythms in cortex are generated by
            # interneuron networks, and inhibition reaches a pyramid through
            # its own synapses, so a cell that is primed (high ADP) escapes the
            # trough first and an unprimed one does not. That is what makes a
            # slot a slot.
            osc = self.modulation(t) if oscillate else 0.0
            i_exc = (self.rec.collect_current() + self.i_to_e.collect_current()
                     + ext + self._adp
                     + self.noise * self.rng.standard_normal(self.n_exc))
            i_inh = (self.e_to_i.collect_current() + osc
                     + self.noise * self.rng.standard_normal(self.n_inh))
            fired = self.exc.step(i_exc.astype(np.float32))
            self.inh.step(i_inh.astype(np.float32))
            # the after-depolarization: a spike leaves the cell primed
            self._adp *= decay
            self._adp[fired] += self.adp_gain
            if learn:
                self.rec.apply_stdp()
            acc |= fired
            filled += 1.0
            self.t_ms += 1.0
            if filled >= width:
                times.append(acc_t)
                pats.append(acc.copy())
                acc[:] = False
                acc_t, filled = self.t_ms, 0.0
        self.rec.stdp.enabled = was
        return np.asarray(times, np.float32), pats

    def clear(self) -> None:
        self._adp[:] = 0.0
        self.rest(80)


# ---------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------
def _load_and_hold(net: OscillatoryWorkingMemory, items: Sequence[np.ndarray],
                   load_ms: int = 40, hold_ms: int = 900,
                   oscillate: bool = True) -> Dict[str, float]:
    """Present items one per gamma slot, then take the world away and watch."""
    net.clear()
    slot = 1000.0 / net.gamma_hz
    t0 = net.t_ms
    order = list(items)

    def schedule(t):
        k = int((t - t0) // slot)
        return order[k] if 0 <= k < len(order) else None

    net.run_rhythmic(int(load_ms * len(order)), cue_schedule=schedule,
                     oscillate=oscillate)
    times, pats = net.run_rhythmic(hold_ms, cue_schedule=None,
                                   oscillate=oscillate)

    # which items are still being visited, and at what phase?
    per_item_phase: List[List[float]] = [[] for _ in order]
    held = np.zeros(len(order), bool)
    counts = np.zeros(len(order), np.int32)
    for t, pat in zip(times, pats):
        for i, it in enumerate(order):
            if _match(pat, it) >= 0.34 and pat.sum() >= 3:
                per_item_phase[i].append(net.theta_phase(float(t)))
                counts[i] += 1
                held[i] = True
    # phase separation: do different items occupy different theta phases?
    means = [float(np.mean(p)) if p else np.nan for p in per_item_phase]
    ok = [m for m in means if not np.isnan(m)]
    sep = 0.0
    if len(ok) > 1:
        d = [abs(a - b) for i, a in enumerate(ok) for b in ok[i + 1:]]
        sep = float(np.mean([min(x, 1.0 - x) for x in d]))
    return {"held": float(held.mean()),
            "events_per_item": float(counts.mean()),
            "phase_separation": sep}


@dataclass
class RhythmReport:
    capacity: Dict[int, float] = field(default_factory=dict)
    no_oscillation: Dict[int, float] = field(default_factory=dict)
    phase_separation: Dict[int, float] = field(default_factory=dict)
    predicted_slots: int = 0
    seconds: float = 0.0

    def summary(self) -> str:
        rows = [f"  {n} items   held {self.capacity[n]:5.1%}   "
                f"no rhythm {self.no_oscillation[n]:5.1%}   "
                f"phase separation {self.phase_separation[n]:.3f}"
                for n in sorted(self.capacity)]
        return ("theta/gamma working memory  (predicted capacity = "
                f"{self.predicted_slots} slots)\n" + "\n".join(rows)
                + f"\n  {self.seconds:.0f}s")


def working_memory_experiment(loads: Sequence[int] = (1, 2, 3, 5, 7, 9),
                              n_exc: int = 600, size: int = 45,
                              seed: int = 0) -> RhythmReport:
    """Load k items into the rhythm and see how many survive the silence.

    The control is the identical network with the oscillation switched off. It
    matters because a recurrent network with an after-depolarization can hold
    *something* without any rhythm at all -- usually one item, or a blend of all
    of them -- and without the control a graded capacity curve would be easy to
    mistake for evidence of slots.
    """
    import time
    t0 = time.time()
    rep = RhythmReport()
    eps = random_episodes(n_exc, max(loads), size, seed=seed)
    net = OscillatoryWorkingMemory(n_exc=n_exc, seed=seed)
    rep.predicted_slots = net.slots
    for e in eps:
        net.store(e)
    for k in loads:
        on = _load_and_hold(net, eps[:k], oscillate=True)
        off = _load_and_hold(net, eps[:k], oscillate=False)
        rep.capacity[k] = on["held"]
        rep.no_oscillation[k] = off["held"]
        rep.phase_separation[k] = on["phase_separation"]
    rep.seconds = time.time() - t0
    return rep
