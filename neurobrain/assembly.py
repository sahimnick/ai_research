"""
assembly.py
===========

**Episodic memory made of spiking neurons**, replacing the vector arithmetic
that stood in for it.

What was wrong with the old memory
----------------------------------
Everything above the sensory layers in this project was linear algebra wearing a
biological name. ``PalliumLayer`` stored a memory as a row of a matrix and
recalled it with ``W @ v`` and a k-winners-take-all. That is a content-
addressable memory and it works, but if asked *"is this episodic memory made of
neurons?"* the answer was no: there were no cells, no spikes, no synapses, and
nothing that could fail the way tissue fails.

What is here instead
--------------------
:class:`SpikingEpisodicMemory` is a recurrent network of Izhikevich cells --
excitatory regular-spiking pyramids and fast-spiking interneurons -- wired by
:class:`~neurobrain.synapse.SynapseBundle` with conduction delays and STDP. A
memory is not written anywhere. It is **an assembly**: a set of cells that fired
together while the episode was present, and whose recurrent synapses were
strengthened by spike timing until the set can re-ignite itself.

Three things follow from that, and each is measured rather than asserted:

* **Pattern completion.** Drive a *fraction* of an assembly and the recurrent
  excitation recruits the rest -- the cells that were never stimulated fire
  anyway. That is the only honest test of an attractor, because the cued cells
  firing proves nothing: they were told to fire. Every number reported here is
  about the **hidden** cells.

* **Inhibition sets the sparseness.** The fast-spiking pool is driven by the
  excitatory pool and inhibits it back, so an assembly ignites without the whole
  network catching fire. There is no k-WTA anywhere; the sparseness is a
  consequence of the E/I loop, and if the loop is mistuned the network either
  dies or seizes. Both failures are visible in :meth:`diagnose`.

* **Replay.** With no input at all, background noise reactivates stored
  assemblies -- the network spontaneously visits its own memories, which is what
  hippocampal replay is. Whether that beats chance is measured, not claimed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .neuron import Population
from .synapse import STDPConfig, SynapseBundle


def _overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of the smaller set that the two sets share."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    both = float((a & b).sum())
    denom = float(min(a.sum(), b.sum())) or 1.0
    return both / denom


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------
class SpikingEpisodicMemory:
    """A recurrent E/I network whose memories are Hebbian cell assemblies.

    Parameters
    ----------
    n_exc, n_inh:
        Pyramidal and interneuron counts. The 4:1 ratio is the cortical one.
    p_rec:
        Connection probability within the excitatory pool -- the synapses that
        carry the memory.
    drive:
        Current injected into a cell that the episode names, in the same units
        the Izhikevich model uses (a regular-spiking cell needs about 5 to fire).
    """

    def __init__(self, n_exc: int = 600, n_inh: int = 150, p_rec: float = 0.25,
                 drive: float = 25.0, w_rec: float = 1.0, w_ei: float = 5.0,
                 w_ie: float = 6.0, noise: float = 4.0, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.n_exc, self.n_inh = int(n_exc), int(n_inh)
        self.drive, self.noise = float(drive), float(noise)

        self.exc = Population(self.n_exc, "regular_spiking", rng=rng, jitter=0.02)
        self.inh = Population(self.n_inh, "fast_spiking", rng=rng, jitter=0.02)

        # Recurrent excitation is the only plastic pathway: the memory lives in
        # pyramid-to-pyramid synapses, exactly as the Hebbian account says.
        # normalize=True makes potentiation competitive -- a cell has a fixed
        # synaptic budget, so joining one assembly costs it strength in others,
        # which is what stops the first memory from swallowing the network.
        self.rec = SynapseBundle.random(
            self.exc, self.exc, p=p_rec, w_scale=w_rec, delay_range=(1, 4),
            rng=rng, stdp=STDPConfig(a_plus=0.060, a_minus=0.008, w_max=10.0,
                                     normalize=True, target_in=220.0))
        # a_plus >> a_minus is not a free choice. With the symmetric default
        # (a_minus slightly larger, which is right for learning input
        # selectivity) two cells firing at the same rate produce as many
        # post-before-pre pairs as pre-before-post, the two cancel, and no
        # assembly forms at all: measured, within-assembly weights came out
        # 1.02 against 1.00 outside, and completion was flat zero. Assembly
        # formation needs potentiation to dominate for CO-ACTIVE cells, which
        # is what the ratio below delivers -- 3.4x separation after storage.
        # The E/I loop. Neither of these learns: inhibition here is a stability
        # mechanism, not a memory.
        self.e_to_i = SynapseBundle.random(
            self.exc, self.inh, p=0.20, w_scale=w_ei, delay_range=(1, 2),
            rng=rng, stdp=STDPConfig(enabled=False))
        self.i_to_e = SynapseBundle.random(
            self.inh, self.exc, p=0.30, w_scale=w_ie, delay_range=(1, 2),
            rng=rng, stdp=STDPConfig(enabled=False))
        self.episodes: List[np.ndarray] = []

    # -- running the tissue -------------------------------------------------
    def run(self, cue: Optional[np.ndarray] = None, ms: int = 150,
            learn: bool = False, noise: Optional[float] = None
            ) -> np.ndarray:
        """Advance the network and return each cell's spike count.

        ``cue`` is a boolean mask over the excitatory pool -- the cells the
        world is currently driving. Everything else is up to the network.
        """
        n = float(self.noise if noise is None else noise)
        ext = np.zeros(self.n_exc, np.float32)
        if cue is not None:
            ext[np.asarray(cue, bool)] = self.drive
        counts = np.zeros(self.n_exc, np.int32)
        was = self.rec.stdp.enabled
        self.rec.stdp.enabled = bool(learn)
        for _ in range(int(ms)):
            i_exc = (self.rec.collect_current() + self.i_to_e.collect_current()
                     + ext + n * self.rng.standard_normal(self.n_exc))
            i_inh = (self.e_to_i.collect_current()
                     + n * self.rng.standard_normal(self.n_inh))
            counts += self.exc.step(i_exc.astype(np.float32))
            self.inh.step(i_inh.astype(np.float32))
            if learn:
                self.rec.apply_stdp()
        self.rec.stdp.enabled = was
        return counts

    def active(self, counts: np.ndarray, k: Optional[int] = None,
               min_spikes: int = 2) -> np.ndarray:
        """Which cells count as part of the current pattern."""
        c = np.asarray(counts)
        if k is None:
            return c >= min_spikes
        idx = np.argsort(-c)[:int(k)]
        out = np.zeros(self.n_exc, bool)
        out[idx[c[idx] >= min_spikes]] = True
        return out

    # -- storing and recalling ---------------------------------------------
    def store(self, pattern: np.ndarray, ms: int = 220, repeats: int = 5
              ) -> np.ndarray:
        """Present an episode until its cells wire together."""
        p = np.asarray(pattern, bool)
        for _ in range(int(repeats)):
            self.run(cue=p, ms=ms, learn=True)
            self.rest(60)
        self.episodes.append(p)
        return p

    def rest(self, ms: int = 60) -> None:
        """Let the network settle so one episode does not smear into the next."""
        self.run(cue=None, ms=int(ms), learn=False, noise=0.4)

    # 150 ms is not a shrug at a number. Run the recall longer and the
    # recurrent excitation stops being recall and becomes spread: at 500 ms the
    # trained network "completes" 100% of hidden cells but also fires 97% of the
    # cells that belong to no episode, and the UNTRAINED control completes 83%.
    # That is a seizure scoring well on a badly chosen metric. The window below
    # is where completion is driven by the stored weights and not by time.
    def recall(self, cue: np.ndarray, ms: int = 150,
               k: Optional[int] = None) -> np.ndarray:
        counts = self.run(cue=np.asarray(cue, bool), ms=ms, learn=False)
        return self.active(counts, k=k)

    def completion(self, episode: np.ndarray, fraction: float = 0.5,
                   ms: int = 150, seed: int = 0) -> Dict[str, float]:
        """The honest attractor test: do the cells NOT cued fire anyway?

        The cued cells are excluded from the score. They were driven; their
        firing is not evidence of anything. What matters is the rest of the
        assembly, and the false-alarm rate among cells that belong to no part of
        this episode at all.
        """
        rng = np.random.default_rng(seed)
        ep = np.asarray(episode, bool)
        members = np.flatnonzero(ep)
        n_cue = max(1, int(round(fraction * len(members))))
        cue_idx = rng.choice(members, n_cue, replace=False)
        cue = np.zeros(self.n_exc, bool)
        cue[cue_idx] = True
        hidden = ep & ~cue
        outside = ~ep

        got = self.recall(cue, ms=ms)
        self.rest(80)
        return {
            "completed": float(got[hidden].mean()) if hidden.any() else 0.0,
            "false_alarm": float(got[outside].mean()) if outside.any() else 0.0,
            "active": float(got.mean()),
        }

    # -- sleep --------------------------------------------------------------
    def replay(self, ms: int = 1200, window: int = 60, noise: float = 6.0,
               k: int = 40, min_active: int = 15) -> List[int]:
        """Spontaneous reactivation with no input: which episode is being visited?

        Returns one index per window -- the stored episode the window's activity
        matches best, or ``-1`` for nothing above threshold.

        MEASURED RESULT: this does not work, and the first version of the metric
        hid that. ``_overlap`` divides by the smaller of the two sets, so a
        window in which **two** cells happened to fire, one of them belonging to
        an episode, scored 0.5 and counted as a replay event -- which produced a
        confident-looking "33% of windows replay a memory" out of pure noise.
        With ``min_active`` demanding a real population event, the true numbers
        are: at low noise nothing fires at all, and at noise high enough to
        ignite cells the activity is unstructured and the untrained control
        matches it. The assemblies here need external drive to ignite; they do
        not reactivate themselves. Reported as a failure rather than tuned until
        a number looked good.
        """
        out: List[int] = []
        for _ in range(max(1, int(ms) // int(window))):
            counts = self.run(cue=None, ms=window, learn=False, noise=noise)
            pat = self.active(counts, k=k)
            if pat.sum() < int(min_active) or not self.episodes:
                out.append(-1)
                continue
            scores = [_overlap(pat, ep) for ep in self.episodes]
            best = int(np.argmax(scores))
            out.append(best if scores[best] >= 0.5 else -1)
        return out

    # -- is the tissue alive and not on fire? -------------------------------
    def diagnose(self, ms: int = 200) -> Dict[str, float]:
        counts = self.run(cue=None, ms=ms, learn=False)
        rate = float(counts.sum()) / (self.n_exc * ms / 1000.0)
        return {"spontaneous_hz": rate,
                "fraction_active": float((counts > 0).mean()),
                "mean_weight": float(self.rec.weight.mean()),
                "max_weight": float(self.rec.weight.max())}


def random_episodes(n_exc: int, n_episodes: int, size: int, seed: int = 0
                    ) -> List[np.ndarray]:
    """Sparse random episodes, with a cap on how much any two may share."""
    rng = np.random.default_rng(seed)
    eps: List[np.ndarray] = []
    while len(eps) < n_episodes:
        p = np.zeros(n_exc, bool)
        p[rng.choice(n_exc, size, replace=False)] = True
        if all(_overlap(p, q) < 0.35 for q in eps):
            eps.append(p)
    return eps


# ---------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------
@dataclass
class AssemblyReport:
    """Pattern completion by spikes, against the controls that make it mean
    something."""

    completed: Dict[float, float] = field(default_factory=dict)
    false_alarm: Dict[float, float] = field(default_factory=dict)
    untrained_completed: Dict[float, float] = field(default_factory=dict)
    replay_hits: float = 0.0
    replay_chance: float = 0.0
    spontaneous_hz: float = 0.0
    seconds: float = 0.0

    def summary(self) -> str:
        rows = [f"  cue {int(f*100):3d}%  hidden cells recruited "
                f"{self.completed[f]:5.1%}   untrained {self.untrained_completed[f]:5.1%}"
                f"   false alarm {self.false_alarm[f]:5.1%}"
                for f in sorted(self.completed)]
        return ("spiking episodic memory\n" + "\n".join(rows) +
                f"\n  replay: {self.replay_hits:.1%} of windows land on a stored "
                f"episode (chance {self.replay_chance:.1%})"
                f"\n  spontaneous rate {self.spontaneous_hz:.1f} Hz, "
                f"{self.seconds:.0f}s")


def assembly_experiment(n_exc: int = 600, n_episodes: int = 4, size: int = 60,
                        fractions: Sequence[float] = (0.25, 0.5, 0.75),
                        seed: int = 0) -> AssemblyReport:
    """Store episodes in tissue, then cue fragments of them.

    The control is the same network with the same episodes **never stored** --
    identical wiring, identical cues, no STDP. Without it the numbers would be
    meaningless, because a recurrent network with random weights already spreads
    activity from any cue.
    """
    import time
    t0 = time.time()
    eps = random_episodes(n_exc, n_episodes, size, seed=seed)

    ctrl = SpikingEpisodicMemory(n_exc=n_exc, seed=seed)
    ctrl.episodes = list(eps)                       # known, but never wired in
    rep = AssemblyReport()
    for f in fractions:
        rep.untrained_completed[f] = float(np.mean(
            [ctrl.completion(e, fraction=f, seed=seed + i)["completed"]
             for i, e in enumerate(eps)]))

    net = SpikingEpisodicMemory(n_exc=n_exc, seed=seed)
    for e in eps:
        net.store(e)
    for f in fractions:
        res = [net.completion(e, fraction=f, seed=seed + i)
               for i, e in enumerate(eps)]
        rep.completed[f] = float(np.mean([r["completed"] for r in res]))
        rep.false_alarm[f] = float(np.mean([r["false_alarm"] for r in res]))

    hits = net.replay()
    rep.replay_hits = float(np.mean([h >= 0 for h in hits]))
    ctrl_hits = ctrl.replay()
    rep.replay_chance = float(np.mean([h >= 0 for h in ctrl_hits]))
    rep.spontaneous_hz = net.diagnose()["spontaneous_hz"]
    rep.seconds = time.time() - t0
    return rep
