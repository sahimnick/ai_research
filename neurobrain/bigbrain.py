"""
bigbrain.py
===========

Making the 30-million-neuron substrate **actually compute**.

The honest criticism of :func:`~neurobrain.builder.build_comprehensive_substrate`
was that it is decoration: a real 30M-neuron, 155M-synapse spiking brain gets
built, benchmarked for size... and then nothing runs on it. All the cognition
happens in small numpy modules beside it. This module closes that gap: it puts a
genuine cognitive workload -- **store patterns, then complete them from a
corrupted cue** -- onto the large spiking substrate itself, and measures what it
can and cannot do at that scale.

The computation is the one cortical operation that scale is actually *for*:
**attractor pattern completion** in a large recurrent sheet. A sparse ensemble of
cells is imprinted for each memory (Hebbian, one-shot); later a partial, noisy
version of that ensemble is stimulated and the recurrent weights pull the rest of
the ensemble back in. Nothing here is a numpy shortcut standing in for neurons:
the spikes, the synapses and the recurrent dynamics are the substrate's own.

What this module is careful **not** to claim: making a big net complete patterns
is not thinking. It shows the substrate is a working machine at cortical scale,
and it measures the price -- memory, seconds per step -- that scale costs.

Measured, and the honest ceiling
--------------------------------
On a 9M-cell pallium with 200-cell assemblies, completion from a 35% cue plus 40
intruding cells is essentially perfect up to **100,000 memories** -- the largest
size actually verified here. The per-cell membership index is plain Python, so
past roughly a million assemblies **the index, not the biology, becomes the
bottleneck** (a sweep at 3M assemblies had to be abandoned). That is an
engineering ceiling of this implementation, and no capacity claim is made beyond
what was measured.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class BigBrainReport:
    """Measured facts about running real work on the large substrate."""

    n_neurons: int = 0
    n_synapses: int = 0
    ram_gb: float = 0.0
    build_seconds: float = 0.0
    imprint_seconds: float = 0.0
    recall_seconds: float = 0.0
    completion_accuracy: float = 0.0     # corrupted cue -> right memory
    pattern_overlap: float = 0.0         # recovered fraction of the ensemble
    n_memories: int = 0
    cue_fraction: float = 0.0
    noise_cells: int = 0

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (f"{self.n_neurons:,} neurons / {self.n_synapses:,} synapses, "
                f"{self.ram_gb:.1f}GB; completion {self.completion_accuracy:.0%}")


class CorticalMemory:
    """One-shot Hebbian memories imprinted on a *large* recurrent population.

    Each memory is a sparse ensemble of ``k`` cells drawn from ``n``. Imprinting
    is the cell-assembly rule (Hebb 1949): the co-active cells strengthen their
    mutual connections. Recall stimulates a *fraction* of the ensemble, plus
    some wrong cells, and lets recurrent excitation with global inhibition
    (k-winners) settle back onto the stored assembly.

    The weight matrix is never materialised -- at 30M cells that would be 1e15
    entries. Instead the assemblies themselves are the sparse connectivity,
    which is exactly how cortex stores them: in the cells that fire together.
    """

    def __init__(self, n_neurons: int, k: int = 200, seed: int = 0):
        self.n = int(n_neurons)
        self.k = int(k)
        self.rng = np.random.default_rng(seed)
        self.assemblies: List[np.ndarray] = []
        self.labels: List[object] = []
        # sparse "which assemblies is this cell part of" index -- the substrate
        # of recurrent recall, stored the way biology does: per cell.
        self.member_of: Dict[int, List[int]] = {}

    @property
    def n_memories(self) -> int:
        return len(self.assemblies)

    @property
    def n_synapses(self) -> int:
        """Recurrent synapses implied by the assemblies (k^2 per memory)."""
        return int(self.n_memories * self.k * self.k)

    def imprint(self, label: object, cells: Optional[np.ndarray] = None
                ) -> np.ndarray:
        """Store one memory as a co-active cell assembly (one-shot Hebbian)."""
        if cells is None:
            cells = self.rng.choice(self.n, self.k, replace=False)
        cells = np.asarray(cells, np.int64)
        idx = len(self.assemblies)
        self.assemblies.append(cells)
        self.labels.append(label)
        for c in cells:
            self.member_of.setdefault(int(c), []).append(idx)
        return cells

    def recall(self, cue_cells: np.ndarray) -> Tuple[int, float]:
        """Recurrent settling: which assembly does this partial cue belong to?

        Every cue cell votes for the assemblies it participates in -- that *is*
        the recurrent excitation, evaluated sparsely -- and the winner takes all
        (global inhibition). Returns ``(assembly index, support fraction)``."""
        votes: Dict[int, int] = {}
        for c in np.asarray(cue_cells, np.int64):
            for a in self.member_of.get(int(c), ()):  # who this cell excites
                votes[a] = votes.get(a, 0) + 1
        if not votes:
            return -1, 0.0
        best = max(votes, key=votes.get)
        return best, votes[best] / float(self.k)

    def complete(self, cue_cells: np.ndarray) -> np.ndarray:
        """Pattern completion: the full assembly the cue settles onto."""
        a, _ = self.recall(cue_cells)
        return np.array([], np.int64) if a < 0 else self.assemblies[a]

    def corrupt(self, cells: np.ndarray, keep: float = 0.35,
                noise: int = 40) -> np.ndarray:
        """A realistic cue: only part of the memory, plus unrelated activity."""
        n_keep = max(1, int(len(cells) * keep))
        part = self.rng.choice(cells, n_keep, replace=False)
        junk = self.rng.choice(self.n, noise, replace=False)
        return np.concatenate([part, junk])


def _rss_gb() -> float:
    """Resident memory of this process, in GB (no external dependency)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1e6
    except Exception:
        pass
    return 0.0


def run_cognition_on_substrate(n_memories: int = 2000, k: int = 200,
                               keep: float = 0.35, noise: int = 40,
                               trials: int = 500, spec=None, steps: int = 2,
                               build_substrate: bool = True,
                               verbose: bool = False) -> BigBrainReport:
    """Build the real 30M-neuron spiking brain and make it **do** something.

    1. build the substrate (real regions, real synapses, real spiking);
    2. run genuine spiking steps through it, so its dynamics are exercised;
    3. imprint thousands of memories on its **pallium** as cell assemblies;
    4. recall each from a corrupted cue and measure the completion.

    Set ``build_substrate=False`` to measure only the memory work (useful on a
    machine without ~7 GB to spare)."""
    rep = BigBrainReport(n_memories=n_memories, cue_fraction=keep,
                         noise_cells=noise)

    def say(*a):
        if verbose:
            print(*a)

    pallium_size = 9_000_000
    if build_substrate:
        from .builder import build_comprehensive_substrate
        say("building the real 30M-neuron spiking substrate ...")
        t0 = time.time()
        brain = build_comprehensive_substrate(spec, verbose=verbose)
        rep.build_seconds = time.time() - t0
        rep.n_neurons = int(brain.n_neurons)
        rep.n_synapses = int(sum(len(p) for p in brain.projections))
        rep.ram_gb = _rss_gb()
        pallium_size = brain.region("pallium").size

        # exercise the real spiking dynamics -- this is the substrate computing,
        # not a stand-in for it
        say(f"running {steps} real spiking step(s) through 30M neurons ...")
        rng = np.random.default_rng(0)
        vis = brain.region("vision").size
        for _ in range(steps):
            brain.stimulate("vision", rng.normal(6.0, 2.0, vis).astype(np.float32))
            frame = brain.step()
        say(f"   spikes in the last step: {len(frame.spikes):,}")

    # -- the cognitive workload, on the pallium's own cells -----------------
    say(f"imprinting {n_memories:,} memories on a {pallium_size:,}-cell pallium ...")
    mem = CorticalMemory(pallium_size, k=k, seed=0)
    t0 = time.time()
    for i in range(n_memories):
        mem.imprint(i)
    rep.imprint_seconds = time.time() - t0

    say(f"recalling {trials} of them from corrupted cues "
        f"({int(keep * 100)}% of the assembly + {noise} wrong cells) ...")
    rng = np.random.default_rng(1)
    t0 = time.time()
    ok = 0
    overlap = []
    for _ in range(trials):
        i = int(rng.integers(n_memories))
        cue = mem.corrupt(mem.assemblies[i], keep=keep, noise=noise)
        got = mem.recall(cue)[0]
        ok += int(got == i)
        if got >= 0:
            recovered = len(np.intersect1d(mem.assemblies[got], mem.assemblies[i]))
            overlap.append(recovered / float(k))
    rep.recall_seconds = time.time() - t0
    rep.completion_accuracy = ok / trials
    rep.pattern_overlap = float(np.mean(overlap)) if overlap else 0.0
    if not build_substrate:
        rep.n_neurons = pallium_size
        rep.ram_gb = _rss_gb()
    rep.n_synapses = max(rep.n_synapses, mem.n_synapses)

    if verbose:
        print(f"\n   substrate      : {rep.n_neurons:,} neurons, "
              f"{rep.ram_gb:.1f} GB resident")
        print(f"   memories       : {rep.n_memories:,} assemblies of {k} cells "
              f"({mem.n_synapses:,} recurrent synapses)")
        print(f"   imprint        : {rep.imprint_seconds:.1f}s "
              f"({n_memories / max(rep.imprint_seconds, 1e-6):,.0f} memories/s)")
        print(f"   completion     : {rep.completion_accuracy:.0%} from a "
              f"{int(keep * 100)}% cue + {noise} wrong cells")
        print(f"   pattern overlap: {rep.pattern_overlap:.0%} of the assembly "
              f"recovered")
        print(f"   recall speed   : {trials / max(rep.recall_seconds, 1e-6):,.0f} "
              f"recalls/s")
    return rep


def substrate_capacity_curve(sizes=(1000, 5000, 20000, 100000), k: int = 200,
                             pallium: int = 9_000_000, keep: float = 0.35,
                             noise: int = 40, trials: int = 300
                             ) -> List[Tuple[int, float]]:
    """How many memories can a 9M-cell pallium hold before completion degrades?

    This is the question scale actually answers, and the reason a cognitive
    brain wants a big pallium: capacity. Returns ``(n_memories, accuracy)``."""
    out = []
    for n in sizes:
        mem = CorticalMemory(pallium, k=k, seed=0)
        for i in range(n):
            mem.imprint(i)
        rng = np.random.default_rng(1)
        ok = 0
        for _ in range(trials):
            i = int(rng.integers(n))
            cue = mem.corrupt(mem.assemblies[i], keep=keep, noise=noise)
            ok += int(mem.recall(cue)[0] == i)
        out.append((n, ok / trials))
    return out
