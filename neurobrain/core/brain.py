"""
brain.py
========

The **brain environment** that ties everything together.

It owns the regions, the projections (synapse bundles) between them, a global
clock, a :class:`WorldModel`, and a recorder that remembers the path activity
took -- which is exactly what the live visualizer replays.

The simulation loop for one millisecond step is:

    1. collect external stimulation queued for sensory regions
    2. collect synaptic current arriving now (respecting axonal delays)
    3. advance every neuron population (Izhikevich update)
    4. let every plastic synapse learn from the spikes (STDP)
    5. let the world model interpret the new global activity
    6. record spikes and active pathways for visualisation

Everything is vectorised per population, so the loop stays fast even for
thousands of neurons.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .neuron import Population
from .region import Region, default_layout, ROLE_SENSORY
from .synapse import STDPConfig, SynapseBundle
from ..world.concepts import WorldModel


@dataclass
class SpikeFrame:
    """One recorded step of the simulation (what the visualizer replays)."""

    t: int
    spikes: np.ndarray                 # global ids of neurons that fired
    activation: np.ndarray             # global decaying-activation vector
    active_concepts: List[str] = field(default_factory=list)


class Brain:
    """A network of regions plus a world model: a small, living brain.

    Parameters
    ----------
    seed:
        Master RNG seed for full reproducibility.
    dt:
        Integration time step in milliseconds.
    noise:
        Std-dev of background current injected into every neuron each step.
        A little noise keeps the network spontaneously active, like real tissue.
    """

    def __init__(self, seed: int = 0, dt: float = 1.0, noise: float = 1.5):
        self.rng = np.random.default_rng(seed)
        self.dt = float(dt)
        self.noise = float(noise)
        # Constant baseline current injected into every neuron each step -- the
        # "always-on electricity" that keeps the brain spontaneously active even
        # with no external input (its resting / default-mode activity). The
        # runtime loop tunes this; 0 keeps the classic present()-style behaviour.
        self.tonic = 0.0

        self.regions: Dict[str, Region] = {}
        self.projections: List[SynapseBundle] = []
        self._region_order: List[str] = []

        self.t = 0
        self.n_neurons = 0
        self._finalized = False

        # When False, synapses (STDP) do not learn. Set off during
        # recall/inference so testing doesn't rewrite what was taught.
        self.plastic = True
        # Emergent transition learning in the world model. Off by default: the
        # teacher gives temporal lessons explicitly (see Teacher.teach_sequence)
        # so predictions stay clean. Turn on to let the brain discover
        # regularities on its own from whatever it observes.
        self.learn_transitions = False

        # Top-down modulation hooks (used by attention + imagination):
        #   region_gain[name] multiplies a region's driving current (attend to
        #     a modality by turning its gain up and others down);
        #   top_down[name] is an additive per-neuron current (a concept the
        #     brain is imagining or attending to is gently pushed from the top).
        self.region_gain: Dict[str, float] = {}
        self.top_down: Dict[str, np.ndarray] = {}

        # external stimulation queued per region: name -> deque of (n,) currents
        self._pending: Dict[str, Deque[np.ndarray]] = defaultdict(deque)

        # recording
        self.history: List[SpikeFrame] = []
        self.recording = False

        self.world: Optional[WorldModel] = None

    # -- construction -----------------------------------------------------
    def add_region(self, region: Region) -> Region:
        """Add a region and assign it a slice of the global neuron id space."""
        if region.name in self.regions:
            raise ValueError(f"Region {region.name!r} already exists.")
        region.gid_start = self.n_neurons
        self.n_neurons += region.size
        self.regions[region.name] = region
        self._region_order.append(region.name)
        self._finalized = False
        return region

    def region(self, name: str) -> Region:
        return self.regions[name]

    def connect(
        self,
        src: str,
        dst: str,
        p: float = 0.1,
        w_scale: float = 1.0,
        delay_range: Tuple[int, int] = (1, 5),
        stdp: Optional[STDPConfig] = None,
        plastic: bool = True,
        n_edges: Optional[int] = None,
    ) -> SynapseBundle:
        """Create a random projection from region ``src`` to region ``dst``.

        Connectivity is set either by probability ``p`` or, at scale, by an
        explicit ``n_edges`` synapse count.
        """
        s, d = self.regions[src], self.regions[dst]
        cfg = stdp or STDPConfig()
        cfg.enabled = plastic and cfg.enabled
        bundle = SynapseBundle.random(
            s.population, d.population, p=p, w_scale=w_scale,
            delay_range=delay_range, rng=self.rng, stdp=cfg, n_edges=n_edges,
        )
        bundle.src_name, bundle.dst_name = src, dst  # tag for the visualizer
        self.projections.append(bundle)
        return bundle

    def finalize(self, activation_threshold: float = 0.2) -> "Brain":
        """Lock the layout and create the world model. Call before running."""
        default_layout(list(self.regions.values()))
        self.world = WorldModel(self.n_neurons, activation_threshold)
        self._finalized = True
        return self

    # -- global views -----------------------------------------------------
    def global_activation(self) -> np.ndarray:
        """Concatenate every region's activation into one (n_neurons,) vector."""
        out = np.empty(self.n_neurons, dtype=np.float32)
        for name in self._region_order:
            r = self.regions[name]
            out[r.gid_start:r.gid_start + r.size] = r.population.activation
        return out

    def global_spikes(self) -> np.ndarray:
        """Global ids of every neuron that fired on the last step."""
        ids = []
        for name in self._region_order:
            r = self.regions[name]
            local = np.nonzero(r.population.spiked)[0]
            if len(local):
                ids.append(local + r.gid_start)
        return np.concatenate(ids) if ids else np.array([], dtype=np.int64)

    def gid_range(self, region_name: str) -> Tuple[int, int]:
        r = self.regions[region_name]
        return r.gid_start, r.gid_start + r.size

    def gids(self, region_name: str, local_indices: Sequence[int]) -> np.ndarray:
        """Convert local indices in a region to global neuron ids."""
        r = self.regions[region_name]
        return np.asarray(local_indices, dtype=np.int64) + r.gid_start

    # -- stimulation ------------------------------------------------------
    def stimulate(self, region_name: str, currents: np.ndarray) -> None:
        """Queue external current for a sensory region.

        ``currents`` is either a single (n,) vector or a (T, n) sequence; it is
        appended to that region's input queue and consumed one row per step.
        """
        r = self.regions[region_name]
        arr = np.asarray(currents, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[1] != r.size:
            raise ValueError(
                f"Stimulus width {arr.shape[1]} != region size {r.size}."
            )
        for row in arr:
            self._pending[region_name].append(row.copy())

    def clear_stimuli(self) -> None:
        self._pending.clear()

    def _external_current(self) -> Dict[str, np.ndarray]:
        """Pop one step of queued stimulation for each region."""
        ext: Dict[str, np.ndarray] = {}
        for name, queue in self._pending.items():
            if queue:
                ext[name] = queue.popleft()
        return ext

    # -- the simulation loop ---------------------------------------------
    def step(self) -> SpikeFrame:
        """Advance the whole brain by one time step and return the frame."""
        if not self._finalized:
            self.finalize()

        ext = self._external_current()

        # 1. Build the *driving* current (noise + external + manual) per region.
        currents: Dict[str, np.ndarray] = {}
        tonic = np.float32(self.tonic)
        for name, r in self.regions.items():
            if self.noise:
                I = self.rng.normal(0.0, self.noise, r.size).astype(np.float32)
            else:
                I = np.zeros(r.size, dtype=np.float32)
            if name in ext:
                I = I + ext[name]
            manual = getattr(r.population, "_manual_I", None)
            if manual is not None:
                I = I + manual
                r.population._manual_I = np.zeros(r.size, dtype=np.float32)
            currents[name] = I

        # 2. Synaptic currents (delays handled inside each bundle).
        for bundle in self.projections:
            syn = bundle.collect_current(self.dt)
            currents[bundle.dst_name] += syn

        # 3. Attention (gain on the driving signal), then the constant tonic
        #    baseline, then top-down (attention/imagination) injection, then step.
        for name, r in self.regions.items():
            I = currents[name]
            gain = self.region_gain.get(name, 1.0)
            if gain != 1.0:
                I *= np.float32(gain)
            if tonic:
                I += tonic
            td = self.top_down.get(name)
            if td is not None:
                I += td
            r.population.step(I, self.dt)

        # 4. Plasticity (skipped when the brain is frozen for inference).
        if self.plastic:
            for bundle in self.projections:
                bundle.apply_stdp()

        # 5. World model interprets the new state.
        activation = self.global_activation()
        active_concepts: List[str] = []
        if self.world is not None:
            active_concepts = self.world.observe(
                activation, learn=self.plastic and self.learn_transitions
            )

        # 6. Record.
        frame = SpikeFrame(
            t=self.t,
            spikes=self.global_spikes(),
            activation=activation,
            active_concepts=active_concepts,
        )
        if self.recording:
            self.history.append(frame)
        self.t += 1
        return frame

    def run(self, steps: int, record: bool = True) -> List[SpikeFrame]:
        """Run ``steps`` time steps, returning the recorded frames."""
        prev = self.recording
        self.recording = record
        frames = [self.step() for _ in range(steps)]
        self.recording = prev
        return frames

    # -- high level: "show it something" ---------------------------------
    def present(
        self,
        region_name: str,
        stimulus: np.ndarray,
        settle: int = 30,
        record: bool = True,
    ) -> List[SpikeFrame]:
        """Present an encoded stimulus and let activity propagate.

        Queues ``stimulus`` into ``region_name`` and runs for
        ``len(stimulus) + settle`` steps so the signal can flow all the way
        from the sensory region to the motor/output regions.
        """
        stimulus = np.atleast_2d(stimulus)
        self.stimulate(region_name, stimulus)
        return self.run(len(stimulus) + settle, record=record)

    # -- structural plasticity: a growing brain --------------------------
    def grow_region(self, name: str, k: int,
                    kind: Optional[str] = None, wire: bool = True) -> int:
        """Grow ``k`` new neurons in a region (neurogenesis) and wire them in.

        New cells are appended at the end of the region, so existing neurons,
        synapses and concepts stay valid; the global id space and the world
        model's concept ids are then re-based. ``wire`` sprouts fresh synapses
        onto/from the newborns on every projection that touches this region, so
        the new neurons can actually be driven and can drive others.
        """
        if k <= 0:
            return 0
        region = self.regions[name]
        old_n = region.size
        region.population.grow(k, kind or region.excitatory_type)
        region.size += k

        boundary = region.gid_start + old_n     # old-space shift boundary
        self.n_neurons = 0
        for rn in self._region_order:
            r = self.regions[rn]
            r.gid_start = self.n_neurons
            self.n_neurons += r.size

        if self.world is not None:
            for c in self.world.concepts.values():
                if len(c.gids) and int(c.gids.min()) >= boundary:
                    c.gids = c.gids + k
            self.world.n_neurons = self.n_neurons

        for bundle in self.projections:
            bundle.resize_traces()
        if wire:
            self._wire_newborns(name, old_n, k)
        return k

    def _wire_newborns(self, name: str, old_n: int, k: int) -> None:
        """Sprout synapses onto and from the newly-grown neurons."""
        new_local = np.arange(old_n, old_n + k)
        for b in self.projections:
            med = float(np.median(b.weight)) if len(b.weight) else 1.0
            if b.dst_name == name:                      # give newborns inputs
                d_in = int(np.clip(len(b) // max(1, old_n), 1, 40))
                m = d_in * k
                pre = self.rng.integers(0, b.source.n, m, dtype=np.int32)
                post = np.repeat(new_local, d_in).astype(np.int32)
                b.add_synapses(pre, post, np.full(m, med, np.float32),
                               self.rng.integers(1, 4, m))
            if b.src_name == name:                      # give newborns outputs
                d_out = int(np.clip(len(b) // max(1, old_n), 1, 40))
                m = d_out * k
                pre = np.repeat(new_local, d_out).astype(np.int32)
                post = self.rng.integers(0, b.target.n, m, dtype=np.int32)
                b.add_synapses(pre, post, np.full(m, med, np.float32),
                               self.rng.integers(1, 4, m))

    def synaptogenesis(self, rate: float = 0.5, min_activation: float = 0.1,
                       max_new: int = 4000) -> int:
        """Grow new synapses between currently co-active neurons (Hebbian).

        Structural version of "cells that fire together wire together": on each
        plastic projection, neurons that are active right now sprout a few new
        connections. Call it periodically while the brain is being taught.
        """
        total = 0
        for b in self.projections:
            if not b.stdp.enabled:
                continue
            pre_pool = np.nonzero(b.source.activation > min_activation)[0]
            post_pool = np.nonzero(b.target.activation > min_activation)[0]
            if len(pre_pool) == 0 or len(post_pool) == 0:
                continue
            m = int(min(max_new, rate * min(len(pre_pool), len(post_pool))))
            if m <= 0:
                continue
            pre = self.rng.choice(pre_pool, m).astype(np.int32)
            post = self.rng.choice(post_pool, m).astype(np.int32)
            w = np.full(m, float(np.median(b.weight)) if len(b.weight) else 1.0,
                        dtype=np.float32)
            total += b.add_synapses(pre, post, w, self.rng.integers(1, 4, m))
        return total

    def prune_synapses(self, threshold: float = 0.05) -> int:
        """Remove weak synapses across the whole brain. Returns count removed."""
        return sum(b.prune(threshold) for b in self.projections)

    def structure_stats(self) -> Dict[str, object]:
        """Current neuron/synapse counts (watch these grow while teaching)."""
        return {
            "neurons": self.n_neurons,
            "synapses": int(sum(len(b) for b in self.projections)),
            "regions": {n: r.size for n, r in self.regions.items()},
        }

    # -- utilities --------------------------------------------------------
    def reset_state(self, keep_weights: bool = True) -> None:
        """Reset neuron dynamics (and optionally the recorded history)."""
        for r in self.regions.values():
            r.population.reset()
        self.clear_stimuli()
        self.history.clear()
        self.t = 0
        if self.world is not None:
            self.world.reset_history()

    def population_of(self, region_name: str) -> Population:
        return self.regions[region_name].population

    def describe(self) -> str:  # pragma: no cover - cosmetic
        lines = [f"Brain: {len(self.regions)} regions, "
                 f"{self.n_neurons} neurons, "
                 f"{sum(len(p) for p in self.projections)} synapses"]
        for name in self._region_order:
            r = self.regions[name]
            lines.append(f"  [{r.role:11}] {name:14} {r.size:4d} neurons "
                         f"@ {r.position}")
        return "\n".join(lines)
