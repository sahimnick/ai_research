"""
synapse.py
==========

Connections between neurons, and how they *learn*.

A real synapse is not a wire. It has a **weight** (how strongly a pre-synaptic
spike pushes the post-synaptic cell), a **delay** (axons take time), and it
**changes with experience**. The change rule implemented here is
Spike-Timing-Dependent Plasticity (STDP), the closest thing biology has to a
learning law:

    * pre fires shortly *before* post   -> strengthen  (it helped cause the spike)
    * pre fires shortly *after*  post   -> weaken      (it arrived too late)

This is Hebbian ("cells that fire together, wire together") and is *local* --
no global error signal, no backpropagation. That matches the user's request to
teach the brain by *induction / experience* rather than the usual supervised
gradient method.

Connectivity is stored as a sparse edge list in NumPy arrays so that a single
vectorised call delivers all spikes and applies all weight updates.

**Synaptic tagging and capture.** Pair-based STDP on its own has a well known
defect: it is *memoryless between events*. Every coincidence, however accidental,
moves the weight by the same amount, so a synapse driven by noise random-walks
just as far as one driven by a real, repeating regularity. Real synapses do not
work this way (Frey & Morris, 1997). A plasticity event leaves a **tag** behind
-- physically, a local calcium transient and the CaMKII / protein-synthesis
machinery it switches on -- which *decays over minutes*. Only if the tag is still
standing when enough plasticity-related product is around does the change get
**captured** into the late, protein-dependent phase and become permanent;
otherwise the early change simply washes out.

That is implemented here as :class:`TagConfig`, adding two arrays beside
``weight``:

    * ``calcium[k]`` -- a signed, exponentially-decaying deposit. Every STDP
      event adds its own ``dw`` to it. Consistent correlations *accumulate*
      (same sign, again and again, faster than the decay); accidental ones
      cancel, because random pre/post orderings deposit ``+`` and ``-`` in equal
      measure. The tag is therefore a low-pass filter on *correlation*, not on
      activity.
    * ``w_stable[k]`` -- the consolidated (late-phase) weight. When ``|calcium|``
      crosses the capture threshold, part of the tag is written into it.

The live ``weight`` then relaxes back toward ``w_stable``. So a one-off
coincidence produces a change that fades, while a regularity that keeps
recurring over a long window is written into the stable component and kept.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .neuron import Population


@dataclass
class STDPConfig:
    """Parameters of the spike-timing-dependent plasticity rule."""

    a_plus: float = 0.010      # potentiation learning rate (pre-before-post)
    a_minus: float = 0.012     # depression learning rate (post-before-pre)
    tau_plus: float = 20.0     # ms, potentiation time constant
    tau_minus: float = 20.0    # ms, depression time constant
    w_min: float = 0.0         # weights never go negative (sign set by cell type)
    w_max: float = 10.0        # saturating ceiling keeps the network stable
    enabled: bool = True
    # -- competitive learning ------------------------------------------------
    # When ``normalize`` is on, each post-synaptic neuron's total incoming
    # (excitatory) weight is rescaled back to ``target_in`` after every update.
    # This conserved-resource rule turns plain Hebbian potentiation into
    # *competitive* learning: strengthening one synapse necessarily weakens the
    # others onto the same cell, so each neuron becomes selective for one input
    # pattern instead of a few loud patterns capturing everything.
    normalize: bool = False
    target_in: float = 12.0
    # Renormalise only every N steps. Normalisation is a slow homeostatic
    # constraint, so doing it every few steps instead of every step barely
    # changes the result but is much cheaper on large grounding bundles.
    normalize_every: int = 1


@dataclass
class TagConfig:
    """Synaptic tagging and capture -- the late, protein-dependent phase.

    Classic pair-based STDP treats every coincidence alike, so noise random-walks
    a weight just as far as a real regularity does. Biology separates the two by
    time: an event leaves a **tag** (a calcium transient and the machinery it
    triggers) which decays over minutes, and only a tag that is still standing
    gets **captured** into a permanent change.

    Attributes
    ----------
    tau_calcium:
        Decay time of the calcium deposit, in ms. Long relative to the STDP
        windows (~20 ms) -- that gap is the whole point: it lets the tag average
        over many events instead of reacting to one.
    deposit:
        How much of each STDP weight change enters the tag.
    threshold:
        Capture threshold. Below it the deposit decays away unread and the
        early-phase change is simply forgotten.
    capture:
        Fraction of a standing tag written into the consolidated weight.
    tau_labile:
        Decay time of the *un*-consolidated part of the weight, i.e. how fast
        the live weight relaxes back toward ``w_stable``. Early-phase LTP that is
        never captured washes out on this timescale.
    """

    enabled: bool = False
    tau_calcium: float = 600.0
    deposit: float = 1.0
    threshold: float = 0.20
    capture: float = 0.05
    tau_labile: float = 300.0


@dataclass
class SynapseBundle:
    """All synapses from one :class:`Population` to another (possibly itself).

    Stored as parallel arrays:
        pre[k], post[k], weight[k], delay[k]  describe the k-th connection.

    A ring buffer (``_spike_queue``) implements axonal conduction delays: a
    spike emitted now is deposited into the target's current-buffer ``delay``
    steps in the future.
    """

    source: Population
    target: Population
    pre: np.ndarray            # int index into source
    post: np.ndarray           # int index into target
    weight: np.ndarray         # float, >= 0
    delay: np.ndarray          # int, in steps
    stdp: STDPConfig = field(default_factory=STDPConfig)
    tags: TagConfig = field(default_factory=TagConfig)

    # eligibility traces (exponentially-decaying record of recent spikes)
    _pre_trace: np.ndarray = field(init=False, repr=False)
    _post_trace: np.ndarray = field(init=False, repr=False)
    _spike_queue: np.ndarray = field(init=False, repr=False)
    _qpos: int = field(init=False, repr=False, default=0)
    _sign: np.ndarray = field(init=False, repr=False)
    # tagging state; allocated lazily so a 60M-edge bundle with tagging off
    # does not pay 480 MB for two arrays it never reads.
    calcium: Optional[np.ndarray] = field(init=False, repr=False, default=None)
    w_stable: Optional[np.ndarray] = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        # Compact dtypes so tens of millions of synapses stay in a few hundred
        # MB: int32 endpoints (ok up to 2.1 billion neurons), float32 weights.
        self.pre = np.asarray(self.pre, dtype=np.int32)
        self.post = np.asarray(self.post, dtype=np.int32)
        self.weight = np.asarray(self.weight, dtype=np.float32)
        self.delay = np.asarray(self.delay, dtype=np.int16)

        n_edges = len(self.pre)
        if not (len(self.post) == len(self.weight) == len(self.delay) == n_edges):
            raise ValueError("pre/post/weight/delay must be the same length.")

        self._pre_trace = np.zeros(self.source.n, dtype=np.float32)
        self._post_trace = np.zeros(self.target.n, dtype=np.float32)

        # Inhibitory pre-synaptic neurons deliver negative current.
        self._sign = np.where(self.source.inhibitory[self.pre],
                              np.float32(-1.0), np.float32(1.0))

        # Ring buffer for conduction delays.
        self._max_delay = int(self.delay.max()) + 1 if n_edges else 1
        self._spike_queue = np.zeros(
            (self._max_delay, self.target.n), dtype=np.float32
        )

    # -- runtime ----------------------------------------------------------
    def collect_current(self, dt: float = 1.0) -> np.ndarray:
        """Emit currents for spikes that arrive *now* and enqueue new ones.

        Returns the synaptic current to add to the target population this step.
        """
        src_spikes = self.source.spiked

        # 1. Enqueue currents caused by neurons that spiked this step.
        if src_spikes.any():
            firing_edges = src_spikes[self.pre]
            if firing_edges.any():
                idx = np.nonzero(firing_edges)[0]
                contrib = self.weight[idx] * self._sign[idx]
                slots = (self._qpos + self.delay[idx]) % self._max_delay
                # scatter-add into the future slots of the ring buffer
                np.add.at(self._spike_queue, (slots, self.post[idx]), contrib)

        # 2. Read out the currents scheduled to arrive now, then clear slot.
        current = self._spike_queue[self._qpos].copy()
        self._spike_queue[self._qpos] = 0.0
        self._qpos = (self._qpos + 1) % self._max_delay

        # 3. Update eligibility traces (used by STDP below).
        decay_pre = np.exp(-dt / self.stdp.tau_plus)
        decay_post = np.exp(-dt / self.stdp.tau_minus)
        self._pre_trace *= decay_pre
        self._post_trace *= decay_post
        self._pre_trace[src_spikes] += 1.0
        self._post_trace[self.target.spiked] += 1.0
        return current

    def apply_stdp(self) -> None:
        """Update weights from the current spike/trace state (call after step).

        Uses the standard pair-based online rule:
            * on a *post* spike: w += a_plus  * pre_trace   (pre came first)
            * on a *pre*  spike: w -= a_minus * post_trace  (post came first)
        Only excitatory synapses are plastic here (a common simplification).
        """
        if not self.stdp.enabled:
            return

        post_fired = self.target.spiked[self.post]
        pre_fired = self.source.spiked[self.pre]
        plastic = self._sign > 0  # only excitatory synapses learn

        # When tagging is on we need the *signed* change of this step, because
        # that sign is what the calcium deposit integrates. The fast path (no
        # tagging) never allocates it.
        tagging = self.tags.enabled
        dw = np.zeros(len(self.weight), np.float32) if tagging else None

        if post_fired.any():
            m = post_fired & plastic
            d = self.stdp.a_plus * self._pre_trace[self.pre[m]]
            self.weight[m] += d
            if tagging:
                dw[m] += d
        if pre_fired.any():
            m = pre_fired & plastic
            d = self.stdp.a_minus * self._post_trace[self.post[m]]
            self.weight[m] -= d
            if tagging:
                dw[m] -= d

        np.clip(self.weight, self.stdp.w_min, self.stdp.w_max, out=self.weight)

        if tagging:
            self.apply_tagging(dw)

        if self.stdp.normalize:
            self._norm_counter = getattr(self, "_norm_counter", 0) + 1
            if self._norm_counter >= self.stdp.normalize_every:
                self._norm_counter = 0
                self._normalize_incoming(plastic)

    # -- synaptic tagging and capture -------------------------------------
    def _ensure_tag_state(self) -> None:
        """Allocate the calcium / consolidated-weight arrays on first use."""
        n = len(self.weight)
        if self.calcium is None or len(self.calcium) != n:
            self.calcium = np.zeros(n, dtype=np.float32)
            self.w_stable = self.weight.copy()

    def apply_tagging(self, dw: np.ndarray) -> None:
        """One step of tagging and capture, given this step's weight changes.

        Three things happen, in the order biology does them:

        1. **Deposit and decay.** The tag integrates the signed change and leaks
           away with ``tau_calcium``. Because the deposit is *signed*, a synapse
           whose pre/post ordering is random contributes ``+`` and ``-`` in equal
           measure and its tag stays near zero however busy it is; a synapse with
           a consistent temporal relation to its target accumulates.
        2. **Capture.** Where the tag stands above threshold, part of it is
           written into ``w_stable`` -- the late, protein-dependent phase.
        3. **Wash-out.** The live weight relaxes toward ``w_stable``. A change
           that was never captured therefore disappears on ``tau_labile``, which
           is exactly what early-phase LTP does without protein synthesis.
        """
        self._ensure_tag_state()
        cfg = self.tags
        self.calcium *= np.float32(np.exp(-1.0 / max(cfg.tau_calcium, 1e-6)))
        self.calcium += np.float32(cfg.deposit) * dw

        cap = np.abs(self.calcium) > cfg.threshold
        if cap.any():
            self.w_stable[cap] += np.float32(cfg.capture) * self.calcium[cap]
            np.clip(self.w_stable, self.stdp.w_min, self.stdp.w_max,
                    out=self.w_stable)

        lam = np.float32(1.0 - np.exp(-1.0 / max(cfg.tau_labile, 1e-6)))
        self.weight += lam * (self.w_stable - self.weight)
        np.clip(self.weight, self.stdp.w_min, self.stdp.w_max, out=self.weight)

    @property
    def consolidated(self) -> np.ndarray:
        """The late-phase weight -- what survives when the labile part washes out.

        Falls back to the live weight when tagging was never switched on, so
        callers can read it unconditionally."""
        return self.weight if self.w_stable is None else self.w_stable

    def _ensure_plastic_cache(self) -> None:
        """Cache the indices/targets of plastic (excitatory) synapses.

        These do not change except when synapses are added or pruned, so
        recomputing them every step (via np.nonzero) was a needless cost --
        dominant when a grounding bundle has hundreds of thousands of edges.
        """
        if getattr(self, "_plastic_idx", None) is None:
            idx = np.nonzero(self._sign > 0)[0].astype(np.int64)
            self._plastic_idx = idx
            self._plastic_post = self.post[idx]

    def _normalize_incoming(self, plastic: np.ndarray) -> None:
        """Rescale each post neuron's incoming excitatory weights to target_in.

        Implements competitive (conserved-total) Hebbian learning: what one
        synapse gains, its siblings onto the same neuron lose.
        """
        self._ensure_plastic_cache()
        idx = self._plastic_idx
        if not len(idx):
            return
        post = self._plastic_post
        sums = np.zeros(self.target.n, dtype=np.float32)
        np.add.at(sums, post, self.weight[idx])
        scale = np.ones(self.target.n, dtype=np.float32)
        nz = sums > 1e-9
        scale[nz] = np.float32(self.stdp.target_in) / sums[nz]
        self.weight[idx] *= scale[post]

    # -- construction helpers --------------------------------------------
    @classmethod
    def random(
        cls,
        source: Population,
        target: Population,
        p: float = 0.1,
        w_scale: float = 1.0,
        delay_range: Tuple[int, int] = (1, 5),
        rng: Optional[np.random.Generator] = None,
        stdp: Optional[STDPConfig] = None,
        self_connections: bool = False,
        max_edges: int = 60_000_000,
        n_edges: Optional[int] = None,
    ) -> "SynapseBundle":
        """Create a random sparse projection.

        Connectivity is either the classic probability ``p`` (number of synapses
        ``p * source.n * target.n``) or, more convenient at scale, an explicit
        ``n_edges`` count. Rather than materialising the (impossible at scale)
        dense ``source.n x target.n`` mask, the edges are *sampled* directly:
        random pre and post indices. A hard ``max_edges`` cap protects memory.
        """
        rng = rng if rng is not None else np.random.default_rng()
        if n_edges is None:
            n_edges = int(p * float(source.n) * float(target.n))
        n_edges = int(min(max(n_edges, 0), max_edges))

        pre = rng.integers(0, source.n, n_edges, dtype=np.int32)
        post = rng.integers(0, target.n, n_edges, dtype=np.int32)
        if source is target and not self_connections and n_edges:
            # Resample the few self-loops onto a neighbour.
            loops = pre == post
            if loops.any():
                post[loops] = (post[loops] + 1) % target.n

        weight = np.abs(rng.normal(w_scale, 0.3 * w_scale, n_edges)
                        ).astype(np.float32)
        lo, hi = delay_range
        delay = rng.integers(lo, hi + 1, n_edges)
        return cls(source, target, pre, post, weight, delay,
                   stdp or STDPConfig())

    # -- structural plasticity (growth & pruning) ------------------------
    def add_synapses(self, pre, post, weight, delay) -> int:
        """Grow new synapses (synaptogenesis). Indices are local to the pops."""
        pre = np.asarray(pre, dtype=np.int32)
        post = np.asarray(post, dtype=np.int32)
        if len(pre) == 0:
            return 0
        weight = np.asarray(weight, dtype=np.float32)
        delay = np.asarray(delay, dtype=np.int16)
        self.pre = np.concatenate([self.pre, pre])
        self.post = np.concatenate([self.post, post])
        self.weight = np.concatenate([self.weight, weight])
        self.delay = np.concatenate([self.delay, delay])
        new_sign = np.where(self.source.inhibitory[pre],
                            np.float32(-1.0), np.float32(1.0))
        self._sign = np.concatenate([self._sign, new_sign])
        if self.calcium is not None:      # new synapses start untagged
            self.calcium = np.concatenate(
                [self.calcium, np.zeros(len(pre), np.float32)])
            self.w_stable = np.concatenate([self.w_stable, weight])
        # grow the delay ring buffer if a longer axon just appeared
        md = int(self.delay.max()) + 1
        if md > self._max_delay:
            self._spike_queue = np.zeros((md, self.target.n), dtype=np.float32)
            self._qpos = 0
            self._max_delay = md
        self._plastic_idx = None      # invalidate the plastic-index cache
        return len(pre)

    def prune(self, threshold: float = 0.05) -> int:
        """Remove weak excitatory synapses (use-it-or-lose-it). Returns count."""
        weak = (self._sign > 0) & (self.weight < threshold)
        n = int(weak.sum())
        if n:
            keep = ~weak
            self.pre = self.pre[keep]
            self.post = self.post[keep]
            self.weight = self.weight[keep]
            self.delay = self.delay[keep]
            self._sign = self._sign[keep]
            if self.calcium is not None:
                self.calcium = self.calcium[keep]
                self.w_stable = self.w_stable[keep]
            self._plastic_idx = None   # invalidate the plastic-index cache
        return n

    def resize_traces(self) -> None:
        """Re-sync trace/queue sizes after the source or target grew."""
        if len(self._pre_trace) < self.source.n:
            self._pre_trace = np.concatenate([self._pre_trace, np.zeros(
                self.source.n - len(self._pre_trace), dtype=np.float32)])
        if len(self._post_trace) < self.target.n:
            self._post_trace = np.concatenate([self._post_trace, np.zeros(
                self.target.n - len(self._post_trace), dtype=np.float32)])
        if self._spike_queue.shape[1] < self.target.n:
            pad = np.zeros((self._spike_queue.shape[0],
                           self.target.n - self._spike_queue.shape[1]),
                           dtype=np.float32)
            self._spike_queue = np.concatenate([self._spike_queue, pad], axis=1)

    def weight_matrix(self) -> np.ndarray:
        """Dense signed weight matrix (source.n x target.n) -- handy for plots.

        Only meaningful for small populations; guarded so it cannot try to
        allocate a matrix for a million-neuron region.
        """
        cells = self.source.n * self.target.n
        if cells > 5_000_000:
            raise MemoryError(
                f"weight_matrix() would allocate {cells:,} cells. "
                "Use the sparse pre/post/weight arrays directly at this scale."
            )
        w = np.zeros((self.source.n, self.target.n), dtype=np.float32)
        w[self.pre, self.post] = self.weight * self._sign
        return w

    def __len__(self) -> int:
        return len(self.pre)
