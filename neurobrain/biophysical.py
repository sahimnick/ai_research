"""
biophysical.py
==============

A **higher-fidelity, continuous-time, biophysical** circuit -- the honest
answer to three ways the fast engine was *not* like a real brain:

  * **biochemistry / real ion channels** -- Hodgkin-Huxley neurons with actual
    voltage-gated Na+ and K+ conductances (gating variables m, h, n), so the
    action potential is *generated* by ion flow, not declared by ``v >= 30``.
  * **neurotransmitters** -- conductance-based synapses with the real receptor
    families and their reversal potentials and kinetics: **AMPA** (fast
    glutamate), **NMDA** (slow, voltage-gated by the Mg2+ block -- a genuine
    coincidence detector), **GABA-A** (fast inhibition), **GABA-B** (slow).
  * **dendritic computation** -- neurons are not points: a dendritic compartment
    integrates its inputs *nonlinearly* (NMDA), so clustered inputs sum
    supralinearly (a dendritic spike), exactly as in real pyramidal cells.
  * **glia** -- an astrocyte per neuron that takes up glutamate (the tripartite
    synapse) and slowly regulates excitability (glial homeostasis).
  * **neuromodulation** -- a dopamine signal gates plasticity (a three-factor
    rule: eligibility x dopamine), how the brain actually does reward learning.

and, underlying all of it:

  * **continuous time** -- everything is a differential equation integrated with
    a small time step (~0.025 ms), not a fixed 1 ms tick.

Honest scope: this is faithful but *expensive* (ODEs per compartment per
receptor). It is a small-circuit, high-fidelity model to demonstrate the real
mechanisms -- not the engine that runs ten million neurons. Both live in the
same package on purpose: the fast engine for scale, this for fidelity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Hodgkin-Huxley gating kinetics (standard squid-axon rate functions, in 1/ms)
# ---------------------------------------------------------------------------
def _safe(expr_num: np.ndarray, denom: np.ndarray) -> np.ndarray:
    # avoids 0/0 at the singular points of the HH alpha/beta functions
    out = np.where(np.abs(denom) < 1e-7, 1.0, denom)
    return expr_num / out


def alpha_n(v):  return _safe(0.01 * (v + 55.0), 1.0 - np.exp(-(v + 55.0) / 10.0))
def beta_n(v):   return 0.125 * np.exp(-(v + 65.0) / 80.0)
def alpha_m(v):  return _safe(0.1 * (v + 40.0), 1.0 - np.exp(-(v + 40.0) / 10.0))
def beta_m(v):   return 4.0 * np.exp(-(v + 65.0) / 18.0)
def alpha_h(v):  return 0.07 * np.exp(-(v + 65.0) / 20.0)
def beta_h(v):   return 1.0 / (1.0 + np.exp(-(v + 35.0) / 10.0))


# receptor kinetics: (decay tau in ms, reversal potential in mV)
RECEPTORS = {
    "AMPA":  (2.0,   0.0),     # fast excitatory (glutamate)
    "NMDA":  (100.0, 0.0),     # slow excitatory, voltage-gated (glutamate)
    "GABAA": (6.0,  -70.0),    # fast inhibitory
    "GABAB": (150.0, -90.0),   # slow inhibitory
}


def nmda_mg_block(v: np.ndarray) -> np.ndarray:
    """Fraction of NMDA channels unblocked at voltage ``v`` (Jahr-Stevens, 1mM Mg).

    This is the coincidence detector: near rest the channel is Mg2+-blocked;
    only when the cell is *already depolarised* (post active) does presynaptic
    glutamate get through -- the molecular basis of Hebbian "fire together".
    """
    return 1.0 / (1.0 + 0.28 * np.exp(-0.062 * v))


@dataclass
class BiophysicalCircuit:
    """A small population of Hodgkin-Huxley neurons with real synapses & glia.

    Parameters
    ----------
    n:           number of neurons.
    dendrite:    give each neuron a nonlinear dendritic compartment.
    glia:        enable astrocyte glutamate uptake + slow homeostasis.
    dt:          integration step in ms (continuous-time).
    spike_model: "HH" for full Hodgkin-Huxley Na/K channels (most faithful but
                 stiff -> needs dt~0.025 and 6 exp/step), or "adex" for the
                 adaptive exponential integrate-and-fire spike generator. AdEx
                 is the one place we *drop* biological detail on purpose: it
                 keeps the real conductance synapses, dendrite, glia and
                 continuous time, but replaces the stiff, expensive channel
                 machinery with a smooth exponential spike -- ~10x faster at a
                 larger dt, with the rest of the biology intact.
    """

    n: int
    dendrite: bool = True
    glia: bool = True
    dt: float = 0.025
    spike_model: str = "HH"

    # -- speed knobs: drop biology that does NOT shape growth/learning/understanding
    use_gabab: bool = True     # GABA-B: slow inhibition, negligible for learning
    glia_every: int = 1        # update the slow astrocyte only every N steps
    growth: bool = False       # enable structural synaptic growth + pruning

    # AdEx parameters (used only when spike_model == "adex")
    VT: float = -52.0
    deltaT: float = 2.0
    Vpeak: float = 20.0
    Vreset: float = -60.0
    a_w: float = 0.1
    b_w: float = 1.2
    tau_w: float = 120.0

    # membrane / channel constants (uF/cm^2, mS/cm^2, mV)
    C: float = 1.0
    gNa: float = 120.0
    gK: float = 36.0
    gL: float = 0.3
    ENa: float = 50.0
    EK: float = -77.0
    EL: float = -54.4

    @classmethod
    def fast(cls, n: int, **kw) -> "BiophysicalCircuit":
        """A deliberately-cheaper-but-still-biological preset.

        Keeps the biology that *builds* perception and memory -- conductance
        synapses, the NMDA coincidence detector, the nonlinear dendrite, dopamine
        gating, continuous time, and structural synaptic growth -- and drops only
        what does not shape learning: the stiff Hodgkin-Huxley channel machinery
        (-> smooth AdEx spike at a larger dt), slow GABA-B inhibition, and the
        per-step astrocyte update (glia are slow, so they are integrated every
        few steps). Net effect: much faster, same learning behaviour.
        """
        params = dict(spike_model="adex", dt=0.1, use_gabab=False,
                      glia_every=8, growth=True)
        params.update(kw)
        return cls(n, **params)

    def __post_init__(self):
        n = self.n
        self.adex = (self.spike_model.lower() == "adex")
        # the receptors actually simulated (GABA-B is optional -- slow, and
        # irrelevant to the fast learning/understanding demos)
        self._receptors = {r: kv for r, kv in RECEPTORS.items()
                           if self.use_gabab or r != "GABAB"}
        self._step_i = 0
        self._uptake = np.ones(n)
        # soma state
        self.v = np.full(n, -65.0)
        if self.adex:
            self.w = np.zeros(n)                 # adaptation current
        else:
            self.m = alpha_m(self.v) / (alpha_m(self.v) + beta_m(self.v))
            self.h = alpha_h(self.v) / (alpha_h(self.v) + beta_h(self.v))
            self.nch = alpha_n(self.v) / (alpha_n(self.v) + beta_n(self.v))
        self.spiked = np.zeros(n, dtype=bool)
        self._above = np.zeros(n, dtype=bool)   # for edge-detecting spikes

        # synaptic conductances per receptor, at soma and (optionally) dendrite
        self.g = {r: np.zeros(n) for r in RECEPTORS}
        self.gd = {r: np.zeros(n) for r in RECEPTORS} if self.dendrite else None

        # dendritic compartment (passive + NMDA), coupled to soma
        if self.dendrite:
            self.vd = np.full(n, -65.0)
            self.gc = 0.5          # axial coupling soma<->dendrite
            self.gLd = 0.1
        # astrocyte: slow glutamate pool + homeostatic gain
        if self.glia:
            self.astro = np.zeros(n)       # local glutamate seen by astrocyte
            self.glia_gain = np.ones(n)    # slow homeostatic multiplier
        # neuromodulation
        self.dopamine = 0.0                # global reward signal (0..1)
        self.time = 0.0

        # -- structural plasticity: a growing recurrent synapse set
        # spike traces (pre-before-post -> LTP eligibility) and co-activity
        self.pre_trace = np.zeros(n)
        self.slow_trace = np.zeros(n)      # longer trace -> detects sequences
        self.tau_trace = 20.0              # ms  (fast, for coincidence)
        self.tau_slow = 60.0               # ms  (slow, for A-then-D chains)
        self.syn_pre = np.zeros(0, dtype=int)
        self.syn_post = np.zeros(0, dtype=int)
        self.syn_w = np.zeros(0, dtype=np.float32)
        self.syn_rec = np.zeros(0, dtype=int)     # 0=AMPA, 1=NMDA
        self.syn_elig = np.zeros(0, dtype=np.float32)
        self.syn_is_shortcut = np.zeros(0, dtype=bool)
        # SPARSE co-activity / sequence stores (scale to many neurons: never n*n)
        self._coact: Dict[Tuple[int, int], float] = {}   # simultaneous pairs
        self._seq: Dict[Tuple[int, int], float] = {}     # source-then-dest counts
        self._seqlag: Dict[Tuple[int, int], float] = {}  # summed source->dest lag
        self._dfire = np.zeros(n, dtype=np.float32)       # how often each fired
        self._last_spike = np.full(n, -10 ** 9, dtype=np.int64)  # for lag/order
        self.coact_every = 2               # accumulate every few steps (sparse)
        self.coact_cap = 400_000           # bound the sparse-store memory
        self.seq_lag = 80                  # a source may precede a dest by <= this

    # -- delivering neurotransmitter -------------------------------------
    def release(self, receptor: str, idx, weight=1.0, dendritic=False) -> None:
        """A presynaptic terminal releases transmitter onto neurons ``idx``.

        Increments the post-synaptic conductance of the given receptor type.
        """
        g = self.gd[receptor] if (dendritic and self.dendrite) else self.g[receptor]
        g[np.asarray(idx)] += weight

    # -- one continuous-time step ----------------------------------------
    def step(self, I_ext: Optional[np.ndarray] = None) -> np.ndarray:
        dt = self.dt
        v = self.v
        if I_ext is None:
            I_ext = np.zeros(self.n)

        # --- deliver last step's spikes along the grown synapses (axonal delay)
        if self.syn_pre.size:
            self._deliver()

        # --- glia: astrocytes take up glutamate & set a slow homeostatic gain.
        # The astrocyte is slow, so we integrate it only every ``glia_every``
        # steps (with a proportionally larger dt) -- a big saving, no biology lost
        exc_glut = self.g["AMPA"] + self.g["NMDA"]
        if self.glia:
            if self._step_i % self.glia_every == 0:
                gdt = dt * self.glia_every
                self.astro += gdt * (0.05 * exc_glut - 0.002 * self.astro)
                target = 0.4       # chronically active -> astrocyte damps gain
                self.glia_gain += gdt * 0.0005 * (target - self.astro)
                self.glia_gain = np.clip(self.glia_gain, 0.3, 1.5)
                self._uptake = 1.0 / (1.0 + 0.3 * self.astro)
            uptake = self._uptake
        else:
            uptake, self.glia_gain = 1.0, np.ones(self.n)

        # --- synaptic currents at the soma (conductance-based)
        I_syn = np.zeros(self.n)
        for r, (tau, Erev) in self._receptors.items():
            g = self.g[r]
            gate = nmda_mg_block(v) if r == "NMDA" else 1.0
            scale = uptake if r in ("AMPA", "NMDA") else 1.0
            I_syn += -g * gate * scale * self.glia_gain * (v - Erev)
            self.g[r] = g - dt * g / tau                 # decay (continuous)

        # --- dendritic compartment: nonlinear (NMDA) integration
        I_dend = 0.0
        if self.dendrite:
            vd = self.vd
            Id = -self.gLd * (vd - self.EL)
            for r, (tau, Erev) in self._receptors.items():
                gd = self.gd[r]
                gate = nmda_mg_block(vd) if r == "NMDA" else 1.0
                Id += -gd * gate * (vd - Erev)
                self.gd[r] = gd - dt * gd / tau
            Id += -self.gc * (vd - v)                    # coupling to soma
            self.vd = vd + dt * Id / self.C
            I_dend = self.gc * (self.vd - v)             # current into soma

        if not self.adex:
            # --- Hodgkin-Huxley soma: spike is generated by Na+/K+ channels
            I_Na = self.gNa * self.m ** 3 * self.h * (v - self.ENa)
            I_K = self.gK * self.nch ** 4 * (v - self.EK)
            I_Lk = self.gL * (v - self.EL)
            dv = (-I_Na - I_K - I_Lk + I_syn + I_dend + I_ext) / self.C
            self.v = v + dt * dv
            self.m += dt * (alpha_m(v) * (1 - self.m) - beta_m(v) * self.m)
            self.h += dt * (alpha_h(v) * (1 - self.h) - beta_h(v) * self.h)
            self.nch += dt * (alpha_n(v) * (1 - self.nch) - beta_n(v) * self.nch)
            above = self.v > 0.0
            self.spiked = above & ~self._above
            self._above = above
        else:
            # --- AdEx soma: smooth exponential spike + adaptation (not stiff)
            exp_arg = np.clip((v - self.VT) / self.deltaT, -30.0, 20.0)
            I_exp = self.gL * self.deltaT * np.exp(exp_arg)
            dv = (-self.gL * (v - self.EL) + I_exp - self.w
                  + I_syn + I_dend + I_ext) / self.C
            self.v = v + dt * dv
            self.w += dt * (self.a_w * (v - self.EL) - self.w) / self.tau_w
            self.spiked = self.v >= self.Vpeak
            self.v[self.spiked] = self.Vreset       # reset after the spike
            self.w[self.spiked] += self.b_w         # spike-triggered adaptation

        # --- plasticity bookkeeping: spike traces, STDP eligibility, co-activity
        self.pre_trace *= np.exp(-dt / self.tau_trace)
        self.slow_trace *= np.exp(-dt / self.tau_slow)
        self.pre_trace[self.spiked] += 1.0
        self.slow_trace[self.spiked] += 1.0
        if self.syn_pre.size:
            self._update_eligibility()
        if self.growth and self.spiked.any():
            if self._step_i % self.coact_every == 0:
                self._accumulate_pairs()      # increments _dfire, _coact, _seq
            self._last_spike[self.spiked] = self._step_i

        self._step_i += 1
        self.time += dt
        return self.spiked

    # -- neuromodulation --------------------------------------------------
    def set_reward(self, dopamine: float) -> None:
        """Set the dopamine level (gates three-factor plasticity)."""
        self.dopamine = float(dopamine)

    def three_factor_dw(self, eligibility: np.ndarray,
                        lr: float = 0.1) -> np.ndarray:
        """Weight change = lr * dopamine * eligibility (reward-modulated STDP).

        Plasticity is *gated* by the neuromodulator: the same coincidence leaves
        only a fleeting eligibility trace, and becomes a lasting weight change
        only if dopamine says it mattered.
        """
        return lr * self.dopamine * eligibility

    def reset_state(self) -> None:
        """Clear dynamic state (voltage, conductances, adaptation, traces) but
        KEEP the learned structure -- the grown synapses and their weights. Use
        it to test recall from a fresh start after a training bout."""
        self.v[:] = -65.0
        if self.adex:
            self.w[:] = 0.0
        for r in self._receptors:
            self.g[r][:] = 0.0
            if self.dendrite:
                self.gd[r][:] = 0.0
        if self.dendrite:
            self.vd[:] = -65.0
        if self.glia:
            self.astro[:] = 0.0
            self.glia_gain[:] = 1.0
        self.pre_trace[:] = 0.0
        self.slow_trace[:] = 0.0
        self.spiked[:] = False
        if not self.adex:
            self._above[:] = False

    # -- structural plasticity: a synapse set that grows and prunes ---------
    @property
    def n_synapses(self) -> int:
        return int(self.syn_pre.size)

    def add_synapses(self, pre, post, w: float = 0.4, receptor: str = "AMPA",
                     shortcut: bool = False) -> None:
        """Create synapses pre->post (skips self-loops and duplicates)."""
        pre = np.atleast_1d(np.asarray(pre, dtype=int))
        post = np.atleast_1d(np.asarray(post, dtype=int))
        keep = pre != post
        pre, post = pre[keep], post[keep]
        if pre.size == 0:
            return
        existing = set(zip(self.syn_pre.tolist(), self.syn_post.tolist()))
        new = [(a, b) for a, b in zip(pre.tolist(), post.tolist())
               if (a, b) not in existing]
        if not new:
            return
        na = np.array(new)
        rec = 1 if receptor == "NMDA" else 0
        self.syn_pre = np.concatenate([self.syn_pre, na[:, 0]])
        self.syn_post = np.concatenate([self.syn_post, na[:, 1]])
        self.syn_w = np.concatenate(
            [self.syn_w, np.full(len(new), w, np.float32)])
        self.syn_rec = np.concatenate(
            [self.syn_rec, np.full(len(new), rec, int)])
        self.syn_elig = np.concatenate(
            [self.syn_elig, np.zeros(len(new), np.float32)])
        self.syn_is_shortcut = np.concatenate(
            [self.syn_is_shortcut, np.full(len(new), shortcut, bool)])

    def _deliver(self) -> None:
        """Neurons that spiked last step release transmitter onto their targets."""
        active = self.spiked[self.syn_pre]
        if not active.any():
            return
        post = self.syn_post[active]
        w = self.syn_w[active]
        rec = self.syn_rec[active]
        for r_id, name in ((0, "AMPA"), (1, "NMDA")):
            sel = rec == r_id
            if sel.any():
                np.add.at(self.g[name], post[sel], w[sel])

    def _update_eligibility(self) -> None:
        """Pre-before-post coincidence leaves an eligibility trace (STDP)."""
        post_fired = self.spiked[self.syn_post]
        if post_fired.any():
            self.syn_elig[post_fired] += self.pre_trace[self.syn_pre[post_fired]]
        self.syn_elig *= np.exp(-self.dt / self.tau_trace)

    def consolidate(self, lr: float = 0.02) -> None:
        """Turn eligibility into lasting weight change, gated by dopamine.

        The three-factor rule: coincidence (eligibility) x reward (dopamine).
        Without dopamine the traces simply fade; with it, co-active synapses
        strengthen -- how a real synapse writes a memory.
        """
        self.syn_w += lr * self.dopamine * self.syn_elig
        np.clip(self.syn_w, 0.0, 4.0, out=self.syn_w)

    def _accumulate_pairs(self) -> None:
        """Sparse update of co-activity (who fires *together*) and sequence
        evidence (who fired a few steps *before* the neurons firing now).

        Ordering is read from each neuron's last-spike time, so it is robust to
        trace time-constants: a neuron is a *partner* if it is active now, and a
        *source* for D if its last spike was 2..seq_lag steps ago. Only the small
        active set is touched, so this scales to many neurons -- never n x n.
        """
        fired = np.flatnonzero(self.spiked)
        if fired.size == 0:
            return
        self._dfire[fired] += 1.0
        lag = self._step_i - self._last_spike
        partners = np.flatnonzero(self.spiked | (lag <= 1))          # simultaneous
        sources = np.flatnonzero((~self.spiked) & (lag >= 2) &
                                 (lag <= self.seq_lag))              # preceded D
        for d in fired.tolist():
            for p in partners.tolist():
                if p != d:
                    self._coact[(p, d)] = self._coact.get((p, d), 0.0) + 1.0
            for s in sources.tolist():
                self._seq[(s, d)] = self._seq.get((s, d), 0.0) + 1.0
                self._seqlag[(s, d)] = self._seqlag.get((s, d), 0.0) + int(lag[s])
        if len(self._coact) > self.coact_cap:
            self._coact = dict(sorted(self._coact.items(),
                                      key=lambda kv: -kv[1])[:self.coact_cap])
        if len(self._seq) > self.coact_cap:
            self._seq = dict(sorted(self._seq.items(),
                                    key=lambda kv: -kv[1])[:self.coact_cap])

    def grow_synapses(self, max_add: int = 40, w0: float = 0.3,
                      frac: float = 0.5) -> int:
        """Synaptogenesis: wire up neuron pairs that keep firing together.

        Reads the sparse co-activity store and adds synapses between the most
        co-active, not-yet-connected pairs -- activity-dependent structural
        growth, the way experience physically grows connections. Returns how
        many synapses were added.
        """
        if not self.growth or not self._coact:
            return 0
        items = sorted(self._coact.items(), key=lambda kv: -kv[1])
        top = items[0][1]
        existing = set(zip(self.syn_pre.tolist(), self.syn_post.tolist()))
        add_pre, add_post = [], []
        for (a, b), val in items:
            if val < frac * top:
                break
            if (a, b) not in existing:
                add_pre.append(a); add_post.append(b); existing.add((a, b))
                if len(add_pre) >= max_add:
                    break
        if add_pre:
            self.add_synapses(add_pre, add_post, w=w0, receptor="AMPA")
        self._coact = {k: v * 0.5 for k, v in items}        # fade the memory
        return len(add_pre)

    def grow_shortcuts(self, max_add: int = 20, reliability: float = 0.5,
                       w0: float = 0.6) -> int:
        """Grow a *shortcut* when a source reliably leads to a destination.

        If activity in neuron ``S`` is consistently followed (a few synapses
        later) by neuron ``D`` firing, a direct S->D synapse is added -- so a
        path the brain travels again and again through 10 or 100 or 1000
        synapses gets a fast express lane, the way a practised route becomes
        automatic (procedural chunking / systems consolidation). The whole
        original chain is left intact and the shortcut is only facilitatory, so
        function is unchanged -- it just arrives sooner. Returns shortcuts added.
        """
        if not self.growth or not self._seq:
            return 0
        existing = set(zip(self.syn_pre.tolist(), self.syn_post.tolist()))
        # among the *reliable* source->dest pairs, prefer the LONGEST-range ones
        # -- those skip the most synapses, which is the point of a shortcut
        cand = []
        for (s, d), val in self._seq.items():
            rel = val / (self._dfire[d] + 1.0)      # how reliably S precedes D
            if rel >= reliability and (s, d) not in existing:
                mean_lag = self._seqlag[(s, d)] / val
                cand.append((mean_lag, rel, s, d))
        cand.sort(reverse=True)                     # longest lag (deepest) first
        add_pre, add_post = [], []
        for _, _, s, d in cand[:max_add]:
            add_pre.append(s); add_post.append(d)
        if add_pre:
            self.add_synapses(add_pre, add_post, w=w0, receptor="AMPA",
                              shortcut=True)
        return len(add_pre)

    @property
    def n_shortcuts(self) -> int:
        return int(self.syn_is_shortcut.sum())

    def prune_synapses(self, thresh: float = 0.05) -> int:
        """Remove synapses that stayed weak -- use it or lose it. Returns count."""
        if self.syn_w.size == 0:
            return 0
        keep = (self.syn_w >= thresh) | self.syn_is_shortcut   # keep grown shortcuts
        removed = int((~keep).sum())
        self.syn_pre = self.syn_pre[keep]
        self.syn_post = self.syn_post[keep]
        self.syn_w = self.syn_w[keep]
        self.syn_rec = self.syn_rec[keep]
        self.syn_elig = self.syn_elig[keep]
        self.syn_is_shortcut = self.syn_is_shortcut[keep]
        return removed

    def live(self, steps: int, stim=None, dopamine: float = 0.0,
             consolidate_every: int = 50, grow_every: int = 2000,
             max_add: int = 200, prune_thresh: float = 0.05,
             shortcuts: bool = True) -> Dict[str, int]:
        """Run the circuit as a *living loop*: integrate continuously, and
        periodically consolidate, grow and prune structure -- the biophysical
        counterpart of the game-loop runtime, so the circuit keeps building
        itself while it runs. ``stim`` is an array or a callable ``t -> I_ext``.
        Returns final structure counts.
        """
        self.set_reward(dopamine)
        for t in range(steps):
            self.step(stim(t) if callable(stim) else stim)
            if consolidate_every and (t + 1) % consolidate_every == 0:
                self.consolidate()
            if grow_every and (t + 1) % grow_every == 0:
                self.grow_synapses(max_add=max_add)
                self.prune_synapses(prune_thresh)
                if shortcuts:
                    self.grow_shortcuts()
        return {"neurons": self.n, "synapses": self.n_synapses,
                "shortcuts": self.n_shortcuts}

    # -- helpers ----------------------------------------------------------
    def run(self, steps: int, I_ext=None, record: bool = True):
        """Integrate ``steps`` steps; return recorded soma voltage (n, steps)."""
        rec = np.zeros((self.n, steps)) if record else None
        for t in range(steps):
            self.step(I_ext)
            if record:
                rec[:, t] = self.v
        return rec
