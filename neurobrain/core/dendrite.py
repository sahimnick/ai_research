"""
dendrite.py
===========

A **dendritic tree** -- the mechanism that lets one neuron carry *thousands* of
synapses, the way a real cortical pyramidal cell does.

Point-neuron models give each cell a handful of inputs. Real neurons are the
opposite: a single layer-5 pyramidal cell has a branching dendritic tree with
**~7,000 synapses**, organised into many small branches. Each branch is a
*nonlinear subunit*: when enough synapses on the **same branch** are active
together, the branch fires a local **NMDA / dendritic spike** and its output is
*supralinear* -- much bigger than the sum of the parts. Scatter the same inputs
across different branches and you get only a linear sum. So the tree is not just
"more wires": it is a two-layer computation (branches, then soma), exactly the
picture from Poirazi & Mel (2003) and Larkum's cortical work.

This module implements that faithfully **and** vectorised, so the synapse count
per neuron reaches the biological range on a tractable cortical column:

    tree = build_cortical_column(n_neurons=2000, branches=64, syn_per_branch=110)
    tree.synapses_per_neuron        # ~7040  -- near a real pyramidal cell
    tree.supralinearity_index()     # >1     -- clustered input spikes the branch

Honest scale note: biological density on the *whole* 30M-neuron brain would be
~2e11 synapses -- far too many to store densely. The dendritic tree gives real
biological fan-in **per column**; the full brain stays sparse. That is a memory
truth, not a modelling shortcut.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


@dataclass
class DendriticTree:
    """A population of ``n`` neurons, each with a tree of ``branches`` dendritic
    subunits carrying ``syn_per_branch`` synapses apiece.

    State arrays are ``(n, branches, syn_per_branch)``:
      * ``pre`` -- which presynaptic neuron each synapse listens to;
      * ``w``   -- its weight.

    Integration is two-layer: synapses sum **within** a branch, the branch adds a
    supralinear NMDA plateau if that local sum crosses threshold, then branches
    sum at the soma. Clustered (co-active) input on one branch is amplified;
    scattered input is not -- the real dendritic nonlinearity.
    """

    n: int
    branches: int
    syn_per_branch: int
    pre: np.ndarray
    w: np.ndarray
    theta: float = 2.2          # per-branch NMDA-spike threshold (local sum)
    plateau: float = 6.0        # size of the dendritic (NMDA) plateau
    gain: float = 3.0           # sharpness of the branch nonlinearity
    soma_threshold: float = 8.0

    # -- sizes ---------------------------------------------------------------
    @property
    def n_synapses(self) -> int:
        return int(self.n * self.branches * self.syn_per_branch)

    @property
    def synapses_per_neuron(self) -> int:
        return int(self.branches * self.syn_per_branch)

    # -- the two-layer dendritic computation ---------------------------------
    def branch_input(self, pre_activity: np.ndarray) -> np.ndarray:
        """Weighted synaptic sum on every branch: shape ``(n, branches)``."""
        gathered = pre_activity[self.pre] * self.w          # (n, B, S)
        return gathered.sum(axis=2)

    def branch_output(self, branch_in: np.ndarray) -> np.ndarray:
        """Per-branch nonlinearity: a linear leak **plus** a supralinear NMDA
        plateau once the branch's local input crosses ``theta``."""
        plateau = self.plateau * _sigmoid(self.gain * (branch_in - self.theta))
        return branch_in + plateau

    def soma_current(self, pre_activity: np.ndarray) -> np.ndarray:
        """Total dendritic current arriving at each soma: shape ``(n,)``."""
        return self.branch_output(self.branch_input(pre_activity)).sum(axis=1)

    def forward(self, pre_activity: np.ndarray) -> np.ndarray:
        """One step: presynaptic activity ``(n,)`` -> somatic spikes ``(n,)``."""
        return (self.soma_current(pre_activity) > self.soma_threshold)

    # -- the biological signature: supralinear dendritic integration ---------
    def supralinearity_index(self, drive: float = 1.0, k: int = 20) -> float:
        """Put ``k`` active inputs on ONE branch vs. spread across ``k`` branches
        and compare the somatic current. A real dendrite amplifies the clustered
        case (ratio > 1); a point neuron would give exactly 1.0."""
        clustered = np.zeros((self.branches, self.syn_per_branch), np.float32)
        clustered[0, :k] = drive
        scattered = np.zeros((self.branches, self.syn_per_branch), np.float32)
        for b in range(k):
            scattered[b % self.branches, 0] = drive
        c = self.branch_output(clustered.sum(1)).sum()
        s = self.branch_output(scattered.sum(1)).sum()
        return float(c / max(s, 1e-6))


def build_cortical_column(n_neurons: int = 2000, branches: int = 64,
                          syn_per_branch: int = 110, seed: int = 0,
                          w_scale: float = 0.5, target_sparsity: float = 0.15,
                          input_rate: float = 0.05) -> DendriticTree:
    """Wire a cortical column whose neurons have a **biological** synapse count.

    ``branches * syn_per_branch`` synapses per neuron (default ~7,040, the range
    of a real pyramidal cell). Presynaptic partners are drawn within the column
    (recurrent cortical connectivity). The soma threshold is then calibrated so
    that a typical input makes about ``target_sparsity`` of the column fire --
    the homeostatic set-point real cortex holds, so the column codes sparsely
    instead of firing all-or-none."""
    rng = np.random.default_rng(seed)
    shape = (n_neurons, branches, syn_per_branch)
    pre = rng.integers(0, n_neurons, size=shape).astype(np.int32)
    w = (rng.gamma(2.0, 0.5, size=shape).astype(np.float32) * w_scale)
    tree = DendriticTree(n=n_neurons, branches=branches,
                         syn_per_branch=syn_per_branch, pre=pre, w=w)
    # homeostatic calibration of the somatic threshold to a sparse set-point,
    # pooled over several input patterns so the set-point is stable
    drives = [tree.soma_current((rng.random(n_neurons) < input_rate)
                                .astype(np.float32)) for _ in range(8)]
    tree.soma_threshold = float(np.quantile(np.concatenate(drives),
                                            1.0 - target_sparsity))
    return tree


def dendritic_density_report(n_neurons: int = 2000, branches: int = 64,
                             syn_per_branch: int = 110) -> dict:
    """Build a column and report how close its fan-in is to biology, plus the
    honest memory cost and why the *whole* brain cannot be this dense."""
    tree = build_cortical_column(n_neurons, branches, syn_per_branch)
    bytes_ = tree.n_synapses * (4 + 4)          # int32 index + float32 weight
    BIO_PER_NEURON = 7000                        # ~pyramidal cell
    return {
        "neurons": n_neurons,
        "synapses_per_neuron": tree.synapses_per_neuron,
        "biological_fraction": tree.synapses_per_neuron / BIO_PER_NEURON,
        "total_synapses": tree.n_synapses,
        "ram_mb": round(bytes_ / 1e6, 1),
        "supralinearity_index": round(tree.supralinearity_index(), 2),
        "whole_brain_30M_at_bio_density": 30_000_000 * BIO_PER_NEURON,
    }
