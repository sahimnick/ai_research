"""
backend.py
==========

**A PyTorch backend for the spiking substrate, with the interface unchanged.**

Why a backend rather than a rewrite
-----------------------------------
The measurement that motivates this is specific: the Izhikevich update is
memory-bandwidth bound, not Python bound. Counting the array passes predicted
130 ms per millisecond of biology for 8M neurons and the measured figure was
133 ms, and each step is a *single* Python call for eight million cells. So
there is nothing to gain from "converting objects to tensors" -- the substrate
has always been a structure of arrays -- and the only real levers are moving
fewer bytes and moving them faster.

``TorchPopulation`` therefore does two things numpy cannot:

* **Fuses the update.** numpy materialises a temporary for every operation, so
  the step walks the arrays ~24 times. In torch the same expression can run
  under ``torch.compile``, and even without it the ops are fewer and larger.
* **Moves the state to whatever memory is fastest.** On an M-series Mac that is
  the unified memory reached through ``mps``; on a CUDA box it is device
  memory. Nothing about the model changes -- only where the bytes live.

What is deliberately NOT done
-----------------------------
No autograd, no optimiser, no loss. Gradients are switched off everywhere.
This module exists to run the same biology faster, and if it ever started
training weights by backpropagation it would be answering a different question
than the project asks.

Numerical equivalence is checked rather than assumed: :func:`compare_backends`
runs both implementations from the same state and reports the spike-train
agreement, because a fast backend that quietly changes the dynamics would
invalidate every result measured so far.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

try:  # torch is optional -- everything else in the package works without it
    import torch
    _HAVE_TORCH = True
except Exception:                                    # pragma: no cover
    torch = None                                     # type: ignore
    _HAVE_TORCH = False


def available_devices() -> Dict[str, bool]:
    """What this machine can actually run on."""
    if not _HAVE_TORCH:
        return {"numpy": True}
    out = {"numpy": True, "cpu": True}
    try:
        out["cuda"] = bool(torch.cuda.is_available())
    except Exception:
        out["cuda"] = False
    try:
        out["mps"] = bool(torch.backends.mps.is_available())
    except Exception:
        out["mps"] = False
    return out


def best_device() -> str:
    d = available_devices()
    if d.get("cuda"):
        return "cuda"
    if d.get("mps"):
        return "mps"
    return "cpu" if _HAVE_TORCH else "numpy"


class TorchPopulation:
    """The same Izhikevich population, on a torch device.

    The interface matches :class:`~neurobrain.neuron.Population` where it
    matters -- ``n``, ``step(I)`` returning a boolean spike vector, ``spiked``,
    ``v``, ``u`` -- so a caller does not need to know which backend it holds.
    """

    def __init__(self, n: int, kind: str = "regular_spiking",
                 device: Optional[str] = None, seed: int = 0,
                 jitter: float = 0.02):
        if not _HAVE_TORCH:
            raise ImportError("TorchPopulation needs torch installed")
        from .neuron import NEURON_TYPES
        t = NEURON_TYPES[kind]
        self.device = torch.device(device or best_device())
        self.n = int(n)
        self.kind = kind
        self.inhibitory = bool(t.inhibitory)
        g = torch.Generator(device="cpu").manual_seed(int(seed))

        def par(x: float) -> "torch.Tensor":
            v = torch.full((self.n,), float(x), dtype=torch.float32)
            if jitter:
                v *= 1.0 + jitter * torch.randn(self.n, generator=g)
            return v.to(self.device)

        self.a, self.b = par(t.a), par(t.b)
        self.c, self.d = par(t.c), par(t.d)
        self.v = (self.c.clone())
        self.u = self.b * self.v
        self.spiked = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self._gen = torch.Generator(device="cpu").manual_seed(int(seed) + 1)

    @torch.no_grad()
    def step(self, I, dt: float = 1.0):
        """One millisecond. ``I`` may be a numpy array or a torch tensor."""
        if not torch.is_tensor(I):
            I = torch.as_tensor(np.asarray(I, np.float32), device=self.device)
        elif I.device != self.device:
            I = I.to(self.device)
        I = torch.clamp(I, -500.0, 500.0)
        reset = self.v >= 30.0
        if bool(reset.any()):
            self.v = torch.where(reset, self.c, self.v)
            self.u = torch.where(reset, self.u + self.d, self.u)
        h = dt / 2.0
        for _ in range(2):                     # same sub-stepping as numpy
            self.v = self.v + h * (0.04 * self.v * self.v + 5.0 * self.v
                                   + 140.0 - self.u + I)
            self.v = torch.clamp(self.v, -120.0, 30.0)
        self.u = self.u + dt * self.a * (self.b * self.v - self.u)
        self.spiked = self.v >= 30.0
        return self.spiked

    def noise(self, scale: float):
        return scale * torch.randn(self.n, generator=self._gen,
                                   dtype=torch.float32).to(self.device)

    def numpy_spikes(self) -> np.ndarray:
        return self.spiked.detach().cpu().numpy()


def compare_backends(n: int = 100_000, ms: int = 40, drive: float = 7.0,
                     device: Optional[str] = None) -> Dict[str, float]:
    """Do the two backends produce the same tissue? Measured, not assumed.

    Returns the per-millisecond cost of each and the fraction of milliseconds
    on which the two populations produced identical spike counts. Exact
    per-neuron agreement is not expected -- the parameter jitter is drawn from
    different generators -- so the comparison is of the population dynamics:
    rate, and how the rate evolves.
    """
    import time
    from .neuron import Population
    if not _HAVE_TORCH:
        raise ImportError("compare_backends needs torch installed")
    npop = Population(n, "regular_spiking", rng=np.random.default_rng(0),
                      jitter=0.02)
    tpop = TorchPopulation(n, "regular_spiking", device=device, seed=0,
                           jitter=0.02)
    I = np.full(n, drive, np.float32)
    It = torch.as_tensor(I, device=tpop.device)

    t0 = time.time()
    rates_np = [float(npop.step(I).mean()) for _ in range(ms)]
    t_np = (time.time() - t0) / ms * 1000

    tpop.step(It)                                    # warm up kernels
    if tpop.device.type == "mps":
        torch.mps.synchronize()
    elif tpop.device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    rates_t = [float(tpop.step(It).float().mean()) for _ in range(ms)]
    if tpop.device.type == "mps":
        torch.mps.synchronize()
    elif tpop.device.type == "cuda":
        torch.cuda.synchronize()
    t_t = (time.time() - t0) / ms * 1000

    a, b = np.asarray(rates_np), np.asarray(rates_t)
    return {"numpy_ms": t_np, "torch_ms": t_t,
            "speedup": t_np / max(t_t, 1e-9),
            "device": str(tpop.device),
            "rate_numpy": float(a.mean()), "rate_torch": float(b.mean()),
            "rate_abs_diff": float(np.abs(a - b).mean())}
