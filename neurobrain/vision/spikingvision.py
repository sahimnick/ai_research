"""
spikingvision.py
================

Perception of **real** handwritten digits by **actual spiking neurons**.

This closes the sharpest honest gap in the project. The headline result -- "the
brain grows its own categories on real MNIST at ~90%" -- was produced by
:mod:`realworld`, which contains **zero spiking neurons**: it is cosine matching
between prototype vectors. Scientifically respectable as competitive learning,
but it is not a spiking network, and calling it one would be a lie.

Here the same job is done with the project's own Izhikevich :class:`Population`:

  1. **Retina / LGN.** Pixel intensity becomes an injected **current**. Bright
     pixels drive their cell harder, so information is carried by *firing rate*
     -- rate coding, exactly as retinal ganglion cells do.

  2. **V1 spiking layer.** Currents propagate through learned weights into a
     population of Izhikevich neurons that **actually integrate and fire** over
     a time window. The code for an image is its **spike-count vector**: how
     many times each cell fired while looking. Nothing here is a dot product
     standing in for a neuron; the neurons run.

  3. **Competitive Hebbian categories on spikes.** Category cells grow (ART-style
     vigilance) from the *spike counts*, not from the pixels. Learning is
     Hebbian between what fired and the winning category cell.

Everything about the pipeline is spiking except the final vigilance comparison,
which is a winner-take-all -- the operation an inhibitory microcircuit performs.

**Honest expectation, stated before the measurement:** spiking rate codes throw
away information that real-valued vectors keep (a spike count over a short window
is a coarse, quantised, noisy version of the drive). A drop relative to the
89.5% non-spiking number is expected and is reported as measured, not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.neuron import Population


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v.astype(np.float32) if n < 1e-9 else (v / n).astype(np.float32)


class SpikingRetina:
    """Pixels -> injected current. Rate coding, with a contrast step first.

    Region: retina / LGN. Bright input means more current means a higher firing
    rate; that is the whole of the encoding, and it is what real ganglion cells
    do."""

    def __init__(self, gain: float = 22.0, baseline: float = 2.0):
        self.gain = float(gain)
        self.baseline = float(baseline)

    def currents(self, image: np.ndarray) -> np.ndarray:
        x = np.asarray(image, np.float32).reshape(-1) / 255.0
        x = x - x.mean()                       # centre-surround style contrast
        s = float(x.std()) or 1.0
        x = x / s
        return (self.baseline + self.gain * np.maximum(x, 0.0)).astype(np.float32)


class SpikingFeatureLayer:
    """A population of Izhikevich cells that really integrates and fires.

    The image's current vector is projected through ``W`` (learned, competitive
    Hebbian) into ``n_cells`` neurons, which are then **stepped for
    ``window`` milliseconds**. The output is the spike-count vector.
    """

    def __init__(self, n_in: int, n_cells: int = 220, window: int = 60,
                 k: int = 22, seed: int = 0, neuron_type: str = "regular_spiking"):
        rng = np.random.default_rng(seed)
        self.n_in, self.n_cells, self.window, self.k = n_in, n_cells, window, k
        self.W = np.abs(rng.normal(0, 1.0, (n_cells, n_in))).astype(np.float32)
        self.W /= np.maximum(np.linalg.norm(self.W, axis=1, keepdims=True), 1e-6)
        self.duty = np.full(n_cells, k / n_cells, np.float32)
        self.pop = Population(n_cells, neuron_type, rng=rng, jitter=0.02)
        # An Izhikevich RS cell only grades its rate between roughly I=8 and
        # I=35, so the drive has to be mapped INTO that band or every cell fires
        # the same number of times and the code carries nothing.
        self.i_floor, self.i_span = 8.0, 30.0

    def drive(self, currents: np.ndarray) -> np.ndarray:
        """Synaptic drive after **divisive normalization**.

        Raw ``W @ x`` has almost no dynamic range (all cells overlap a
        non-negative input about equally), so a rate code cannot form. Cortex
        solves this with lateral/feedback inhibition, which divides each cell's
        drive by the population's activity. That is what happens here: subtract
        the population mean, divide by its spread, then map into the band where
        the neuron actually grades its firing rate."""
        raw = self.W @ _unit(currents)
        z = (raw - raw.mean()) / (raw.std() + 1e-6)          # divisive normalization
        z = z - 2.0 * (self.duty - self.k / self.n_cells)    # homeostatic threshold
        return (self.i_floor + self.i_span * np.clip(z, 0.0, 3.0) / 3.0
                ).astype(np.float32) * (z > 0)

    def spike_counts(self, currents: np.ndarray) -> np.ndarray:
        """Run the neurons for the window and count spikes -- the real code."""
        d = self.drive(currents)
        self.pop.reset()
        counts = np.zeros(self.n_cells, np.float32)
        for _ in range(self.window):
            counts += self.pop.step(d).astype(np.float32)
        return counts

    def learn(self, currents: np.ndarray, counts: np.ndarray,
              lr: float = 0.02) -> None:
        """Competitive Hebbian: the cells that actually FIRED move their weights
        toward the input that made them fire."""
        fired = counts > 0
        if not fired.any():
            return
        x = _unit(currents)
        w = counts[fired][:, None] / max(float(counts.max()), 1.0)
        self.W[fired] += lr * w * (x[None, :] - self.W[fired])
        self.W[fired] = np.maximum(self.W[fired], 0.0)
        self.W[fired] /= np.maximum(
            np.linalg.norm(self.W[fired], axis=1, keepdims=True), 1e-6)
        self.duty *= (1.0 - lr)
        self.duty[fired] += lr

    def train(self, images: np.ndarray, retina: SpikingRetina, epochs: int = 1,
              lr: float = 0.02, seed: int = 0) -> "SpikingFeatureLayer":
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            for i in rng.permutation(len(images)):
                cur = retina.currents(images[i])
                self.learn(cur, self.spike_counts(cur), lr=lr)
        return self


class SpikingCategoryMap:
    """Category cells grown from **spike counts** (ART-style vigilance).

    Identical in spirit to :class:`~neurobrain.realworld.GrowingCategoryMap`,
    but its input is what the spiking layer actually fired, not raw pixels."""

    def __init__(self, dim: int, vigilance: float = 0.62, lr: float = 0.12,
                 max_cells: int = 1500):
        self.dim, self.vigilance, self.lr = dim, float(vigilance), float(lr)
        self.max_cells = int(max_cells)
        self.W = np.zeros((0, dim), np.float32)
        self.cell_label: np.ndarray = np.zeros(0, int)

    @property
    def n_categories(self) -> int:
        return int(len(self.W))

    def learn(self, x: np.ndarray) -> int:
        xn = _unit(x)
        if len(self.W) == 0:
            self.W = xn[None].copy()
            return 0
        sim = self.W @ xn
        w = int(sim.argmax())
        if sim[w] < self.vigilance and len(self.W) < self.max_cells:
            self.W = np.vstack([self.W, xn])
            return len(self.W) - 1
        self.W[w] += self.lr * (xn - self.W[w])
        self.W[w] = _unit(self.W[w])
        return w

    def best(self, X: np.ndarray) -> np.ndarray:
        return (X @ self.W.T).argmax(1)

    def name_cells(self, X: np.ndarray, labels: np.ndarray) -> None:
        """Labels only NAME the clusters that already grew; they never train."""
        win = self.best(X)
        self.cell_label = np.full(len(self.W), -1, int)
        for c in range(len(self.W)):
            m = win == c
            if m.any():
                self.cell_label[c] = np.bincount(labels[m]).argmax()

    def recognise(self, X: np.ndarray) -> np.ndarray:
        return self.cell_label[self.best(X)]


@dataclass
class SpikingDigitBrain:
    """A digit recogniser whose perception is done by spiking neurons."""

    retina: SpikingRetina
    layer: SpikingFeatureLayer
    cortex: SpikingCategoryMap
    test_accuracy: float = 0.0
    n_categories: int = 0
    mean_spikes: float = 0.0
    silent_fraction: float = 0.0

    def code(self, image: np.ndarray) -> np.ndarray:
        """The spike-count vector for one image -- the brain's actual percept."""
        return self.layer.spike_counts(self.retina.currents(image))

    def recognise(self, image: np.ndarray) -> int:
        return int(self.cortex.recognise(_unit(self.code(image))[None])[0])


def build_spiking_digit_brain(n_train: int = 6000, n_test: int = 1500,
                              n_cells: int = 220, window: int = 24,
                              epochs: int = 1, vigilance: float = 0.62,
                              verbose: bool = False) -> SpikingDigitBrain:
    """Train the spiking perception on REAL MNIST and score it honestly.

    Labels are used only to *name* the categories that grew by themselves, never
    to train -- the same rule as the non-spiking version, so the two numbers are
    comparable."""
    from ..sensing.realworld import load_mnist

    def say(*a):
        if verbose:
            print(*a)

    trx, trY, tex, teY = load_mnist(n_train=n_train, n_test=n_test)
    retina = SpikingRetina()
    layer = SpikingFeatureLayer(n_in=trx[0].size, n_cells=n_cells,
                                window=window, seed=0)

    say(f"V1: {n_cells} Izhikevich cells integrating for {window} ms per image ...")
    layer.train(trx[:min(2500, len(trx))], retina, epochs=epochs)

    say("encoding the training set as SPIKE COUNTS ...")
    Xtr = np.array([_unit(layer.spike_counts(retina.currents(im))) for im in trx],
                   np.float32)
    counts_raw = np.array([c.sum() for c in
                           (layer.spike_counts(retina.currents(im))
                            for im in trx[:400])])

    say("growing category cells from the spikes ...")
    cortex = SpikingCategoryMap(dim=Xtr.shape[1], vigilance=vigilance)
    for i in range(len(Xtr)):
        cortex.learn(Xtr[i])
    cortex.name_cells(Xtr, trY)

    say("scoring on held-out real digits ...")
    Xte = np.array([_unit(layer.spike_counts(retina.currents(im))) for im in tex],
                   np.float32)
    acc = float(np.mean(cortex.recognise(Xte) == teY))

    brain = SpikingDigitBrain(retina, layer, cortex, acc, cortex.n_categories,
                              float(counts_raw.mean()),
                              float(np.mean([(_ == 0).mean() for _ in
                                             [np.array([c for c in Xte[0]])]])))
    # a genuine sparsity read-out: fraction of cells silent for a typical image
    silent = np.mean([(layer.spike_counts(retina.currents(im)) == 0).mean()
                      for im in tex[:200]])
    brain.silent_fraction = float(silent)
    if verbose:
        print(f"   held-out accuracy (SPIKING)  : {acc:.1%}")
        print(f"   categories grown             : {cortex.n_categories}")
        print(f"   mean spikes per image        : {brain.mean_spikes:.0f}")
        print(f"   cells silent for an image    : {brain.silent_fraction:.0%} "
              f"(sparse code)")
    return brain


def spiking_vs_rate_comparison(n_train: int = 6000, n_test: int = 1500,
                               verbose: bool = False) -> Dict[str, float]:
    """The honest side-by-side: real spiking neurons vs. the vector version.

    Same data, same amount of it, same 'labels only name the clusters' rule."""
    import numpy as _np
    from ..sensing.realworld import load_mnist, GrowingCategoryMap, contrast_normalise

    spk = build_spiking_digit_brain(n_train=n_train, n_test=n_test,
                                    verbose=verbose)
    trx, trY, tex, teY = load_mnist(n_train=n_train, n_test=n_test)
    X = contrast_normalise(trx)
    m = GrowingCategoryMap(dim=X.shape[1], vigilance=0.65, lr=0.1)
    m.train(X, epochs=2)
    m.name_cells(X, trY)
    rate_acc = float(_np.mean(m.recognise(tex, 0.0) == teY))
    out = {"spiking_accuracy": spk.test_accuracy,
           "vector_accuracy": rate_acc,
           "gap": rate_acc - spk.test_accuracy,
           "spiking_categories": float(spk.n_categories),
           "mean_spikes_per_image": spk.mean_spikes,
           "silent_fraction": spk.silent_fraction}
    if verbose:
        print(f"\n   SPIKING neurons : {out['spiking_accuracy']:.1%}")
        print(f"   vector version  : {out['vector_accuracy']:.1%}")
        print(f"   cost of being spiking: {out['gap']:+.1%}")
    return out
