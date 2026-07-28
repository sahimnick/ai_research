"""
sensors.py
==========

Sense organs: they turn *stimuli from the world* into *currents into neurons*.

The brain never sees pixels or sound waves directly. A retina, a cochlea, a
fingertip -- each is a transducer that converts one kind of physical signal
into spikes. This module provides the same for the simulator: encoders that map

    text  / image / sound / arbitrary vector   ->   a current pattern

over the neurons of a *sensory* region. An encoder returns a 2-D array of shape
``(T, n_neurons)``: ``T`` time-windows, each a current vector. Feeding that
sequence into the brain is what "showing it something" means.

Design goals:
    * Same interface for every modality (``encode``) so regions are pluggable.
    * Sparse, structured codes (only a few neurons strongly active per input)
      -- this is how real sensory codes look and it makes assemblies separable.
    * Deterministic given a seed, so demos are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np


ArrayLike = Union[Sequence[float], np.ndarray]


class Encoder:
    """Base class. Subclasses implement :meth:`encode`."""

    def __init__(self, n_neurons: int, amplitude: float = 12.0,
                 window: int = 20):
        #: number of neurons in the target sensory region
        self.n_neurons = int(n_neurons)
        #: peak input current for a fully-active neuron
        self.amplitude = float(amplitude)
        #: how many simulation steps one input symbol is held for
        self.window = int(window)

    def encode(self, data) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- helpers ----------------------------------------------------------
    def _hold(self, current: np.ndarray, window: Optional[int] = None
              ) -> np.ndarray:
        """Repeat a single current vector for ``window`` steps -> (T, n)."""
        w = window if window is not None else self.window
        return np.tile(current, (w, 1))

    @staticmethod
    def _sparsify(current: np.ndarray, k: Optional[int],
                  fill: Optional[float] = None) -> np.ndarray:
        """Keep only the ``k`` strongest neurons active (k-winners-take-all).

        Sensory codes in the brain are sparse, and sparsity is what makes
        competitive grounding *selective*: a dense pattern lights up every
        assembly, a sparse one only its own. This enforces that.

        If ``fill`` is given, the winners are all set to that value (a binary
        code). This is important for spiky inputs like a pure tone, where only
        one or two frequency bins carry real energy -- a binary code still gives
        the whole winning band a strong, reliable drive so a concept can bind.
        """
        if not k or k >= current.size:
            return current
        winners = np.argsort(current)[-k:]
        out = np.zeros_like(current)
        out[winners] = fill if fill is not None else current[winners]
        return out


class VectorEncoder(Encoder):
    """Encode an arbitrary real-valued feature vector by *population coding*.

    Each input feature is assigned a Gaussian bump of neurons; the feature's
    value shifts the bump's centre, so similar values excite overlapping
    neurons (graceful generalisation).
    """

    def __init__(self, n_neurons: int, n_features: int, **kw):
        super().__init__(n_neurons, **kw)
        self.n_features = int(n_features)
        self._per = max(1, n_neurons // max(1, n_features))
        self._sigma = max(1.0, self._per / 6.0)

    def encode(self, data: ArrayLike) -> np.ndarray:
        x = np.asarray(data, dtype=np.float64).ravel()
        if x.size != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {x.size}."
            )
        # normalise each feature to 0..1 for placement
        x = np.clip(x, 0.0, 1.0)
        current = np.zeros(self.n_neurons)
        positions = np.arange(self.n_neurons)
        for f, val in enumerate(x):
            centre = f * self._per + val * (self._per - 1)
            bump = np.exp(-0.5 * ((positions - centre) / self._sigma) ** 2)
            current = np.maximum(current, bump)
        return self._hold(self.amplitude * current)


class ImageEncoder(Encoder):
    """Encode a 2-D grayscale image as a retina-like current map.

    Like the real retina, this uses **opponent (ON/OFF) coding** (``opponent=True``,
    the default): half the neurons are ON cells that fire for *bright* pixels and
    half are OFF cells that fire for *dark* pixels. This matters because darkness
    is the *absence* of light -- with a purely brightness-driven code an all-dark
    image would produce no spikes at all and could never be grounded. OFF cells
    give "dark" something concrete to fire, so it can become a concept.

    Brightness is treated as an absolute value in ``[0, 1]`` (arrays with values
    above 1 are assumed to be 0..255 and scaled). Set ``opponent=False`` for a
    plain bright-only retina.
    """

    def __init__(self, n_neurons: int, opponent: bool = True,
                 active_k: int = 14, **kw):
        super().__init__(n_neurons, **kw)
        self.opponent = opponent
        self.active_k = int(active_k)
        per_channel = n_neurons // 2 if opponent else n_neurons
        self.side = max(1, int(np.floor(np.sqrt(per_channel))))

    def _resize(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape
        ys = (np.linspace(0, h - 1, self.side)).astype(int)
        xs = (np.linspace(0, w - 1, self.side)).astype(int)
        return img[np.ix_(ys, xs)]

    def encode(self, image: ArrayLike) -> np.ndarray:
        img = np.asarray(image, dtype=np.float64)
        if img.ndim == 3:  # collapse colour channels to luminance
            img = img.mean(axis=2)
        if img.ndim != 2:
            raise ValueError("ImageEncoder expects a 2-D (or H×W×C) array.")
        if img.max() > 1.0:  # looks like 0..255 -> scale to 0..1
            img = img / 255.0
        bright = np.clip(self._resize(img), 0.0, 1.0).ravel()

        current = np.zeros(self.n_neurons)
        k = self.side * self.side
        current[:k] = bright * self.amplitude               # ON cells
        if self.opponent:
            half = self.n_neurons // 2
            current[half:half + k] = (1.0 - bright) * self.amplitude  # OFF cells
        current = self._sparsify(current, self.active_k, fill=self.amplitude)
        return self._hold(current)


class SoundEncoder(Encoder):
    """Encode a 1-D waveform as a cochlea-like tonotopic current pattern.

    A short-time magnitude spectrum is computed and binned into ``n_neurons``
    frequency channels, low frequency -> low index. Louder band = more current.
    Optionally emits several time-windows so the brain hears the sound unfold.
    """

    def __init__(self, n_neurons: int, sample_rate: int = 16000,
                 frames: int = 1, active_k: int = 12, **kw):
        super().__init__(n_neurons, **kw)
        self.sample_rate = int(sample_rate)
        self.frames = int(frames)
        self.active_k = int(active_k)

    def _spectrum(self, chunk: np.ndarray) -> np.ndarray:
        mag = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk))))
        # bin the spectrum down to one value per neuron (log-ish spacing)
        edges = np.linspace(0, len(mag), self.n_neurons + 1).astype(int)
        binned = np.array([
            mag[edges[i]:max(edges[i] + 1, edges[i + 1])].mean()
            for i in range(self.n_neurons)
        ])
        m = binned.max()
        return (binned / m) if m > 0 else binned

    def encode(self, waveform: ArrayLike) -> np.ndarray:
        wave = np.asarray(waveform, dtype=np.float64).ravel()
        if wave.size < 2:
            raise ValueError("SoundEncoder needs a waveform of length >= 2.")
        chunks = np.array_split(wave, self.frames)
        windows: List[np.ndarray] = []
        for chunk in chunks:
            spec = self._spectrum(chunk)
            current = self._sparsify(self.amplitude * spec, self.active_k,
                                     fill=self.amplitude)
            windows.append(self._hold(current))
        return np.concatenate(windows, axis=0)


class TextEncoder(Encoder):
    """Encode text as a sequence of sparse, word/char-specific current patterns.

    Every symbol (word or character) is hashed to a fixed, sparse set of active
    neurons -- a Sparse Distributed Representation. The same symbol always lights
    the same neurons, so an assembly can form for it, while different symbols
    overlap only a little. One time-window is emitted per symbol, so the brain
    reads the text left to right.
    """

    def __init__(self, n_neurons: int, level: str = "word",
                 active_bits: int = 8, seed: int = 7, **kw):
        super().__init__(n_neurons, **kw)
        if level not in ("word", "char"):
            raise ValueError("level must be 'word' or 'char'.")
        self.level = level
        self.active_bits = int(min(active_bits, n_neurons))
        self.seed = int(seed)

    def _bits_for(self, symbol: str) -> np.ndarray:
        # Deterministic hash -> a reproducible RNG -> a fixed active set.
        h = abs(hash((self.seed, symbol.lower()))) % (2 ** 32)
        rng = np.random.default_rng(h)
        return rng.choice(self.n_neurons, size=self.active_bits, replace=False)

    def symbols(self, text: str) -> List[str]:
        return text.split() if self.level == "word" else list(text)

    def encode(self, text: str) -> np.ndarray:
        windows: List[np.ndarray] = []
        for sym in self.symbols(text):
            if not sym.strip():
                continue
            current = np.zeros(self.n_neurons)
            current[self._bits_for(sym)] = self.amplitude
            windows.append(self._hold(current))
        if not windows:
            return np.zeros((1, self.n_neurons))
        return np.concatenate(windows, axis=0)
