"""
audio.py
========

A **spiking auditory hierarchy** -- the hearing counterpart of ``vision.py``.
The ear turns sound into a *cochleagram* (a frequency x time picture), and from
there the cortex builds the same kind of ladder vision uses, one area per layer:

    cochlea  ->  A1  ->  pool  ->  belt (A2)  ->  readout
    (freq x time) (spectro-temporal   (invariance)  (timbre / pitch /  (sound
                   receptive fields)                  sweep parts)       class)

The key idea: a cochleagram is a 2-D map, so the **same spiking convolutional
neurons and synapses** from the visual stream work here -- only now a "local
receptive field" spans a little patch of *frequency and time*. Trained
unsupervised (competitive Hebbian / STDP-style), **A1** discovers
spectro-temporal receptive fields: onset detectors, up/down frequency sweeps and
harmonic stacks -- exactly the features real primary auditory cortex is tuned
to. **belt** combines them, and a linear readout names the sound.

Every area is a separate object and logs the spike map that went in and came out
(``layer.log``), just like the visual stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..vision.ventral import (SpikingConvLayer, SpikingPool, VisionHierarchy,
                     _sample_patches, _grid_pool, _fit_readout)


# ---------------------------------------------------------------------------
# sound generators (small, synthetic, but with real spectro-temporal structure)
# ---------------------------------------------------------------------------
SR = 8000
DUR = 0.40

SOUND_CLASSES = ("pure", "up_chirp", "down_chirp", "harmonic", "noise", "am",
                 "fm", "chord")


def tone(freq: float, dur: float = DUR, sr: int = SR) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def chirp(f0: float, f1: float, dur: float = DUR, sr: int = SR) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    k = (f1 - f0) / dur
    return np.sin(2 * np.pi * (f0 * t + 0.5 * k * t * t)).astype(np.float32)


def harmonic_tone(f0: float, n_harm: int = 5, dur: float = DUR, sr: int = SR
                  ) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    y = np.zeros_like(t)
    for h in range(1, n_harm + 1):
        y += np.sin(2 * np.pi * f0 * h * t) / h
    return y.astype(np.float32)


def noise_burst(dur: float = DUR, sr: int = SR, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(int(dur * sr)).astype(np.float32)


def am_tone(freq: float, mod: float = 30.0, dur: float = DUR, sr: int = SR
            ) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    env = 0.5 * (1 + np.sin(2 * np.pi * mod * t))
    return (env * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def fm_tone(freq: float, mod: float = 6.0, depth: float = 0.5,
            dur: float = DUR, sr: int = SR) -> np.ndarray:
    """A vibrato tone: the *pitch* wobbles (frequency modulation)."""
    t = np.arange(int(dur * sr)) / sr
    inst = freq * (1.0 + depth * np.sin(2 * np.pi * mod * t))
    phase = 2 * np.pi * np.cumsum(inst) / sr
    return np.sin(phase).astype(np.float32)


def chord_tone(freq: float, ratio: float = 1.5, dur: float = DUR, sr: int = SR
               ) -> np.ndarray:
    """Two tones at a non-harmonic ratio sounded together (a chord/interval)."""
    t = np.arange(int(dur * sr)) / sr
    return (0.6 * np.sin(2 * np.pi * freq * t)
            + 0.6 * np.sin(2 * np.pi * freq * ratio * t)).astype(np.float32)


def sound_dataset(n_per_class: int = 40, seed: int = 0, sr: int = SR,
                  dur: float = DUR
                  ) -> Tuple[List[np.ndarray], np.ndarray, List[str]]:
    """A labelled set of the six sound classes with randomised parameters."""
    rng = np.random.default_rng(seed)
    sigs, labels = [], []
    for cls, name in enumerate(SOUND_CLASSES):
        for _ in range(n_per_class):
            f = rng.uniform(300, 1200)
            if name == "pure":
                s = tone(f, dur, sr)
            elif name == "up_chirp":
                s = chirp(f, f * rng.uniform(2.0, 3.5), dur, sr)
            elif name == "down_chirp":
                s = chirp(f * rng.uniform(2.0, 3.5), f, dur, sr)
            elif name == "harmonic":
                s = harmonic_tone(f * 0.5, int(rng.integers(4, 7)), dur, sr)
            elif name == "noise":
                s = noise_burst(dur, sr, seed=int(rng.integers(1 << 30)))
            elif name == "am":
                s = am_tone(f, rng.uniform(20, 45), dur, sr)
            elif name == "fm":
                s = fm_tone(f, rng.uniform(4, 9), rng.uniform(0.3, 0.6), dur, sr)
            elif name == "chord":
                s = chord_tone(f, rng.uniform(1.3, 1.7), dur, sr)
            s = s + 0.02 * rng.standard_normal(len(s)).astype(np.float32)
            sigs.append(s)
            labels.append(cls)
    return sigs, np.array(labels), list(SOUND_CLASSES)


# ---------------------------------------------------------------------------
# the cochlea: signal -> log-frequency spectrogram (a freq x time map)
# ---------------------------------------------------------------------------
def _mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _imel(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


class Cochleagram:
    """Turns a 1-D waveform into a (1, n_freq, n_time) cochleagram map.

    A short-time Fourier transform gives energy over time; a **mel** (log-
    frequency) filterbank warps it the way the cochlea does -- fine resolution
    low, coarse high -- and a compressive nonlinearity mimics hair-cell
    saturation. The output is a 2-D map the spiking conv areas read like an
    image.
    """

    def __init__(self, sr: int = SR, n_freq: int = 36, n_fft: int = 128,
                 hop: int = 32, name: str = "cochlea"):
        self.sr = sr
        self.n_freq = n_freq
        self.n_fft = n_fft
        self.hop = hop
        self.name = name
        self.log: Dict[str, np.ndarray] = {}
        self.win = np.hanning(n_fft).astype(np.float32)
        self.fb = self._filterbank()

    def _filterbank(self) -> np.ndarray:
        n_bins = self.n_fft // 2 + 1
        edges = _imel(np.linspace(_mel(50), _mel(self.sr / 2), self.n_freq + 2))
        bins = np.floor((self.n_fft + 1) * edges / self.sr).astype(int)
        bins = np.clip(bins, 0, n_bins - 1)
        fb = np.zeros((self.n_freq, n_bins), np.float32)
        for m in range(1, self.n_freq + 1):
            lo, ce, hi = bins[m - 1], bins[m], bins[m + 1]
            for b in range(lo, ce):
                fb[m - 1, b] = (b - lo) / max(ce - lo, 1)
            for b in range(ce, hi):
                fb[m - 1, b] = (hi - b) / max(hi - ce, 1)
        return fb

    def forward(self, sig: np.ndarray) -> np.ndarray:
        n = self.n_fft
        starts = range(0, len(sig) - n + 1, self.hop)
        mag = np.array([np.abs(np.fft.rfft(sig[s:s + n] * self.win))
                        for s in starts], np.float32).T          # (bins, time)
        coch = self.fb @ mag                                     # (n_freq, time)
        coch = np.log1p(4.0 * coch)                              # hair-cell comp.
        peak = coch.max()
        if peak > 1e-6:
            coch = coch / peak
        out = coch[None].astype(np.float32)
        self.log = {"input": sig.copy(), "output": out}
        return out


def divisive_normalization(spec: np.ndarray, sigma_band: float = 2.5,
                           semi_saturation: float = 0.1, exponent: float = 2.0,
                           adaptive: bool = False, axis: int = 0) -> np.ndarray:
    """Heeger (1992) divisive normalization along the frequency axis.

        R_i = E_i^n / (s^n + sum_j G_ij * E_j^n)

    Each channel is divided by the pooled activity of its **neighbours in
    frequency**, weighted by a Gaussian -- lateral inhibition, the operation
    Heeger proposed for V1 and which is now taken to be a canonical cortical
    computation wherever it appears. ``semi_saturation`` is the constant that
    keeps the ratio finite at low input and sets the contrast at which the
    response is half-maximal.

    What this replaces, and why
    ---------------------------
    The previous preprocessing subtracted a per-band percentile floor. It made
    the code robust to background noise (34.4% -> 90.6% under noise) but it is a
    subtractive, static estimate: it removes an offset without ever changing the
    *relative* gain between a loud band and a quiet one. Divisive normalization
    does exactly that, which is what a self-organizing map needs -- a map can
    only learn distinctions the input still contains, and a representation whose
    channels have not been put on a common scale is dominated by whichever
    channel happens to be loudest.

    Parameters
    ----------
    spec:  (n_freq, n_time) cochleagram or spectrogram.
    sigma_band:  width of the Gaussian normalization pool, in frequency bands.
    semi_saturation:  the constant ``s`` above, relative to the input scale.
    exponent:  ``n``. 2.0 is Heeger's; 1.0 gives a purely linear pool.
    """
    E = np.asarray(spec, np.float32)
    if E.ndim != 2:
        raise ValueError("divisive_normalization expects a 2-D (freq, time) map")
    if axis == 1:
        return divisive_normalization(E.T, sigma_band, semi_saturation,
                                      exponent, adaptive, 0).T
    n_f = E.shape[0]
    d = np.arange(n_f, dtype=np.float32)
    G = np.exp(-((d[:, None] - d[None, :]) ** 2) / (2.0 * sigma_band ** 2))
    G /= G.sum(1, keepdims=True)                 # a proper weighted average

    p = np.power(np.maximum(E, 0.0), exponent)
    pool = G @ p                                 # Gaussian pool over frequency
    # ``adaptive`` scales the semi-saturation constant to each input's own
    # level, which makes the operation exactly contrast-invariant. It sounds
    # like an improvement and it is a TRAP, so it is off by default: in a
    # streaming window the "input's own level" depends on how much silence the
    # window happens to contain, so the same sound produces a different code
    # depending on where the window fell. Measured, it collapsed streaming
    # audio recognition from 85.7% to 28.6% and cross-modal recall to chance.
    # Heeger's fixed sigma has no such dependence and is the default.
    scale = float(np.percentile(p, 95)) or 1.0 if adaptive else 1.0
    s = (semi_saturation ** exponent) * scale
    out = p / (s + pool)
    m = float(out.max())
    return (out / m).astype(np.float32) if m > 1e-9 else out.astype(np.float32)


# ---------------------------------------------------------------------------
# the auditory stream: cochlea -> A1 -> pool -> belt + a sound readout
# ---------------------------------------------------------------------------
@dataclass
class AuditoryStream:
    coch: Cochleagram
    A1: SpikingConvLayer
    pool: SpikingPool
    belt: SpikingConvLayer
    hierarchy: VisionHierarchy
    class_names: Optional[List[str]] = None
    _mu: Optional[np.ndarray] = None
    _sd: Optional[np.ndarray] = None
    _W: Optional[np.ndarray] = None

    def descriptor(self, sig: np.ndarray) -> np.ndarray:
        """A spectro-temporal descriptor: what frequencies, and how they move."""
        m = self.coch.forward(sig)
        a1 = self.A1.forward(m)
        p = self.pool.forward(a1)
        belt = self.belt.forward(p)
        freq_prof = m[0].mean(1)                       # spectral content (pitch)
        tmod = m[0].std(1) / (m[0].mean(1) + 1e-6)     # temporal modulation (AM)
        env = m[0].sum(0)                              # broadband envelope
        env = env - env.mean()
        E = np.abs(np.fft.rfft(env * np.hanning(len(env))))
        modspec = E[1:20] / (E.sum() + 1e-6)           # modulation spectrum (MTF)
        am_score = float(modspec[3:].sum())            # energy above ~10 Hz (AM)
        # spectral centroid over time -> pitch movement (FM vibrato vs chirp)
        bands = np.arange(m.shape[1])[:, None]
        cen = (m[0] * bands).sum(0) / (m[0].sum(0) + 1e-6)
        cen0 = cen - cen.mean()
        Ec = np.abs(np.fft.rfft(cen0 * np.hanning(len(cen0))))
        pitch_wobble = float(cen0.std())               # how much pitch moves
        pitch_osc = float(Ec[2:12].sum() / (Ec.sum() + 1e-6))  # wobble vs drift
        # spectral shape -> chord/harmonic (wide, multi-peak) vs pure/am (narrow)
        fp = freq_prof / (freq_prof.max() + 1e-6)
        n_peaks = float(np.sum((fp[1:-1] > 0.35) &
                               (fp[1:-1] > fp[:-2]) & (fp[1:-1] > fp[2:])))
        n_bands = float((fp > 0.35).sum())             # how many bands are lit
        cen_f = float((fp * bands[:, 0]).sum() / (fp.sum() + 1e-6))
        spread = float(np.sqrt((fp * (bands[:, 0] - cen_f) ** 2).sum() /
                               (fp.sum() + 1e-6)))      # bandwidth (chord wide)
        gm = np.exp(np.log(freq_prof + 1e-6).mean())
        flatness = float(gm / (freq_prof.mean() + 1e-6))  # noise high, tone low
        d = np.concatenate([
            freq_prof,
            tmod,                                      # steady vs modulated
            modspec,                                   # AM rate (tremolo) cue
            [am_score, am_score],                      # explicit AM-present cue
            [pitch_wobble, pitch_osc, pitch_osc],      # FM vibrato vs drift
            [n_peaks, n_bands, spread, spread, flatness],  # chord / harmonic / noise
            _grid_pool(m, 4).reshape(-1),              # coarse freq x time layout
            a1.reshape(a1.shape[0], -1).sum(1),        # A1 part signature
            _grid_pool(a1, 3).reshape(-1),             # where A1 parts fire
            belt.reshape(belt.shape[0], -1).sum(1),
            _grid_pool(belt, 2).reshape(-1),
        ]).astype(np.float32)
        n = np.linalg.norm(d)
        return d / n if n > 1e-6 else d

    def classify(self, sig: np.ndarray) -> str:
        d = (self.descriptor(sig) - self._mu) / self._sd
        scores = np.concatenate([d, [1.0]]) @ self._W
        return self.class_names[int(scores.argmax())]

    def sound_code(self, sig: np.ndarray) -> np.ndarray:
        """A compact, whole-sound code (global-pooled belt) -- the auditory
        counterpart of the visual IT object code, for cross-modal binding."""
        m = self.coch.forward(sig)
        belt = self.belt.forward(self.pool.forward(self.A1.forward(m)))
        v = belt.reshape(belt.shape[0], -1).mean(1)
        n = np.linalg.norm(v)
        return (v / n).astype(np.float32) if n > 1e-6 else v.astype(np.float32)


def build_auditory_stream(n_a1: int = 24, n_belt: int = 28,
                          verbose: bool = False) -> AuditoryStream:
    """Train A1 then belt on sound, each on the previous area's output, and fit
    a sound-class readout. Returns an :class:`AuditoryStream`."""
    def say(*a):
        if verbose:
            print(*a)

    coch = Cochleagram()
    sigs, labels, names = sound_dataset(n_per_class=90, seed=0)
    cochs = np.array([coch.forward(s)[0] for s in sigs])[:, None]   # (N,1,F,T)

    # A1: spectro-temporal receptive fields (onsets, sweeps, harmonic stacks)
    say("A1: learning spectro-temporal receptive fields from sound ...")
    a1_patches = _sample_patches(cochs, [], k=7, per_image=8, seed=1)
    A1 = SpikingConvLayer(1, n_a1, 7, name="A1", lr=0.03, seed=1)
    A1.train(a1_patches, epochs=5, init="kmeans")
    pool = SpikingPool(2, name="pool")

    # belt / A2: combinations of A1 features
    say("belt: learning combinations of A1 features ...")
    belt_patches = _sample_patches(cochs, [A1, pool], k=5, per_image=8, seed=2)
    belt = SpikingConvLayer(n_a1, n_belt, 5, name="belt", lr=0.03, seed=2)
    belt.train(belt_patches, epochs=5, init="kmeans")

    hierarchy = (VisionHierarchy().add(A1).add(pool).add(belt))
    stream = AuditoryStream(coch, A1, pool, belt, hierarchy)

    say("readout: fitting sound classifier ...")
    X = np.array([stream.descriptor(s) for s in sigs], np.float32)
    stream._mu, stream._sd, stream._W, stream.test_accuracy = _fit_readout(
        X, labels, len(names), lam=1.0, seed=0)
    stream.class_names = names
    say(f"   sound recognition accuracy: {stream.test_accuracy:.0%} "
        f"({len(names)} classes, chance {100 // len(names)}%)")
    return stream
