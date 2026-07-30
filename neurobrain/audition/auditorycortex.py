"""
auditorycortex.py
=================

**A predictive A1 with temporal receptive fields**, replacing the auditory
self-organizing map that did not work.

Why the map was the wrong idea
------------------------------
Test A settled it. A nearest-prototype read-out was run on three
representations of the same sounds:

    raw cochleagram, no filters at all : 66.7%
    designed Gabor belt               : 79.2%
    self-organized map                : 58.3%

The map scores **below doing nothing**. A layer that leaves the input worse than
it found it is not an under-trained layer; it is the wrong layer. Three further
facts say why:

1. **The window was far too short.** One cochleagram step is 4 ms and the
   belt's receptive field spanned 10 of them -- 40 ms of a 400 ms event, about
   10%. A 400 ms frequency sweep is very nearly a straight line inside a 40 ms
   window, so the feature that *defines* the classes was invisible to the
   filters that were supposed to learn it.

2. **The weights were non-negative, so no cell could have an OFF subfield.**
   Every filter could only say "energy here", never "energy here and not
   there". Filters restricted to the positive orthant overlap heavily by
   construction, which is also why they came out so alike.

3. **The objective was reconstruction, not prediction.** A map asks *which
   prototype does this patch resemble*. The classes here differ by
   ``x(t) -> x(t+1)`` -- a rising glide, a falling glide, a modulation -- so the
   question worth asking is *given this, what comes next*.

What is built here
------------------
:class:`PredictiveA1`, following those three points in order:

* **Spectrotemporal receptive fields.** Each cell sees ``n_lags`` frames of
  history across all frequency bands -- a frequency-by-time field long enough to
  contain a sweep, not a square patch of spectrogram.

* **ON and OFF channels.** The cochleagram is split into the positive and
  negative parts of its temporal derivative, the way auditory onset and offset
  cells split it. A non-negative weight on the OFF channel *is* an inhibitory
  subfield, so the sign problem is solved by the input representation rather
  than by allowing negative synapses.

* **Learning by prediction error.** Each cell encodes the recent past into a
  sparse state, that state predicts the next frame, and the synapses change by
  the local product of the state and the error:

      r    = relu(W_enc @ context)
      pred = W_dec.T @ r
      err  = target - pred
      W_dec += lr * outer(r, err)                     # state x error
      W_enc += lr * outer(relu(W_dec @ err) * (r>0), context)

  Both updates are products of quantities present at the synapse -- the second
  is the error arriving back through the decoder, which is the anatomical
  feedback path. There is no global objective and no backpropagation through
  time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..vision.widev1 import _unit


# ---------------------------------------------------------------------------
# The ear's ON/OFF split
# ---------------------------------------------------------------------------
def on_off_channels(coch: np.ndarray, smooth: int = 2) -> np.ndarray:
    """Split a cochleagram into onset (ON) and offset (OFF) channels.

    Returns ``(2, n_freq, n_time)``: the positive and negative parts of the
    temporal derivative of each frequency band. This is what auditory onset and
    offset cells do, and it is what lets a **non-negative** weight express an
    inhibitory subfield: weight on the OFF channel means "and the energy here
    must be falling"."""
    E = np.asarray(coch, np.float32)
    if smooth > 1:
        k = np.ones(smooth, np.float32) / smooth
        E = np.stack([np.convolve(row, k, mode="same") for row in E])
    d = np.diff(E, axis=1, prepend=E[:, :1])
    return np.stack([np.maximum(d, 0.0), np.maximum(-d, 0.0)]).astype(np.float32)


def spectrotemporal_context(ch: np.ndarray, n_lags: int, step: int = 1
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """Slice an ON/OFF map into (context, next-frame) pairs.

    ``context`` is ``n_lags`` frames of history flattened over both channels and
    all bands -- the frequency-by-time field. ``target`` is the frame that comes
    after it, which is what the cell has to predict."""
    C, F, T = ch.shape
    if n_lags >= T:
        # A 400 ms clip at a 4 ms hop is only 97 frames, so asking for 100 lags
        # asks for more history than exists. The old code fell through to a
        # zero-filled context and the decoder then diverged (prediction error
        # 4.9e7, accuracy at chance). Fail loudly instead of silently.
        raise ValueError(
            f"n_lags={n_lags} needs more than {T} frames of signal; the clip "
            f"is only {T} frames long.")
    ctx, tgt = [], []
    for t in range(n_lags, T, step):
        ctx.append(ch[:, :, t - n_lags:t].reshape(-1))
        tgt.append(ch[:, :, t].reshape(-1))
    if not ctx:
        z = np.zeros((1, C * F * n_lags), np.float32)
        return z, np.zeros((1, C * F), np.float32)
    return np.asarray(ctx, np.float32), np.asarray(tgt, np.float32)


# ---------------------------------------------------------------------------
# A1
# ---------------------------------------------------------------------------
class PredictiveA1:
    """Cells with long spectrotemporal fields that learn by predicting.

    Parameters
    ----------
    n_freq:   number of cochlear bands.
    n_units:  size of the layer.
    n_lags:   how many frames of history a cell sees. At a 4 ms hop, 32 lags is
              128 ms -- long enough for a glide to be a glide.
    sparsity: target fraction of units active, enforced homeostatically.
    """

    def __init__(self, n_freq: int = 36, n_units: int = 128, n_lags: int = 32,
                 sparsity: float = 0.12, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n_freq, self.n_units, self.n_lags = int(n_freq), int(n_units), int(n_lags)
        self.sparsity = float(sparsity)
        self.n_in = 2 * self.n_freq * self.n_lags
        self.n_out = 2 * self.n_freq
        self.W_enc = np.abs(rng.normal(0, 1, (self.n_units, self.n_in))).astype(np.float32)
        self.W_enc /= np.maximum(np.linalg.norm(self.W_enc, axis=1, keepdims=True), 1e-6)
        # The DECODER is deliberately NOT kept on the unit sphere. Normalizing
        # both matrices leaves nothing to tie the prediction's scale to the
        # target's: measured, the prediction came out 35.8x too large and the
        # normalized error sat above 10 forever. The delta rule finds the right
        # scale by itself provided it is allowed to, so these weights are free
        # and start small, with a light decay instead of a hard norm.
        self.W_dec = (0.01 * np.abs(rng.normal(0, 1, (self.n_units, self.n_out)))
                      ).astype(np.float32)
        self.decay = 1e-4
        self.duty = np.full(self.n_units, sparsity, np.float32)
        self._e2 = self._y2 = 0.0        # last step's squared norms, for train()
        self.rng = rng

    # -- response ----------------------------------------------------------
    def encode(self, context: np.ndarray) -> np.ndarray:
        """A sparse state: k-winners of the drive, homeostatically balanced."""
        x = np.asarray(context, np.float32)
        n = float(np.linalg.norm(x))
        if n > 1e-9:
            x = x / n
        drive = self.W_enc @ x - 0.5 * (self.duty - self.sparsity)
        k = max(2, int(self.n_units * self.sparsity))
        idx = np.argpartition(drive, -k)[-k:]
        r = np.zeros(self.n_units, np.float32)
        r[idx] = np.maximum(drive[idx], 0.0)
        # divisive normalization at the soma: the state carries WHICH cells are
        # active and in what proportion, not how loud the input happened to be.
        m = float(np.linalg.norm(r))
        return (r / m).astype(np.float32) if m > 1e-9 else r

    def predict(self, context: np.ndarray) -> np.ndarray:
        return self.W_dec.T @ self.encode(context)

    # -- learning by prediction error --------------------------------------
    def learn(self, context: np.ndarray, target: np.ndarray,
              lr: float = 0.03) -> float:
        """One predictive step. Returns the normalized prediction error."""
        x = np.asarray(context, np.float32)
        y = np.asarray(target, np.float32)
        r = self.encode(x)
        err = y - self.W_dec.T @ r
        act = r > 0
        if act.any():
            self.W_dec[act] += lr * np.outer(r[act], err)
            self.W_dec *= (1.0 - self.decay)
            np.maximum(self.W_dec, 0.0, out=self.W_dec)
            # the error arriving back through the decoder -- the feedback path
            back = np.maximum(self.W_dec @ err, 0.0) * act
            if back.any():
                self.W_enc[act] += lr * np.outer(back[act], x)
                np.maximum(self.W_enc, 0.0, out=self.W_enc)
                self.W_enc /= np.maximum(np.linalg.norm(self.W_enc, axis=1,
                                                        keepdims=True), 1e-6)
        self.duty *= (1.0 - lr)
        self.duty += lr * act
        # kept for :meth:`train` to pool -- see the note there on why the mean
        # of per-step ratios is not usable on real recordings
        self._e2 = float(err @ err)
        self._y2 = float(y @ y)
        return float(np.linalg.norm(err) / (np.linalg.norm(y) + 1e-9))

    def train(self, cochleagrams: Sequence[np.ndarray], epochs: int = 4,
              lr: float = 0.03, step: int = 2, seed: int = 0) -> List[float]:
        """Returns the curve of **pooled** normalized error, one point per epoch.

        Pooled -- ``sqrt(sum||err||^2 / sum||y||^2)`` -- rather than the mean of
        the per-step ratios :meth:`learn` returns, and the difference is not
        cosmetic. A synthetic tone is loud in every frame, so per-frame
        ``||err||/||y||`` is well behaved and averaging it is fine. A real field
        recording is mostly *silence*: a car-horn clip is one second of horn in
        four seconds of street, and in those frames ``||y||`` is ~0 while the
        prediction is not, so the ratio is divided by the 1e-9 guard and returns
        a number near 1e9. Averaged, a handful of silent frames set the whole
        epoch's figure. Measured on 363 ESC-50 clips this reported
        **3,425,472 -> 1,723,145**, which says nothing except that some frames
        were quiet.

        Pooling weights each frame by how much sound was actually in it, which
        is the standard normalized RMSE and is exactly what a scale error shows
        up in: the 35.8x-too-large prediction that motivated the free decoder
        still reads as ~35. Learning itself was never affected -- the delta rule
        uses the raw error and never saw this ratio.
        """
        rng = np.random.default_rng(seed)
        pairs = []
        for c in cochleagrams:
            ctx, tgt = spectrotemporal_context(on_off_channels(c), self.n_lags,
                                               step)
            pairs.extend(zip(ctx, tgt))
        curve = []
        for _ in range(int(epochs)):
            e2 = y2 = 0.0
            for i in rng.permutation(len(pairs)):
                self.learn(pairs[i][0], pairs[i][1], lr=lr)
                e2 += self._e2
                y2 += self._y2
            curve.append(float(np.sqrt(e2 / (y2 + 1e-12))))
        return curve

    # -- the code a downstream area reads ----------------------------------
    def code(self, coch: np.ndarray, step: int = 2) -> np.ndarray:
        """Pool the predictive state over the whole sound.

        The state says *what kind of continuation this is*, so pooling it over
        time gives a description of the sound's dynamics rather than of its
        spectrum."""
        ctx, _ = spectrotemporal_context(on_off_channels(coch), self.n_lags,
                                         step)
        R = np.array([self.encode(c) for c in ctx], np.float32)
        if not len(R):
            return np.zeros(2 * self.n_units, np.float32)
        # mean AND max: what happened on average, and what happened at all
        return _unit(np.concatenate([R.mean(0), R.max(0)]))


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------
@dataclass
class A1Report:
    """Predictive A1 against every earlier auditory representation."""

    raw_cochleagram: float = 0.0
    designed_gabor: float = 0.0
    self_organized_map: float = 0.0
    predictive_a1: float = 0.0
    prediction_error_curve: List[float] = field(default_factory=list)
    rf_ms: float = 0.0

    def summary(self) -> str:
        return (f"raw cochleagram (no filters) : {self.raw_cochleagram:.1%}\n"
                f"designed Gabor belt          : {self.designed_gabor:.1%}\n"
                f"self-organized map           : {self.self_organized_map:.1%}\n"
                f"PREDICTIVE A1 ({self.rf_ms:.0f} ms field): "
                f"{self.predictive_a1:.1%}")


def predictive_a1_experiment(n_units: int = 128, n_lags: int = 32,
                             n_dev: int = 40, n_eval: int = 6, epochs: int = 4,
                             seed: int = 0, verbose: bool = False) -> A1Report:
    """Build a predictive A1 and score it against everything that came before.

    The development set is a **disjoint draw** from the evaluation set, and the
    read-out is the same nearest-prototype rule used for every other auditory
    number in this project, so the four rows are comparable."""
    from .audio import sound_dataset, Cochleagram
    from ..sensing.streams import AuditoryBelt
    from ..learning.selforganize import randomise_filters, develop_v1
    from ..vision.widev1 import _nearest_prototype

    coch = Cochleagram()
    ev, evl, names = sound_dataset(n_per_class=n_eval, seed=5)
    evl = np.asarray(evl)
    C = [coch.forward(s)[0] for s in ev]
    perm = np.random.default_rng(11).permutation(len(C))
    half = len(C) // 2
    tr, te = perm[:half], perm[half:]

    def score(X) -> float:
        X = np.array([_unit(np.asarray(x, np.float32).reshape(-1)) for x in X],
                     np.float32)
        return float(np.mean(_nearest_prototype(X[tr], evl[tr], X[te],
                                                len(names)) == evl[te]))

    rep = A1Report()
    rep.rf_ms = n_lags * coch.hop / coch.sr * 1000.0
    rep.raw_cochleagram = score(C)

    belt = AuditoryBelt(n_freq=coch.n_freq, divisive=True, seed=seed)
    rep.designed_gabor = score([belt.code(c) for c in C])

    dev, _, _ = sound_dataset(n_per_class=n_dev, seed=77)
    devC = [coch.forward(s)[0] for s in dev]

    som = AuditoryBelt(n_freq=coch.n_freq, divisive=True, seed=seed)
    sl = [x for c in (som.adapt(cc) for cc in devC) for x in som.slices(c)]
    randomise_filters(som.layer, seed=seed)
    develop_v1(som.layer, sl, epochs=6, seed=seed)
    rep.self_organized_map = score([som.code(c) for c in C])

    a1 = PredictiveA1(n_freq=coch.n_freq, n_units=n_units, n_lags=n_lags,
                      seed=seed)
    rep.prediction_error_curve = a1.train(devC, epochs=epochs, seed=seed)
    rep.predictive_a1 = score([a1.code(c) for c in C])

    if verbose:
        print(f"   prediction error: " + " -> ".join(
            f"{v:.3f}" for v in rep.prediction_error_curve))
        print()
        print(rep.summary())
    return rep


# ---------------------------------------------------------------------------
# B2 / B3: predicting the LATENT, and a second stage above A1
# ---------------------------------------------------------------------------
class LatentPredictiveA1(PredictiveA1):
    """A1 that predicts its **own future state**, not the future waveform.

    The difference matters and is the point of the experiment. Predicting the
    next cochleagram frame asks the cell to reproduce the raw signal, including
    every part of it that is noise and every part that no history could have
    told you. Predicting the next *latent* asks only that the cell's own
    description of the world stay consistent a moment from now -- which is what
    a cortical area can actually be held to, and what makes a representation
    become invariant rather than merely accurate.

    Implementation: the target is the state this same layer assigns to the
    context ``horizon`` frames ahead. That target is a moving one early in
    development, so it is held **frozen** during each epoch (an old copy of the
    weights supplies it), which is the standard way to keep a self-predicting
    system from collapsing onto a constant.
    """

    def __init__(self, *a, horizon: int = 8, **kw):
        super().__init__(*a, **kw)
        self.horizon = int(horizon)
        self.n_out = self.n_units
        rng = np.random.default_rng(kw.get("seed", 0) + 5)
        self.W_dec = (0.01 * np.abs(rng.normal(0, 1, (self.n_units, self.n_out)))
                      ).astype(np.float32)
        self._frozen: Optional[np.ndarray] = None

    def _target_state(self, context: np.ndarray) -> np.ndarray:
        """The frozen copy's view of a later context -- the prediction target."""
        W = self._frozen if self._frozen is not None else self.W_enc
        x = np.asarray(context, np.float32)
        n = float(np.linalg.norm(x))
        if n > 1e-9:
            x = x / n
        drive = W @ x
        k = max(2, int(self.n_units * self.sparsity))
        idx = np.argpartition(drive, -k)[-k:]
        r = np.zeros(self.n_units, np.float32)
        r[idx] = np.maximum(drive[idx], 0.0)
        m = float(np.linalg.norm(r))
        return (r / m).astype(np.float32) if m > 1e-9 else r

    def train(self, cochleagrams: Sequence[np.ndarray], epochs: int = 4,
              lr: float = 0.03, step: int = 2, seed: int = 0) -> List[float]:
        rng = np.random.default_rng(seed)
        ctxs: List[np.ndarray] = []
        for c in cochleagrams:
            ctx, _ = spectrotemporal_context(on_off_channels(c), self.n_lags,
                                             step)
            ctxs.append(ctx)
        curve = []
        for _ in range(int(epochs)):
            self._frozen = self.W_enc.copy()      # a stable target this epoch
            errs = []
            order = [(a, i) for a, arr in enumerate(ctxs)
                     for i in range(len(arr) - self.horizon)]
            for j in rng.permutation(len(order)):
                a, i = order[j]
                errs.append(self.learn(ctxs[a][i],
                                       self._target_state(ctxs[a][i + self.horizon]),
                                       lr=lr))
            curve.append(float(np.mean(errs)) if errs else 1.0)
        return curve


class PredictiveA2:
    """A second stage over A1: a longer window, and its own prediction.

    A1 sees ~100-400 ms and reports what kind of continuation is happening. A2
    reads a **sequence of A1 states** and predicts how that sequence continues,
    so its window is the sequence's window -- long enough for the identity of a
    whole sound rather than of a fragment.

    This is the standard cortical arrangement (short local fields below, long
    integrative fields above) and the reason for building it is measured rather
    than assumed: one layer with a 128 ms field reached 58.0% where a designed
    bank reached 85.0%, and the shortfall is exactly the kind a single timescale
    produces."""

    def __init__(self, a1: PredictiveA1, n_units: int = 96, n_lags: int = 8,
                 sparsity: float = 0.12, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.a1 = a1
        self.n_units, self.n_lags = int(n_units), int(n_lags)
        self.sparsity = float(sparsity)
        self.n_in = a1.n_units * self.n_lags
        self.W_enc = np.abs(rng.normal(0, 1, (self.n_units, self.n_in))).astype(np.float32)
        self.W_enc /= np.maximum(np.linalg.norm(self.W_enc, axis=1, keepdims=True), 1e-6)
        self.W_dec = (0.01 * np.abs(rng.normal(0, 1, (self.n_units, a1.n_units)))
                      ).astype(np.float32)
        self.decay = 1e-4
        self.duty = np.full(self.n_units, sparsity, np.float32)

    def _a1_sequence(self, coch: np.ndarray, step: int = 2) -> np.ndarray:
        ctx, _ = spectrotemporal_context(on_off_channels(coch), self.a1.n_lags,
                                         step)
        return np.array([self.a1.encode(c) for c in ctx], np.float32)

    def encode(self, seq_window: np.ndarray) -> np.ndarray:
        x = np.asarray(seq_window, np.float32).reshape(-1)
        n = float(np.linalg.norm(x))
        if n > 1e-9:
            x = x / n
        drive = self.W_enc @ x - 0.5 * (self.duty - self.sparsity)
        k = max(2, int(self.n_units * self.sparsity))
        idx = np.argpartition(drive, -k)[-k:]
        r = np.zeros(self.n_units, np.float32)
        r[idx] = np.maximum(drive[idx], 0.0)
        m = float(np.linalg.norm(r))
        return (r / m).astype(np.float32) if m > 1e-9 else r

    def learn(self, window: np.ndarray, target: np.ndarray,
              lr: float = 0.03) -> float:
        r = self.encode(window)
        err = target - self.W_dec.T @ r
        act = r > 0
        if act.any():
            self.W_dec[act] += lr * np.outer(r[act], err)
            self.W_dec *= (1.0 - self.decay)
            np.maximum(self.W_dec, 0.0, out=self.W_dec)
            back = np.maximum(self.W_dec @ err, 0.0) * act
            if back.any():
                x = np.asarray(window, np.float32).reshape(-1)
                n = float(np.linalg.norm(x)) or 1.0
                self.W_enc[act] += lr * np.outer(back[act], x / n)
                np.maximum(self.W_enc, 0.0, out=self.W_enc)
                self.W_enc /= np.maximum(np.linalg.norm(self.W_enc, axis=1,
                                                        keepdims=True), 1e-6)
        self.duty *= (1.0 - lr)
        self.duty += lr * act
        return float(np.linalg.norm(err) / (np.linalg.norm(target) + 1e-9))

    def train(self, cochleagrams: Sequence[np.ndarray], epochs: int = 4,
              lr: float = 0.03, step: int = 2, seed: int = 0) -> List[float]:
        rng = np.random.default_rng(seed)
        seqs = [self._a1_sequence(c, step) for c in cochleagrams]
        curve = []
        for _ in range(int(epochs)):
            pairs = [(s, t) for s, seq in enumerate(seqs)
                     for t in range(self.n_lags, len(seq))]
            e = []
            for j in rng.permutation(len(pairs)):
                s, t = pairs[j]
                e.append(self.learn(seqs[s][t - self.n_lags:t], seqs[s][t],
                                    lr=lr))
            curve.append(float(np.mean(e)) if e else 1.0)
        return curve

    def code(self, coch: np.ndarray, step: int = 2) -> np.ndarray:
        seq = self._a1_sequence(coch, step)
        if len(seq) <= self.n_lags:
            return np.zeros(2 * self.n_units, np.float32)
        R = np.array([self.encode(seq[t - self.n_lags:t])
                      for t in range(self.n_lags, len(seq))], np.float32)
        return _unit(np.concatenate([R.mean(0), R.max(0)]))


# ---------------------------------------------------------------------------
# C1 / C3 / C4: a contrastive objective, sparser code, three-factor rule
# ---------------------------------------------------------------------------
class ContrastivePredictiveA1(PredictiveA1):
    """A1 that asks *is this the same event continuing?* instead of *what comes next?*

    Why the objective had to change
    -------------------------------
    Reconstructive prediction plateaued. Across every variant tried -- signal
    target, latent target, one stage, two stages -- the prediction error stopped
    at ~0.88, explaining about an eighth of the target, and accuracy sat at
    62.0% against 85.0% for the hand-designed bank. Nothing about the
    architecture moved it, which is the signature of a wrong objective rather
    than a wrong network: a decoder asked to reproduce a future vector can
    satisfy itself by getting the scale roughly right, and never has to discover
    *what stays the same* about a sound.

    The contrastive question cannot be satisfied that way. The layer is asked to
    tell a genuine continuation of the sound it is listening to from a
    continuation of some **other** sound. To win that, the code has to carry
    whatever is invariant about this event and discard whatever is not -- which
    is the definition of a representation.

    Why this is still local (and three-factor)
    ------------------------------------------
    The InfoNCE gradient collapses to something a synapse can compute. Writing
    ``p = W_pred z_t`` for the predicted future code and softmax over the
    positive and the negatives:

        delta   = 1 - softmax(p . z_pos)          # a scalar surprise
        dW_pred = lr * delta * outer(z_pos - sum_k w_k z_neg_k, z_t)
        dW_enc  = lr * delta * outer(feedback * (r > 0), x)

    ``delta`` is one number broadcast to the whole layer -- a neuromodulatory
    "that was/was not the real continuation" signal -- and every weight change
    is ``pre x post x delta``. That is exactly the three-factor rule, and it is
    what dopamine and acetylcholine are usually taken to supply.

    Sparsity is also lowered to 5%, from the 12% used before: real A1 population
    activity is nearer 1-5%, and a sparser code separates events better.
    """

    def __init__(self, *a, horizon: int = 8, n_negatives: int = 8,
                 temperature: float = 0.1, sparsity: float = 0.05,
                 init: str = "random", plastic: bool = True, **kw):
        kw["sparsity"] = sparsity
        super().__init__(*a, **kw)
        self.horizon = int(horizon)
        self.n_negatives = int(n_negatives)
        self.temperature = float(temperature)
        self.plastic = bool(plastic)
        if init == "gabor":
            # the developmental prior: fields shaped before any sound arrives
            self.W_enc = spectrotemporal_gabor_bank(
                self.n_units, self.n_freq, self.n_lags, seed=kw.get("seed", 0))
        rng = np.random.default_rng(kw.get("seed", 0) + 11)
        # the predictor maps a code to the code it expects a moment later
        self.W_pred = (0.1 * rng.normal(0, 1, (self.n_units, self.n_units))
                       ).astype(np.float32)
        self.W_dec = np.zeros((self.n_units, 1), np.float32)   # unused here

    def _score(self, p: np.ndarray, z: np.ndarray) -> float:
        return float(p @ z) / self.temperature

    def learn_contrastive(self, ctx: np.ndarray, ctx_pos: np.ndarray,
                          ctx_negs: Sequence[np.ndarray], lr: float = 0.05
                          ) -> float:
        """One contrastive step. Returns the probability given to the truth."""
        z = self.encode(ctx)
        z_pos = self.encode(ctx_pos)
        z_negs = [self.encode(c) for c in ctx_negs]
        p = self.W_pred @ z

        scores = np.array([self._score(p, z_pos)]
                          + [self._score(p, n) for n in z_negs], np.float32)
        scores -= scores.max()
        w = np.exp(scores)
        w /= w.sum()
        delta = float(1.0 - w[0])            # the scalar surprise

        # dW_pred: pull the prediction toward the true continuation and away
        # from the impostors, weighted by how plausible each impostor looked
        target = z_pos - sum(float(w[k + 1]) * n for k, n in enumerate(z_negs))
        self.W_pred += lr * delta * np.outer(target, z)

        # dW_enc: the same scalar gates a Hebbian product at the encoder
        act = z > 0
        if act.any() and self.plastic:
            back = np.maximum(self.W_pred.T @ target, 0.0) * act
            if back.any():
                self._encoder_update(np.asarray(ctx, np.float32), act, back,
                                     lr, delta)
        self.duty *= (1.0 - lr)
        self.duty += lr * act
        return float(w[0])

    # -- where the impostors come from -------------------------------------
    def _contexts(self, cochleagrams: Sequence[np.ndarray], step: int
                  ) -> Tuple[List[np.ndarray], List[int]]:
        ctxs, keep = [], []
        for k, c in enumerate(cochleagrams):
            ctx, _ = spectrotemporal_context(on_off_channels(c), self.n_lags,
                                             step)
            if len(ctx) > self.horizon:
                ctxs.append(ctx)
                keep.append(k)
        return ctxs, keep

    def _negatives(self, ctxs: Sequence[np.ndarray], a: int, i: int,
                   rng, source: str, labels=None) -> List[np.ndarray]:
        """Draw impostors.

        ``source`` decides what the layer is being asked to tell apart, and it
        turns out to decide everything (see the C-series notes):

        ``"cross"``      any other clip -- the standard CPC choice. Solvable by
                         clip-identity nuisance: level, noise instance, room.
        ``"same_class"`` another clip **of the same sound**, so clip identity
                         still helps but the class does not.
        ``"within"``     a distant moment of **this same clip**, so every
                         clip-level cue is shared by truth and impostor and only
                         *where in the event we are* can win.
        ``"mixed"``      half within-clip, half cross.
        """
        negs = []
        n_here = len(ctxs[a])
        # A 400 ms clip at step 3 is only ~27 contexts long, so a margin of
        # 2*horizon leaves no legal moment at all. The first version guarded
        # with ``len > 4*horizon`` and, when that failed, fell through to a
        # cross-clip draw -- silently, so "within" and "cross" produced
        # identical numbers and I nearly reported them as a real result. The
        # margin now scales with what the clip can actually offer.
        margin = max(2, min(2 * self.horizon, n_here // 4))
        for m in range(self.n_negatives):
            use_within = (source == "within"
                          or (source == "mixed" and m % 2 == 0))
            if use_within and n_here > 3 * margin:
                j = int(rng.integers(n_here))
                for _ in range(12):
                    if abs(j - (i + self.horizon)) > margin:
                        break
                    j = int(rng.integers(n_here))
                negs.append(ctxs[a][j])
                continue
            b = int(rng.integers(len(ctxs)))
            for _ in range(16):
                same = labels is not None and labels[b] == labels[a]
                ok = (b != a) and (same if source == "same_class" else True)
                if ok:
                    break
                b = int(rng.integers(len(ctxs)))
            negs.append(ctxs[b][int(rng.integers(len(ctxs[b])))])
        return negs

    def _encoder_update(self, x: np.ndarray, act: np.ndarray,
                        back: np.ndarray, lr: float, delta: float) -> None:
        """Rate-coded three-factor update. E2 replaces this with a timed one."""
        n = float(np.linalg.norm(x)) or 1.0
        self.W_enc[act] += lr * delta * np.outer(back[act], x / n)
        np.maximum(self.W_enc, 0.0, out=self.W_enc)
        self.W_enc /= np.maximum(np.linalg.norm(self.W_enc, axis=1,
                                                keepdims=True), 1e-6)

    def train(self, cochleagrams: Sequence[np.ndarray], epochs: int = 4,
              lr: float = 0.05, step: int = 2, seed: int = 0,
              source: str = "cross", labels=None) -> List[float]:
        """Returns the curve of P(correct continuation) -- chance is 1/(1+K)."""
        rng = np.random.default_rng(seed)
        ctxs, keep = self._contexts(cochleagrams, step)
        if not ctxs:
            return [0.0]
        lab = None if labels is None else [labels[k] for k in keep]
        curve = []
        for _ in range(int(epochs)):
            hits = []
            pairs = [(a, i) for a, arr in enumerate(ctxs)
                     for i in range(len(arr) - self.horizon)]
            for j in rng.permutation(len(pairs)):
                a, i = pairs[j]
                negs = self._negatives(ctxs, a, i, rng, source, lab)
                hits.append(self.learn_contrastive(ctxs[a][i],
                                                   ctxs[a][i + self.horizon],
                                                   negs, lr=lr))
            curve.append(float(np.mean(hits)))
        return curve

    def probe(self, cochleagrams: Sequence[np.ndarray], source: str,
              labels=None, step: int = 2, seed: int = 0, n: int = 400) -> float:
        """P(correct) with NO learning -- what can this code already tell apart?"""
        rng = np.random.default_rng(seed)
        ctxs, keep = self._contexts(cochleagrams, step)
        if not ctxs:
            return 0.0
        lab = None if labels is None else [labels[k] for k in keep]
        out = []
        for _ in range(int(n)):
            a = int(rng.integers(len(ctxs)))
            i = int(rng.integers(len(ctxs[a]) - self.horizon))
            negs = self._negatives(ctxs, a, i, rng, source, lab)
            p = self.W_pred @ self.encode(ctxs[a][i])
            s = np.array([self._score(p, self.encode(ctxs[a][i + self.horizon]))]
                         + [self._score(p, self.encode(c)) for c in negs],
                         np.float32)
            s -= s.max()
            w = np.exp(s)
            out.append(float(w[0] / w.sum()))
        return float(np.mean(out))


# ---------------------------------------------------------------------------
# C2: temporal pooling -- an invariant sound object, built by a slow trace
# ---------------------------------------------------------------------------
class TemporalPool:
    """Slow units above A1 that bind a sound's successive states into one thing.

    The problem left by C1
    ----------------------
    Contrastive A1 learns a good *momentary* code: it can tell a real
    continuation of this sound from an impostor. But a 400 ms event passes
    through many momentary states, and pooling them with a mean and a max --
    which is what :meth:`PredictiveA1.code` does -- treats the sound as a bag of
    instants. Nothing in that ties instant *t* to instant *t+1* as belonging to
    the same object.

    The rule
    --------
    Foldiak's trace rule (1991), which is the standard account of how complex
    cells acquire invariance from simple cells and how view-invariant cells
    appear in IT. The post-synaptic term is not the unit's activity now but a
    trace of its recent activity:

        trace_j <- (1 - a) trace_j + a y_j
        dW_ji    = lr * trace_j * z_i

    Because the trace outlasts the response, a unit that won on frame *t* is
    still partly "on" for the learning step at frame *t+1*, so **whatever the
    input does next gets bound onto the same unit**. Successive states of one
    event are pulled together; states of different events, separated by silence
    and by the trace decaying, are not.

    The trace is the same quantity the synapse module already carries as
    ``calcium`` -- a decaying post-synaptic mark left by activity, which is what
    makes this an STDP-family rule rather than an engineering trick.

    ``slow=False`` disables the trace (``dW = lr * y_j * z_i``, plain
    competitive Hebbian) and is the control that isolates what slowness buys.
    """

    def __init__(self, n_in: int, n_units: int = 64, sparsity: float = 0.10,
                 trace: float = 0.35, slow: bool = True, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n_in, self.n_units = int(n_in), int(n_units)
        self.sparsity, self.slow = float(sparsity), bool(slow)
        self.alpha = float(trace)
        self.W = np.abs(rng.normal(0, 1, (self.n_units, self.n_in))).astype(np.float32)
        self.W /= np.maximum(np.linalg.norm(self.W, axis=1, keepdims=True), 1e-6)
        self.duty = np.full(self.n_units, self.sparsity, np.float32)

    # -- response ----------------------------------------------------------
    def respond(self, z: np.ndarray) -> np.ndarray:
        drive = self.W @ np.asarray(z, np.float32) - 0.5 * (self.duty - self.sparsity)
        k = max(1, int(self.n_units * self.sparsity))
        idx = np.argpartition(drive, -k)[-k:]
        y = np.zeros(self.n_units, np.float32)
        y[idx] = np.maximum(drive[idx], 0.0)
        m = float(np.linalg.norm(y))
        return (y / m).astype(np.float32) if m > 1e-9 else y

    # -- learning ----------------------------------------------------------
    def learn_sequence(self, Z: Sequence[np.ndarray], lr: float = 0.02) -> float:
        """Walk one sound in order. Order is the whole point -- do not shuffle."""
        tr = np.zeros(self.n_units, np.float32)
        stab = []
        prev = None
        for z in Z:
            y = self.respond(z)
            tr = (1.0 - self.alpha) * tr + self.alpha * y if self.slow else y
            post = tr if self.slow else y
            act = post > 0
            if act.any():
                self.W[act] += lr * np.outer(post[act], np.asarray(z, np.float32))
                np.maximum(self.W, 0.0, out=self.W)
                self.W /= np.maximum(np.linalg.norm(self.W, axis=1, keepdims=True), 1e-6)
            self.duty *= (1.0 - lr)
            self.duty += lr * (y > 0)
            if prev is not None:
                stab.append(float(prev @ y))
            prev = y
        return float(np.mean(stab)) if stab else 0.0

    def train(self, sequences: Sequence[Sequence[np.ndarray]], epochs: int = 4,
              lr: float = 0.02, seed: int = 0) -> List[float]:
        """Returns the curve of within-sound stability -- cos(y_t, y_{t+1})."""
        rng = np.random.default_rng(seed)
        curve = []
        for _ in range(int(epochs)):
            s = [self.learn_sequence(sequences[i], lr=lr)
                 for i in rng.permutation(len(sequences))]
            curve.append(float(np.mean(s)))
        return curve

    # -- the code ----------------------------------------------------------
    def code(self, Z: Sequence[np.ndarray]) -> np.ndarray:
        if not len(Z):
            return np.zeros(2 * self.n_units, np.float32)
        Y = np.array([self.respond(z) for z in Z], np.float32)
        return _unit(np.concatenate([Y.mean(0), Y.max(0)]))


def a1_states(a1: PredictiveA1, coch: np.ndarray, step: int = 3) -> np.ndarray:
    """The ordered sequence of A1 states for one sound."""
    ctx, _ = spectrotemporal_context(on_off_channels(coch), a1.n_lags, step)
    return np.array([a1.encode(c) for c in ctx], np.float32)


# ---------------------------------------------------------------------------
# D1: a developmental prior -- Gabor-shaped fields that are still plastic
# ---------------------------------------------------------------------------
def spectrotemporal_gabor_bank(n_units: int, n_freq: int, n_lags: int,
                               seed: int = 0) -> np.ndarray:
    """``(n_units, 2 * n_freq * n_lags)`` non-negative spectrotemporal Gabors.

    A cortex does not start from noise. Orientation columns are present in V1
    before a kitten opens its eyes, and frequency-sweep selectivity is present
    in A1 before it hears much; experience *refines* an inherited layout rather
    than inventing one. This function supplies that layout for A1.

    Each field is a Gabor in the frequency-by-lag plane -- an oriented ripple
    whose tilt is a sweep rate, whose wavelength is a modulation rate, and whose
    envelope sets bandwidth and duration. A tilted field IS a frequency sweep
    detector; a vertical one is an onset detector; a horizontal one is a
    sustained tone detector.

    The weights must stay non-negative, but the fields are signed. Both are
    possible at once because the input has been split into ON and OFF channels:
    the positive lobe is carried by the ON channel and the negative lobe by the
    OFF channel, which is exactly how a real cell with an inhibitory sideband
    is wired. This is the point raised earlier -- that non-negative weights
    forbid OFF subfields -- answered properly rather than by allowing negative
    weights.
    """
    rng = np.random.default_rng(seed)
    ff, tt = np.mgrid[0:n_freq, 0:n_lags].astype(np.float32)
    W = np.zeros((int(n_units), 2, n_freq, n_lags), np.float32)
    for i in range(int(n_units)):
        f0 = rng.uniform(0.15, 0.85) * n_freq
        t0 = rng.uniform(0.25, 0.95) * n_lags     # biased to the recent past
        th = rng.uniform(0.0, np.pi)              # sweep direction
        lam = rng.uniform(0.25, 0.8) * n_freq
        sf = rng.uniform(0.12, 0.30) * n_freq     # bandwidth
        st = rng.uniform(0.15, 0.45) * n_lags     # duration
        phase = rng.choice([0.0, np.pi / 2])
        df, dt = (ff - f0), (tt - t0)
        u = df * np.cos(th) + dt * np.sin(th) * (n_freq / max(n_lags, 1))
        g = (np.exp(-(df ** 2) / (2 * sf ** 2) - (dt ** 2) / (2 * st ** 2))
             * np.cos(2 * np.pi * u / lam + phase))
        g += 0.03 * rng.standard_normal(g.shape)  # no two cells identical
        W[i, 0] = np.maximum(g, 0.0)              # ON  subfield
        W[i, 1] = np.maximum(-g, 0.0)             # OFF subfield
    W = W.reshape(int(n_units), -1)
    W /= np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-6)
    return W


class MultiScaleA1:
    """Several contrastive A1 banks with different integration windows.

    B1 measured that simply lengthening one cell's window makes things worse
    (64 ms 62%, 128 ms 56%, 256 ms 53%), which is easy to misread as "long
    windows do not help". A cortex does not lengthen one cell's window; it keeps
    fast cells and adds slower ones alongside. Best time constants in A1 cluster
    near 20-60 ms, in the belt near 100-300 ms, and later still slower, all
    running at once. This class does the same: separate banks at separate
    windows, read out together.
    """

    def __init__(self, n_freq: int, lags: Sequence[int] = (8, 16, 32),
                 n_units: int = 96, seed: int = 0, **kw):
        self.banks = [ContrastivePredictiveA1(n_freq=n_freq, n_units=n_units,
                                              n_lags=int(L), seed=seed + k, **kw)
                      for k, L in enumerate(lags)]
        self.lags = list(lags)

    def train(self, cochleagrams, epochs: int = 6, lr: float = 0.05,
              step: int = 3, seed: int = 0, **kw) -> List[float]:
        return [b.train(cochleagrams, epochs=epochs, lr=lr, step=step,
                        seed=seed, **kw)[-1] for b in self.banks]

    def code(self, coch: np.ndarray, step: int = 3) -> np.ndarray:
        return _unit(np.concatenate([b.code(coch, step=step) for b in self.banks]))


# ---------------------------------------------------------------------------
# D3: a read-out neuron, so the metric is not doing the deciding
# ---------------------------------------------------------------------------
def linear_probe(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray,
                 yte: np.ndarray, ridge: float = 1e-2) -> float:
    """Accuracy of one layer of weights read off the population.

    Nearest-prototype asks whether the classes are separated *in the code's own
    cosine metric*, which is a strong requirement and not the one a brain has to
    meet: a downstream cell learns its own weights over the population. This is
    that cell -- one linear layer, fitted in closed form -- and it says how much
    class information is present at all, independently of whether the code
    happens to be arranged so that raw similarity finds it.
    """
    Xtr = np.asarray(Xtr, np.float64).reshape(len(Xtr), -1)
    Xte = np.asarray(Xte, np.float64).reshape(len(Xte), -1)
    ytr, yte = np.asarray(ytr), np.asarray(yte)
    classes = np.unique(ytr)
    Y = (ytr[:, None] == classes[None, :]).astype(np.float64)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    d = A.shape[1]
    if d <= len(A):
        W = np.linalg.solve(A.T @ A + ridge * d * np.eye(d), A.T @ Y)
        P = B @ W
    else:                                   # kernel form when d > n
        K = A @ A.T
        al = np.linalg.solve(K + ridge * d * np.eye(len(A)), Y)
        P = (B @ A.T) @ al
    return float(np.mean(classes[np.argmax(P, axis=1)] == yte))


# ---------------------------------------------------------------------------
# E1: A2 -- the same question asked of sequences instead of moments
# ---------------------------------------------------------------------------
class SequenceContrastiveA2(ContrastivePredictiveA1):
    """A second stage that discriminates FUTURE SEGMENTS, not future frames.

    A1 asks "is this the same event continuing?" of one moment. A2 asks it of a
    stretch: it reads a window of A1 states -- roughly 100 ms of cortical
    history -- and has to tell the genuine continuation of that stretch from a
    stretch of some other sound.

    This is deliberately **not** the reconstruction hierarchy that failed three
    times (B3, and the A2 built on the reconstructive A1). Nothing here tries to
    reproduce A1's activity. The objective is the same contrastive one that was
    the only objective in this project ever measured to optimize, applied one
    level up, which is the change the previous hierarchy attempts lacked.
    """

    def __init__(self, a1: ContrastivePredictiveA1, n_units: int = 128,
                 n_states: int = 8, horizon: int = 8, n_negatives: int = 8,
                 temperature: float = 0.1, sparsity: float = 0.05,
                 seed: int = 0):
        super().__init__(n_freq=1, n_units=n_units, n_lags=1, horizon=horizon,
                         n_negatives=n_negatives, temperature=temperature,
                         sparsity=sparsity, seed=seed)
        self.a1 = a1
        self.n_states = int(n_states)
        self.n_in = self.n_states * a1.n_units
        rng = np.random.default_rng(seed + 3)
        self.W_enc = np.abs(rng.normal(0, 1, (self.n_units, self.n_in))).astype(np.float32)
        self.W_enc /= np.maximum(np.linalg.norm(self.W_enc, axis=1, keepdims=True), 1e-6)
        self.W_dec = np.zeros((self.n_units, 1), np.float32)

    def _contexts(self, cochleagrams: Sequence[np.ndarray], step: int
                  ) -> Tuple[List[np.ndarray], List[int]]:
        ctxs, keep = [], []
        for k, c in enumerate(cochleagrams):
            S = a1_states(self.a1, c, step=step)
            if len(S) <= self.n_states + self.horizon:
                continue
            ctxs.append(np.array([S[t - self.n_states:t].reshape(-1)
                                  for t in range(self.n_states, len(S))],
                                 np.float32))
            keep.append(k)
        return ctxs, keep

    def code(self, coch: np.ndarray, step: int = 3) -> np.ndarray:
        S = a1_states(self.a1, coch, step=step)
        if len(S) <= self.n_states:
            return np.zeros(2 * self.n_units, np.float32)
        R = np.array([self.encode(S[t - self.n_states:t].reshape(-1))
                      for t in range(self.n_states, len(S))], np.float32)
        return _unit(np.concatenate([R.mean(0), R.max(0)]))


# ---------------------------------------------------------------------------
# E2: spike timing, and the fourth factor
# ---------------------------------------------------------------------------
class STDPContrastiveA1(ContrastivePredictiveA1):
    """The same layer, but the synapse is changed by TIMING rather than by rate.

    Everything upstream of this class treats the context as a vector: a synapse
    sees "how much" arrived, not "when". But the context is a frequency-by-LAG
    field, so *when* is already written into the input -- column ``l`` of the
    field carries what happened ``(n_lags - l) * hop`` milliseconds ago. That is
    enough to run real spike-timing-dependent plasticity without inventing a
    clock.

    Each afferent is treated as spiking at the time of its own lag column. The
    post-synaptic cell spikes once, at a latency set by its drive -- a strongly
    driven cell fires early, which is the standard latency code in A1. Then, for
    each synapse,

        dt   = t_post - t_pre
        dt > 0  (pre before post, causal)      -> potentiate, A+ exp(-dt/tau+)
        dt < 0  (pre after post, acausal)      -> depress,    A- exp(+dt/tau-)

    and the whole window is multiplied by the same scalar surprise ``delta``
    that C4 introduced. So the update has all four factors the cortex is thought
    to use: **pre-synaptic spike, post-synaptic spike, their relative timing,
    and a neuromodulatory scalar** -- the eligibility-trace-plus-dopamine
    picture, with the tag decaying over the STDP window exactly as in the
    synaptic-tagging model already implemented in :mod:`neurobrain.synapse`.
    """

    def __init__(self, *a, tau_plus: float = 20.0, tau_minus: float = 24.0,
                 a_plus: float = 1.0, a_minus: float = 0.55,
                 hop_ms: float = 4.0, threshold: float = 0.5,
                 axonal_delay: float = 2.0, **kw):
        super().__init__(*a, **kw)
        self.tau_plus, self.tau_minus = float(tau_plus), float(tau_minus)
        self.a_plus, self.a_minus = float(a_plus), float(a_minus)
        self.hop_ms, self.threshold = float(hop_ms), float(threshold)
        self.axonal_delay = float(axonal_delay)
        # Arrival time of every afferent, in ms relative to "now": the lag
        # column it sits in. Shape matches a flattened (2, n_freq, n_lags).
        self.lag_time = -(self.n_lags - 1
                          - np.arange(self.n_lags, dtype=np.float32)) * self.hop_ms
        self.t_pre = np.tile(self.lag_time, 2 * self.n_freq)

    def stdp_window(self, t_post: float) -> np.ndarray:
        dt = t_post - self.t_pre
        w = np.where(dt >= 0.0,
                     self.a_plus * np.exp(-np.abs(dt) / self.tau_plus),
                     -self.a_minus * np.exp(-np.abs(dt) / self.tau_minus))
        return w.astype(np.float32)

    def spike_time(self, cell: int, xn: np.ndarray) -> float:
        """When this cell fires: integrate the field in real time until threshold.

        A cell whose evidence arrives early crosses threshold early, so the
        afferents that arrive AFTER it has fired are acausal for it and get
        depressed. Setting the latency from the drive alone cannot produce that
        -- every afferent is in the past of "now", so every dt comes out
        positive and the depressing half of the window never fires at all. That
        was the first version, and it made STDP into a recency kernel with the
        interesting half missing."""
        c = (self.W_enc[cell].reshape(2, self.n_freq, self.n_lags)
             * xn.reshape(2, self.n_freq, self.n_lags)).sum(axis=(0, 1))
        tot = float(c.sum())
        if tot <= 1e-12:
            return float(self.lag_time[-1] + self.axonal_delay)
        cum = np.cumsum(c) / tot
        l = int(np.searchsorted(cum, self.threshold))
        l = min(l, self.n_lags - 1)
        return float(self.lag_time[l] + self.axonal_delay)

    def _encoder_update(self, x: np.ndarray, act: np.ndarray,
                        back: np.ndarray, lr: float, delta: float) -> None:
        n = float(np.linalg.norm(x)) or 1.0
        xn = (x / n).astype(np.float32)
        for cell in np.flatnonzero(act):
            # pre-synaptic activity gates it: a silent afferent has no spike to
            # be timed, so it gets no change however favourable the window is
            self.W_enc[cell] += (lr * delta * float(back[cell])
                                 * self.stdp_window(self.spike_time(cell, xn))
                                 * xn)
        np.maximum(self.W_enc, 0.0, out=self.W_enc)
        self.W_enc /= np.maximum(np.linalg.norm(self.W_enc, axis=1,
                                                keepdims=True), 1e-6)


# ---------------------------------------------------------------------------
# E4: the positives were the problem, not the layers
# ---------------------------------------------------------------------------
def vocal_tract_warp(coch: np.ndarray, alpha: float) -> np.ndarray:
    """Stretch the frequency axis -- a different-sized vocal tract (E5-A).

    Rolling the spectrum by a couple of bins, which is what the first version
    did, moves everything by the same amount and leaves formant SPACING intact,
    so it barely touches speaker identity: measured, speaker probe went UP
    (54.7% -> 63.3%) alongside digit. What actually distinguishes talkers is the
    length of the vocal tract, which scales formant frequencies multiplicatively
    -- so the transform that models it is a stretch of the frequency axis, not a
    shift. This is vocal tract length perturbation, and on a log-frequency
    cochleagram it is a resampling with factor ``alpha``.
    """
    E = np.asarray(coch, np.float32)
    F = E.shape[0]
    src = np.clip(np.arange(F, dtype=np.float32) / max(float(alpha), 1e-3),
                  0.0, F - 1.0)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, F - 1)
    w = (src - lo).astype(np.float32)[:, None]
    return ((1.0 - w) * E[lo] + w * E[hi]).astype(np.float32)


def speaker_normalize(coch: np.ndarray) -> np.ndarray:
    """Divide out the clip's own long-term spectral envelope (E5-B).

    A talker's vocal tract imposes a fixed spectral colouring on everything they
    say, so it appears in the cochleagram as a per-band constant across the
    whole clip. Removing the per-band long-term average therefore removes most
    of the talker while leaving the movement -- the formant transitions that
    carry the word -- untouched.

    This is not the percentile-floor adaptation that :class:`AuditoryBelt` runs
    (that one tracks the noise floor and is level-driven); it is normalization
    ALONG FREQUENCY by the clip's own average, which is what the classical
    speaker-normalization step does and what cortical adaptation to a sustained
    spectrum would do over a longer window.
    """
    E = np.asarray(coch, np.float32)
    m = E.mean(axis=1, keepdims=True)
    return (E / (m + 1e-6)).astype(np.float32)


def nuisance_transform(coch: np.ndarray, rng, level: float = 6.0,
                       noise: float = 0.08, stretch: float = 0.15,
                       shift: int = 2, warp: float = 0.0) -> np.ndarray:
    """The same sound heard again under conditions the ear is built to discard.

    E3 showed what goes wrong when the only positive available is *the future of
    this same recording*: everything constant within the recording -- who is
    speaking, the room, the level -- is rewarded, and on real speech the layer
    learns the speaker roughly twice as fast as the word (54.7% vs 28.6%).

    The fix cannot be "positive = another clip of the same class", because that
    reads the label and stops being self-supervised. What a developing cortex
    actually gets is the same event under **physical variation it has no reason
    to keep**: loudness changes with distance, noise comes and goes, a talker
    speeds up and slows down, and a voice sits higher or lower in the spectrum.
    Those four are applied here. Each is a transformation the auditory periphery
    already spends machinery discarding -- gain control, adaptation, and the
    tonotopic shift-invariance of a log-frequency axis.
    """
    E = np.asarray(coch, np.float32)
    out = E * float(np.exp(rng.uniform(-1, 1) * np.log(level) / 2))
    if noise > 0:
        out = out + noise * float(out.mean()) * rng.random(out.shape).astype(np.float32)
    if stretch > 0:                                   # a faster or slower talker
        T = out.shape[1]
        f = 1.0 + float(rng.uniform(-stretch, stretch))
        idx = np.clip(np.round(np.arange(T) * f).astype(int), 0, T - 1)
        out = out[:, idx]
    if shift:                                         # a higher or lower voice
        out = np.roll(out, int(rng.integers(-shift, shift + 1)), axis=0)
    if warp > 0:                                      # a different vocal tract
        out = vocal_tract_warp(out, 1.0 + float(rng.uniform(-warp, warp)))
    return out.astype(np.float32)


class InvariantContrastiveA1(ContrastivePredictiveA1):
    """A1 whose positives carry the event but not the nuisance.

    ``positive``:

    ``"future"``   the C1 objective -- the future of this same recording. Keeps
                   whatever is constant about the recording, which is the E3
                   failure.
    ``"nuisance"`` the same moment of the same event under a physical transform
                   the ear discards. Label-free, and the only one of the three
                   a developing animal could actually run.
    ``"label"``    a moment from a different recording **of the same class**.
                   This READS THE LABEL and is therefore not self-supervised; it
                   is here as a ceiling, to say how much of the gap is the
                   objective and how much is everything else.
    """

    def __init__(self, *a, positive: str = "nuisance", n_views: int = 2,
                 warp: float = 0.0, normalize: bool = False, **kw):
        super().__init__(*a, **kw)
        self.positive = str(positive)
        self.n_views = int(n_views)
        self.warp = float(warp)            # E5-A: vocal tract length
        self.normalize = bool(normalize)   # E5-B: divide out the talker first

    def _prep(self, coch: np.ndarray) -> np.ndarray:
        return speaker_normalize(coch) if self.normalize else coch

    def code(self, coch: np.ndarray, step: int = 2) -> np.ndarray:
        return super().code(self._prep(coch), step=step)

    def _views(self, cochleagrams: Sequence[np.ndarray], step: int, rng
               ) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
        """Contexts for the clip and for a nuisance-transformed copy of it."""
        base, alt, keep = [], [], []
        for k, c0 in enumerate(cochleagrams):
            c = self._prep(c0)
            ctx, _ = spectrotemporal_context(on_off_channels(c), self.n_lags, step)
            if len(ctx) <= self.horizon:
                continue
            t = self._prep(nuisance_transform(c0, rng, warp=self.warp))
            ctx2, _ = spectrotemporal_context(on_off_channels(t), self.n_lags, step)
            if len(ctx2) <= self.horizon:
                continue
            base.append(ctx)
            alt.append(ctx2)
            keep.append(k)
        return base, alt, keep

    def train(self, cochleagrams: Sequence[np.ndarray], epochs: int = 4,
              lr: float = 0.05, step: int = 2, seed: int = 0,
              source: str = "cross", labels=None) -> List[float]:
        if self.positive == "future":
            return super().train(cochleagrams, epochs=epochs, lr=lr, step=step,
                                 seed=seed, source=source, labels=labels)
        rng = np.random.default_rng(seed)
        if self.positive == "label" and labels is None:
            raise ValueError("positive='label' is the supervised ceiling and "
                             "needs labels; it is not self-supervised")
        curve = []
        for _ in range(int(epochs)):
            base, alt, keep = self._views(cochleagrams, step, rng)
            if not base:
                return [0.0]
            lab = None if labels is None else [labels[k] for k in keep]
            hits = []
            pairs = [(a, i) for a, arr in enumerate(base) for i in range(len(arr))]
            for j in rng.permutation(len(pairs)):
                a, i = pairs[j]
                if self.positive == "hybrid":
                    # the cortex plausibly has BOTH: continuity in time, and
                    # invariance across hearings of the same kind of event
                    if rng.random() < 0.5 and i + self.horizon < len(base[a]):
                        pos = base[a][i + self.horizon]
                    else:
                        frac = i / max(len(base[a]) - 1, 1)
                        pos = alt[a][min(int(round(frac * (len(alt[a]) - 1))),
                                         len(alt[a]) - 1)]
                elif self.positive == "nuisance":
                    # the SAME moment of the same event, heard differently
                    frac = i / max(len(base[a]) - 1, 1)
                    pos = alt[a][min(int(round(frac * (len(alt[a]) - 1))),
                                     len(alt[a]) - 1)]
                else:                                   # supervised ceiling
                    same = [b for b in range(len(base)) if lab[b] == lab[a] and b != a]
                    if not same:
                        continue
                    b = int(rng.choice(same))
                    pos = base[b][int(rng.integers(len(base[b])))]
                negs = self._negatives(base, a, i, rng, source, lab)
                hits.append(self.learn_contrastive(base[a][i], pos, negs, lr=lr))
            curve.append(float(np.mean(hits)) if hits else 0.0)
        return curve


def adapted_code(a1: PredictiveA1, coch: np.ndarray, step: int = 3,
                 tau: float = 12.0) -> np.ndarray:
    """Pool A1 states after adapting away whatever holds still.

    Speaker identity is nearly constant across a recording; the word is not. A
    cortex removes the sustained and keeps the change -- that is what adaptation
    is for, and it is already in this project as the percentile floor in
    :class:`~neurobrain.streams.AuditoryBelt`. Applied to the A1 state sequence,
    it subtracts a slow running mean, so what survives is how the state MOVED.
    """
    S = a1_states(a1, coch, step=step)
    if not len(S):
        return np.zeros(2 * a1.n_units, np.float32)
    alpha = 1.0 / max(tau, 1.0)
    slow = np.zeros(S.shape[1], np.float32)
    D = np.empty_like(S)
    for t, s in enumerate(S):
        slow = (1.0 - alpha) * slow + alpha * s
        D[t] = s - slow
    return _unit(np.concatenate([np.abs(D).mean(0), D.max(0)]))
