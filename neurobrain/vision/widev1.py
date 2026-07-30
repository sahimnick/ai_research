"""
widev1.py
=========

**Width instead of depth**, and a **50 ms window instead of an instant**.

Where this comes from
---------------------
v0.21 measured something uncomfortable: stacking spiking layers made perception
*worse*. V1->V2->V3 scored 39.2% while a single wide spiking layer scored 63.4%
and the non-spiking vector shortcut scored 89.5%. The representation improved up
the stack (separability +0.144 -> +0.282) but the accuracy did not follow --
each narrow stage threw away more than the next one could rebuild.

The obvious reading is that the layers were too *thin*. Biology does not solve
vision with a deep stack of narrow layers; it solves it with enormous width at
every stage. Roughly 1e6 fibres leave each human optic nerve and roughly 1.4e8
neurons receive them in V1 -- an expansion of about **two orders of magnitude**
at the very first cortical stage. Nothing in this project had ever been that
wide, so "spiking is worse" had never actually been tested against a
biologically-shaped V1.

Three things are changed here, and each is measured on its own:

1. **Width.** A hypercolumn architecture: the image is tiled with 10x10
   receptive fields, and every location gets a *column* of cells that all look
   at the same patch through different filters. At 1024 cells over 49 locations
   that is ~21 cells per location; at 4096 it is ~84. Width is swept
   (256 / 1024 / 4096) so the effect of width alone is visible.

2. **Time.** The code is no longer an instantaneous response but a **windowed
   firing rate** over 50 ms -- the timescale over which a cortical neuron's
   output is actually read by its targets. With membrane noise present (real
   cells sit in a high-conductance state and their rate estimates are noisy), a
   longer window averages that noise down.

   Measured, and it is a **split verdict**. The window does what it was supposed
   to do to noise: code instability falls from 0.028 at 5 ms to 0.008 at 80 ms.
   It does **not** improve static digit accuracy -- 15 ms scores 82.6% +/- 0.4
   and 50 ms scores 82.9% +/- 0.5 over three seeds, which is the same number.
   The window's real payoff is noise and motion, not sharper static vision, and
   claiming otherwise would mean reading a 0.3-point difference against a
   0.5-point spread as a result.

3. **Motion.** A window long enough to contain a *change* can encode the change.
   A fraction of the cells are wired as **Reichardt / Barlow-Levick detectors**:
   they receive their own patch now, multiplied by a neighbouring patch delayed
   by one frame, so the two coincide only when the stimulus sweeps from the
   neighbour toward them at the matching speed. This is measured against a
   static snapshot that *cannot possibly* carry direction, so the test cannot be
   passed by accident.

   Measured: 4-way direction reaches 72.0% against a 28.5% static control
   (chance 25%). The sharper test is left-versus-right, where the two sequences
   contain the **identical set of frames in reversed order** so that nothing but
   temporal order distinguishes them: 75.5% against a ~chance static control.
   The window really does carry movement. But almost all of it comes from the
   window itself -- switching the delay-line detectors off only costs 1.5 points
   (72.0% -> 70.5%). Spike-frequency adaptation, which makes a cell's response
   depend on what it was doing a moment ago, is apparently enough; the explicit
   correlators add little on top.

Local normalization
-------------------
One design change carries much of the result. Previous spiking layers divided
each cell's drive by the **whole population's** activity. Cortex does not: the
normalization pool of a V1 cell is its own neighbourhood (Carandini & Heeger).
Global normalization lets a bright region anywhere in the image suppress cells
everywhere else, which destroys retinotopy -- the same failure that showed up as
the collapsed pooling in :mod:`spikinghierarchy`. Here normalization happens
**inside each hypercolumn**, so cells compete with the other filters looking at
*their* patch and with nobody else, and a separate contrast gate keeps blank
patches from being amplified into noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.neuron import Population


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v.astype(np.float32) if n < 1e-9 else (v / n).astype(np.float32)


# ---------------------------------------------------------------------------
# The wide layer
# ---------------------------------------------------------------------------
class WideV1:
    """A wide, retinotopic, spiking V1: many cells per location, 10x10 fields.

    Parameters
    ----------
    n_cells:
        Total number of Izhikevich cells. They are distributed round-robin over
        the receptive-field locations, so this divided by the number of
        locations is the size of one hypercolumn.
    rf:
        Side of the receptive field in pixels. Kept **wide** (10) on purpose:
        the point of the experiment is more cells looking at the *same* amount
        of retina, not more cells looking at less.
    window_ms:
        Integration window. The code is the firing rate over this window.
    noise:
        Standard deviation of the per-millisecond membrane current noise. Real
        cortical neurons are bombarded continuously and their rates are noisy
        estimates; without this the "a longer window is less noisy" claim would
        be untestable because the simulation would be exactly repeatable.
    motion_fraction:
        Fraction of cells wired as direction detectors (own patch now + a
        neighbouring patch delayed).
    lateral_iters:
        Rounds of recurrent inhibition between neighbouring hypercolumns.
        **Defaults to 0, because measurement said so.** The idea is sound and
        the circuit is real -- see :meth:`_lateral` -- but on this task it costs
        accuracy at every width tried (1024: 79.8% -> 73.4%; 4096: 90.8% ->
        85.9%; 16384: 85.0% -> 85.4%) and it lowers separability every time.
        It is kept, documented and switchable rather than deleted, because the
        negative result is worth as much as a positive one.
    """

    def __init__(self, image_shape: Tuple[int, int] = (28, 28), n_cells: int = 1024,
                 rf: int = 10, stride: int = 3, window_ms: int = 50,
                 noise: float = 1.5, motion_fraction: float = 0.25,
                 lag_ms: int = 4, lateral_iters: int = 0,
                 lateral_keep: float = 0.30, lateral_gain: float = 0.5,
                 lateral_hard: bool = False,
                 seed: int = 0):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.H, self.W_img = image_shape
        self.rf, self.stride = int(rf), int(stride)
        self.n_cells, self.window_ms = int(n_cells), int(window_ms)
        self.noise, self.lag_ms = float(noise), int(lag_ms)
        self.lateral_iters = int(lateral_iters)
        self.lateral_keep = float(lateral_keep)
        self.lateral_gain = float(lateral_gain)
        self.lateral_hard = bool(lateral_hard)

        # -- receptive-field locations: a retinotopic tiling -----------------
        ys = list(range(0, self.H - rf + 1, stride))
        xs = list(range(0, self.W_img - rf + 1, stride))
        self.anchors = np.array([(y, x) for y in ys for x in xs], int)
        self.n_pos = len(self.anchors)
        # rows/cols of the location grid. These must be tracked separately: the
        # anchor index is iy * n_cols + ix, so a single "side" is only correct
        # for a square input. It is not square when this layer is used on a
        # cochleagram (frequency x time), where the partner-location arithmetic
        # would otherwise wrap onto the wrong neighbour.
        self.n_rows, self.n_cols = len(ys), len(xs)
        self.side = len(ys)
        # every cell belongs to one location; consecutive cells fill different
        # columns so a hypercolumn always holds a full set of filters
        self.cell_pos = (np.arange(self.n_cells) % self.n_pos).astype(int)
        self.per_column = self.n_cells // self.n_pos

        self.Wt = self._seed_filters(rng)
        self.duty = np.full(self.n_cells, 0.12, np.float32)
        self.pop = Population(self.n_cells, "regular_spiking", rng=rng,
                              jitter=0.02)
        self.i_floor, self.i_span = 8.0, 30.0

        # -- direction-selective wiring (Reichardt pairing) ------------------
        n_motion = int(self.n_cells * motion_fraction)
        self.motion_cells = rng.permutation(self.n_cells)[:n_motion]
        # each detector's partner sits one step away in one of four directions
        dirs = np.array([(0, 1), (0, -1), (1, 0), (-1, 0)])
        self.cell_dir = np.full(self.n_cells, -1, int)
        self.partner = np.arange(self.n_cells) * 0 + self.cell_pos
        for j, c in enumerate(self.motion_cells):
            d = j % 4
            self.cell_dir[c] = d
            gy, gx = divmod(int(self.cell_pos[c]), self.n_cols)
            py = min(max(gy - int(dirs[d][0]), 0), self.n_rows - 1)
            px = min(max(gx - int(dirs[d][1]), 0), self.n_cols - 1)
            self.partner[c] = py * self.n_cols + px
        self.w_lag = 0.8

    # -- filters ------------------------------------------------------------
    def _seed_filters(self, rng: np.random.Generator) -> np.ndarray:
        """Gabor-seeded receptive fields, varied *within* each hypercolumn.

        A real hypercolumn holds every orientation, both phases and several
        spatial frequencies for one patch of retina. That variety is what makes
        width useful: 84 cells per location are only worth having if they are
        not 84 copies of the same filter."""
        rf = self.rf
        yy, xx = np.mgrid[0:rf, 0:rf] - (rf - 1) / 2.0
        Wt = np.zeros((self.n_cells, rf * rf), np.float32)
        for i in range(self.n_cells):
            j = i // self.n_pos           # index within this cell's column
            n_orient = 12
            th = np.pi * (j % n_orient) / n_orient
            phase = 0.0 if (j // n_orient) % 2 == 0 else np.pi / 2
            lam = 3.0 + 1.8 * ((j // (2 * n_orient)) % 3)
            sigma = rf / 4.0
            if (j // (6 * n_orient)) % 4 == 3:
                # a non-oriented centre-surround cell, as in the V1 blobs
                r2 = xx ** 2 + yy ** 2
                g = np.exp(-r2 / (2 * sigma ** 2)) - 0.6 * np.exp(
                    -r2 / (2 * (1.8 * sigma) ** 2))
            else:
                xr = xx * np.cos(th) + yy * np.sin(th)
                yr = -xx * np.sin(th) + yy * np.cos(th)
                g = (np.exp(-(xr ** 2 + 0.7 * yr ** 2) / (2 * sigma ** 2))
                     * np.cos(2 * np.pi * xr / lam + phase))
            g = g + 0.05 * rng.standard_normal(g.shape)   # no two cells alike
            Wt[i] = np.maximum(g.reshape(-1), 0.0)
        Wt /= np.maximum(np.linalg.norm(Wt, axis=1, keepdims=True), 1e-6)
        return Wt

    # -- the forward path ---------------------------------------------------
    def patches(self, image: np.ndarray) -> np.ndarray:
        """(n_pos, rf*rf) -- what each location sees."""
        img = np.asarray(image, np.float32)
        if img.max() > 1.5:
            img = img / 255.0
        rf = self.rf
        return np.stack([img[y:y + rf, x:x + rf].reshape(-1)
                         for y, x in self.anchors]).astype(np.float32)

    def _raw(self, P: np.ndarray) -> np.ndarray:
        """Filter response of every cell at its own location."""
        return np.einsum("cf,cf->c", self.Wt, P[self.cell_pos])

    def _raw_at(self, P: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """Filter response of every cell at an arbitrary location (for lags)."""
        return np.einsum("cf,cf->c", self.Wt, P[pos])

    def _normalise(self, raw: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Divisive normalization **within the hypercolumn**, then a contrast gate.

        Competition is local: a cell is divided by what the other filters looking
        at *its* patch are doing. Then the whole column is scaled by how much
        contrast its patch actually carries, so an empty corner of the image is
        not amplified into a confident-looking response."""
        z = np.empty_like(raw)
        # per-position mean/std via bincount -- one pass, no Python loop
        cnt = np.bincount(self.cell_pos, minlength=self.n_pos).astype(np.float32)
        s = np.bincount(self.cell_pos, weights=raw, minlength=self.n_pos)
        s2 = np.bincount(self.cell_pos, weights=raw ** 2, minlength=self.n_pos)
        mu = (s / np.maximum(cnt, 1)).astype(np.float32)
        var = np.maximum(s2 / np.maximum(cnt, 1) - mu ** 2, 1e-12).astype(np.float32)
        z = (raw - mu[self.cell_pos]) / np.sqrt(var)[self.cell_pos]

        energy = np.linalg.norm(P, axis=1)
        gate = (energy / (energy.max() + 1e-6)).astype(np.float32)
        return z * gate[self.cell_pos]

    # -- recurrent lateral inhibition ---------------------------------------
    def _lateral(self, z: np.ndarray) -> np.ndarray:
        """Recurrent inhibition **between** neighbouring hypercolumns.

        The divisive normalization above is competition *inside* one column --
        the filters looking at one patch competing with each other. Nothing was
        competing *across* space, so a feature that spans several patches was
        reported by all of them and the code wasted its cells saying the same
        thing many times.

        Real V1 does not work that way. Its inhibitory interneurons reach
        laterally across neighbouring columns, and the effect is recurrent
        rather than one-shot: the population settles over a few milliseconds
        into a state where only the locally strongest responses survive.

        Implemented as ``iters`` rounds of, over each cell's 3x3 spatial
        neighbourhood: subtract a fraction of the neighbourhood's activity
        (shunting inhibition), then keep only the top ``keep_frac`` and silence
        the rest (the winner-take-all an inhibitory microcircuit enforces)."""
        if not self.lateral_iters:
            return z
        nf = max(self.n_cells // self.n_pos, 1)
        n = nf * self.n_pos
        head, tail = z[:n], z[n:]
        g = head.reshape(nf, self.n_rows, self.n_cols)
        for _ in range(self.lateral_iters):
            pos_act = np.maximum(g, 0.0).sum(0)                # (rows, cols)
            pad = np.pad(pos_act, 1, mode="edge")
            nbr = sum(pad[i:i + self.n_rows, j:j + self.n_cols]
                      for i in range(3) for j in range(3)) / 9.0
            g = g - self.lateral_gain * (nbr / max(nf, 1))[None]
            # 3x3 winner-take-all: survive only if you beat your neighbourhood
            gp = np.pad(g, ((0, 0), (1, 1), (1, 1)), mode="edge")
            block = np.stack([gp[:, i:i + self.n_rows, j:j + self.n_cols]
                              for i in range(3) for j in range(3)])
            flat = np.ascontiguousarray(
                block.reshape(-1, self.n_rows, self.n_cols))
            kk = int(round((1.0 - self.lateral_keep) * (flat.shape[0] - 1)))
            thr = np.partition(flat, kk, axis=0)[kk]
            if self.lateral_hard:
                g = np.where(g >= thr[None], g, 0.0)
            else:
                # SOFT threshold, and it matters. Zeroing everything below the
                # neighbourhood's cut-off raised within-class similarity
                # (0.593 -> 0.642) but raised BETWEEN-class similarity more
                # (0.474 -> 0.537), so separability fell (+0.119 -> +0.106) and
                # accuracy with it (76.8% -> 74.0%). A hard mask throws away the
                # graded part of the response, which is where the identity of
                # the stimulus lives. Real inhibition is subtractive and graded;
                # this keeps the ordering while still silencing the losers.
                g = np.maximum(g - thr[None], 0.0)
        return np.concatenate([g.reshape(-1), tail]).astype(np.float32)

    def drive(self, image: np.ndarray) -> np.ndarray:
        """Injected current for a static image."""
        P = self.patches(image)
        z = self._normalise(self._raw(P), P)
        z = self._lateral(z)
        z = z - 2.0 * (self.duty - 0.12)          # homeostatic threshold
        return ((self.i_floor + self.i_span * np.clip(z, 0.0, 3.0) / 3.0)
                * (z > 0)).astype(np.float32)

    # -- temporal integration ----------------------------------------------
    def rate(self, image: np.ndarray, window_ms: Optional[int] = None
             ) -> np.ndarray:
        """**Windowed firing rate** (Hz) for a static image.

        This is the code the rest of the brain reads. It is not a single
        instantaneous response: the cells integrate for the whole window under
        continuous membrane noise, and the rate is the average of what they did.
        """
        return self.rate_over([image], window_ms=window_ms)

    def rate_over(self, frames: Sequence[np.ndarray],
                  window_ms: Optional[int] = None) -> np.ndarray:
        """Windowed firing rate (Hz) over a **sequence** of frames.

        The window is divided among the frames and the population runs straight
        through without resetting, so the rate integrates the whole trajectory.
        Direction-selective cells add their partner location's response from
        ``lag_ms`` earlier: when the stimulus is sweeping toward them at the
        matching speed the delayed and the direct input arrive together and the
        cell fires hard; in the opposite direction they never coincide."""
        T = int(window_ms or self.window_ms)
        n_f = len(frames)
        Ps = [self.patches(f) for f in frames]
        # Lateral inhibition runs ONCE PER FRAME, not once per millisecond.
        # It is a spatial computation over a drive that does not change within a
        # frame, so doing it inside the time loop was 50x the work AND wrong
        # about the timescale: recurrent inhibition settles in a few ms and then
        # holds. Profiled, it was the whole cost -- 57 ms per image against 7.5.
        base = np.stack([self._lateral(self._normalise(self._raw(P), P))
                         for P in Ps])
        lagged = np.stack([self._lateral(
            self._normalise(self._raw_at(P, self.partner), P)) for P in Ps])

        self.pop.reset()
        counts = np.zeros(self.n_cells, np.float32)
        has_dir = self.cell_dir >= 0
        for t in range(T):
            f = min(int(t * n_f / T), n_f - 1)
            z = base[f].copy()
            tl = t - self.lag_ms
            if tl >= 0 and has_dir.any():
                fl = min(int(tl * n_f / T), n_f - 1)
                # A Reichardt detector CORRELATES; it does not add. A sum fires
                # for either input alone and so is not direction-selective at
                # all -- measured, it scored *below* the plain layer. The
                # product fires only when the delayed neighbour signal and the
                # direct one arrive together, which happens for one direction
                # and one speed. Supralinear coincidence like this is what a
                # dendritic NMDA plateau does (cf. :mod:`dendrite`).
                coinc = (np.maximum(z, 0.0)
                         * np.maximum(lagged[fl], 0.0))
                z[has_dir] += self.w_lag * coinc[has_dir]
            z = z - 2.0 * (self.duty - 0.12)
            I = ((self.i_floor + self.i_span * np.clip(z, 0.0, 3.0) / 3.0)
                 * (z > 0)).astype(np.float32)
            if self.noise > 0:
                I = I + self.rng.normal(0.0, self.noise, self.n_cells
                                        ).astype(np.float32)
            counts += self.pop.step(I).astype(np.float32)
        return (counts * (1000.0 / T)).astype(np.float32)     # spikes/s

    # -- learning -----------------------------------------------------------
    def learn(self, image: np.ndarray, rate: np.ndarray, lr: float = 0.02
              ) -> None:
        """Competitive Hebbian refinement of the seeded filters.

        Only the cells that actually fired move, and each moves toward the patch
        *it* was looking at -- learning is local to the receptive field, as it
        must be when every cell sees a different part of the image."""
        fired = rate > 0
        if not fired.any():
            return
        P = self.patches(image)
        x = P[self.cell_pos[fired]]
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-6)
        w = (rate[fired] / max(float(rate.max()), 1.0))[:, None]
        self.Wt[fired] += lr * w * (x - self.Wt[fired])
        np.maximum(self.Wt[fired], 0.0, out=self.Wt[fired])
        self.Wt[fired] /= np.maximum(
            np.linalg.norm(self.Wt[fired], axis=1, keepdims=True), 1e-6)
        self.duty *= (1.0 - lr)
        self.duty[fired] += lr

    def train(self, images: Sequence[np.ndarray], epochs: int = 1,
              lr: float = 0.02, seed: int = 0) -> "WideV1":
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            for i in rng.permutation(len(images)):
                self.learn(images[i], self.rate(images[i]), lr=lr)
        return self

    # -- read-outs ----------------------------------------------------------
    def code(self, image: np.ndarray) -> np.ndarray:
        return _unit(self.rate(image))

    # Same name the spiking hierarchy exposes, so this layer is a drop-in front
    # end anywhere that one is used (see :mod:`endtoend`).
    def top_code(self, image: np.ndarray) -> np.ndarray:
        return self.code(image)

    # -- complex cells: pooling away one axis of the location grid ----------
    def pooling_index(self, axis: str = "rows") -> Tuple[np.ndarray, int]:
        """Group cells so that one axis of the retinotopic grid is pooled away.

        This is the **complex cell** operation: sum over cells that share a
        filter but sit at different locations along one axis, and the code stops
        caring *where* along that axis the feature was, while still caring which
        feature it is.

        It matters most when this layer is used on a cochleagram, where the
        vertical axis is frequency. Sound classes here differ by *pattern* -- a
        rising sweep, an amplitude modulation -- while their absolute pitch is
        randomised over an octave and a half, so a frequency-specific code
        cannot generalise. Measured on the 8-class sound set: position-specific
        37.5%, pooled across frequency **75.0%** (raw cochleagram 25.0%,
        chance 12.5%).

        Returns ``(group_of_cell, n_groups)``."""
        rows = self.cell_pos // self.n_cols
        cols = self.cell_pos % self.n_cols
        filt = np.arange(self.n_cells) // self.n_pos
        if axis == "both":
            # keep only WHICH filter fired -- fully shift-invariant. Measured on
            # the 8-class sound set this beat pooling frequency alone on both
            # conditions (pre-cut 75.0% -> 87.5%, streaming 33.3% -> 46.7%),
            # because the detected onset is never exactly on time and a
            # time-specific code pays for every millisecond of that error.
            key = filt.astype(np.int64)
        else:
            keep = cols if axis == "rows" else rows
            key = keep.astype(np.int64) * (filt.max() + 2) + filt
        _, inv = np.unique(key, return_inverse=True)
        return inv.astype(int), int(inv.max()) + 1

    def pooled_code(self, rate: np.ndarray, index: np.ndarray, n_groups: int
                    ) -> np.ndarray:
        """Apply a :meth:`pooling_index` to a rate vector."""
        return _unit(np.bincount(index, weights=np.asarray(rate, np.float64),
                                 minlength=n_groups).astype(np.float32))

    def sparsity(self, images: Sequence[np.ndarray]) -> float:
        """Fraction of cells silent for a typical image."""
        return float(np.mean([(self.rate(im) == 0).mean() for im in images]))


# ---------------------------------------------------------------------------
# Motion stimuli
# ---------------------------------------------------------------------------
DIRECTIONS = ("right", "left", "down", "up")
_SHIFT = {"right": (0, 1), "left": (0, -1), "down": (1, 0), "up": (-1, 0)}


def drift_sequence(image: np.ndarray, direction: str, n_frames: int = 6,
                   speed: int = 1) -> List[np.ndarray]:
    """A stimulus sweeping across the retina, centred on the same middle frame.

    The sequence is built so that **its middle frame is identical for every
    direction**. A static snapshot therefore carries exactly zero information
    about which way the thing is moving, which is what makes the comparison
    below a real test rather than a demonstration."""
    dy, dx = _SHIFT[direction]
    mid = (n_frames - 1) / 2.0
    out = []
    for i in range(n_frames):
        k = int(round((i - mid) * speed))
        out.append(np.roll(np.roll(np.asarray(image), k * dy, axis=0),
                           k * dx, axis=1))
    return out


def static_middle(image: np.ndarray, n_frames: int = 6) -> List[np.ndarray]:
    """The same stimulus, held still -- the control for the motion test."""
    return [np.asarray(image)] * n_frames


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
class PopulationAdaptation:
    """What a cell stops transmitting: the part of its input that never changes.

    Measured problem. On MNIST the wide V1 code is excellent -- 1-NN 0.818 over
    ten digits. On CIFAR-10 photographs the *same* layer scores 1-NN **0.168**
    against a chance of 0.100, and raw opponent pixels beat it at 0.308. A
    spiking layer of 1024 cells doing worse than the pixels it reads is not a
    hard-problem result; it is a coding fault.

    The fault is a **common component**. 63.5% of cells are silent for a typical
    photograph and the rest respond similarly to all of them, so the cosine
    between any two codes is dominated by the part they share and neighbourhood
    structure disappears into it. MNIST never showed this because a digit is a
    high-contrast figure on an empty field, which is already mean-subtracted by
    construction; a photograph is not.

    Subtracting the population's running mean fixes it -- 1-NN 0.168 -> 0.313,
    5-NN 0.168 -> 0.320, matching raw pixels while keeping the prototype read-out
    (0.363 -> 0.338 subtractive, 0.387 with the divisive term as well).

    This is not a normalisation trick bolted on; it is the operation the rest of
    the project already runs everywhere else and vision was the one place
    missing it. :class:`AuditoryBelt` subtracts a per-band floor for exactly this
    reason ("a channel that stops transmitting its own long-run floor stops
    transmitting the background with it"), and
    :class:`~neurobrain.workspace.GlobalWorkspace` takes ``adapt=True``.

    Kept **online** -- a running mean and variance updated per code, not batch
    statistics -- because a mind meets its world one frame at a time and never
    has the dataset. ``divisive=True`` adds the per-cell gain control, which is
    better for class-mean read-outs and slightly worse for nearest-neighbour;
    the default is subtractive because the association area stores exemplars.
    """

    def __init__(self, n: int, tau: float = 0.01, divisive: bool = False,
                 eps: float = 1e-3):
        self.mu = np.zeros(int(n), np.float32)
        self.var = np.ones(int(n), np.float32)
        self.tau = float(tau)          # how fast the cell forgets its baseline
        self.divisive = bool(divisive)
        self.eps = float(eps)
        self.n_seen = 0

    def observe(self, r: np.ndarray) -> None:
        """One code goes past; the baseline drifts toward it."""
        r = np.asarray(r, np.float32)
        # a fast start, so the first few codes are not compared against zeros
        a = max(self.tau, 1.0 / (self.n_seen + 1))
        d = r - self.mu
        self.mu += a * d
        self.var += a * (d * d - self.var)
        self.n_seen += 1

    def apply(self, r: np.ndarray) -> np.ndarray:
        """The code with its baseline removed -- what actually differs."""
        out = np.asarray(r, np.float32) - self.mu
        if self.divisive:
            out = out / (np.sqrt(self.var) + self.eps)
        return out.astype(np.float32)

    def __call__(self, r: np.ndarray, learn: bool = True) -> np.ndarray:
        if learn:
            self.observe(r)
        return self.apply(r)


def _nearest_prototype(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray,
                       n_class: Optional[int] = None) -> np.ndarray:
    """Class-mean read-out. Deliberately the simplest possible decoder so the
    number reflects the *code*, not a classifier's cleverness.

    Note this one **is** supervised -- the labels build the prototypes -- so it
    is reported only as a secondary figure. The project's standard rule is
    :func:`grown_readout` below.

    **Returns label values, not row indices.** This used to iterate
    ``range(n_class)`` and so silently assumed labels were ``0..n_class-1``.
    Handing it ESC-50's *nature* classes (labels 10-19) with ``n_class=10``
    built ten all-zero prototypes and scored **exactly 0.000** -- not an
    informative failure but a bug wearing a result's clothes. Class identity is
    now taken from the labels themselves whenever they fall outside
    ``range(n_class)``, so any label set works and the old callers, whose
    labels *are* ``0..n_class-1``, are unchanged."""
    ytr = np.asarray(ytr)
    present = np.unique(ytr)
    if (n_class is not None and len(present)
            and present.min() >= 0 and present.max() < n_class):
        classes = np.arange(n_class)      # legacy contract: index == label
    else:
        classes = present                 # arbitrary label set
    proto = np.stack([_unit(Xtr[ytr == c].mean(0)) if (ytr == c).any()
                      else np.zeros(Xtr.shape[1], np.float32)
                      for c in classes])
    return classes[(Xte @ proto.T).argmax(1)]


# The ladder has to reach low as well as high. Dense rate codes need the top
# of it; the SPARSE codes of the global workspace sit at cosine ~0.08 to each
# other once adaptation removes their common component, and every rung above
# 0.6 gives each image its own category -- so with a ladder starting at 0.60
# nothing qualified and the read-out returned 0 categories and 0% accuracy.
VIGILANCE_LADDER = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50,
                    0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)


def grown_readout(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray,
                  yte: np.ndarray, max_frac: float = 1.0 / 3.0
                  ) -> Tuple[float, int, float]:
    """The project's standard read-out: categories grow, labels only *name* them.

    Returns ``(accuracy, n_categories, vigilance)``.

    Two rules keep this honest, and both matter:

    * **Labels never train.** Category cells are grown from the code alone by
      ART-style vigilance; the labels are applied afterwards, by majority vote,
      to say what each cluster turned out to be.
    * **Compression is capped.** Raising vigilance far enough makes every
      training example its own category, at which point the "category system" is
      just a nearest-neighbour table over the training set and its accuracy says
      nothing about whether the code is any good. (Measured: at vigilance 0.92
      this code grew 2347 categories from 2500 images and scored 91.6% -- a
      number that must not be quoted.) So vigilance is raised only while the
      category count stays below ``max_frac`` of the training set, and the
      count is returned with the accuracy so the compression is always visible.

    Vigilance is chosen by the category count alone, which uses no labels.
    """
    from .spikingvision import SpikingCategoryMap

    best = (0.0, 0, 0.0)
    for v in VIGILANCE_LADDER:
        m = SpikingCategoryMap(dim=Xtr.shape[1], vigilance=v, max_cells=6000)
        for x in Xtr:
            m.learn(x)
        if m.n_categories <= max_frac * len(Xtr):
            m.name_cells(Xtr, ytr)
            best = (float(np.mean(m.recognise(Xte) == yte)), m.n_categories, v)
    return best


@dataclass
class WideV1Report:
    """What the wide layer achieved, per condition."""

    width_accuracy: Dict[int, float] = field(default_factory=dict)
    width_categories: Dict[int, int] = field(default_factory=dict)
    window_accuracy: Dict[int, float] = field(default_factory=dict)
    window_cv: Dict[int, float] = field(default_factory=dict)
    motion_moving: float = 0.0
    motion_static: float = 0.0
    motion_no_detectors: float = 0.0
    best_width: int = 0
    best_accuracy: float = 0.0
    sparsity: float = 0.0


def _encode(layer: WideV1, images: Sequence[np.ndarray],
            window_ms: Optional[int] = None) -> np.ndarray:
    return np.array([_unit(layer.rate(im, window_ms=window_ms))
                     for im in images], np.float32)


def width_ablation(widths: Sequence[int] = (256, 1024, 4096),
                   n_train: int = 2500, n_test: int = 1000,
                   window_ms: int = 50, n_fit: int = 400,
                   verbose: bool = False) -> Dict[int, Dict[str, float]]:
    """Does making V1 four times wider actually buy anything?

    Same data, same window, same read-outs, same receptive-field size; only the
    number of cells per hypercolumn changes."""
    from ..sensing.realworld import load_mnist

    trx, trY, tex, teY = load_mnist(n_train=n_train, n_test=n_test)
    out: Dict[int, Dict[str, float]] = {}
    for w in widths:
        layer = WideV1(n_cells=w, window_ms=window_ms, seed=0)
        layer.train(trx[:n_fit], epochs=1)
        Xtr, Xte = _encode(layer, trx), _encode(layer, tex)
        acc, ncat, vig = grown_readout(Xtr, trY, Xte, teY)
        cen = float(np.mean(_nearest_prototype(Xtr, trY, Xte, 10) == teY))
        out[w] = {"grown": acc, "centroid": cen, "categories": float(ncat),
                  "vigilance": vig, "per_column": float(layer.per_column),
                  "silent": layer.sparsity(tex[:60])}
        if verbose:
            print(f"   V1 width {w:>5} cells "
                  f"({layer.per_column:>3}/hypercolumn x {layer.n_pos} "
                  f"locations): grown {acc:.1%} ({ncat} categories), "
                  f"centroid {cen:.1%}")
    return out


# -- what actually earned the improvement ------------------------------------

def architecture_decomposition(n_train: int = 2500, n_test: int = 1000,
                               n_fit: int = 400, verbose: bool = False
                               ) -> Dict[str, Dict[str, float]]:
    """Change one thing at a time and see which change carried the result.

    The user's three directives were width, a longer window, and better
    plasticity. Applying all of them at once and reporting the total would hide
    which of them worked, so they are separated here, each measured against the
    same data with the same read-out:

      1. the v0.20 spiking layer as it was -- 220 cells, whole-image receptive
         field, global divisive normalization, 24 ms;
      2. the same layer widened to 1024 cells;
      3. the same layer with the window stretched to 50 ms;
      4. the wide V1 of this module: 10x10 local receptive fields tiled
         retinotopically, normalization pooled *within* the hypercolumn.

    The baseline is **re-measured here** rather than quoted from v0.20, because
    the training-set size and the vigilance protocol both changed; comparing to
    the old published 63.4% directly would be comparing two different
    experiments."""
    from ..sensing.realworld import load_mnist
    from .spikingvision import SpikingFeatureLayer, SpikingRetina

    trx, trY, tex, teY = load_mnist(n_train=n_train, n_test=n_test)

    def old_layer(n_cells: int, window: int) -> Tuple[np.ndarray, np.ndarray]:
        r = SpikingRetina()
        f = SpikingFeatureLayer(n_in=trx[0].size, n_cells=n_cells,
                                window=window, seed=0)
        f.train(trx[:n_fit], r, epochs=1)
        enc = lambda ims: np.array(
            [_unit(f.spike_counts(r.currents(im))) for im in ims], np.float32)
        return enc(trx), enc(tex)

    rows: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
        "1_baseline_220_24ms": old_layer(220, 24),
        "2_plus_width_1024_24ms": old_layer(1024, 24),
        "3_plus_window_1024_50ms": old_layer(1024, 50),
    }
    wide = WideV1(n_cells=1024, window_ms=50, seed=0)
    wide.train(trx[:n_fit], epochs=1)
    rows["4_plus_local_rf_and_norm"] = (_encode(wide, trx), _encode(wide, tex))

    out: Dict[str, Dict[str, float]] = {}
    for name, (A, B) in rows.items():
        acc, ncat, vig = grown_readout(A, trY, B, teY)
        out[name] = {"grown": acc, "categories": float(ncat),
                     "centroid": float(np.mean(
                         _nearest_prototype(A, trY, B, 10) == teY))}
        if verbose:
            print(f"   {name:<28} grown {acc:.1%} ({ncat:>4} cat)  "
                  f"centroid {out[name]['centroid']:.1%}")
    return out


def window_ablation(windows: Sequence[int] = (5, 15, 50), n_cells: int = 1024,
                    n_train: int = 2500, n_test: int = 1000, n_fit: int = 400,
                    n_repeat: int = 8, verbose: bool = False
                    ) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Does a longer integration window really reduce the noise, and help?

    Two things are measured for each window length:

      * **code instability** -- present the *same* image several times and see
        how far the percept moves: ``1 - mean cosine`` between repeats. This is
        the direct measurement of spike noise, and it is only meaningful because
        the cells carry membrane noise.

        A per-cell coefficient of variation was tried first and had to be
        thrown away: it averages over whichever cells happen to be firing, and
        a longer window wakes up more marginal cells, so the *set* being
        averaged changes with the very thing being measured. It duly reported
        that 50 ms was noisier than 15 ms. The whole-code distance has no such
        selection effect -- every cell is in it at every window.
      * held-out accuracy with the same read-out as everywhere else.
    """
    from ..sensing.realworld import load_mnist

    trx, trY, tex, teY = load_mnist(n_train=n_train, n_test=n_test)
    acc: Dict[int, float] = {}
    cv: Dict[int, float] = {}
    for T in windows:
        # Each condition gets its OWN layer, trained and read at its own window.
        # Sharing one layer trained at the longest window silently favours that
        # window -- the filters were refined under it -- so the comparison would
        # not be about the window at all.
        layer = WideV1(n_cells=n_cells, window_ms=T, seed=0)
        layer.train(trx[:n_fit], epochs=1)
        drift = []
        for im in tex[:24]:
            reps = np.stack([_unit(layer.rate(im, window_ms=T))
                             for _ in range(n_repeat)])
            S = reps @ reps.T
            iu = np.triu_indices(n_repeat, 1)
            drift.append(1.0 - float(S[iu].mean()))
        cv[T] = float(np.mean(drift))
        Xtr, Xte = _encode(layer, trx, T), _encode(layer, tex, T)
        acc[T] = grown_readout(Xtr, trY, Xte, teY)[0]
        if verbose:
            print(f"   window {T:>3} ms: accuracy {acc[T]:.1%}, "
                  f"code CV {cv[T]:.3f}")
    return acc, cv


def motion_experiment(n_cells: int = 1024, n_train: int = 400, n_test: int = 200,
                      window_ms: int = 50, n_frames: int = 6,
                      speed: Optional[int] = None, lag_ms: Optional[int] = None,
                      seed: int = 0, verbose: bool = False) -> Dict[str, float]:
    """Can the 50 ms window carry **movement**, which an instant cannot?

    Digits drift in one of four directions. Because every sequence is built
    around the *same* middle frame, a static snapshot contains no direction
    information at all -- the static control is a hard floor at chance, not a
    weak baseline. Three conditions:

      ``moving``         windowed rate over the drifting sequence, with the
                         Reichardt-paired direction cells wired in;
      ``no_detectors``   the same drifting sequence with the delayed pairing
                         switched off, so only the cells' own adaptation could
                         carry direction;
      ``static``         the middle frame held still -- the impossibility check.

    Speed and lag are **matched to the geometry** rather than picked freely. A
    detector's partner location sits ``stride`` pixels away, so the stimulus is
    drifted ``stride`` pixels per frame and the delay is set to one frame
    duration; then the neighbour's signal and the direct one land together for
    exactly one direction. Measured with a mismatched lag (4 ms against a 25 ms
    crossing time) the detectors did nothing at all -- the delayed input was
    simply the same frame again -- which is the honest reason these two numbers
    are derived and not tuned."""
    from ..sensing.realworld import load_mnist

    trx, _, tex, _ = load_mnist(n_train=n_train, n_test=n_test)
    probe = WideV1(n_cells=8, seed=0)
    speed = int(speed if speed is not None else probe.stride)
    lag_ms = int(lag_ms if lag_ms is not None else round(window_ms / n_frames))

    def dataset(layer: WideV1, images, rs, moving: bool):
        X, y = [], []
        for im in images:
            d = int(rs.integers(4))
            frames = (drift_sequence(im, DIRECTIONS[d], n_frames, speed)
                      if moving else static_middle(im, n_frames))
            X.append(_unit(layer.rate_over(frames, window_ms=window_ms)))
            y.append(d)
        return np.array(X, np.float32), np.array(y)

    def score(moving: bool, detectors: bool) -> float:
        layer = WideV1(n_cells=n_cells, window_ms=window_ms, seed=0,
                       lag_ms=lag_ms,
                       motion_fraction=0.6 if detectors else 0.0)
        Xtr, ytr = dataset(layer, trx, np.random.default_rng(seed), moving)
        Xte, yte = dataset(layer, tex, np.random.default_rng(seed + 1), moving)
        return float(np.mean(_nearest_prototype(Xtr, ytr, Xte, 4) == yte))

    out = {"moving": score(True, True),
           "no_detectors": score(True, False),
           "static": score(False, True),
           "chance": 0.25}
    out["motion_gain"] = out["moving"] - out["static"]
    out["detector_gain"] = out["moving"] - out["no_detectors"]
    if verbose:
        print(f"   drifting + direction detectors : {out['moving']:.1%}")
        print(f"   drifting, detectors off        : {out['no_detectors']:.1%}")
        print(f"   static middle frame (control)  : {out['static']:.1%} "
              f"(chance 25%)")
    return out


@dataclass
class WideDigitBrain:
    """A digit recogniser whose perception is a wide, spiking, retinotopic V1."""

    layer: WideV1
    cortex: object                       # SpikingCategoryMap
    accuracy: float = 0.0
    n_categories: int = 0
    vigilance: float = 0.0
    silent_fraction: float = 0.0

    def code(self, image: np.ndarray) -> np.ndarray:
        return self.layer.code(image)

    def recognise(self, image: np.ndarray) -> int:
        return int(self.cortex.recognise(self.code(image)[None])[0])


def build_wide_digit_brain(n_cells: int = 1024, window_ms: int = 50,
                           n_train: int = 2500, n_test: int = 1000,
                           n_fit: int = 400, verbose: bool = False
                           ) -> WideDigitBrain:
    """Train the wide spiking V1 on REAL MNIST and score it under the house rule.

    Labels only *name* the categories that grew by themselves, and vigilance is
    capped so the category count stays a real compression of the training set --
    see :func:`grown_readout` for why both rules matter."""
    from ..sensing.realworld import load_mnist
    from .spikingvision import SpikingCategoryMap

    trx, trY, tex, teY = load_mnist(n_train=n_train, n_test=n_test)
    layer = WideV1(n_cells=n_cells, window_ms=window_ms, seed=0)
    if verbose:
        print(f"   {n_cells} cells, {layer.per_column}/hypercolumn over "
              f"{layer.n_pos} locations, {layer.rf}x{layer.rf} fields, "
              f"{window_ms} ms window")
    layer.train(trx[:n_fit], epochs=1)
    Xtr, Xte = _encode(layer, trx), _encode(layer, tex)
    acc, ncat, vig = grown_readout(Xtr, trY, Xte, teY)

    cortex = SpikingCategoryMap(dim=Xtr.shape[1], vigilance=vig, max_cells=6000)
    for x in Xtr:
        cortex.learn(x)
    cortex.name_cells(Xtr, trY)
    brain = WideDigitBrain(layer, cortex, acc, ncat, vig,
                           layer.sparsity(tex[:60]))
    if verbose:
        print(f"   held-out accuracy : {acc:.1%}")
        print(f"   categories grown  : {ncat} from {len(trx)} images "
              f"({len(trx) / max(ncat, 1):.1f}x compression, vigilance {vig})")
        print(f"   cells silent for an image: {brain.silent_fraction:.0%}")
    return brain


def build_wide_v1_report(n_train: int = 2500, n_test: int = 1000,
                         verbose: bool = False) -> WideV1Report:
    """Run every wide-V1 measurement and collect it in one honest report."""
    def say(*a):
        if verbose:
            print(*a)

    rep = WideV1Report()
    say("width ablation (256 / 1024 / 4096 cells, 10x10 fields) ...")
    w = width_ablation(n_train=n_train, n_test=n_test, verbose=verbose)
    rep.width_accuracy = {k: v["grown"] for k, v in w.items()}
    rep.width_categories = {k: int(v["categories"]) for k, v in w.items()}
    rep.best_width = max(rep.width_accuracy, key=rep.width_accuracy.get)
    rep.best_accuracy = rep.width_accuracy[rep.best_width]
    rep.sparsity = float(np.mean([v["silent"] for v in w.values()]))

    say("temporal-window ablation (5 / 15 / 50 ms) ...")
    rep.window_accuracy, rep.window_cv = window_ablation(
        n_train=n_train, n_test=n_test, verbose=verbose)

    say("motion coding ...")
    m = motion_experiment(verbose=verbose)
    rep.motion_moving = m["moving"]
    rep.motion_static = m["static"]
    rep.motion_no_detectors = m["no_detectors"]
    return rep
