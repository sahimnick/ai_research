"""
streams.py
==========

The brain against **continuous, unsegmented, real-world sensory streams** --
a moving eye on a large cluttered scene, and an ear on a soundscape that nobody
has cut into clips.

Why this is the hard test
-------------------------
Everything measured up to v0.22 was fed **pre-segmented** input: a 28x28 image
containing exactly one centred digit, or a 400 ms waveform containing exactly
one sound. Both of those are gifts. Real sensation arrives as an unbroken
stream in which *the boundaries are not given*: an eye lands somewhere on a
large scene and has to work out whether it is looking at an object at all, and
an ear receives a continuous pressure wave in which events start and stop
without announcement.

A system that scores well on cut-up data and collapses on a stream has not
solved perception; it has solved a dataset. So the same machinery is run here
with nothing pre-cut, and the cost of that is measured explicitly.

What is built
-------------
:class:`SaccadicEye`
    A real eye, not a crop. It has a **fovea** (full resolution over a small
    patch) and a **periphery** (heavily blurred and downsampled over the whole
    scene), it chooses where to look next from a peripheral **saliency** map,
    and it never stops moving: during each fixation the image drifts by a pixel
    or two (**fixational drift** -- real eyes do this, and vision fades without
    it). Because the drift happens *inside* the 50 ms window, the stream feeds
    :meth:`~neurobrain.widev1.WideV1.rate_over` directly.

:class:`ContinuousEar`
    A cochlea sliding along a long waveform in fixed hops. It receives a
    soundscape built from **real sound events** dropped at random times into
    background noise, sometimes overlapping, with silence in between -- and it
    is not told when anything begins. Onsets are found the way the auditory
    system finds them, from a **rising energy derivative** with a refractory
    period, and the events it finds are then identified.

:class:`StreamingBrain`
    Both senses on one clock, projecting into the one shared cortical code from
    :mod:`workspace`. When the eye is fixating an object at the same moment the
    ear hears an event, the two codes are bound. The test afterwards is
    cross-modal: play the **sound alone** and ask what was seen with it.

Every number this module reports is measured against the corresponding
pre-segmented condition, so the cost of streaming is never hidden inside a
total.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v.astype(np.float32) if n < 1e-9 else (v / n).astype(np.float32)


def _blur(img: np.ndarray, k: int = 3) -> np.ndarray:
    """Cheap separable box blur -- the periphery's optics."""
    out = img.astype(np.float32)
    pad = k // 2
    for axis in (0, 1):
        p = np.pad(out, [(pad, pad) if a == axis else (0, 0)
                         for a in range(2)], mode="edge")
        acc = np.zeros_like(out)
        for i in range(k):
            sl = [slice(None)] * 2
            sl[axis] = slice(i, i + out.shape[axis])
            acc += p[tuple(sl)]
        out = acc / k
    return out


# ---------------------------------------------------------------------------
# A large scene made of real objects
# ---------------------------------------------------------------------------
@dataclass
class Scene:
    """A big canvas with real images placed on it, plus clutter.

    This is the "large size" the eye has to deal with: the object is a small
    part of a much bigger field, most of which is background, and its position
    is not given to anybody."""

    canvas: np.ndarray
    positions: List[Tuple[int, int]]      # (row, col) of each object's centre
    labels: List[int]
    tile: int = 28

    @property
    def shape(self) -> Tuple[int, int]:
        return self.canvas.shape

    def label_at(self, r: int, c: int, tol: Optional[int] = None) -> int:
        """Which object is at this point, or -1 for background."""
        tol = tol if tol is not None else self.tile // 2
        for (pr, pc), lab in zip(self.positions, self.labels):
            if abs(pr - r) <= tol and abs(pc - c) <= tol:
                return int(lab)
        return -1


def build_scene(images: np.ndarray, labels: np.ndarray, size: int = 256,
                n_objects: int = 12, clutter: float = 0.08,
                seed: int = 0) -> Scene:
    """Scatter real images across a large canvas, on a noisy background.

    Objects are placed without overlapping so that ground truth is unambiguous;
    everything else -- where they are, what the eye does about it -- is left for
    the system to work out."""
    rng = np.random.default_rng(seed)
    tile = images.shape[1]
    canvas = (clutter * 255.0 * rng.random((size, size))).astype(np.float32)
    positions: List[Tuple[int, int]] = []
    labs: List[int] = []
    tries = 0
    while len(positions) < n_objects and tries < n_objects * 200:
        tries += 1
        r = int(rng.integers(tile, size - tile))
        c = int(rng.integers(tile, size - tile))
        if any(abs(r - pr) < 1.6 * tile and abs(c - pc) < 1.6 * tile
               for pr, pc in positions):
            continue
        i = int(rng.integers(len(images)))
        patch = images[i].astype(np.float32)
        y, x = r - tile // 2, c - tile // 2
        canvas[y:y + tile, x:x + tile] = np.maximum(
            canvas[y:y + tile, x:x + tile], patch)
        positions.append((r, c))
        labs.append(int(labels[i]))
    return Scene(canvas, positions, labs, tile)


# ---------------------------------------------------------------------------
# The eye
# ---------------------------------------------------------------------------
@dataclass
class Fixation:
    """One glance: where the eye landed, what it saw, what was really there."""

    row: int
    col: int
    frames: List[np.ndarray]          # the drifting foveal sequence
    true_label: int                   # -1 when the eye landed on background
    t_ms: int = 0


class SaccadicEye:
    """A fovea, a periphery, saliency-driven saccades and fixational drift.

    The eye is deliberately *not* given object positions. It computes a coarse
    saliency map from its blurred periphery -- local contrast, which is what
    superior-colliculus-driven orienting actually responds to -- and jumps to
    peaks it has not recently visited (**inhibition of return**, without which
    any saliency-driven system stares at the brightest thing forever)."""

    def __init__(self, scene: Scene, fovea: int = 28, drift: int = 1,
                 n_drift_frames: int = 6, periphery_ds: int = 8,
                 ior_radius: int = 34, seed: int = 0):
        self.scene = scene
        self.fovea = int(fovea)
        self.drift = int(drift)
        self.n_drift_frames = int(n_drift_frames)
        self.periphery_ds = int(periphery_ds)
        self.ior_radius = int(ior_radius)
        self.rng = np.random.default_rng(seed)
        self.visited: List[Tuple[int, int]] = []
        self.t_ms = 0
        self._saliency = self._compute_saliency()

    # -- the periphery ------------------------------------------------------
    def peripheral_view(self) -> np.ndarray:
        """What the eye sees outside the fovea: blurred and downsampled."""
        b = _blur(self.scene.canvas, k=5)
        d = self.periphery_ds
        h = (b.shape[0] // d) * d
        w = (b.shape[1] // d) * d
        return b[:h, :w].reshape(h // d, d, w // d, d).mean((1, 3))

    def _compute_saliency(self) -> np.ndarray:
        """Local contrast in the periphery -- where something might be."""
        p = self.peripheral_view()
        loc = _blur(p, k=3)
        sal = np.abs(p - loc) + 0.25 * p
        return sal.astype(np.float32)

    # -- where to look next -------------------------------------------------
    def next_target(self) -> Tuple[int, int]:
        """Pick the most salient place not recently looked at."""
        sal = self._saliency.copy()
        d = self.periphery_ds
        for (vr, vc) in self.visited[-24:]:
            gr, gc = vr // d, vc // d
            rad = max(1, self.ior_radius // d)
            r0, r1 = max(0, gr - rad), min(sal.shape[0], gr + rad + 1)
            c0, c1 = max(0, gc - rad), min(sal.shape[1], gc + rad + 1)
            sal[r0:r1, c0:c1] = -1e9              # inhibition of return
        flat = int(np.argmax(sal))
        gr, gc = divmod(flat, sal.shape[1])
        jitter = self.rng.integers(-d // 2, d // 2 + 1, 2)
        r = int(np.clip(gr * d + d // 2 + jitter[0], self.fovea,
                        self.scene.shape[0] - self.fovea))
        c = int(np.clip(gc * d + d // 2 + jitter[1], self.fovea,
                        self.scene.shape[1] - self.fovea))
        return r, c

    # -- one glance ---------------------------------------------------------
    def fixate(self, row: int, col: int) -> Fixation:
        """Hold the fovea here and let it drift, returning the frame sequence.

        The drift is the point: a real retina fed a perfectly stationary image
        stops responding within a second or two (Troxler fading), and the drift
        is also what puts *motion* inside the integration window."""
        f = self.fovea
        half = f // 2
        frames = []
        for i in range(self.n_drift_frames):
            dr = int(self.rng.integers(-self.drift, self.drift + 1))
            dc = int(self.rng.integers(-self.drift, self.drift + 1))
            r = int(np.clip(row + dr, half, self.scene.shape[0] - half - 1))
            c = int(np.clip(col + dc, half, self.scene.shape[1] - half - 1))
            frames.append(self.scene.canvas[r - half:r - half + f,
                                            c - half:c - half + f].copy())
        self.visited.append((row, col))
        self.t_ms += 50
        return Fixation(row, col, frames, self.scene.label_at(row, col),
                        self.t_ms)

    def foveate(self, row: int, col: int, n_steps: int = 2
                ) -> Tuple[int, int]:
        """Correct the landing point so the thing lands **on the fovea**.

        This exists because of a measured failure. A saliency saccade lands
        *near* an object, not on its centre, and the wide V1 code is
        retinotopic -- a cell only ever sees its own patch -- so it has no
        translation invariance whatever. Recognition of freely-viewed objects
        collapsed to 17.2% against 78.3% for centred crops purely because the
        digit was sitting off-centre in the fovea.

        The fix is the one biology uses: a fast **corrective saccade** driven by
        retinal error. The centre of mass of the current foveal image says which
        way the target is off-centre, and the eye steps that way. It uses only
        what is on the retina -- no object position, no label."""
        f, half = self.fovea, self.fovea // 2
        r, c = int(row), int(col)
        for _ in range(n_steps):
            r = int(np.clip(r, half, self.scene.shape[0] - half - 1))
            c = int(np.clip(c, half, self.scene.shape[1] - half - 1))
            patch = self.scene.canvas[r - half:r - half + f,
                                      c - half:c - half + f]
            m = patch - patch.mean()
            m = np.maximum(m, 0.0)
            tot = float(m.sum())
            if tot < 1e-6:
                break
            yy, xx = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
            cy = float((m * yy).sum() / tot)
            cx = float((m * xx).sum() / tot)
            dy, dx = int(round(cy - half)), int(round(cx - half))
            if abs(dy) < 1 and abs(dx) < 1:
                break
            r, c = r + dy, c + dx
        return (int(np.clip(r, half, self.scene.shape[0] - half - 1)),
                int(np.clip(c, half, self.scene.shape[1] - half - 1)))

    def free_view(self, n_saccades: int = 20, correct: bool = True
                  ) -> List[Fixation]:
        """Look around the scene on its own for a while.

        ``correct=False`` disables the corrective saccade, which is how the
        cost of *not* foveating is measured rather than assumed."""
        out = []
        for _ in range(n_saccades):
            r, c = self.next_target()
            if correct:
                r, c = self.foveate(r, c)
            out.append(self.fixate(r, c))
        return out


# ---------------------------------------------------------------------------
# The ear
# ---------------------------------------------------------------------------
@dataclass
class SoundEvent:
    """One thing that happened in the soundscape, and when."""

    start: int            # sample index
    end: int
    label: int


@dataclass
class Soundscape:
    """A long waveform with real events buried in it and nothing marked."""

    wave: np.ndarray
    events: List[SoundEvent]
    sr: int
    class_names: List[str]


def build_soundscape(n_events: int = 24, gap_s: Tuple[float, float] = (0.15, 0.6),
                     noise: float = 0.05, seed: int = 0) -> Soundscape:
    """Drop real sound events into background noise at unannounced times."""
    from .audio import sound_dataset, SR

    rng = np.random.default_rng(seed)
    sigs, labels, names = sound_dataset(n_per_class=8, seed=seed)
    order = rng.permutation(len(sigs))[:n_events]
    chunks: List[np.ndarray] = []
    events: List[SoundEvent] = []
    pos = 0
    lead = int(0.2 * SR)
    chunks.append(np.zeros(lead, np.float32))
    pos += lead
    for i in order:
        s = sigs[i].astype(np.float32)
        events.append(SoundEvent(pos, pos + len(s), int(labels[i])))
        chunks.append(s)
        pos += len(s)
        g = int(rng.uniform(*gap_s) * SR)
        chunks.append(np.zeros(g, np.float32))
        pos += g
    wave = np.concatenate(chunks)
    wave = wave + noise * rng.standard_normal(len(wave)).astype(np.float32)
    return Soundscape(wave.astype(np.float32), events, SR, names)


class ContinuousEar:
    """A cochlea sliding along a stream, finding its own event boundaries.

    Nothing tells this class where a sound starts. It computes the cochlear
    energy over time and looks for a **rising derivative** past a threshold --
    the same cue auditory onset cells use -- with a refractory period so one
    event is not reported many times."""

    def __init__(self, sr: int = 8000, hop_ms: int = 10, win_ms: int = 50,
                 refractory_ms: int = 250, threshold: float = 2.5):
        from .audio import Cochleagram

        self.coch = Cochleagram(sr=sr)
        self.sr = sr
        self.hop = max(1, int(sr * hop_ms / 1000))
        self.win = max(64, int(sr * win_ms / 1000))
        self.refractory = int(sr * refractory_ms / 1000)
        self.threshold = float(threshold)

    def energy_trace(self, wave: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Cochlear energy sampled along the stream, and the sample index."""
        starts = np.arange(0, max(1, len(wave) - self.win), self.hop)
        e = np.array([float(np.abs(wave[s:s + self.win]).mean())
                      for s in starts], np.float32)
        return e, starts

    def detect_onsets(self, wave: np.ndarray) -> List[int]:
        """Find event starts from the rising energy, with a refractory period."""
        e, starts = self.energy_trace(wave)
        if len(e) < 3:
            return []
        d = np.diff(e, prepend=e[0])
        pos = d[d > 0]
        thr = (pos.mean() + self.threshold * pos.std()) if len(pos) else np.inf
        out: List[int] = []
        last = -10 ** 9
        for i in range(1, len(d)):
            if d[i] > thr and starts[i] - last > self.refractory:
                out.append(int(starts[i]))
                last = int(starts[i])
        return out

    def listen(self, wave: np.ndarray, at: int, dur_s: float = 0.40
               ) -> np.ndarray:
        """Take the cochleagram of the window beginning at a detected onset."""
        n = int(dur_s * self.sr)
        seg = wave[at:at + n]
        if len(seg) < n:
            seg = np.pad(seg, (0, n - len(seg)))
        return self.coch.forward(seg)[0]


def onset_scores(found: Sequence[int], truth: Sequence[SoundEvent],
                 tol_ms: float = 120.0, sr: int = 8000) -> Dict[str, float]:
    """Precision / recall of the ear's self-found boundaries.

    This is the number that says whether streaming audio works at all: if the
    boundaries are wrong, everything downstream is classifying the wrong
    window."""
    tol = tol_ms * sr / 1000.0
    used = set()
    hits = 0
    for f in found:
        for j, ev in enumerate(truth):
            if j not in used and abs(f - ev.start) <= tol:
                used.add(j)
                hits += 1
                break
    prec = hits / max(len(found), 1)
    rec = hits / max(len(truth), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"precision": prec, "recall": rec, "f1": f1,
            "n_found": float(len(found)), "n_true": float(len(truth))}


# ---------------------------------------------------------------------------
# Both senses, one clock, one shared code
# ---------------------------------------------------------------------------
@dataclass
class StreamReport:
    """What streaming actually cost, measured against pre-segmented input."""

    # vision
    saccades: int = 0
    on_object_rate: float = 0.0
    chance_on_object: float = 0.0
    streaming_vision_accuracy: float = 0.0
    uncorrected_vision_accuracy: float = 0.0
    presegmented_vision_accuracy: float = 0.0
    # hearing
    onset_precision: float = 0.0
    onset_recall: float = 0.0
    onset_f1: float = 0.0
    streaming_audio_accuracy: float = 0.0
    presegmented_audio_accuracy: float = 0.0
    raw_cochleagram_accuracy: float = 0.0
    # both
    crossmodal_recall: float = 0.0
    crossmodal_chance: float = 0.0
    bound_pairs: int = 0

    def summary(self) -> str:
        return (
            f"vision  : streaming {self.streaming_vision_accuracy:.1%} "
            f"(no corrective saccade {self.uncorrected_vision_accuracy:.1%}) "
            f"vs pre-cut {self.presegmented_vision_accuracy:.1%}\n"
            f"hearing : streaming {self.streaming_audio_accuracy:.1%} vs "
            f"pre-cut {self.presegmented_audio_accuracy:.1%} "
            f"(raw cochleagram {self.raw_cochleagram_accuracy:.1%}, "
            f"onset F1 {self.onset_f1:.2f})\n"
            f"binding : sound -> seen object {self.crossmodal_recall:.1%} "
            f"(chance {self.crossmodal_chance:.1%})")


class AuditoryBelt:
    """A wide spiking layer over the cochleagram -- the ear's version of V1.

    Two ideas make this work, and both were forced by measurement:

    * **The cochleagram is a stream, not a picture.** Fed as one static image
      the layer scored 25.0%, no better than matching raw cochleagrams. Fed as a
      *sequence* of frequency-by-short-time slices -- which is how the ear
      actually receives it -- a rising sweep becomes literal upward motion, the
      same delayed-coincidence detectors from :mod:`widev1` fire on it, and the
      score rises to 37.5%.
    * **Pitch is not identity.** The classes here differ by pattern (a sweep, a
      modulation, a harmonic stack) while their absolute frequency is randomised
      across an octave and a half. A frequency-specific code cannot generalise
      over that, so the responses are **pooled across frequency** -- the complex
      cell operation. 37.5% -> 75.0%. Pooling across **time as well**, so the
      code says only *which* feature occurred and neither where nor when,
      reaches **87.5%**: a detected onset is never exactly on time, and a
      time-specific code pays for every millisecond of that error.

    One fix that did **not** work is worth recording. The obvious answer to a
    late onset is to walk back to the foot of the energy rise -- the ear's
    version of the corrective saccade that rescued vision here. Measured, it
    made alignment *worse* (mean error 28.2 ms -> 38.9 ms) because it lands in
    the silence before the event. Shift-invariant pooling solved the same
    problem properly, so the backtracking was dropped rather than tuned.
    """

    def __init__(self, n_freq: int = 36, slice_w: int = 16, step: int = 8,
                 n_cells: int = 1024, window_ms: int = 50, n_frames: int = 11,
                 adapt_bands: bool = True, floor_pct: float = 30.0,
                 floor_gain: float = 0.8, divisive: bool = False,
                 sigma_band: float = 2.5, semi_saturation: float = 0.1,
                 seed: int = 0):
        from .widev1 import WideV1

        self.slice_w, self.step = int(slice_w), int(step)
        self.adapt_bands = bool(adapt_bands)
        self.floor_pct, self.floor_gain = float(floor_pct), float(floor_gain)
        # Heeger (1992) divisive normalization along frequency. OFF by default,
        # and the default is set by measurement rather than preference: it is a
        # clear win on cleanly cut clips (8-class designed bank 70.8% -> 79.2%)
        # and a clear loss on the STREAM (85.7% -> 42.9%), because a divisive
        # rule with a fixed semi-saturation constant is level-dependent and a
        # streamed window's level varies with how much silence it caught. The
        # subtractive floor is level-robust. Turn it on for pre-segmented audio.
        self.divisive = bool(divisive)
        self.sigma_band = float(sigma_band)
        self.semi_saturation = float(semi_saturation)
        self.layer = WideV1(image_shape=(n_freq, slice_w), n_cells=n_cells,
                            rf=10, stride=3, window_ms=window_ms,
                            motion_fraction=0.6,
                            lag_ms=max(1, window_ms // max(n_frames, 1)),
                            seed=seed)
        self.index, self.n_groups = self.layer.pooling_index("both")

    def slices(self, coch: np.ndarray) -> List[np.ndarray]:
        w, st = self.slice_w, self.step
        out = [coch[:, i:i + w] for i in range(0, coch.shape[1] - w + 1, st)]
        return out or [coch[:, :w]]

    def adapt(self, coch: np.ndarray) -> np.ndarray:
        """Per-band adaptation: every frequency channel subtracts its own floor.

        This is auditory-nerve adaptation, and it is what makes the code survive
        a noisy world. A stationary background raises the level in *every* band;
        a channel that stops transmitting its own long-run floor stops
        transmitting the background with it, and what is left is the event.

        It earns its place by measurement. Against a background of 0.05 the
        un-adapted belt fell from 87.5% to 34.4%; adapted it holds 90.6%. On
        three **independent** draws of the sound set the noise condition
        improved every time (+37.5, +12.5, +34.4 points). On clean audio the
        effect is smaller and not uniform -- two draws improved (+15.7, +6.2)
        and one got worse (-6.3) -- which is stated here rather than averaged
        into the headline."""
        from .audio import divisive_normalization

        c = np.asarray(coch, np.float32)
        if self.adapt_bands:
            floor = np.percentile(c, self.floor_pct, axis=1, keepdims=True)
            c = np.maximum(c - self.floor_gain * floor, 0.0)
            m = float(c.max())
            c = (c / m).astype(np.float32) if m > 1e-6 else c.astype(np.float32)
        if self.divisive:
            c = divisive_normalization(c, sigma_band=self.sigma_band,
                                       semi_saturation=self.semi_saturation)
        return c

    def code(self, coch: np.ndarray) -> np.ndarray:
        """Frequency-invariant spiking code for one cochleagram."""
        r = self.layer.rate_over(self.slices(self.adapt(coch)))
        return self.layer.pooled_code(r, self.index, self.n_groups)


class StreamingBrain:
    """Eye and ear on one clock, both projecting into the shared cortical code.

    The eye's wide V1 and the ear's cochlea produce codes in completely
    different spaces; :class:`~neurobrain.workspace.GlobalWorkspace` is what
    makes them comparable, and binding is simply the two codes being in the
    workspace at the same moment."""

    def __init__(self, v1_cells: int = 1024, window_ms: int = 50,
                 dim: int = 512, seed: int = 0):
        from .widev1 import WideV1
        from .workspace import GlobalWorkspace

        self.v1 = WideV1(n_cells=v1_cells, window_ms=window_ms, seed=seed)
        self.ear = ContinuousEar()
        self.belt = AuditoryBelt(n_freq=self.ear.coch.n_freq, seed=seed)
        self.ws = GlobalWorkspace(dim=dim, vigilance=0.25, seed=seed, adapt=True)
        self.window_ms = window_ms
        self.bindings: List[Tuple[np.ndarray, np.ndarray, int]] = []

    # -- one glance / one listen, in the shared code ------------------------
    def see(self, fixation: Fixation) -> np.ndarray:
        """A whole fixation -- the drifting sequence -- becomes one percept.

        The drift frames go through ``rate_over``, so the 50 ms window is doing
        exactly what v0.22 built it for: integrating a short moving sequence
        rather than a frozen snapshot."""
        r = self.v1.rate_over(fixation.frames, window_ms=self.window_ms)
        return self.ws.encode("vision", _unit(r))

    def hear(self, wave: np.ndarray, at: int) -> np.ndarray:
        """Cochlea -> spiking auditory belt -> the shared code."""
        return self.ws.encode("sound", self.belt.code(self.ear.listen(wave, at)))

    def bind(self, v_code: np.ndarray, a_code: np.ndarray, label: int) -> None:
        """Two codes present at the same instant become one association.

        This is Hebbian binding at the level of the workspace: no supervision
        says they belong together, only that they co-occurred."""
        self.bindings.append((v_code, a_code, label))

    def recall_visual_from_sound(self, a_code: np.ndarray) -> Optional[int]:
        """Hear a sound, remember what was seen with it."""
        if not self.bindings:
            return None
        sims = [float(a_code @ a) for _, a, _ in self.bindings]
        return int(self.bindings[int(np.argmax(sims))][2])


def streaming_experiment(n_saccades: int = 60, n_events: int = 24,
                         n_scene_objects: int = 12, scene_size: int = 256,
                         v1_cells: int = 1024, seed: int = 0,
                         verbose: bool = False) -> StreamReport:
    """Run both senses on real streams and price the streaming honestly.

    Each modality is measured twice on the *same* underlying data: once with the
    stream doing its own segmentation, and once with the boundaries handed over.
    The difference is the cost of the thing this module exists to test."""
    from .realworld import load_mnist
    from .audio import sound_dataset
    from .widev1 import _nearest_prototype

    def say(*a):
        if verbose:
            print(*a)

    rep = StreamReport()
    trx, trY, tex, teY = load_mnist(n_train=1200, n_test=400)
    brain = StreamingBrain(v1_cells=v1_cells, seed=seed)

    # ---------------- vision: free viewing a large cluttered scene --------
    say("building a large scene and letting the eye look around it ...")
    scene = build_scene(trx, trY, size=scene_size, n_objects=n_scene_objects,
                        seed=seed)
    eye = SaccadicEye(scene, seed=seed)
    fixations = eye.free_view(n_saccades=n_saccades, correct=True)
    raw_eye = SaccadicEye(scene, seed=seed)
    raw_fix = [f for f in raw_eye.free_view(n_saccades=n_saccades,
                                            correct=False) if f.true_label >= 0]
    on_obj = [f for f in fixations if f.true_label >= 0]
    rep.saccades = len(fixations)
    rep.on_object_rate = len(on_obj) / max(len(fixations), 1)
    tile = scene.tile
    rep.chance_on_object = (n_scene_objects * tile * tile) / float(
        scene_size * scene_size)

    # train the read-out on pre-cut, centred crops (the easy condition) ...
    Xc = np.array([_unit(brain.v1.rate(im)) for im in trx[:600]], np.float32)
    yc = trY[:600]
    Xt = np.array([_unit(brain.v1.rate(im)) for im in tex[:300]], np.float32)
    rep.presegmented_vision_accuracy = float(
        np.mean(_nearest_prototype(Xc, yc, Xt, 10) == teY[:300]))

    # ... and apply the SAME read-out to what the moving eye actually got
    if on_obj:
        Xs = np.array([_unit(brain.v1.rate_over(f.frames)) for f in on_obj],
                      np.float32)
        ys = np.array([f.true_label for f in on_obj])
        rep.streaming_vision_accuracy = float(
            np.mean(_nearest_prototype(Xc, yc, Xs, 10) == ys))
    if raw_fix:
        Xr = np.array([_unit(brain.v1.rate_over(f.frames)) for f in raw_fix],
                      np.float32)
        yr = np.array([f.true_label for f in raw_fix])
        rep.uncorrected_vision_accuracy = float(
            np.mean(_nearest_prototype(Xc, yc, Xr, 10) == yr))
    say(f"   eye landed on an object {rep.on_object_rate:.0%} of saccades "
        f"(chance {rep.chance_on_object:.0%})")

    # ---------------- hearing: a soundscape with no marked boundaries -----
    say("building a soundscape and making the ear find its own onsets ...")
    scape = build_soundscape(n_events=n_events, seed=seed)
    found = brain.ear.detect_onsets(scape.wave)
    s = onset_scores(found, scape.events, sr=scape.sr)
    rep.onset_precision, rep.onset_recall, rep.onset_f1 = (
        s["precision"], s["recall"], s["f1"])

    sigs, labels, names = sound_dataset(n_per_class=8, seed=seed + 5)
    Xa = np.array([brain.belt.code(brain.ear.coch.forward(sg)[0])
                   for sg in sigs], np.float32)
    Xa_raw = np.array([_unit(brain.ear.coch.forward(sg)[0].reshape(-1))
                       for sg in sigs], np.float32)
    n_cls = len(names)
    # pre-segmented: the clip boundaries are given.
    # sound_dataset returns its clips GROUPED BY CLASS, so splitting it down the
    # middle puts classes 0-3 in train and 4-7 in test -- no overlap, and the
    # score is a guaranteed 0.0%. It duly reported 0.0% until this shuffle was
    # added, which is why the split is randomised here.
    perm = np.random.default_rng(seed + 11).permutation(len(Xa))
    Xa, labels = Xa[perm], np.asarray(labels)[perm]
    half = len(Xa) // 2
    # ONE prototype bank for both conditions. Letting the streaming condition
    # use all 64 clips while the pre-segmented one used 32 made streaming look
    # better than pre-cut (61.1% vs 25.0%) -- an artefact of unequal training,
    # not a result. Both now read out from Xa[:half].
    bank_X, bank_y = Xa[:half], labels[:half]
    rep.presegmented_audio_accuracy = float(np.mean(
        _nearest_prototype(bank_X, bank_y, Xa[half:], n_cls) == labels[half:]))
    Xr = Xa_raw[perm]
    rep.raw_cochleagram_accuracy = float(np.mean(
        _nearest_prototype(Xr[:half], bank_y, Xr[half:], n_cls) == labels[half:]))

    # streaming: classify whatever the ear itself decided was an event
    tol = 0.12 * scape.sr
    hits, ok = 0, 0
    for f in found:
        truth = [e for e in scape.events if abs(e.start - f) <= tol]
        if not truth:
            continue                      # a false alarm has no right answer
        hits += 1
        code = brain.belt.code(brain.ear.listen(scape.wave, f))
        pred = int(_nearest_prototype(bank_X, bank_y, code[None], n_cls)[0])
        ok += int(pred == truth[0].label)
    rep.streaming_audio_accuracy = ok / max(hits, 1)
    say(f"   onsets: precision {rep.onset_precision:.2f}, "
        f"recall {rep.onset_recall:.2f}")

    # ---------------- both: bind what was seen and heard together ---------
    say("binding vision and hearing in the ONE shared code ...")
    # The world is consistent: an object of class d always makes a sound of
    # class d % n_cls. Nothing tells the brain this rule; it only ever sees the
    # two codes arrive together.
    pairs = min(len(on_obj), len(found))
    for i in range(pairs):
        v = brain.see(on_obj[i])
        lab = on_obj[i].true_label
        # the sound this object makes: a real clip of the paired class
        idx = int(np.flatnonzero(labels == (lab % n_cls))[0])
        a = brain.ws.encode("sound", Xa[idx])
        brain.bind(v, a, lab)
    rep.bound_pairs = pairs

    # The test must use a sound the brain has NEVER heard, or it is an exact
    # lookup of the very vector that was stored -- which scores 100% and means
    # nothing. Held-out clips of the same classes are used instead.
    ok = probes = 0
    rng_p = np.random.default_rng(seed + 21)
    for i in range(pairs):
        lab = on_obj[i].true_label
        cand = np.flatnonzero(labels == (lab % n_cls))
        cand = [j for j in cand if j != int(cand[0])]      # never the bound one
        if not cand:
            continue
        j = int(cand[int(rng_p.integers(len(cand)))])
        probes += 1
        got = brain.recall_visual_from_sound(brain.ws.encode("sound", Xa[j]))
        ok += int(got is not None and got % n_cls == lab % n_cls)
    rep.crossmodal_recall = ok / max(probes, 1)
    labs = [f.true_label % n_cls for f in on_obj[:pairs]]
    rep.crossmodal_chance = (max(np.bincount(labs)) / len(labs)) if labs else 0.0

    if verbose:
        print()
        print(rep.summary())
    return rep


# ---------------------------------------------------------------------------
# A world where things actually move
# ---------------------------------------------------------------------------
@dataclass
class MovingObject:
    """One object with a position and a velocity, in pixels per frame."""

    label: int
    image: np.ndarray
    row: float
    col: float
    vr: float
    vc: float

    def advance(self, h: int, w: int) -> None:
        """Move, and bounce off the walls so the object stays in the world."""
        self.row += self.vr
        self.col += self.vc
        m = self.image.shape[0] / 2 + 1
        if not (m <= self.row <= h - m):
            self.vr = -self.vr
            self.row = float(np.clip(self.row, m, h - m))
        if not (m <= self.col <= w - m):
            self.vc = -self.vc
            self.col = float(np.clip(self.col, m, w - m))


class MovingScene:
    """A scene that changes because the *world* moves, not just the eye.

    Everything measured so far drifted the retina across a frozen picture. That
    is enough to make a sequence, but it is not motion in the world: the object
    never goes anywhere, nothing enters or leaves, and a tracking system has
    nothing to track. Here each object carries a velocity and the canvas is
    re-rendered every frame.

    This is what the direction-selective cells built in :mod:`widev1` were for.
    Until now they had only the eye's own 1-pixel tremor to work with, which is
    why switching them off cost almost nothing."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, size: int = 256,
                 n_objects: int = 8, speed: float = 3.0, clutter: float = 0.08,
                 seed: int = 0):
        rng = np.random.default_rng(seed)
        self.size, self.clutter, self.rng = int(size), float(clutter), rng
        self.tile = int(images.shape[1])
        self.objects: List[MovingObject] = []
        for _ in range(n_objects):
            i = int(rng.integers(len(images)))
            th = rng.uniform(0, 2 * np.pi)
            self.objects.append(MovingObject(
                int(labels[i]), images[i].astype(np.float32),
                float(rng.uniform(self.tile, size - self.tile)),
                float(rng.uniform(self.tile, size - self.tile)),
                float(speed * np.sin(th)), float(speed * np.cos(th))))
        self._bg = (clutter * 255.0 * rng.random((size, size))).astype(np.float32)
        self.t = 0

    def render(self) -> np.ndarray:
        canvas = self._bg.copy()
        h = self.tile // 2
        for o in self.objects:
            r, c = int(round(o.row)), int(round(o.col))
            y0, x0 = max(r - h, 0), max(c - h, 0)
            y1, x1 = min(y0 + self.tile, self.size), min(x0 + self.tile, self.size)
            canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1],
                                              o.image[:y1 - y0, :x1 - x0])
        return canvas

    def step(self) -> np.ndarray:
        for o in self.objects:
            o.advance(self.size, self.size)
        self.t += 1
        return self.render()

    def scene_now(self) -> Scene:
        """A :class:`Scene` snapshot, so the existing eye can look at it."""
        return Scene(self.render(),
                     [(int(round(o.row)), int(round(o.col))) for o in self.objects],
                     [o.label for o in self.objects], self.tile)

    def nearest(self, r: int, c: int) -> Optional[MovingObject]:
        best, bd = None, 1e9
        for o in self.objects:
            d = abs(o.row - r) + abs(o.col - c)
            if d < bd:
                best, bd = o, d
        return best if bd <= self.tile else None


def smooth_pursuit_experiment(n_objects: int = 8, size: int = 256,
                              speed: float = 3.0, n_frames: int = 6,
                              n_trials: int = 90, v1_cells: int = 1024,
                              seed: int = 0, verbose: bool = False
                              ) -> Dict[str, float]:
    """Can the brain read the **direction a real object is travelling**?

    The eye fixates a moving object and integrates a 50 ms window while the
    object crosses its fovea, so the frames differ because the world moved.
    Direction is quantised into four headings and decoded from the V1 code.

    Two controls make the result meaningful rather than decorative:

      ``static``    the same object rendered motionless -- a hard floor, since a
                    frozen object carries no heading at all;
      ``no_detectors``  the same moving sequence with the delayed-coincidence
                    detectors switched off, which isolates what the explicit
                    motion circuitry contributes over the window alone.
    """
    from .realworld import load_mnist
    from .widev1 import WideV1, _unit, _nearest_prototype

    trx, trY, _, _ = load_mnist(n_train=600, n_test=10)

    def trials(layer: WideV1, moving: bool, rs: np.random.Generator):
        """One trial: fixate a point, and let the object cross the fovea.

        Two bugs in the first version of this had to be fixed, and both
        inverted the result:

        * **The eye tracked the object perfectly**, re-centring on it every
          frame. A perfectly pursued object is *stationary on the retina* --
          it is the background that streams past -- so V1 saw no object motion
          at all and scored 21.1%, below chance. The eye now holds a fixed
          fixation and the object moves through the fovea, which is what a
          direction-selective cell is for.
        * **Heading leaked into the still image.** Objects bounce off walls, so
          after a while position predicts heading, and a *frozen* control scored
          32.2% against a chance of 25%. Each trial now assigns the object a
          fresh random heading, which makes heading independent of everything
          visible in the first frame and puts the control back on the floor
          where a control belongs.
        """
        X, y = [], []
        world = MovingScene(trx, trY, size=size, n_objects=n_objects,
                            speed=speed, seed=int(rs.integers(1 << 30)))
        half = world.tile // 2
        for _ in range(n_trials):
            o = world.objects[int(rs.integers(len(world.objects)))]
            heading = int(rs.integers(4))
            th = heading * (np.pi / 2)
            o.vr, o.vc = float(speed * np.sin(th)), float(speed * np.cos(th))
            # Fixate where the object IS at frame 0, then let it move away.
            # Backing the object up first and fixating ahead of it put the
            # object off-centre in frame 0 by exactly minus its heading, so a
            # single frozen frame predicted the heading at 98.9%. Centring it at
            # frame 0 makes the first frame carry no heading at all, and only
            # the later frames -- the motion itself -- can carry it.
            o.row = float(np.clip(o.row, half + 1, size - half - 2))
            o.col = float(np.clip(o.col, half + 1, size - half - 2))
            rr, cc = int(round(o.row)), int(round(o.col))
            frames = []
            for k in range(n_frames):
                if moving or k == 0:
                    canvas = world.render()
                    frames.append(canvas[rr - half:rr + half,
                                         cc - half:cc + half].copy())
                else:
                    frames.append(frames[0])
                if moving:
                    world.step()
            X.append(_unit(layer.rate_over(frames)))
            y.append(heading)
        return np.array(X, np.float32), np.array(y)

    def score(moving: bool, detectors: bool) -> float:
        layer = WideV1(n_cells=v1_cells, window_ms=50, seed=0,
                       lag_ms=max(1, 50 // n_frames),
                       motion_fraction=0.6 if detectors else 0.0)
        Xtr, ytr = trials(layer, moving, np.random.default_rng(seed))
        Xte, yte = trials(layer, moving, np.random.default_rng(seed + 1))
        return float(np.mean(_nearest_prototype(Xtr, ytr, Xte, 4) == yte))

    out = {"moving": score(True, True),
           "no_detectors": score(True, False),
           "static": score(False, True),
           "chance": 0.25}
    out["motion_gain"] = out["moving"] - out["static"]
    out["detector_gain"] = out["moving"] - out["no_detectors"]
    if verbose:
        print(f"   real object motion + detectors : {out['moving']:.1%}")
        print(f"   real object motion, no detectors: {out['no_detectors']:.1%}")
        print(f"   frozen object (control)         : {out['static']:.1%} "
              f"(chance 25%)")
    return out
