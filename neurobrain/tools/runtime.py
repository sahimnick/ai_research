"""
runtime.py
==========

The **infinite loop** that makes the brain *live* -- a game-engine main loop.

A game engine does not run a scene once; it runs a loop forever, building and
processing every frame (physics, graphics, sound, input, ...). This module does
the same for the brain:

    while running:
        tick()      # one frame ≈ one millisecond of brain time

Two ideas drive it:

1. **A constant baseline current** ("جریان ثابت برق") flows through every neuron
   every frame. Like a power rail that is always on, it keeps the brain
   *spontaneously active* -- there is ongoing activity, an ever-present hum of
   "thought", even with no input. When neurons fire, their (learned, stronger)
   synapses route extra current onward, and the flow from perception to thought
   builds itself, frame by frame.

2. **It never stops learning.** STDP runs on every frame, and structural
   plasticity (synaptogenesis / pruning / neurogenesis) runs periodically, so
   the network keeps reshaping itself from its own ongoing experience.

A light homeostatic controller nudges the baseline current so the brain stays
alive but never seizes (a target firing rate), exactly like the brain's own
excitation/inhibition balance.

Inputs can be fed into the running loop at any moment; the current "thought"
(the strongest word concept, smoothed over recent frames) can be read at any
moment.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable, Deque, Dict, List, Optional

import numpy as np

from ..core.brain import Brain
from ..learning.language import LanguageModel
from ..sensing.sensors import ImageEncoder, SoundEncoder, TextEncoder
from ..learning.teacher import Teacher
from ..cognition.imagination import Imagination
from ..cognition.attention import Attention
from ..learning.curriculum import bright_image, dark_image, high_tone, low_tone


class BrainRuntime:
    """Runs a :class:`Brain` in a perpetual, learning game loop.

    Parameters
    ----------
    brain:        the brain to run (finalized).
    teacher:      teacher for live grounding (created if omitted).
    encoders:     text/vision/sound encoders (created if omitted).
    tonic:        starting baseline current (the "always-on electricity").
    target_rate:  fraction of neurons the homeostat aims to keep firing.
    fps:          loop speed cap (frames per second). None = as fast as possible.
    learn:        run STDP every frame.
    grow:         run structural plasticity periodically.
    """

    def __init__(self, brain: Brain, teacher: Optional[Teacher] = None,
                 encoders: Optional[Dict[str, object]] = None,
                 tonic: float = 3.0, target_rate: float = 0.035,
                 fps: Optional[float] = 60.0, learn: bool = True,
                 grow: bool = True):
        self.brain = brain
        self.teacher = teacher or Teacher(brain, concept_region="memory",
                                         clamp_current=30.0, assembly_size=14)
        enc = encoders or {}
        self.text: TextEncoder = enc.get("text") or TextEncoder(
            brain.region("text").size, level="word", active_bits=16,
            amplitude=18.0, window=24)
        self.vision: ImageEncoder = enc.get("vision") or ImageEncoder(
            brain.region("vision").size, amplitude=18.0, window=24)
        self.sound: SoundEncoder = enc.get("sound") or SoundEncoder(
            brain.region("sound").size, amplitude=18.0, window=24)
        self.language = LanguageModel(brain, self.teacher, self.text)

        # higher cognition living inside the loop
        self.imagination = Imagination(brain)
        self.attention = Attention(brain)

        self.tonic = float(tonic)
        self.target_rate = float(target_rate)
        self.fps = fps
        self.learn = learn
        self.grow = grow

        self.synaptogenesis_every = 120     # frames
        self.prune_every = 900              # frames
        self._tonic_max = 8.0

        self.frame = 0
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()      # guards every brain mutation
        self.callbacks: List[Callable[["BrainRuntime"], None]] = []

        # a smoothed activation vector gives a *stable* current thought
        self._ema = np.zeros(brain.n_neurons, dtype=np.float32)
        self.thought: Optional[str] = None
        self.thought_log: Deque[str] = deque(maxlen=48)
        self._latest: Optional[dict] = None
        # a fresh sensory input captures attention for a short while, quieting
        # the day-dream so the perceived input wins (bottom-up salience).
        self._input_salience = 0

        self.brain.plastic = self.learn
        self.brain.tonic = self.tonic

    # -- one frame --------------------------------------------------------
    def tick(self) -> dict:
        """Advance the brain by one frame and update the live state."""
        with self._lock:
            self.brain.plastic = self.learn
            self.brain.tonic = self.tonic

            # higher cognition: attention (gains) + a top-down current that is
            # the sum of what the brain is imagining and what it is attending to.
            self.attention.apply_gains()
            self.imagination.step(bias=self.attention.imagination_bias())
            # a recent input quiets the imagination so perception wins
            imag_scale = 0.2 if self._input_salience > 0 else 1.0
            if self._input_salience > 0:
                self._input_salience -= 1
            self.brain.top_down["memory"] = (
                imag_scale * self.imagination.top_down_current()
                + self.attention.top_down_current())

            frame = self.brain.step()

            # homeostasis: keep the ongoing firing rate near the target by
            # nudging the baseline current (excitation/inhibition balance).
            rate = len(frame.spikes) / max(1, self.brain.n_neurons)
            self.tonic += 0.4 * (self.target_rate - rate)
            self.tonic = float(np.clip(self.tonic, 0.0, self._tonic_max))

            # periodic structural plasticity -- the brain reshapes itself
            if self.grow and self.frame:
                if self.frame % self.synaptogenesis_every == 0:
                    self.brain.synaptogenesis(rate=0.25, max_new=800)
                if self.frame % self.prune_every == 0:
                    self.brain.prune_synapses(0.02)
                    if self._ema.shape[0] != self.brain.n_neurons:
                        self._resize_ema()

            # smoothed activation -> a stable readout of the current thought
            if self._ema.shape[0] != frame.activation.shape[0]:
                self._resize_ema()
            self._ema *= np.float32(0.85)
            self._ema += np.float32(0.15) * frame.activation
            thought = self._read_thought()
            if thought and thought != self.thought:
                self.thought_log.append(thought)
            self.thought = thought

            self._latest = {
                "frame": self.frame,
                "activation": frame.activation,
                "active_concepts": frame.active_concepts,
                "thought": thought,
                "rate": round(rate, 4),
                "tonic": round(self.tonic, 3),
                "neurons": self.brain.n_neurons,
                "synapses": int(sum(len(p) for p in self.brain.projections)),
            }
            self.frame += 1

        for cb in self.callbacks:
            cb(self)
        return self._latest

    # -- the loop ---------------------------------------------------------
    def run(self, max_frames: Optional[int] = None) -> None:
        """Run the loop (blocking). ``max_frames=None`` runs forever."""
        self.running = True
        target_dt = (1.0 / self.fps) if self.fps else 0.0
        while self.running and (max_frames is None or self.frame < max_frames):
            t0 = time.time()
            self.tick()
            if target_dt:
                rest = target_dt - (time.time() - t0)
                if rest > 0:
                    time.sleep(rest)     # yield so inputs/teaching can run
        self.running = False

    def start(self) -> "BrainRuntime":
        """Start the loop in a background thread."""
        if self._thread and self._thread.is_alive():
            return self
        self.running = True
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self.running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    # -- feeding inputs into the live loop -------------------------------
    def feed(self, region: str, stimulus: np.ndarray) -> None:
        """Inject a stimulus into the running loop (thread-safe)."""
        with self._lock:
            self.brain.stimulate(region, stimulus)
            # this input captures attention for ~45 frames (quiets the daydream)
            self._input_salience = 45

    def feed_text(self, text: str) -> None:
        self.feed("text", self.text.encode(text))

    def feed_modality(self, modality: str, value: str) -> None:
        if modality == "text":
            self.feed("text", self.text.encode(value))
        elif modality == "image":
            img = dark_image() if value.strip().lower() in ("dark", "0") \
                else bright_image()
            self.feed("vision", self.vision.encode(img))
        elif modality == "sound":
            wave = low_tone() if value.strip().lower() in ("low", "quiet") \
                else high_tone()
            self.feed("sound", self.sound.encode(wave))
        else:
            raise ValueError(f"unknown modality {modality!r}")

    # -- live teaching & growth (pause the loop via the lock) ------------
    def teach(self, mode: str, value: str) -> dict:
        with self._lock:
            before = self.brain.structure_stats()
            saved_tonic = self.tonic
            self.tonic = 0.0                      # calm while imprinting
            # silence imagination/attention injection so it can't interfere
            saved_td = self.brain.top_down
            saved_gain = self.brain.region_gain
            self.brain.top_down = {}
            self.brain.region_gain = {}
            value = (value or "").strip()
            if mode == "word":
                for w in value.split():
                    self.language.ground_word(w, repeats=6)
                msg = f"grounded: {value}"
            elif mode == "sequence":
                words = [w for w in value.split() if w]
                self.language.teach_sentence(" ".join(words), repeats=6)
                msg = f"taught order: {' → '.join(words)}"
            else:
                raise ValueError(f"unknown teach mode {mode!r}")
            self.tonic = saved_tonic
            self.brain.top_down = saved_td       # overwritten again next tick
            self.brain.region_gain = saved_gain
            self.brain.reset_state(keep_weights=True)
            self._resize_ema()
            after = self.brain.structure_stats()
        return {"message": msg,
                "grew_neurons": after["neurons"] - before["neurons"],
                "grew_synapses": after["synapses"] - before["synapses"],
                **self.stats()}

    def grow_now(self, kind: str) -> dict:
        with self._lock:
            if kind == "synaptogenesis":
                n = self.brain.synaptogenesis(rate=0.6)
                msg = f"synaptogenesis: +{n} synapses"
            elif kind == "neurogenesis":
                self.brain.grow_region("memory", 40)
                self._resize_ema()
                msg = "neurogenesis: +40 memory neurons"
            elif kind == "prune":
                n = self.brain.prune_synapses(0.03)
                msg = f"pruned {n} weak synapses"
            else:
                raise ValueError(f"unknown growth {kind!r}")
        return {"message": msg, **self.stats()}

    # -- higher cognition controls ---------------------------------------
    def attend(self, modality: Optional[Dict[str, float]] = None,
               concepts: Optional[List[str]] = None,
               weight: float = 1.0) -> dict:
        """Point the multi-faceted attention: a modality mix and/or concepts."""
        with self._lock:
            if modality is not None:
                self.attention.attend_modality(modality)
            if concepts is not None:
                if concepts:
                    self.attention.attend_concepts(concepts, weight=weight)
                else:
                    self.attention.concept_focus = {}
            self.attention.apply_gains()      # take effect immediately
        return self.attention.summary()

    def clear_attention(self) -> dict:
        with self._lock:
            self.attention.clear()
        return self.attention.summary()

    def set_imagination(self, enabled: Optional[bool] = None,
                        strength: Optional[float] = None,
                        temperature: Optional[float] = None) -> dict:
        with self._lock:
            im = self.imagination
            if enabled is not None:
                im.enabled = bool(enabled)
            if strength is not None:
                im.strength = float(strength)
            if temperature is not None:
                im.temperature = float(temperature)
        return {"enabled": self.imagination.enabled,
                "strength": self.imagination.strength,
                "temperature": self.imagination.temperature}

    # -- reading the live state ------------------------------------------
    def stats(self) -> dict:
        return {"neurons": self.brain.n_neurons,
                "synapses": int(sum(len(p) for p in self.brain.projections)),
                "vocabulary": sorted(self.brain.world.word_index)}

    def state(self) -> dict:
        """The latest frame snapshot (for a viewer)."""
        if self._latest is None:
            return {"frame": 0, "thought": None, "rate": 0.0,
                    "tonic": self.tonic, **self.stats()}
        s = dict(self._latest)
        s.pop("activation", None)          # heavy; viewers use node_activation()
        s["vocabulary"] = sorted(self.brain.world.word_index)
        s["imagining"] = self.imagination.imagined_word()
        s["imagination_stream"] = list(self.imagination.stream)[-10:]
        s["imagination_on"] = self.imagination.enabled
        s["attention"] = self.attention.summary()
        return s

    def node_activation(self, gids: np.ndarray) -> List[float]:
        """Latest activation for a set of global neuron ids (for the graph)."""
        if self._latest is None:
            return [0.0] * len(gids)
        act = self._latest["activation"]
        gids = gids[gids < act.shape[0]]
        return [round(float(a), 3) for a in act[gids]]

    def region_rates(self) -> Dict[str, float]:
        if self._latest is None:
            return {n: 0.0 for n in self.brain.regions}
        act = self._latest["activation"]
        return {n: round(float(act[r.gid_start:r.gid_start + r.size].mean()), 4)
                for n, r in self.brain.regions.items()}

    def _read_thought(self, threshold: float = 0.11) -> Optional[str]:
        """The strongest *word* concept in the smoothed activation, if any.

        More sensitive than world.recognise (which uses the strict active-concept
        threshold), so a fed input registers as a thought promptly.
        """
        world = self.brain.world
        ema = self._ema
        best, best_s = None, 0.0
        for name, c in world.concepts.items():
            if c.kind != "word" or len(c.gids) == 0:
                continue
            g = c.gids[c.gids < ema.shape[0]]
            s = float(ema[g].mean()) if len(g) else 0.0
            if s > best_s:
                best, best_s = name, s
        if best and best_s >= threshold:
            c = world.concepts[best]
            return c.word or c.name
        return None

    # -- internals --------------------------------------------------------
    def _resize_ema(self) -> None:
        n = self.brain.n_neurons
        new = np.zeros(n, dtype=np.float32)
        m = min(n, self._ema.shape[0])
        new[:m] = self._ema[:m]
        self._ema = new
