"""
mind.py
=======

Where perception becomes an inner world. The sensory streams (vision -> IT
objects, hearing -> belt sounds) and the multisensory association area give the
brain *percepts*; this module grounds those percepts into **concept assemblies**,
learns a **world model** of how concepts follow one another, **imagines**
sequences of concepts on its own, and uses an imagined goal to steer **top-down
attention** back down into vision.

This is the bridge the earlier versions were missing: the perceptual hierarchies
and the inner world (world model + imagination + attention) were two separate
subsystems. Here they meet -- imagination now runs in the *learned perceptual
space*, and attention can be driven by what the mind expects to see.

    perceive  ->  ground into a concept        (see a shape / hear a sound)
    experience -> learn concept transitions     (the world model)
    imagine   ->  walk the transitions          (an inner train of thought)
    envision  ->  recall a concept's percept    (mental imagery)
    search    ->  imagine a goal, attend to it  (top-down attention)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Mind:
    """Perception grounded into a world model you can imagine with."""

    brain: object                        # MultisensoryBrain (vision + audio + assoc)
    concepts: List[str]                  # concept (shape) names, in order
    sound_of: Dict[str, str] = field(default_factory=dict)   # concept -> sound
    _T: Optional[np.ndarray] = None      # concept transition weights (world model)

    def __post_init__(self):
        from .worldmodel import PredictiveWorldModel
        n = len(self.concepts)
        self._T = np.full((n, n), 0.05, dtype=np.float32)     # small uniform prior
        self._sound_to_concept = {s: c for c, s in self.sound_of.items()}
        # the predictive world model: transitions + successor representation +
        # action-conditioned dynamics + a latent belief (see worldmodel.py).
        self.wm = PredictiveWorldModel()

    def _idx(self, name: str) -> int:
        return self.concepts.index(name)

    # -- grounding: a percept becomes a concept --------------------------
    def see(self, image: np.ndarray) -> str:
        """Recognise the object in an image and return its concept."""
        name = self.brain.vision.classify_object(image)
        return name if name in self.concepts else name

    def hear(self, sound: np.ndarray) -> str:
        """Recognise a sound and return the concept it belongs to."""
        snd = self.brain.audio.classify(sound)
        return self._sound_to_concept.get(snd, snd)

    def see_then_hear(self, image: np.ndarray) -> str:
        """Cross-modal: recognise the shape, then recall the sound it makes."""
        c = self.see(image)
        return self.sound_of.get(c, c)

    def hear_then_see(self, sound: np.ndarray) -> str:
        """Cross-modal: recognise the sound, then recall the shape it belongs to."""
        return self.hear(sound)

    # -- world model: learn how concepts follow one another --------------
    def experience(self, sequence: List[str],
                   actions: Optional[List[str]] = None) -> None:
        """Watch a sequence of concepts go by and strengthen their transitions.
        If ``actions`` are given, the world model also learns the *action-
        conditioned* dynamics (what each action tends to cause)."""
        for a, b in zip(sequence[:-1], sequence[1:]):
            if a in self.concepts and b in self.concepts:
                self._T[self._idx(a), self._idx(b)] += 1.0
        # feed the predictive world model (grows its own state space)
        self.wm.experience(sequence, actions)

    def experience_percepts(self, images: List[np.ndarray]) -> List[str]:
        """Ground a stream of images into concepts and learn their transitions."""
        seq = [self.see(im) for im in images]
        self.experience(seq)
        return seq

    def expected_next(self, concept: str) -> str:
        """The single most likely next concept (the world model's prediction)."""
        pred = self.wm.predict_next(concept)          # predictive model first
        if pred is not None:
            return pred
        return self.concepts[int(self._T[self._idx(concept)].argmax())]

    # -- the predictive world model, surfaced on the mind ----------------
    def predict_next(self, concept: str, action: Optional[str] = None
                     ) -> Optional[str]:
        """What tends to follow ``concept`` (optionally under an ``action``)."""
        return self.wm.predict_next(concept, action)

    def surprise(self, prev: str, cur: str) -> float:
        """How unexpected ``prev -> cur`` was (prediction error, in [0, 1])."""
        return self.wm.surprise(prev, cur)

    def reachable(self, concept: str, top: int = 5):
        """Where a concept tends to lead over many steps (successor rep.)."""
        return self.wm.reachable(concept, top=top)

    def plan(self, start: str, goal: str):
        """A sequence of actions the model expects to lead from start to goal."""
        return self.wm.plan(start, goal)

    # -- imagination: a generative walk over the world model -------------
    def imagine(self, seed: str, steps: int = 6, temperature: float = 0.6,
                rng_seed: int = 0) -> List[str]:
        """Set off from ``seed`` and let one concept call up the next -- an inner
        train of thought that follows what experience made likely."""
        rng = np.random.default_rng(rng_seed)
        cur = self._idx(seed)
        out = [seed]
        for _ in range(steps):
            p = self._T[cur] ** (1.0 / max(temperature, 1e-3))
            p = p / p.sum()
            cur = int(rng.choice(len(p), p=p))
            out.append(self.concepts[cur])
        return out

    def envision(self, concept: str) -> Tuple[np.ndarray, np.ndarray]:
        """Mental imagery: the visual and auditory prototype a concept evokes
        (the association area's learned templates)."""
        c = self._idx(concept)
        return self.brain.assoc.Wv[c].copy(), self.brain.assoc.Wa[c].copy()

    # -- top-down: imagine a goal, then attend to find it ----------------
    def search_for(self, image: np.ndarray, goal: str, beta: float = 10.0) -> str:
        """Look at an image *for* an imagined goal object (world-model-driven
        top-down attention) and report what attention settles on."""
        return self.brain.vision.attend(image, goal, beta=beta)

    def search_expected(self, image: np.ndarray, current: str,
                        beta: float = 10.0) -> str:
        """World-model-guided attention: attend to whatever the world model
        expects to come next after ``current``."""
        return self.search_for(image, self.expected_next(current), beta=beta)

    # -- online growth: meet and learn a brand-new concept ---------------
    def learn_new_concept(self, name: str, sound_name: str,
                          images: List[np.ndarray],
                          sounds: List[np.ndarray]) -> int:
        """Meet a new audio-visual concept and, if it is genuinely novel, grow a
        concept cell for it on the spot (online structural growth). Returns the
        new concept index, or -1 if it was already known."""
        v = np.mean([self.brain.vision_code(im) for im in images], axis=0)
        a = np.mean([self.brain.sound_code(s) for s in sounds], axis=0)
        idx = self.brain.assoc.grow_concept(v, a)
        if idx < 0:
            return -1
        self.concepts.append(name)
        self.sound_of[name] = sound_name
        self._sound_to_concept[sound_name] = name
        n = len(self.concepts)
        T = np.full((n, n), 0.05, dtype=np.float32)
        T[:n - 1, :n - 1] = self._T
        self._T = T
        return idx


def build_mind(pairs: Optional[List[Tuple[str, str]]] = None,
               verbose: bool = False) -> Mind:
    """Build the whole thing: sensory streams + association + world model, and
    measure how well percepts ground into concepts. Returns a :class:`Mind`."""
    from .multimodal import build_multisensory_brain
    from .vision import shape_images
    from .audio import sound_dataset

    def say(*a):
        if verbose:
            print(*a)

    if pairs is None:
        pairs = [("circle", "pure"), ("square", "up_chirp"),
                 ("triangle", "down_chirp"), ("star", "harmonic"),
                 ("cross", "noise"), ("arrow", "am"),
                 ("hexagon", "fm"), ("heart", "chord")]
    brain = build_multisensory_brain(pairs=pairs, verbose=verbose)
    concepts = [s for s, _ in pairs]
    sound_of = {s: a for s, a in pairs}
    mind = Mind(brain, concepts, sound_of)

    # grounding accuracy: does a percept land on the right concept?
    say("grounding: recognising percepts as concepts ...")
    shapes = tuple(concepts)
    imgs, ilab, _ = shape_images(20, brain.vision.size, seed=71, classes=shapes,
                                 noise=0.04, occlude=0.06)
    vg = np.mean([mind.see(im) == concepts[l] for im, l in zip(imgs, ilab)])
    mind.vision_grounding = float(vg)

    sigs, slab, snames = sound_dataset(20, seed=72)
    snd_ok = 0
    for s, l in zip(sigs, slab):
        want = mind._sound_to_concept.get(snames[l])
        if want is not None:
            snd_ok += mind.hear(s) == want
    mind.sound_grounding = snd_ok / sum(
        1 for l in slab if snames[l] in mind._sound_to_concept)

    # cross-modal recall through the world: recognise, then recall the pair
    mind.cross_see_hear = float(np.mean(
        [mind.see_then_hear(im) == sound_of[concepts[l]]
         for im, l in zip(imgs, ilab)]))
    denom = [l for l in slab if snames[l] in mind._sound_to_concept]
    mind.cross_hear_see = float(np.sum(
        [mind.hear_then_see(s) == mind._sound_to_concept[snames[l]]
         for s, l in zip(sigs, slab)
         if snames[l] in mind._sound_to_concept]) / len(denom))

    say(f"   grounding: see->concept {mind.vision_grounding:.0%}, "
        f"hear->concept {mind.sound_grounding:.0%}")
    say(f"   cross-modal: see->hear {mind.cross_see_hear:.0%}, "
        f"hear->see {mind.cross_hear_see:.0%}")
    return mind
