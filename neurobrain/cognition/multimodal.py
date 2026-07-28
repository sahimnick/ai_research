"""
multimodal.py
=============

Where seeing and hearing meet. On top of the visual stream (which ends in IT
object cells) and the auditory stream (which ends in belt sound cells) sits a
**multisensory association area** -- the brain's convergence zone (superior
temporal sulcus / association cortex). It learns, by Hebbian co-occurrence, a
shared pool of **concept cells**, each of which fires for one audio-visual thing
(the *look* of a circle bound to the *sound* it makes).

Because the two senses meet on the same concept cells, the area does what a
convergence zone does: **cross-modal completion**. Show it only the picture and
it recalls the expected sound; play it only the sound and it recalls the
expected shape -- the neural basis of "hear a bell, picture a bell".

Nothing here is a lookup table: the binding lives in the synapses from each
modality onto the shared concept cells (``Wv`` and ``Wa``), learned the same
competitive-Hebbian way the sensory areas learned their features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


def _unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x)
    return (x / n).astype(np.float32) if n > 1e-6 else x.astype(np.float32)


class AssociationArea:
    """A shared pool of concept cells wired to both senses.

    ``Wv`` are the synapses from the visual object code onto the concept cells,
    ``Wa`` from the sound code. ``bind`` presents both senses together and lets a
    concept cell win and tune both its visual and auditory synapses toward what
    it saw and heard -- so afterwards either sense alone can wake the concept and
    read out the other sense's expected pattern.
    """

    def __init__(self, n_vis: int, n_aud: int, n_concept: int,
                 lr: float = 0.15, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.Wv = _rows_unit(rng.standard_normal((n_concept, n_vis)) * 0.1)
        self.Wa = _rows_unit(rng.standard_normal((n_concept, n_aud)) * 0.1)
        self.n_concept = n_concept
        self.lr = lr
        self.wins = np.zeros(n_concept)
        self.v_mu = self.v_sd = self.a_mu = self.a_sd = None

    def set_stats(self, V: np.ndarray, A: np.ndarray) -> None:
        """Learn each modality's feature statistics so the concept cells compare
        codes on an equal footing (divisive normalisation -- what cortex does)."""
        self.v_mu, self.v_sd = V.mean(0), V.std(0) + 1e-6
        self.a_mu, self.a_sd = A.mean(0), A.std(0) + 1e-6

    def prep_v(self, v: np.ndarray) -> np.ndarray:
        return _unit(v if self.v_mu is None else (v - self.v_mu) / self.v_sd)

    def prep_a(self, a: np.ndarray) -> np.ndarray:
        return _unit(a if self.a_mu is None else (a - self.a_mu) / self.a_sd)

    def bind(self, v: np.ndarray, a: np.ndarray) -> int:
        """Co-present a visual code ``v`` and a sound code ``a``; a concept cell
        wins and learns both. Returns the winning concept cell."""
        vn, an = self.prep_v(v), self.prep_a(a)
        drive = self.Wv @ vn + self.Wa @ an - 0.1 * (self.wins /
                                                     (self.wins.sum() + 1))
        win = int(np.argmax(drive))
        self.Wv[win] = _unit(self.Wv[win] + self.lr * (vn - self.Wv[win]))
        self.Wa[win] = _unit(self.Wa[win] + self.lr * (an - self.Wa[win]))
        self.wins[win] += 1
        return win

    def concept_from_vision(self, v: np.ndarray) -> int:
        return int(np.argmax(self.Wv @ self.prep_v(v)))

    def concept_from_sound(self, a: np.ndarray) -> int:
        return int(np.argmax(self.Wa @ self.prep_a(a)))

    def novelty(self, v: np.ndarray, a: np.ndarray) -> float:
        """How poorly the current concept cells explain this audio-visual pair
        (1 = brand new, 0 = already a known concept)."""
        vn, an = self.prep_v(v), self.prep_a(a)
        best = 0.5 * (self.Wv @ vn).max() + 0.5 * (self.Wa @ an).max()
        return float(1.0 - max(best, 0.0))

    def grow_concept(self, v: np.ndarray, a: np.ndarray,
                     novelty_thresh: float = 0.4) -> int:
        """Online structural growth: if a pairing is novel enough, sprout a new
        concept cell tuned to it. Returns the new cell index, or -1 if the pair
        is already covered by an existing concept."""
        if self.novelty(v, a) < novelty_thresh:
            return -1
        self.Wv = np.vstack([self.Wv, self.prep_v(v)]).astype(np.float32)
        self.Wa = np.vstack([self.Wa, self.prep_a(a)]).astype(np.float32)
        self.wins = np.concatenate([self.wins, [1.0]])
        self.n_concept += 1
        return self.n_concept - 1

    def expect_sound(self, v: np.ndarray) -> np.ndarray:
        """See -> imagine: the sound code the vision-evoked concept expects."""
        return self.Wa[self.concept_from_vision(v)]

    def expect_vision(self, a: np.ndarray) -> np.ndarray:
        """Hear -> imagine: the visual code the sound-evoked concept expects."""
        return self.Wv[self.concept_from_sound(a)]


def _rows_unit(W: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(W, axis=1, keepdims=True)
    n[n < 1e-6] = 1.0
    return (W / n).astype(np.float32)


@dataclass
class MultisensoryBrain:
    """Vision + hearing + the association area that binds them."""

    vision: object            # VentralStream
    audio: object             # AuditoryStream
    assoc: AssociationArea
    concept_label: Optional[np.ndarray] = None   # concept cell -> pair index
    pair_names: Optional[List[str]] = None       # "circle+pure", ...

    def vision_code(self, image: np.ndarray) -> np.ndarray:
        return self.vision.it_code(image)

    def sound_code(self, sig: np.ndarray) -> np.ndarray:
        # the full auditory descriptor is the sound's identity (spectral +
        # modulation + belt) -- discriminative enough to name the sound
        return self.audio.descriptor(sig)

    def learn_pair(self, image: np.ndarray, sig: np.ndarray) -> int:
        return self.assoc.bind(self.vision_code(image), self.sound_code(sig))

    def see_then_hear(self, image: np.ndarray) -> str:
        """Given only a picture, name the sound the brain expects (cross-modal)."""
        c = self.assoc.concept_from_vision(self.vision_code(image))
        return self._sound_of(c)

    def hear_then_see(self, sig: np.ndarray) -> str:
        """Given only a sound, name the shape the brain pictures (cross-modal)."""
        c = self.assoc.concept_from_sound(self.sound_code(sig))
        return self._shape_of(c)

    def _shape_of(self, c: int) -> str:
        return self.pair_names[int(self.concept_label[c])].split("+")[0]

    def _sound_of(self, c: int) -> str:
        return self.pair_names[int(self.concept_label[c])].split("+")[1]


def build_multisensory_brain(pairs: Optional[List[Tuple[str, str]]] = None,
                             n_concept: int = 0, n_per_pair: int = 24,
                             verbose: bool = False) -> MultisensoryBrain:
    """Build both sensory streams, bind shape<->sound pairs, and measure
    cross-modal recall. Returns a :class:`MultisensoryBrain`."""
    from ..vision.ventral import build_ventral_stream, shape_images
    from ..audition.audio import build_auditory_stream, sound_dataset

    def say(*a):
        if verbose:
            print(*a)

    if pairs is None:
        pairs = [("circle", "pure"), ("star", "harmonic"), ("square", "up_chirp"),
                 ("triangle", "down_chirp"), ("bar", "am"), ("cross", "noise")]
    shapes = tuple(p[0] for p in pairs)
    pair_names = [f"{s}+{a}" for s, a in pairs]

    say("building the visual stream ...")
    vision = build_ventral_stream(verbose=verbose)
    say("building the auditory stream ...")
    audio = build_auditory_stream(verbose=verbose)

    # pre-compute per-pair visual object codes and sound codes
    imgs, ilab, _ = shape_images(n_per_pair, vision.size, seed=31,
                                 classes=shapes, noise=0.04)
    sigs, slab, snames = sound_dataset(n_per_pair, seed=32)
    say("association area: binding what is seen to what is heard ...")
    n_concept = n_concept or len(pairs)
    n_aud = len(audio.descriptor(sigs[0]))
    assoc = AssociationArea(vision.IT.n_units, n_aud, n_concept)
    brain = MultisensoryBrain(vision, audio, assoc, pair_names=pair_names)

    # each pair's examples are bound onto one concept cell -- repeated Hebbian
    # co-activation, whose fixed point is the mean audio-visual code
    v_by_pair, a_by_pair = {}, {}
    for p, (shape, snd) in enumerate(pairs):
        v_by_pair[p] = [brain.vision_code(im) for im in imgs[ilab == p]]
        a_by_pair[p] = [brain.sound_code(s) for s in
                        [sigs[i] for i in range(len(sigs))
                         if snames[slab[i]] == snd]]
    assoc.set_stats(np.array([v for vs in v_by_pair.values() for v in vs]),
                    np.array([a for as_ in a_by_pair.values() for a in as_]))
    for p in range(len(pairs)):
        assoc.Wv[p] = _unit(np.mean([assoc.prep_v(v) for v in v_by_pair[p]], 0))
        assoc.Wa[p] = _unit(np.mean([assoc.prep_a(a) for a in a_by_pair[p]], 0))
        assoc.wins[p] = len(v_by_pair[p])
    brain.concept_label = np.arange(len(pairs))

    # cross-modal accuracy on fresh examples: see->name sound, hear->name shape
    t_imgs, t_ilab, _ = shape_images(12, vision.size, seed=99,
                                     classes=shapes, noise=0.04)
    v2s = 0
    for im, lab in zip(t_imgs, t_ilab):
        v2s += brain.see_then_hear(im) == pairs[lab][1]
    brain.vision_to_sound_acc = v2s / len(t_imgs)

    t_sigs, t_slab, t_sn = sound_dataset(12, seed=98)
    snd_to_pair = {snd: p for p, (_, snd) in enumerate(pairs)}
    s2v = seen = 0
    for s, lab in zip(t_sigs, t_slab):
        snd = t_sn[lab]
        if snd not in snd_to_pair:           # this sound isn't a bound concept
            continue
        seen += 1
        s2v += brain.hear_then_see(s) == pairs[snd_to_pair[snd]][0]
    brain.sound_to_vision_acc = s2v / max(seen, 1)

    # binding consistency: do vision and sound of the same concept wake the
    # SAME concept cell? (the two senses meeting on one neuron)
    same = 0
    for p, (shape, snd) in enumerate(pairs):
        vi = brain.assoc.concept_from_vision(v_by_pair[p][0])
        ai = brain.assoc.concept_from_sound(a_by_pair[p][0])
        same += vi == ai == p
    brain.binding_consistency = same / len(pairs)
    say(f"   cross-modal: see->hear {brain.vision_to_sound_acc:.0%}, "
        f"hear->see {brain.sound_to_vision_acc:.0%}, "
        f"binding consistency {brain.binding_consistency:.0%}")
    return brain
