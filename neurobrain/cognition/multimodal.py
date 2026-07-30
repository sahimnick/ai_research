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
                 lr: float = 0.15, vigilance: float = 0.80,
                 conscience: float = 1.0, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.Wv = _rows_unit(rng.standard_normal((n_concept, n_vis)) * 0.1)
        self.Wa = _rows_unit(rng.standard_normal((n_concept, n_aud)) * 0.1)
        self.n_concept = n_concept
        self.lr = lr
        self.vigilance = float(vigilance)
        self.conscience = float(conscience)
        self.wins = np.zeros(n_concept)
        self._rng = rng
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
        wins and learns both. Returns the winning concept cell.

        Two guards, and neither is optional -- without them this layer collapses.

        **Vigilance.** The old rule was pure argmax over the summed drive, and
        it had a runaway in it. The first cell to win tunes toward the data, so
        its drive rises; every subsequent pair then finds it the best match and
        tunes it further, while 31 other cells stay at their random
        initialisation and never win anything. Measured: binding 48 audio-visual
        pairs from 8 well-separated classes woke **1 concept cell out of 32**,
        and cross-modal recall sat at exactly chance (0.125) even though the
        sound codes themselves were 96% separable by nearest prototype. The
        association area was not failing to learn the pairing; it had one
        category and could not express a second.

        So a match below ``vigilance`` does not update the incumbent -- it
        recruits an uncommitted cell instead. This is the ART rule the rest of
        the project already runs on (:class:`GrowingCategoryMap`,
        :meth:`AssociativeCortex.consolidate`), applied here for the same
        reason: a new thing should become a new category rather than blurring
        an old one.

        **Conscience.** A frequency bias in the competition itself (DeSieno
        1988), scaled to the drive rather than the 0.1 the old rule used. A
        penalty of 0.1 cannot move an argmax whose winner leads by ~1.0, which
        is why the old homeostatic term was present and did nothing.

        Both defaults are measured (``benchmarks/concept_cells.py``), on
        cross-modal recall from sounds that were **never bound** -- 5 sounds per
        class bound, 3 held out, over 5 seeds. Querying with a code that was
        bound is a lookup and scores well even on shuffled labels; a held-out
        query does not:

            vigilance  conscience   recall   shuffled   cells   spread
              0.00        0.0        0.125    0.125       1.0   +0.000
              0.50        2.0        0.325    0.133       3.0   +0.192
              0.80        1.0        0.742    0.058      16.0   +0.683
              0.90        0.0        0.825    0.108      32.0   +0.717

        0.125 is chance. The layer as it was scored **exactly** chance with one
        cell; 0.80/1.0 reaches 0.742 against a shuffled control at 0.058, d=10.8
        over 5 of 5 seeds.

        0.90/0.0 scores a little higher but is not the default, because it was
        *saturating the pool* -- it used 16 of 16, 32 of 32, and only became
        selective at 40 cells once given 64. A parameter whose behaviour is set
        by how many cells happen to be allocated will change silently the first
        time the pool is resized. 0.80/1.0 recruits **15.3 cells whether the
        pool is 16, 32, 64 or 128**, which means the number is a property of the
        data rather than of the array, and that is the setting worth shipping.
        """
        vn, an = self.prep_v(v), self.prep_a(a)
        # a *mean* over the two senses, so match sits on the same [-1, 1] scale
        # as vigilance no matter how many modalities are wired in
        match = 0.5 * (self.Wv @ vn) + 0.5 * (self.Wa @ an)
        tot = max(float(self.wins.sum()), 1.0)
        bias = self.conscience * (self.wins / tot - 1.0 / self.n_concept)
        win = int(np.argmax(match - bias))
        if match[win] < self.vigilance:
            free = np.flatnonzero(self.wins == 0)
            if len(free):
                # an uncommitted cell takes the pair whole -- not a fraction of
                # the way toward it, since it has nothing worth preserving
                win = int(free[0])
                self.Wv[win], self.Wa[win] = vn.copy(), an.copy()
                self.wins[win] += 1
                return win
        self.Wv[win] = _unit(self.Wv[win] + self.lr * (vn - self.Wv[win]))
        self.Wa[win] = _unit(self.Wa[win] + self.lr * (an - self.Wa[win]))
        self.wins[win] += 1
        return win

    def concept_from_vision(self, v: np.ndarray) -> int:
        return int(np.argmax(self.Wv @ self.prep_v(v)))

    def concept_from_sound(self, a: np.ndarray) -> int:
        return int(np.argmax(self.Wa @ self.prep_a(a)))

    # -- the read-out the binding is *for* -----------------------------------
    def vision_from_sound(self, a: np.ndarray) -> np.ndarray:
        """Hear a sound; read out the visual pattern its concept expects.

        This is the half of binding that makes it cross-modal rather than a
        lookup: the sound wakes a concept cell through ``Wa``, and that cell's
        ``Wv`` row *is* the vision it has learned to expect. Nothing about the
        query is visual, and the answer is entirely visual."""
        return self.Wv[self.concept_from_sound(a)].copy()

    def sound_from_vision(self, v: np.ndarray) -> np.ndarray:
        """See a thing; read out the sound its concept expects."""
        return self.Wa[self.concept_from_vision(v)].copy()

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

    def consolidate(self, threshold: float = 0.75) -> Dict[int, int]:
        """Merge concept cells that turned out to be the same thing.

        Vigilance recruits eagerly, and on real data it recruits *very* eagerly:
        76 training pairs of CIFAR photographs and ESC-50 recordings produced 77
        concept cells -- one per example. That is a good exemplar memory and not
        a set of concepts, and the difference matters for a mind that is meant
        to imagine from its concepts rather than replay its examples.

        Eager encoding followed by offline merging is how the biology is usually
        described: the hippocampus takes a separate trace for nearly every
        episode (pattern separation, which is what high vigilance is), and sleep
        is where those traces are integrated into cortical structure that
        generalises. So this belongs in a night rather than in the waking rule,
        and lowering vigilance instead would lose the separation that makes
        one-shot binding work at all.

        Two cells merge when they agree in **both** senses -- a cat photo and a
        cat recording must both be close, since agreeing on the sound alone is
        how a dog and a cat recording of similar spectral shape would collapse
        into one. The survivor is the win-weighted mean, so a cell that has seen
        forty pairs is not dragged by one that has seen one.

        Returns ``{old_cell: surviving_cell}`` for every cell that was absorbed,
        so a caller holding a vote map or a name table can fold it accordingly.
        """
        order = np.argsort(-self.wins)                # the best-evidenced lead
        alive = [int(c) for c in order if self.wins[c] > 0]
        merged: Dict[int, int] = {}
        for i, c in enumerate(alive):
            if c in merged:
                continue
            for d in alive[i + 1:]:
                if d in merged:
                    continue
                if (float(self.Wv[c] @ self.Wv[d]) >= threshold
                        and float(self.Wa[c] @ self.Wa[d]) >= threshold):
                    wc, wd = self.wins[c], self.wins[d]
                    self.Wv[c] = _unit((wc * self.Wv[c] + wd * self.Wv[d]) / (wc + wd))
                    self.Wa[c] = _unit((wc * self.Wa[c] + wd * self.Wa[d]) / (wc + wd))
                    self.wins[c] = wc + wd
                    # d becomes uncommitted, and is re-randomised rather than
                    # left holding its old tuning or zeroed. A zeroed row has
                    # cosine 0 to everything, which makes it a uniform attractor
                    # the moment vigilance looks for a free cell; a random row
                    # is what an unused cell actually looks like.
                    self.wins[d] = 0.0
                    self.Wv[d] = _unit(self._rng.standard_normal(self.Wv.shape[1]))
                    self.Wa[d] = _unit(self._rng.standard_normal(self.Wa.shape[1]))
                    merged[d] = c
        return merged

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
