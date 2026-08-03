"""The adapter that lets the already-written :class:`~neurobrain.minds.mind.Mind`
run on a real eye and a real ear instead of synthetic shapes and chirps.

Why this file exists
--------------------
``minds/mind.py`` was already an integrated brain -- perception grounded into
concepts, a world model over those concepts, imagination, surprise, planning,
and top-down search. It was never dormant for want of design. It was dormant
because :func:`~neurobrain.minds.mind.build_mind` feeds it ``shape_images`` and
``sound_dataset``: drawn circles and 400 ms chirps. Every number in EVALUATION
§9.18-§9.36 is on real video. The two never met.

``Mind`` asks its ``brain`` for exactly seven things::

    brain.vision.classify_object(image) -> str
    brain.vision.attend(image, cue, beta) -> str
    brain.audio.classify(sound) -> str
    brain.assoc.Wv , brain.assoc.Wa
    brain.assoc.grow_concept(v, a) -> int
    brain.vision_code(image) , brain.sound_code(sound)

:class:`EyeBrain` supplies those seven from :class:`~neurobrain.vision.unified_eye.UnifiedEye`
(224 px, the canvas §9.35 measured at 0.902-0.910) and
:class:`~neurobrain.audition.audio.AuditoryStream` (a real cochleagram, not a
tone generator). ``Mind`` itself is not modified -- not one line.

The visual code, and why averaging is right here but wrong for a tag
--------------------------------------------------------------------
:meth:`vision_code` mean-pools each area's map over space. §9.21 measured that
exact operation at **0.000** and it is worth being precise about why that is not
a contradiction: §9.21 was *locating* -- finding where in a frame a tag sits --
and location is carried entirely by the map's spatial layout, so averaging it
away leaves nothing to correlate. Naming is the other question. "What is this
region" is a question about which features are present, not where, and pooling
is the standard answer to it (it is what ``VentralStream.it_code`` already
does). The two must not be swapped: this class deliberately uses pooling for
``vision_code`` and keeps :func:`~neurobrain.vision.ventral.locate_template`'s
spatial patch for anything positional.

What is measured here, and what is not
--------------------------------------
H23 (``benchmarks/brain_naming.py``) measures ``classify_object`` only.
:meth:`_Vision.attend` is written here because ``Mind.search_for`` calls it and
a half-built adapter would crash rather than measure -- but it is **unmeasured
until H24**, and no claim rests on it in the H23 write-up.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def _pool(maps: Dict[str, np.ndarray], areas: Sequence[str]) -> np.ndarray:
    """Mean-pool each area's map over space and concatenate: the object code."""
    return np.concatenate([np.asarray(maps[a], np.float32).reshape(
        np.asarray(maps[a]).shape[0], -1).mean(1) for a in areas])


class _Vision:
    """The ``brain.vision`` port: name a region, and look for a named one."""

    def __init__(self, owner: "EyeBrain"):
        self.o = owner

    def classify_object(self, image: np.ndarray) -> str:
        """Name a region by which concept cell it wakes.

        This is the association area doing the naming, not a supervised
        read-out -- the same route ``MultisensoryBrain`` uses, pointed at a
        real crop instead of a drawn shape.
        """
        o = self.o
        c = o.assoc.concept_from_vision(o.vision_code(image))
        return o.concepts[c] if 0 <= c < len(o.concepts) else "?"

    def attend(self, image: np.ndarray, cue: str, beta: float = 10.0,
               power: float = 3.0) -> str:
        """Look at ``image`` *for* ``cue``: top-down gain, then name what wins.

        Unmeasured until H24. The point of running it on ``UnifiedEye``'s
        224 px canvas is that ``VentralStream``'s own canvas is 56 px, where
        the V4 map is 7x21 with 100% of cells above 55% of peak -- saturated,
        with no layout for a template to exploit. That saturation, not the
        choice of tag, is what §2 of the plan named as the limiting factor.
        """
        o = self.o
        if cue not in o.concepts:
            return self.classify_object(image)
        M = o.eye._pass(image)
        top = o.area_proto.get(cue)
        if top is None:
            return self.classify_object(image)
        gated = {}
        for a in o.areas:
            m = np.asarray(M[a], np.float32)
            p = top[a]
            p = p / (np.linalg.norm(p) + 1e-6)
            h = np.maximum(np.tensordot(p, m, axes=(0, 0)), 0.0)
            mx = h.max()
            h = (h / mx if mx > 1e-6 else h) ** power
            gated[a] = m * (1.0 + beta * h)[None]
        c = o.assoc.concept_from_vision(_pool(gated, o.areas))
        return o.concepts[c] if 0 <= c < len(o.concepts) else "?"


class _Audio:
    """The ``brain.audio`` port: name a sound."""

    def __init__(self, owner: "EyeBrain"):
        self.o = owner

    def classify(self, sig: np.ndarray) -> str:
        """Name a real recording by which concept cell its sound code wakes.

        Deliberately *not* ``AuditoryStream.classify``: that is a supervised
        read-out fitted on synthetic ``sound_dataset`` classes, and its label
        set is the wrong one. Going through the association area keeps the
        naming in the same currency as vision, which is what makes
        ``see_then_hear`` mean anything.
        """
        o = self.o
        c = o.assoc.concept_from_sound(o.sound_code(sig))
        name = o.concepts[c] if 0 <= c < len(o.concepts) else "?"
        return o.sound_of.get(name, name)


class EyeBrain:
    """A ``brain`` for :class:`~neurobrain.minds.mind.Mind`, built from the eye
    and ear that the benchmarks actually measure.

    Parameters
    ----------
    eye
        A **developed** :class:`~neurobrain.vision.unified_eye.UnifiedEye`.
    ear
        An :class:`~neurobrain.audition.audio.AuditoryStream`.
    assoc
        An :class:`~neurobrain.cognition.multimodal.AssociationArea` sized to
        this eye's and ear's code widths.
    concepts
        Concept names, indexed the same way as ``assoc``'s concept cells.
    sound_of
        concept -> sound name, the arbitrary-but-consistent pairing a word is.
    """

    def __init__(self, eye, ear, assoc, concepts: List[str],
                 sound_of: Optional[Dict[str, str]] = None):
        self.eye = eye
        self.ear = ear
        self.assoc = assoc
        self.concepts = list(concepts)
        self.sound_of = dict(sound_of or {})
        self.areas = tuple(eye.areas)
        #: concept -> per-area pooled prototype, for the top-down gain in
        #: :meth:`_Vision.attend`. Filled by :meth:`remember_prototype`.
        self.area_proto: Dict[str, Dict[str, np.ndarray]] = {}
        self.vision = _Vision(self)
        self.audio = _Audio(self)

    # -- the two codes ---------------------------------------------------
    def vision_code(self, image: np.ndarray) -> np.ndarray:
        """Pooled multi-area object code for one 224 px view. See module docs
        for why pooling is right for naming and wrong for locating."""
        return _pool(self.eye._pass(image), self.areas)

    def sound_code(self, sig: np.ndarray) -> np.ndarray:
        """The ear's whole-sound code (global-pooled belt)."""
        return np.asarray(self.ear.sound_code(sig), np.float32)

    # -- what `attend` needs remembered ----------------------------------
    def remember_prototype(self, concept: str, views: Sequence[np.ndarray]
                           ) -> None:
        """Store a concept's mean per-area channel profile, for top-down gain."""
        acc: Dict[str, List[np.ndarray]] = {a: [] for a in self.areas}
        for v in views:
            M = self.eye._pass(v)
            for a in self.areas:
                m = np.asarray(M[a], np.float32)
                acc[a].append(m.reshape(m.shape[0], -1).mean(1))
        self.area_proto[concept] = {a: np.mean(acc[a], 0) for a in self.areas}
