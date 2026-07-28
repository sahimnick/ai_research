"""
curriculum.py
=============

The **first lessons** -- taught by induction, exactly as requested.

This is where the brain is given a small set of primitive concepts so that it
crosses an initial threshold of general understanding: a handful of grounded
words, two perceptual opposites (light/dark, loud/quiet), cross-modal links
(the *word* "light" and a *bright image* mean the same thing), and a few
temporal regularities of the world (day -> light, night -> dark, one -> two ->
three). None of it uses backpropagation; every association is imprinted by
co-activation and consolidated by STDP.

Run :func:`teach_starter_curriculum` on a brain from :mod:`builder` and you get
back the teacher, the encoders, and a short report of what stuck.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .brain import Brain
from .sensors import ImageEncoder, SoundEncoder, TextEncoder
from .teacher import Lesson, Teacher


# -- little stimulus helpers (stand-ins for real cameras / microphones) ------
def bright_image(side: int = 12) -> np.ndarray:
    """A mostly-bright frame -> the percept 'light'."""
    img = np.ones((side, side)) * 0.9
    img[side // 3: 2 * side // 3, side // 3: 2 * side // 3] = 1.0
    return img


def dark_image(side: int = 12) -> np.ndarray:
    """A mostly-dark frame -> the percept 'dark'."""
    return np.ones((side, side)) * 0.05


def high_tone(n: int = 2048, sr: int = 16000) -> np.ndarray:
    """A loud high-frequency tone -> the percept 'loud/high'."""
    t = np.arange(n) / sr
    return 0.9 * np.sin(2 * np.pi * 3000 * t)


def low_tone(n: int = 2048, sr: int = 16000) -> np.ndarray:
    """A soft low-frequency tone -> the percept 'quiet/low'."""
    t = np.arange(n) / sr
    return 0.3 * np.sin(2 * np.pi * 200 * t)


@dataclass
class CurriculumReport:
    """What the brain learned, and how well it recalls it."""

    concepts: List[str] = field(default_factory=list)
    recall: Dict[str, str] = field(default_factory=dict)   # word -> recognised
    accuracy: float = 0.0
    predictions: Dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        lines = [f"Taught {len(self.concepts)} concepts. "
                 f"Recall accuracy: {self.accuracy:.0%}", ""]
        lines.append("  word shown      -> concept recognised")
        for word, got in self.recall.items():
            mark = "✓" if got == word else "·"
            lines.append(f"  {mark} {word:14} -> {got}")
        if self.predictions:
            lines.append("")
            lines.append("  learned expectations (world model):")
            for a, b in self.predictions.items():
                lines.append(f"    {a:14} -> expects -> {b}")
        return "\n".join(lines)


def teach_starter_curriculum(
    brain: Brain, repeats: int = 8, verbose: bool = True
) -> Tuple[Teacher, Dict[str, object], CurriculumReport]:
    """Teach the brain its first grounded concepts and world regularities.

    Returns ``(teacher, encoders, report)``.
    """
    # Encoders are the sense organs, one per sensory region.
    text = TextEncoder(brain.region("text").size, level="word",
                       active_bits=16, amplitude=18.0, window=24)
    vision = ImageEncoder(brain.region("vision").size, amplitude=18.0,
                          window=24)
    sound = SoundEncoder(brain.region("sound").size, amplitude=18.0, window=24)
    encoders = {"text": text, "vision": vision, "sound": sound}

    teacher = Teacher(brain, concept_region="memory", clamp_current=30.0,
                     assembly_size=14)

    # 1) Ground words + percepts into concept assemblies (single modality each).
    #    Every concept gets a word label, so the world model is also a word model.
    words = ["light", "dark", "sound", "quiet", "cat", "dog", "hello",
             "one", "two", "three", "day", "night"]
    for w in words:
        teacher.allocate_concept(w, word=w)

    text_lessons = [Lesson("text", text.encode(w), w) for w in words]
    teacher.teach(text_lessons, repeats=repeats)

    # 2) Cross-modal grounding: a bright image also means "light", etc.
    cross_modal = [
        Lesson("vision", vision.encode(bright_image()), "light"),
        Lesson("vision", vision.encode(dark_image()), "dark"),
        Lesson("sound", sound.encode(high_tone()), "sound"),
        Lesson("sound", sound.encode(low_tone()), "quiet"),
    ]
    teacher.teach(cross_modal, repeats=repeats)

    # 3) Teach a few regularities of the world (temporal transitions).
    teacher.teach_sequence(["day", "light"], repeats=repeats)
    teacher.teach_sequence(["night", "dark"], repeats=repeats)
    teacher.teach_sequence(["one", "two", "three"], repeats=repeats)

    # 4) Test recall: show each word alone and see which concept wakes.
    report = CurriculumReport(concepts=list(teacher.concepts))
    correct = 0
    for w in words:
        got, _ = teacher.recall("text", text.encode(w), settle=25, record=False)
        report.recall[w] = got or "—"
        if got == w:
            correct += 1
    report.accuracy = correct / len(words) if words else 0.0

    # 5) Report the world-model expectations it formed.
    for a in ("day", "night", "one", "two"):
        nxt = brain.world.predict_next(a)
        if nxt:
            report.predictions[a] = nxt

    if verbose:
        print(report)
    return teacher, encoders, report
