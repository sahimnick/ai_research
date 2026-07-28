"""
curriculum_advanced.py
======================

Second-stage lessons: **counting, categories (classification), and relations.**

All still taught by induction. Three new abilities are built on top of the
starter curriculum:

1. **Counting** -- a longer ordered chain one -> two -> ... -> ten, so the brain
   can carry on a count it is given the start of.

2. **Categories / classification** -- a category ("animal", "number") is its own
   concept assembly that is made to fire *together with each of its members*.
   Afterwards, showing a member ("cat") wakes both the member concept and its
   category, so the brain can answer "what kind of thing is this?".

3. **Relations (opposites)** -- pairs like light/dark are linked so that one
   evokes the other.

Run :func:`teach_advanced_curriculum` on a brain that has already had the
starter curriculum, or use the all-in-one at the bottom.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..core.brain import Brain
from ..sensing.sensors import TextEncoder
from .teacher import Teacher


NUMBERS = ["one", "two", "three", "four", "five",
           "six", "seven", "eight", "nine", "ten"]

CATEGORIES: Dict[str, List[str]] = {
    "animal": ["cat", "dog"],
    "number": ["one", "two", "three"],
    "brightness": ["light", "dark"],
}

# Opposite pairs chosen so they carry no *sequence* role (unlike day, which is
# already taught to be followed by light). Otherwise the single transition table
# would have to mean both "comes next" and "is the opposite of" at once.
OPPOSITES: List[Tuple[str, str]] = [
    ("light", "dark"),
    ("hot", "cold"),
    ("big", "small"),
]


@dataclass
class AdvancedReport:
    counting_ok: bool = False
    counting_chain: List[str] = field(default_factory=list)
    classification: Dict[str, List[str]] = field(default_factory=dict)
    opposites: Dict[str, Optional[str]] = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        lines = ["Advanced curriculum results", "=" * 30]
        lines.append(f"Counting  : {' -> '.join(self.counting_chain)}")
        lines.append("Classify  :")
        for member, cats in self.classification.items():
            lines.append(f"   {member:8} is a  ->  {', '.join(cats) or '?'}")
        lines.append("Opposites :")
        for a, b in self.opposites.items():
            lines.append(f"   opposite of {a:7} ->  {b}")
        return "\n".join(lines)


def _ensure_word(teacher: Teacher, text: TextEncoder, word: str,
                 repeats: int = 6) -> None:
    """Ground a word (give it a concept + imprint it) if it is new."""
    if word.lower() in teacher.brain.world.word_index:
        return
    teacher.allocate_concept(word, word=word)
    teacher.induce("text", text.encode(word), word, repeats=repeats)


def teach_counting(teacher: Teacher, text: TextEncoder,
                   numbers: Sequence[str] = NUMBERS, repeats: int = 6) -> None:
    """Ground the number words and teach the counting order."""
    for w in numbers:
        _ensure_word(teacher, text, w, repeats=repeats)
    teacher.teach_sequence(list(numbers), repeats=repeats)


def teach_categories(teacher: Teacher, text: TextEncoder,
                     categories: Dict[str, List[str]] = CATEGORIES,
                     repeats: int = 6) -> None:
    """Bind each category assembly to fire together with its members."""
    for category, members in categories.items():
        # the category needs its own assembly (marked as a category so it does
        # not compete with words during recognition)
        if category not in teacher.concepts:
            teacher.allocate_concept(category, word=category, kind="category")
        for member in members:
            _ensure_word(teacher, text, member, repeats=repeats)
            # Show the member's word while forcing the CATEGORY to fire, so the
            # member's input learns to also wake the category.
            teacher.induce("text", text.encode(member), category,
                           repeats=repeats)


def teach_opposites(teacher: Teacher, text: TextEncoder,
                    opposites: Sequence[Tuple[str, str]] = OPPOSITES,
                    repeats: int = 5) -> None:
    """Link opposite pairs so each evokes the other (a symmetric relation).

    New opposite words (hot/cold/big/small) are grounded first.
    """
    for a, b in opposites:
        _ensure_word(teacher, text, a, repeats=repeats)
        _ensure_word(teacher, text, b, repeats=repeats)
        teacher.brain.world.learn_transition(a, b, count=float(repeats))
        teacher.brain.world.learn_transition(b, a, count=float(repeats))


def category_baseline(teacher: Teacher, text: TextEncoder,
                      categories: Dict[str, List[str]] = CATEGORIES
                      ) -> Dict[str, float]:
    """Each category's intrinsic excitability to a blank input.

    Some category assemblies fire more readily than others; subtracting this
    baseline isolates the part of a category's response that the *member*
    specifically caused, which makes classification much cleaner.
    """
    import numpy as np
    blank = np.zeros((text.window, teacher.brain.region("text").size),
                     dtype=np.float32)
    scores = teacher.concept_scores("text", blank, settle=22)
    return {c: scores.get(c, 0.0) for c in categories}


def classify(teacher: Teacher, text: TextEncoder, member: str,
             categories: Dict[str, List[str]] = CATEGORIES,
             baseline: Optional[Dict[str, float]] = None,
             top_only: bool = True, threshold: float = 0.05) -> List[str]:
    """Return the category (or categories) that ``member`` belongs to.

    Uses baseline-subtracted assembly activation. With ``top_only`` (default)
    the single best category is returned; otherwise every category whose
    member-driven response clears ``threshold``.
    """
    if baseline is None:
        baseline = category_baseline(teacher, text, categories)
    scores = teacher.concept_scores("text", text.encode(member), settle=22)
    adj = {c: scores.get(c, 0.0) - baseline.get(c, 0.0) for c in categories}
    ranked = sorted(categories, key=lambda c: adj[c], reverse=True)
    if top_only:
        return [ranked[0]] if adj[ranked[0]] > 0 else []
    return [c for c in ranked if adj[c] >= threshold]


def teach_advanced_curriculum(
    brain: Brain,
    teacher: Optional[Teacher] = None,
    text: Optional[TextEncoder] = None,
    verbose: bool = True,
) -> Tuple[Teacher, TextEncoder, AdvancedReport]:
    """Teach counting, categories and relations; return a report.

    ``teacher`` / ``text`` may be reused from the starter curriculum; if omitted,
    fresh ones are created (make sure the starter curriculum ran first so the
    base concepts exist).
    """
    text = text or TextEncoder(brain.region("text").size, level="word",
                               active_bits=16, amplitude=18.0, window=24)
    teacher = teacher or Teacher(brain, concept_region="memory",
                                clamp_current=30.0, assembly_size=14)

    teach_counting(teacher, text)
    teach_categories(teacher, text)
    teach_opposites(teacher, text)

    report = AdvancedReport()

    # counting chain from "one"
    chain, cur, seen = ["one"], "one", {"one"}
    for _ in range(len(NUMBERS)):
        nxt = brain.world.predict_next(cur)
        if not nxt or nxt in seen:
            break
        chain.append(nxt)
        seen.add(nxt)
        cur = nxt
    report.counting_chain = chain
    report.counting_ok = chain[:4] == ["one", "two", "three", "four"]

    # classification (baseline computed once, then reused)
    baseline = category_baseline(teacher, text)
    for category, members in CATEGORIES.items():
        for member in members:
            report.classification[member] = classify(
                teacher, text, member, baseline=baseline)

    # opposites
    for a, b in OPPOSITES:
        report.opposites[a] = brain.world.predict_next(a)

    if verbose:
        print(report)
    return teacher, text, report
