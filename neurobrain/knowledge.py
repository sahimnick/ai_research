"""
knowledge.py
============

A **foundational knowledge** curriculum -- the first education of the brain.

It teaches, all by induction, a starter body of knowledge across three domains:

* **Language** -- a vocabulary sorted by part of speech (nouns, verbs,
  adjectives), and a few sentence structures (subject-verb-object, ...).
* **Mathematics** (school level) -- numbers, the operations, comparison,
  geometry, and a little algebra, plus counting and a few number facts.
* **The physical and non-physical world** -- matter, forces, the elements and
  the sky; and abstract concepts: emotions, mental life, and values.

Knowledge here means four things the brain can actually hold:
    * grounded **words**            (a concept assembly per word),
    * **categories**                (dog -> animal, triangle -> shape),
    * **relations / facts**         (sun -> light, gravity -> fall, plus -> sum),
    * **order / structure**         (one->..->ten, article->noun->verb).

Teach it onto a :func:`build_knowledge_brain`; the concept region grows
(neurogenesis) as the vocabulary does. The result is an active, growing brain
that already knows a little about language, number and the world -- and can
imagine and attend over all of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .brain import Brain
from .sensors import TextEncoder
from .teacher import Lesson, Teacher


# ---------------------------------------------------------------------------
# The foundation (as data)
# ---------------------------------------------------------------------------
CATEGORIES: Dict[str, List[str]] = {
    # language -- parts of speech
    "noun": ["cat", "dog", "tree", "book", "child", "bird"],
    "verb": ["run", "jump", "eat", "read", "grow", "think"],
    "adjective": ["big", "small", "hot", "cold", "bright", "fast"],
    # mathematics
    "number": ["zero", "one", "two", "three", "four", "five"],
    "operation": ["plus", "minus", "times", "divide", "equals"],
    "shape": ["point", "line", "triangle", "square", "circle", "rectangle"],
    "algebra": ["variable", "equation", "function", "unknown"],
    # the physical world
    "element": ["water", "air", "fire", "earth"],
    "state": ["solid", "liquid", "gas"],
    "celestial": ["sun", "moon", "star", "planet"],
    "force": ["gravity", "energy", "heat", "light", "motion"],
    "matter": ["atom", "molecule", "metal", "stone", "mass"],
    # the non-physical world
    "emotion": ["love", "fear", "joy", "sadness", "hope"],
    "mental": ["idea", "thought", "memory", "dream", "mind"],
    "value": ["truth", "justice", "freedom", "good", "right"],
}

OPPOSITES: List[Tuple[str, str]] = [
    ("big", "small"), ("hot", "cold"), ("fast", "slow"), ("bright", "dark"),
    ("good", "bad"), ("true", "false"), ("right", "wrong"), ("up", "down"),
    ("wet", "dry"), ("day", "night"), ("love", "fear"), ("open", "close"),
    ("plus", "minus"), ("odd", "even"),
]

# order / structure: counting, and simple sentence templates
STRUCTURE: Dict[str, List[str]] = {
    "counting": ["one", "two", "three", "four", "five",
                 "six", "seven", "eight", "nine", "ten"],
    "sentence_svo": ["the", "child", "read", "book"],   # subject-verb-object
    "sentence_adj": ["big", "dog", "run"],              # adjective-noun-verb
    "sentence_be": ["the", "sun", "is", "hot"],
    "geometry": ["point", "line", "angle", "triangle"],
    "algebra_path": ["variable", "equation", "function"],
}

# relations / facts:  a -> b  (property, cause, or connection)
FACTS: List[Tuple[str, str]] = [
    ("sun", "light"), ("sun", "hot"), ("fire", "hot"), ("ice", "cold"),
    ("gravity", "fall"), ("cloud", "rain"), ("rain", "wet"), ("water", "wet"),
    ("night", "dark"), ("seed", "tree"), ("bird", "fly"), ("fish", "swim"),
    ("plus", "sum"), ("times", "product"), ("angle", "triangle"),
    ("radius", "circle"), ("question", "answer"), ("cause", "effect"),
    ("love", "joy"), ("fear", "sadness"), ("day", "light"), ("moon", "night"),
]


def _all_words() -> List[str]:
    """Every word the foundation refers to (deduplicated, order-stable)."""
    words: List[str] = []
    for members in CATEGORIES.values():
        words += members
    for a, b in OPPOSITES:
        words += [a, b]
    for seq in STRUCTURE.values():
        words += seq
    for a, b in FACTS:
        words += [a, b]
    return list(dict.fromkeys(words))


@dataclass
class FoundationReport:
    n_words: int = 0
    n_concepts: int = 0
    recall_accuracy: float = 0.0
    samples: Dict[str, str] = field(default_factory=dict)
    facts: Dict[str, str] = field(default_factory=dict)
    counting: List[str] = field(default_factory=list)
    classifications: Dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        L = [f"Foundation taught: {self.n_words} words, {self.n_concepts} "
             f"concepts.  Recall {self.recall_accuracy:.0%}", ""]
        L.append("  counting : " + " → ".join(self.counting))
        L.append("  facts    :")
        for a, b in list(self.facts.items())[:8]:
            L.append(f"     {a:9} → {b}")
        L.append("  classify :")
        for m, c in list(self.classifications.items())[:8]:
            L.append(f"     {m:9} is a {c}")
        return "\n".join(L)


def teach_foundation(
    brain: Brain,
    repeats: int = 3,
    category_repeats: int = 2,
    verbose: bool = True,
) -> Tuple[Teacher, TextEncoder, FoundationReport]:
    """Teach the foundational knowledge onto ``brain``. Returns teacher/encoder/report."""
    text = TextEncoder(brain.region("text").size, level="word",
                       active_bits=20, amplitude=18.0, window=24)
    teacher = Teacher(brain, concept_region="memory", clamp_current=30.0,
                     assembly_size=14)

    words = _all_words()
    if verbose:
        print(f"Grounding {len(words)} words ...")
    for w in words:
        teacher.allocate_concept(w, word=w)
    teacher.teach([Lesson("text", text.encode(w), w) for w in words],
                  repeats=repeats)

    # categories (marked so they don't compete with words at recognition time)
    if verbose:
        print("Teaching categories ...")
    for category, members in CATEGORIES.items():
        if category not in teacher.concepts:
            teacher.allocate_concept(category, word=category, kind="category")
        for member in members:
            teacher.induce("text", text.encode(member), category,
                           repeats=category_repeats)

    # relations: opposites (both ways) and facts (one way)
    if verbose:
        print("Teaching relations, facts and structure ...")
    for a, b in OPPOSITES:
        brain.world.learn_transition(a, b, count=float(repeats))
        brain.world.learn_transition(b, a, count=float(repeats))
    for a, b in FACTS:
        brain.world.learn_transition(a, b, count=float(repeats))

    # order / structure (also strengthens the neural forward links)
    for seq in STRUCTURE.values():
        teacher.teach_sequence(seq, repeats=repeats)

    report = _evaluate(brain, teacher, text, words)
    if verbose:
        print("\n" + str(report))
    return teacher, text, report


# ---------------------------------------------------------------------------
# Evaluation / report
# ---------------------------------------------------------------------------
def _classify(teacher: Teacher, text: TextEncoder, member: str,
              baseline: Dict[str, float]) -> str:
    scores = teacher.concept_scores("text", text.encode(member), settle=22)
    adj = {c: scores.get(c, 0.0) - baseline.get(c, 0.0) for c in CATEGORIES}
    best = max(adj, key=adj.get)
    return best if adj[best] > 0 else "?"


def _evaluate(brain: Brain, teacher: Teacher, text: TextEncoder,
              words: List[str]) -> FoundationReport:
    rep = FoundationReport(n_words=len(words), n_concepts=len(teacher.concepts))

    # recall on a representative sample (full set would be slow)
    rng = np.random.default_rng(0)
    sample = list(rng.choice(words, size=min(40, len(words)), replace=False))
    correct = 0
    for w in sample:
        got, _ = teacher.recall("text", text.encode(w), settle=20, record=False)
        rep.samples[w] = got or "—"
        if got == w:
            correct += 1
    rep.recall_accuracy = correct / len(sample)

    # facts the brain now expects
    for a, _ in FACTS[:10]:
        nxt = brain.world.predict_next(a)
        if nxt:
            rep.facts[a] = nxt

    # counting chain
    chain, cur, seen = ["one"], "one", {"one"}
    for _ in range(10):
        nxt = brain.world.predict_next(cur)
        if not nxt or nxt in seen:
            break
        chain.append(nxt)
        seen.add(nxt)
        cur = nxt
    rep.counting = chain

    # classification of a few members
    blank = np.zeros((text.window, brain.region("text").size), dtype=np.float32)
    blank_scores = teacher.concept_scores("text", blank, settle=22)
    baseline = {c: blank_scores.get(c, 0.0) for c in CATEGORIES}
    for member in ["dog", "run", "three", "triangle", "sun", "love", "water"]:
        if member in teacher.concepts:
            rep.classifications[member] = _classify(teacher, text, member,
                                                    baseline)
    return rep
