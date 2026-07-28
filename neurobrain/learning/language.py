"""
language.py
===========

A **simple language layer** on top of the brain.

The world model already grounds single words to concept assemblies and learns
which concept follows which. This module lifts that to whole phrases: it turns a
sentence into the sequence of concepts the brain recognises (understanding), and
turns a prompt into a continuation by walking the learned transitions
(generation). It is deliberately small -- a bridge between text and the neural
concepts, not a large language model -- but it demonstrates the two halves of
language on a purely inductive, spiking substrate:

    * comprehension:  text  -> spikes -> concepts   ("what did it mean?")
    * production:     concept -> expected next -> word  ("what comes next?")

Everything is taught by induction (see :mod:`teacher`); nothing is trained by
gradient descent.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from ..core.brain import Brain
from ..sensing.sensors import TextEncoder
from .teacher import Teacher


class LanguageModel:
    """Understand and continue text using a taught brain.

    Parameters
    ----------
    brain:      a finalized brain with a world model.
    teacher:    the teacher used to ground new words.
    text_encoder: the encoder mapping words to the text sense.
    """

    def __init__(self, brain: Brain, teacher: Teacher,
                 text_encoder: Optional[TextEncoder] = None):
        self.brain = brain
        self.teacher = teacher
        self.text = text_encoder or TextEncoder(
            brain.region("text").size, level="word",
            active_bits=16, amplitude=18.0, window=24,
        )

    # -- vocabulary -------------------------------------------------------
    def known_words(self) -> List[str]:
        return sorted(self.brain.world.word_index)

    def ground_word(self, word: str, repeats: int = 6) -> None:
        """Give a new word its own concept assembly and imprint it."""
        if word.lower() in self.brain.world.word_index:
            return
        self.teacher.allocate_concept(word, word=word)
        self.teacher.induce("text", self.text.encode(word), word,
                            repeats=repeats)

    # -- comprehension ----------------------------------------------------
    def understand(self, sentence: str) -> List[Tuple[str, Optional[str]]]:
        """Read a sentence word by word and report the concept each evokes.

        Returns a list of ``(word, recognised_concept_or_None)``. A word the
        brain has never been taught comes back as ``None`` ("I don't know it").
        """
        out: List[Tuple[str, Optional[str]]] = []
        for word in self.text.symbols(sentence):
            if not word.strip():
                continue
            if word.lower() in self.brain.world.word_index:
                got, _ = self.teacher.recall("text", self.text.encode(word),
                                             settle=20, record=False)
            else:
                got = None
            out.append((word, got))
        return out

    def comprehension_rate(self, sentence: str) -> float:
        """Fraction of a sentence's words the brain recognises correctly."""
        results = self.understand(sentence)
        if not results:
            return 0.0
        correct = sum(1 for w, c in results if c == w.lower())
        return correct / len(results)

    # -- teaching phrases -------------------------------------------------
    def teach_sentence(self, sentence: str, repeats: int = 6) -> List[str]:
        """Teach a phrase: ground any new words, then learn its word order.

        The sequence of the sentence's concepts is taught as transitions, so the
        brain can later continue the phrase. Returns the concept sequence.
        """
        words = [w.lower() for w in self.text.symbols(sentence) if w.strip()]
        for w in words:
            self.ground_word(w)
        # Teach the temporal order of the concepts (a -> b -> c ...).
        self.teacher.teach_sequence(words, repeats=repeats)
        return words

    def teach_corpus(self, sentences: Sequence[str], repeats: int = 6) -> None:
        for s in sentences:
            self.teach_sentence(s, repeats=repeats)

    # -- production -------------------------------------------------------
    def predict_next_word(self, word: str) -> Optional[str]:
        """The word the brain expects to follow ``word``."""
        concept = self.brain.world.concept_for_word(word)
        name = concept.name if concept else word.lower()
        nxt = self.brain.world.predict_next(name)
        if not nxt:
            return None
        c = self.brain.world.concepts.get(nxt)
        return (c.word or c.name) if c else nxt

    def continue_text(self, prompt: str, length: int = 5) -> str:
        """Continue a prompt by repeatedly predicting the next word.

        Stops early if the brain has no expectation or starts repeating.
        """
        words = [w.lower() for w in self.text.symbols(prompt) if w.strip()]
        if not words:
            return prompt
        produced = list(words)
        current = words[-1]
        seen = set(words)
        for _ in range(length):
            nxt = self.predict_next_word(current)
            if not nxt or nxt in seen:
                break
            produced.append(nxt)
            seen.add(nxt)
            current = nxt
        return " ".join(produced)

    def respond(self, prompt: str) -> str:
        """A minimal 'answer': recognise the prompt, then say what's expected."""
        understood = [c for _, c in self.understand(prompt) if c]
        nxt = self.predict_next_word(understood[-1]) if understood else None
        if nxt:
            return nxt
        return "?"
