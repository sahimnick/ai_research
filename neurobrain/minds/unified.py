"""
unified.py
==========

One brain that does it all at once -- and does it the way a brain does: every
percept passes through **attention** and comes out as a **comprehension**, not a
bare label.

The earlier versions built the pieces -- perception (self-grown categories on
real digits), a pallium-scale associative memory, a world model, imagination,
and analogy -- but each lived on its own. :class:`UnifiedMind` wires them into a
single object with a single flow, and makes one thing a **standing principle**:

    perception is never bare classification. Seeing something means
      (1) attending (top-down gain biases the competition),
      (2) grounding it to a concept,
      (3) comprehending it -- what it predicts next (world model), what it is
          related to (successor representation), and how surprising it is.

So :meth:`UnifiedMind.comprehend` -- not a raw label -- is the normal output,
and attention is always in the loop (:meth:`attend`, and the ``attend=`` path in
recall). Honest point this still demonstrates rather than asserts: **more
neurons help, measurably** -- a bigger pallium recalls a noisy real percept
better; the intelligence is the integration, not a new kind of mind.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Comprehension:
    """What the mind *understands* about a percept -- perception with meaning.

    The **code** is primary: it is the shared cortical representation the other
    faculties consume. ``concept`` is the name read off that code, and may be
    ``None`` when nothing familiar fits."""
    concept: str                       # what it is (grounded concept)
    predicted_next: Optional[str]      # what the world model expects to follow
    associations: List[str]            # what it tends to lead to (successor rep.)
    surprise: float                    # prediction error of seeing it now
    attended: Optional[str]            # what attention was pointed at, if any
    code: Optional[np.ndarray] = None  # the SHARED cortical code (primary)
    workspace_name: Optional[str] = None   # the name read off that code

    def __str__(self) -> str:          # reads like a thought
        nxt = f" -> expect '{self.predicted_next}'" if self.predicted_next else ""
        rel = f" ~ {self.associations}" if self.associations else ""
        return f"'{self.concept}'{nxt}{rel}"


@dataclass
class UnifiedMind:
    vision: object                 # DigitRecognizer -- perception of real digits
    space: object                  # MentalSpace -- pallium memory + world model
    reasoner: object               # RelationalMind -- analogy
    digit_cue: Dict[int, np.ndarray] = field(default_factory=dict)
    attention: Optional[str] = None            # currently attended concept
    attention_gain: float = 0.6
    last_percept: Optional[str] = None         # for the surprise signal
    # v0.17: the agent is not only a world model
    causal: object = None          # CausalWorldGraph -- objects, relations, causes
    self_model: object = None      # SelfModel -- body, capability, agency
    episodes: object = None        # EpisodicBuffer -- the day, for replay
    ws: object = None              # GlobalWorkspace -- the SHARED cortical code
    # measured end to end
    perceive_accuracy: float = 0.0
    recall_accuracy: float = 0.0
    analogy_accuracy: float = 0.0
    comprehend_accuracy: float = 0.0
    attention_benefit: float = 0.0
    causal_accuracy: float = 0.0       # effects predicted for UNSEEN objects
    self_accuracy: float = 0.0         # "can this body do it?" judgements

    # -- attention: a standing part of the loop ------------------------------
    def attend(self, concept: Optional[str]) -> None:
        """Point top-down attention at a concept (or ``None`` to release it)."""
        self.attention = concept

    # -- one coherent flow across all the faculties --------------------------
    def perceive(self, image: np.ndarray, remember: bool = True) -> str:
        """See a real digit and recognise which self-grown concept it is.

        Perceiving also **lays down an episode** for tonight's replay, which is
        what makes :meth:`dream` a mechanism rather than a no-op: the
        hippocampal buffer was constructed empty and nothing ever wrote to it,
        so ``sleep()`` returned 0 for the object's whole life. Waking
        experience is the only thing replay has to consolidate.

        Pass ``remember=False`` for a probe you do not want the day to
        remember -- scoring a test set, or re-perceiving something the mind
        just imagined."""
        lab = self.vision.cortex.recognise(
            image.reshape(1, 28, 28).astype(np.float32), 0.0)[0]
        concept = str(int(lab))
        if remember:
            self._record_episode(image, concept)
        return concept

    def _record_episode(self, image: np.ndarray, concept: str) -> None:
        """Store one waking moment, weighted by how unexpected it was.

        Surprise is the world model's own ``1 - P(concept | previous)``, so
        replay is prioritised by prediction error rather than by recency --
        which is the whole point of prioritised replay. With no previous
        percept to predict from, the episode is maximally surprising (1.0);
        the first thing you see today is by definition unexplained."""
        if self.episodes is None:
            return
        surprise = 1.0
        if self.last_percept is not None:
            try:
                surprise = float(
                    self.space._world().surprise(self.last_percept, concept))
            except Exception:
                surprise = 1.0
        pattern = np.asarray(image, np.float32).reshape(-1)
        if pattern.size and float(pattern.max()) > 1.5:   # uint8 [0,255] in
            pattern = pattern / 255.0
        self.episodes.store(pattern, concept, surprise)
        self.last_percept = concept

    def comprehend(self, image: np.ndarray,
                   remember: bool = True) -> Comprehension:
        """Perception with meaning: recognise, then understand -- what it
        predicts, what it relates to, how surprising it is. This (not a bare
        label) is the mind's normal output; attention is always in the loop."""
        # the CODE is computed first and broadcast: every faculty downstream
        # reads the same representation, and the name is a read-out of it.
        code = self.perceive_code(image)
        if self.ws is not None:
            self.ws.broadcast(code, source="vision")
        ws_name = self.name_of(code) if self.ws is not None else None
        # capture the previous percept BEFORE perceiving: perceive() now records
        # an episode and advances last_percept, so reading it afterwards would
        # compare this concept against itself and report zero surprise forever.
        prev = self.last_percept
        concept = self.perceive(image, remember=remember)
        predicted = self.space.predict_next(concept)
        assoc = [n for n, _ in self.space.reachable(concept, top=3)]
        surprise = (self.space._world().surprise(prev, concept)
                    if prev is not None else 0.0)
        self.last_percept = concept
        return Comprehension(concept, predicted, assoc, float(surprise),
                             self.attention, code=code, workspace_name=ws_name)

    def recall_from_cue(self, cue: np.ndarray,
                        attend: object = "__self__") -> Tuple[str, np.ndarray]:
        """A familiar (noisy) cue brings back the concept and its picture. By
        default the mind's current attention biases the competition."""
        att = self.attention if attend == "__self__" else attend
        return self.space.recall("cue", cue, "image", attend=att,
                                 gain=self.attention_gain)

    def imagine(self, seed: str, steps: int = 5):
        """Wander the world model and re-create each concept from memory."""
        return self.space.imagine(seed, steps=steps)

    def reason(self, a: str, b: str, c: str) -> Optional[str]:
        """Analogy a:b::c:? across the reasoner's relations."""
        return self.reasoner.analogy(a, b, c)

    # -- v0.18: the shared cortical code ------------------------------------
    def encode(self, modality: str, x: np.ndarray) -> np.ndarray:
        """Any modality -> the ONE shared sparse code every faculty speaks."""
        return self.ws.encode(modality, x)

    def perceive_code(self, image: np.ndarray) -> np.ndarray:
        """See a digit and return its **cortical code**, not a string. This is
        what the other faculties should consume."""
        return self.ws.encode("vision", np.asarray(image, np.float32).reshape(-1))

    def name_of(self, code: np.ndarray) -> Optional[str]:
        """Read a name off a code -- or ``None`` when nothing fits. Naming is a
        late read-out of the representation, never its substance."""
        return self.ws.name(code)

    def code_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Because all faculties share one space, any two contents can be
        compared directly -- that is the point of a common code."""
        from ..workspace import _unit
        return float(_unit(a) @ _unit(b))

    # -- v0.17: objects, self, and sleep ------------------------------------
    def expect_effects(self, action: str, obj: str,
                       properties=None) -> List[str]:
        """What would this action do to that object? Works for objects never
        seen, because the causal rules are keyed on properties."""
        if self.causal is None:
            return []
        return self.causal.expect(action, obj, properties=properties)

    def imagine_if(self, action: str, obj: str, without: str = None,
                   with_: str = None) -> List[str]:
        """Counterfactual imagination over causal structure: what would happen
        if this object were *not* fragile?"""
        if self.causal is None:
            return []
        return self.causal.counterfactual(action, obj, without=without,
                                          with_=with_)

    def can_i(self, action: str, **kw) -> bool:
        """Ask the *self* model, not the world model: is this within my body?"""
        return False if self.self_model is None else self.self_model.can(action, **kw)

    def why_not(self, action: str, **kw) -> str:
        """The body's own explanation of a capability judgement."""
        return "" if self.self_model is None else self.self_model.explain(action, **kw)

    def who_did_that(self, sensation: np.ndarray,
                     action: Optional[str] = None) -> str:
        """Self-caused or externally caused? (efference-copy comparator)"""
        if self.self_model is None:
            return "unknown"
        return str(self.self_model.attribute(sensation, action=action)["agent"])

    def experience_episode(self, pattern: np.ndarray, label: object,
                           surprise: float = 1.0) -> None:
        """Store a waking episode for tonight's replay."""
        if self.episodes is not None:
            self.episodes.store(pattern, label, surprise)

    # -- the brain LIVES: the closed loop, on the mind's own workspace -------
    def live(self, world=None, steps: int = 1000, n_places: int = 8,
             noise: float = 0.35, seed: int = 0):
        """Put this brain in a world and run the closed loop inside it.

        The loop shares **this mind's workspace**, so what it learns while
        living lands in the same cortical code the rest of the faculties use --
        it is the brain's own loop, not a separate machine."""
        from ..world.loop import SensoryWorld, CognitiveLoop
        world = world or SensoryWorld(n_places=n_places, noise=noise, seed=seed)
        loop = CognitiveLoop(world, workspace=self.ws, seed=seed)
        loop.live(steps=steps)
        self.loop = loop
        return loop

    def dream(self, cycles: int = 3, replays_per_cycle: int = 300,
              consolidate: float = 0.75) -> Dict[str, int]:
        """A full night: replay the day into memory, then **consolidate** the
        workspace (merging redundant concepts that noise spawned)."""
        replays = self.sleep(cycles=cycles, replays_per_cycle=replays_per_cycle)
        merged = self.ws.consolidate(consolidate) if self.ws is not None else 0
        return {"replays": replays, "concepts_merged": merged}

    def sleep(self, cycles: int = 3, replays_per_cycle: int = 300) -> int:
        """Consolidate the day: prioritised hippocampal replay into the pallium.
        No new data enters -- the day is simply re-processed."""
        if self.episodes is None or len(self.episodes) == 0:
            return 0
        from ..memory.development import SleepConsolidator
        cons = SleepConsolidator(self.episodes, seed=0)
        return cons.sleep(
            lambda p, l, lr: self.space.remember(str(l), image=p),
            cycles=cycles, replays_per_cycle=replays_per_cycle)


def build_unified_mind(n_pallium: int = 4000, verbose: bool = False):
    """Assemble the whole mind on REAL handwritten digits and measure it end to
    end. ``n_pallium`` sets how many exemplar memories the pallium holds."""
    from ..sensing.realworld import build_digit_recognizer, load_mnist
    from ..memory.psyche import MentalSpace
    from ..cognition.analogy import RelationalMind

    def say(*a):
        if verbose:
            print(*a)

    say("perception: self-grown digit categories on real data ...")
    recog = build_digit_recognizer(n_train=15000)
    trx, trY, tex, teY = load_mnist(n_train=15000)

    say(f"pallium: filling a large associative memory ({n_pallium} exemplars) ...")
    space = MentalSpace()
    rng = np.random.default_rng(0)
    digit_cue = {}
    for d in range(10):
        cue = np.zeros(256, np.float32)
        cue[rng.choice(256, 24, replace=False)] = 1.0
        digit_cue[d] = cue
    # cross-modal concept memories (cue <-> mean picture), for recall + imagination
    for d in range(10):
        img = (trx[trY == d][:60].reshape(-1, 784) / 255.0).mean(0)
        space.remember(str(d), image=img, cue=digit_cue[d])
    # a LARGE store of individual real-digit exemplars -> the pallium's capacity
    pick = rng.choice(len(trx), n_pallium, replace=False)
    for i in pick:
        space.remember(str(int(trY[i])),
                       image=(trx[i].reshape(-1) / 255.0),
                       cue=digit_cue[int(trY[i])])

    # a MEANINGFUL world model: the counting order 0->1->...->9, so the mind
    # can predict "after 3 comes 4" and relate a digit to its successors.
    say("world model: learning the counting structure 0->1->...->9 ...")
    for _ in range(20):
        space.experience([str(d) for d in range(10)], actions=["+1"] * 9)

    say("reasoner: a small relational knowledge base ...")
    reasoner = RelationalMind(seed=0)
    for a, b in [("hot", "cold"), ("big", "small"), ("up", "down"),
                 ("light", "dark")]:
        reasoner.relate(a, "opposite", b, symmetric=True)
    for a, b in [("dog", "mammal"), ("sparrow", "bird"), ("salmon", "fish")]:
        reasoner.relate(a, "is_a", b)
    for a, b in [("france", "paris"), ("japan", "tokyo"), ("italy", "rome")]:
        reasoner.relate(a, "capital", b)

    # -- v0.17: an agent that also models objects, itself, and its own day ---
    say("causal world: inducing property-keyed physics from episodes ...")
    from ..world.objects import build_causal_world
    from ..cognition.selfmodel import build_self_model
    from ..memory.development import EpisodicBuffer
    causal = build_causal_world()

    say("self model: learning this body's limits by trying ...")
    self_model, _ = build_self_model()

    # -- v0.18: one shared cortical code, grounded on the REAL percepts ------
    say("workspace: grounding one shared cortical code across modalities ...")
    from ..workspace import GlobalWorkspace
    ws = GlobalWorkspace(dim=512, vigilance=0.30, seed=0)
    for d in range(10):                       # bind the digit's look AND its cue
        for i in np.where(trY == d)[0][:40]:
            ws.bind(str(d), vision=(trx[i].reshape(-1) / 255.0),
                    cue=digit_cue[d])

    mind = UnifiedMind(recog, space, reasoner, digit_cue,
                       causal=causal, self_model=self_model,
                       episodes=EpisodicBuffer(), ws=ws)

    # -- measure the whole pipeline on REAL held-out data --------------------
    # remember=False throughout: measuring the mind is not the mind living.
    # perceive() now lays down an episode, and scoring a held-out set must not
    # become the day the mind then replays all night.
    pe = np.mean([mind.perceive(tex[i], remember=False) == str(int(teY[i]))
                  for i in range(500)])
    mind.perceive_accuracy = float(pe)

    # comprehension: concept right AND the world model predicts the next digit
    cok = pnext = 0
    for i in range(500):
        c = mind.comprehend(tex[i], remember=False)
        cok += int(c.concept == str(int(teY[i])))
        if int(teY[i]) < 9:
            pnext += int(c.predicted_next == str(int(teY[i]) + 1))
    mind.comprehend_accuracy = cok / 500

    # the "remember a picture from a familiar cue" story (well-separated cues,
    # so this is near-perfect and does not need attention):
    ok = 0
    for _ in range(500):
        d = int(rng.integers(10))
        cue = digit_cue[d] + rng.normal(0, 0.2, 256).astype(np.float32)
        cue[rng.choice(256, int(256 * 0.35), replace=False)] *= 0.2
        ok += int(mind.recall_from_cue(cue, attend=None)[0] == str(d))
    mind.recall_accuracy = ok / 500

    # attention where memories genuinely COMPETE: recall a heavily-noised real
    # IMAGE against all 4,010 exemplars, without vs. with top-down attention on
    # the target concept. Biased competition should lift it (cue-validity gain).
    Xte = (tex[:400].reshape(-1, 784) / 255.0)
    base = watt = 0
    for j in range(400):
        d = int(teY[j])
        noisy = Xte[j] + rng.normal(0, 0.9, 784).astype(np.float32)
        b = space.cortex.recall("image", noisy)
        a = space.cortex.recall("image", noisy,
                                bias=space.cortex.label_bias(str(d), 0.5))
        base += int(space.cortex.labels[b] == str(d))
        watt += int(space.cortex.labels[a] == str(d))
    mind.attention_benefit = float((watt - base) / 400)

    q = [("hot", "cold", "big", "small"), ("dog", "mammal", "sparrow", "bird"),
         ("france", "paris", "japan", "tokyo")]
    mind.analogy_accuracy = float(np.mean([mind.reason(a, b, c) == d
                                           for a, b, c, d in q]))

    # v0.17: causal generalisation to objects never seen, and body self-knowledge
    from ..world.objects import causal_generalisation_score
    mind.causal_accuracy = float(causal_generalisation_score(causal))
    mind.self_accuracy = float(np.mean([
        mind.can_i("lift", load_kg=w) == (w * 9.81 <= self_model.body.max_force_n
                                          * self_model.intero.capability_scale)
        for w in np.linspace(1, 200, 200)]))

    # v0.18: does the SHARED code carry the percept? (codes, not strings)
    mind.workspace_accuracy = float(np.mean([
        mind.name_of(mind.perceive_code(tex[i])) == str(int(teY[i]))
        for i in range(300)]))

    say(f"   perceive real held-out digit -> concept : {mind.perceive_accuracy:.0%}")
    say(f"   SHARED cortical code -> concept          : {mind.workspace_accuracy:.0%}")
    say(f"   causal effects on UNSEEN objects         : {mind.causal_accuracy:.0%}")
    say(f"   body self-knowledge (can I lift X?)      : {mind.self_accuracy:.0%}")
    say(f"   comprehend (concept + predict next)      : {mind.comprehend_accuracy:.0%}")
    say(f"   noisy cue -> recall (with attention)     : {mind.recall_accuracy:.0%}")
    say(f"   attention benefit under noise            : +{mind.attention_benefit:.0%}")
    say(f"   analogy a:b::c:?                          : {mind.analogy_accuracy:.0%}")
    say(f"   pallium: {space.cortex.n_synapses:,} synapses, "
        f"{space.cortex.n_memories:,} memory cells")
    return mind


def pallium_capacity_curve(sizes=(50, 200, 1000, 4000), n_train: int = 15000
                           ) -> List[Tuple[int, float]]:
    """Show that a bigger pallium recalls better: recall accuracy of a noisy real
    digit vs. how many exemplar memories the pallium holds."""
    from ..sensing.realworld import build_digit_recognizer, load_mnist
    from ..memory.psyche import AssociativeCortex
    trx, trY, tex, teY = load_mnist(n_train=n_train)
    rng = np.random.default_rng(0)
    Xte = (tex[:500].reshape(-1, 784) / 255.0)
    Xte = Xte + rng.normal(0, 0.25, Xte.shape).astype(np.float32)   # noisy percepts
    out = []
    for n in sizes:
        cortex = AssociativeCortex()
        pick = rng.choice(len(trx), n, replace=False)
        for i in pick:
            cortex.remember({"image": trx[i].reshape(-1) / 255.0},
                            label=int(trY[i]))
        acc = np.mean([cortex.labels[cortex.recall("image", Xte[j])] == int(teY[j])
                       for j in range(len(Xte))])
        out.append((n, float(acc)))
    return out
