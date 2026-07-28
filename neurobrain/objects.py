"""
objects.py
==========

An **object-centric causal world graph**.

The previous world model was state-centric: opaque states and the transitions
between them. Brains are not organised that way. The ventral stream binds
features into **object files** (Kahneman & Treisman) -- persistent tokens with
*properties*; the parietal stream attaches **affordances**; hippocampus and PFC
bind objects into **relations** ("cup ON table", "human HOLDING cup"); and
experience teaches **what an action does**, cascading through consequences:

    drop(cup)  ->  fall -> impact -> break -> spill

The crucial part is *why* it breaks. A brain does not memorise "cups break". It
learns that **fragile things break on impact** and **containers of liquid
spill** -- causal rules keyed on **properties**, not on identity. That is what
lets it predict, the first time it ever sees a vase, that dropping it will break
and spill.

So causal knowledge here is learned **per property**, by the delta / Rescorla-
Wagner rule -- the classical associative-learning rule that dopamine reward-
prediction error implements in the brain, and which reproduces blocking and
overshadowing. Nothing is hand-coded per object; the rules generalise to objects
the system has never encountered.

    graph.observe(action="drop", obj="cup", effects=["fall", "impact",
                                                     "break", "spill"])
    ...
    graph.expect("drop", "vase")     # never seen before
    # -> ['fall', 'impact', 'break', 'spill']   (from ceramic+fragile+liquid)

Counterfactuals come for free: ask the same question with a property removed.

    graph.counterfactual("drop", "cup", without="fragile")
    # -> no 'break'
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


@dataclass
class ObjectConcept:
    """An object *file*: a token bound to a set of properties and affordances."""

    name: str
    properties: Set[str] = field(default_factory=set)
    affordances: Set[str] = field(default_factory=set)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.name}({', '.join(sorted(self.properties))})"


@dataclass
class Relation:
    """A typed, directed relation between two objects: ``subject rel object``."""

    subject: str
    relation: str
    object: str

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.subject} -{self.relation}-> {self.object}"


class CausalWorldGraph:
    """Objects + relations + **learned, property-keyed causal rules**.

    Causal weights live in ``w[(action, effect)][property]`` and are learned by
    the delta rule::

        p       = sigma( sum of weights of the properties that are present )
        w_prop += lr * (outcome - p)          # for each present property

    which is Rescorla-Wagner / dopaminergic reward-prediction error. Because the
    weights attach to *properties*, an object never seen before inherits the
    right predictions from what it is made of.

    Effects can themselves cause effects (``fall`` -> ``impact`` -> ``break``),
    so prediction is a **cascade**, not a flat lookup.
    """

    def __init__(self, lr: float = 0.25, threshold: float = 0.5):
        self.lr = float(lr)
        self.threshold = float(threshold)
        self.objects: Dict[str, ObjectConcept] = {}
        self.relations: List[Relation] = []
        # (action, effect) -> {property -> weight};  plus a bias term per pair
        self.w: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(dict)
        self.bias: Dict[Tuple[str, str], float] = defaultdict(float)
        # effect -> effect chaining, learned the same way
        self.chain: Dict[Tuple[str, str], float] = defaultdict(float)
        self.actions: Set[str] = set()
        self.effects: Set[str] = set()

    # -- the graph ----------------------------------------------------------
    def add_object(self, name: str, properties: Iterable[str] = (),
                   affordances: Iterable[str] = ()) -> ObjectConcept:
        obj = ObjectConcept(name, set(properties), set(affordances))
        self.objects[name] = obj
        return obj

    def relate(self, subject: str, relation: str, object_: str) -> Relation:
        r = Relation(subject, relation, object_)
        self.relations.append(r)
        return r

    def relations_of(self, name: str, relation: Optional[str] = None
                     ) -> List[Relation]:
        return [r for r in self.relations
                if (r.subject == name or r.object == name)
                and (relation is None or r.relation == relation)]

    def properties_of(self, name: str) -> Set[str]:
        obj = self.objects.get(name)
        return set(obj.properties) if obj else set()

    # -- learning causal structure ------------------------------------------
    @staticmethod
    def _sigma(x: float) -> float:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))

    def _score(self, action: str, effect: str, props: Set[str]) -> float:
        w = self.w[(action, effect)]
        s = self.bias[(action, effect)] + sum(w.get(p, 0.0) for p in props)
        return self._sigma(s)

    def observe(self, action: str, obj: str, effects: Sequence[str],
                properties: Optional[Iterable[str]] = None) -> None:
        """Watch ``action`` applied to ``obj`` and see which effects followed.

        Learning is contrastive: every effect the system knows about is either
        present (target 1) or absent (target 0) in this episode, so the delta
        rule learns both what causes an effect **and what does not**."""
        props = set(properties) if properties is not None \
            else self.properties_of(obj)
        self.actions.add(action)
        seen = set(effects)
        self.effects |= seen
        for effect in self.effects:
            target = 1.0 if effect in seen else 0.0
            p = self._score(action, effect, props)
            delta = self.lr * (target - p)
            self.bias[(action, effect)] += delta
            w = self.w[(action, effect)]
            for prop in props:
                w[prop] = w.get(prop, 0.0) + delta
        # learn effect -> effect chaining from co-occurrence within the episode
        for i, a in enumerate(effects):
            for b in effects[i + 1:]:
                self.chain[(a, b)] += 1.0

    # -- prediction: the causal cascade -------------------------------------
    def expect(self, action: str, obj: str,
               properties: Optional[Iterable[str]] = None,
               with_probabilities: bool = False):
        """What do we expect ``action`` on ``obj`` to cause?

        Works for objects never seen before: the prediction is assembled from
        the object's **properties**. Effects are returned in causal order (the
        order experience showed them cascading)."""
        props = set(properties) if properties is not None \
            else self.properties_of(obj)
        scored = [(e, self._score(action, e, props)) for e in sorted(self.effects)]
        fired = [(e, p) for e, p in scored if p >= self.threshold]
        fired.sort(key=lambda ep: (-self._cascade_rank(ep[0]), -ep[1]))
        if with_probabilities:
            return fired
        return [e for e, _ in fired]

    def _cascade_rank(self, effect: str) -> int:
        """How *early* an effect sits in the causal chain (more things follow it
        than precede it => earlier)."""
        after = sum(c for (a, b), c in self.chain.items() if a == effect)
        before = sum(c for (a, b), c in self.chain.items() if b == effect)
        return int(after - before)

    def why(self, action: str, obj: str, effect: str,
            properties: Optional[Iterable[str]] = None) -> List[Tuple[str, float]]:
        """Which properties are responsible for this prediction (explanation)."""
        props = set(properties) if properties is not None \
            else self.properties_of(obj)
        w = self.w[(action, effect)]
        contrib = [(p, float(w.get(p, 0.0))) for p in props]
        return sorted(contrib, key=lambda pv: -pv[1])

    # -- counterfactual reasoning -------------------------------------------
    def counterfactual(self, action: str, obj: str, without: str = None,
                       with_: str = None) -> List[str]:
        """"What if this object were not fragile?" -- re-run the cascade on a
        modified property set. This is imagination over causal structure."""
        props = self.properties_of(obj)
        if without is not None:
            props = props - {without}
        if with_ is not None:
            props = props | {with_}
        return self.expect(action, obj, properties=props)

    # -- a scene: objects + relations + what an action would do -------------
    def scene_summary(self) -> str:  # pragma: no cover - cosmetic
        lines = [f"objects:   {', '.join(repr(o) for o in self.objects.values())}",
                 f"relations: {', '.join(repr(r) for r in self.relations)}"]
        return "\n".join(lines)


# -- a teaching world: physical episodes the system learns from --------------

#: property-based ground truth of physics, used ONLY to generate experience.
#: The graph never sees these rules -- it must induce them from episodes.
def _true_effects(action: str, props: Set[str]) -> List[str]:
    out: List[str] = []
    if action == "drop":
        out += ["fall", "impact"]
        if "fragile" in props:
            out.append("break")
        if "contains_liquid" in props:
            out.append("spill")
        if "elastic" in props:
            out.append("bounce")
        if "soft" in props:
            out.append("deform")
    elif action == "squeeze":
        if "soft" in props:
            out.append("deform")
        if "fragile" in props:
            out.append("break")
        if "contains_liquid" in props:
            out.append("spill")
    elif action == "push":
        out.append("slide")
        if "elastic" in props:
            out.append("roll")
        if "fragile" in props and "flat" not in props:
            out.append("topple")
    return out


TRAIN_OBJECTS: Dict[str, Set[str]] = {
    "cup":     {"ceramic", "fragile", "contains_liquid"},
    "glass":   {"transparent", "fragile", "contains_liquid"},
    "plate":   {"ceramic", "fragile", "flat"},
    "ball":    {"rubber", "elastic"},
    "pillow":  {"soft", "light"},
    "brick":   {"heavy", "solid"},
    "bottle":  {"plastic", "contains_liquid"},
    "mirror":  {"fragile", "flat", "reflective"},
    "sponge":  {"soft", "absorbent"},
    "can":     {"metal", "contains_liquid", "solid"},
    # Two objects that DE-CORRELATE the cues. Without them, "elastic" only ever
    # appears together with "rubber", so Rescorla-Wagner shares the associative
    # strength between them (overshadowing) and neither cue alone predicts
    # bouncing -- a real effect seen in animal conditioning. Varied experience,
    # not a patched rule, is what lets the brain isolate the true cause.
    "spring":  {"metal", "elastic"},
    "eraser":  {"rubber", "soft"},
}

#: objects the system has NEVER seen -- the generalisation test.
NOVEL_OBJECTS: Dict[str, Set[str]] = {
    "vase":      {"ceramic", "fragile", "contains_liquid"},
    "teapot":    {"ceramic", "fragile", "contains_liquid", "heavy"},
    "balloon":   {"elastic", "light"},
    "cushion":   {"soft", "light", "flat"},
    "window":    {"fragile", "flat", "transparent"},
    "jar":       {"transparent", "fragile", "contains_liquid", "solid"},
    "rubber_duck": {"rubber", "elastic", "light"},
    "carton":    {"contains_liquid", "light"},
}

ACTIONS = ("drop", "squeeze", "push")


def build_causal_world(epochs: int = 40, seed: int = 0,
                       verbose: bool = False) -> CausalWorldGraph:
    """Let the graph *experience* physical episodes and induce the causal rules.

    It only ever sees ``(action, object, effects)`` -- never a rule. Afterwards
    it should predict correctly for objects it has never encountered."""
    rng = np.random.default_rng(seed)
    g = CausalWorldGraph(lr=0.25)
    for name, props in TRAIN_OBJECTS.items():
        g.add_object(name, props)
    # a small scene, so relations are part of the world too
    g.relate("cup", "on", "table")
    g.relate("human", "holding", "cup")
    g.add_object("table", {"solid", "flat", "heavy"})
    g.add_object("human", {"animate", "agent"})

    names = list(TRAIN_OBJECTS)
    for _ in range(epochs):
        rng.shuffle(names)
        for name in names:
            for action in ACTIONS:
                props = TRAIN_OBJECTS[name]
                g.observe(action, name, _true_effects(action, props), props)
    if verbose:
        print(f"  learned {len(g.effects)} effect types over {len(ACTIONS)} actions")
    return g


def causal_generalisation_score(g: CausalWorldGraph,
                                objects: Optional[Dict[str, Set[str]]] = None
                                ) -> float:
    """Exact-match accuracy of the predicted effect *set* on objects the graph
    has never seen -- the honest test of whether it induced real causal rules
    instead of memorising instances."""
    objects = objects if objects is not None else NOVEL_OBJECTS
    ok = 0
    total = 0
    for name, props in objects.items():
        for action in ACTIONS:
            pred = set(g.expect(action, name, properties=props))
            true = set(_true_effects(action, props))
            ok += int(pred == true)
            total += 1
    return ok / max(total, 1)
