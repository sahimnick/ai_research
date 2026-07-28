"""
perceptloop.py
==============

**Phase 3: the whole chain joined.** And **Phase 4: perception that knows when
it does not know**.

    pixels -> V1 spikes -> V2 fragments -> object file -> properties
           -> affordances -> causal model -> imagination

Every one of those stages already existed in this project and every one was
measured on its own bench. What did not exist was the *chain*: the object file
that carries a percept from the eyes into :mod:`objects`, so that seeing
something and knowing what it affords are the same act.

The object file
---------------
:class:`ObjectFile` is the join. The name comes from Kahneman and Treisman's
object file: a temporary, updateable record that binds the features currently
attributed to one thing, keeps its identity while it moves, and is what
properties get attached to. Here it holds

    * the V1+V2 code -- the percept itself, in spikes;
    * where it is and how confident the identification is;
    * the properties read off it, and through them the affordances and the
      causal expectations from :class:`~neurobrain.objects.CausalWorldGraph`.

Properties are **not** read from a label. They are recalled from the nearest
codes the brain has already bound to properties, so an object never seen before
gets the properties of what it most resembles -- which is what lets the causal
graph generalise to it.

Phase 4: active perception
--------------------------
The part that makes this perception rather than classification. After each
glance the brain asks *how sure am I*, and if the answer is "not enough" it does
not guess -- it **moves the eye and looks again**, then fuses the new evidence
with the old.

Two things make this a real mechanism rather than a slogan:

* **The uncertainty is its own.** It comes from the spread of the match over
  candidate identities (the entropy of the posterior), not from being told the
  answer is wrong.
* **The next look does not repeat.** The eye goes to a part it has not seen --
  novelty-driven sampling, which is what inhibition of return buys an eye. Note
  the honest limit of this: the policy is *novelty*, not a calculation of which
  view would be most discriminative. It is not claimed to be the latter.

Three controls separate the two things that could be doing the work:

    ``one_glance``      guess from the first look. The floor.
    ``random_replace``  the same number of extra glances, sampled **with
                        replacement** -- an eye with no memory of where it has
                        been. Isolates the value of inhibition of return.
    ``random_sweep``    the same number of extra glances covering distinct parts,
                        but a **fixed** budget with no stopping rule. Isolates
                        the value of deciding for oneself when to stop.

Active perception has to beat all three, and what it beats each by says which
mechanism earned the gain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..vision.widev1 import _unit


# ---------------------------------------------------------------------------
# The object file
# ---------------------------------------------------------------------------
@dataclass
class ObjectFile:
    """One thing the brain is currently holding on to."""

    code: np.ndarray                      # the V1+V2 percept, in spikes
    row: int = 0
    col: int = 0
    identity: Optional[str] = None
    confidence: float = 0.0
    entropy: float = 1.0
    glances: int = 1
    properties: Set[str] = field(default_factory=set)
    affordances: List[str] = field(default_factory=list)
    expectations: Dict[str, List[str]] = field(default_factory=dict)

    def fuse(self, code: np.ndarray, w: float = 1.0) -> None:
        """Add another glance to the same file -- evidence accumulates.

        This is the object file's whole purpose: two looks at one thing update
        one record, instead of producing two unrelated percepts."""
        self.code = _unit(self.code * self.glances + w * _unit(code))
        self.glances += 1

    def __repr__(self) -> str:                       # pragma: no cover
        p = ",".join(sorted(self.properties)) or "-"
        return (f"<ObjectFile {self.identity or '?'} conf={self.confidence:.2f} "
                f"glances={self.glances} props=[{p}]>")


# ---------------------------------------------------------------------------
# Perception all the way through to affordances
# ---------------------------------------------------------------------------
AFFORDANCE_OF = {
    "fragile": "can break", "elastic": "can bounce", "heavy": "resists lifting",
    "soft": "can squash", "rigid": "keeps its shape", "contains-liquid":
    "can spill", "rollable": "can roll", "light": "easy to lift",
}


class PerceptualObjectMind:
    """The full chain, joined and runnable.

    ``stream`` is any front end exposing ``code(image)`` -- the self-organized
    V1+V2 of :mod:`v2binding` is the intended one. ``graph`` is a
    :class:`~neurobrain.objects.CausalWorldGraph`, which supplies properties,
    effects and counterfactuals once an identity is available."""

    def __init__(self, stream, graph, vigilance: float = 0.35,
                 temperature: float = 20.0):
        self.stream = stream
        self.graph = graph
        self.vigilance = float(vigilance)
        # Softmax sharpness for turning cosine matches into a posterior. It has
        # to be calibrated, not guessed: at T=8 the mean confidence over real
        # glances was 0.21, so a vigilance of 0.35 rejected 94% of them and the
        # brain said "I don't know" to almost everything. Use `calibrate`.
        self.temperature = float(temperature)
        self.names: List[str] = []
        self.protos: List[np.ndarray] = []
        self.props: List[Set[str]] = []

    # -- learning what things look like ------------------------------------
    def bind(self, image: np.ndarray, name: str,
             properties: Iterable[str] = ()) -> None:
        """Attach a percept to a name and its properties.

        Labels here play the same restricted role they play everywhere in this
        project: they *name* a code the perception produced. They never shape
        the code."""
        c = _unit(self.stream.code(image))
        if name in self.names:
            i = self.names.index(name)
            self.protos[i] = _unit(self.protos[i] + 0.25 * (c - self.protos[i]))
            self.props[i] |= set(properties)
        else:
            self.names.append(name)
            self.protos.append(c)
            self.props.append(set(properties))

    # -- one glance --------------------------------------------------------
    def _posterior(self, code: np.ndarray) -> np.ndarray:
        if not self.protos:
            return np.zeros(0, np.float32)
        s = np.array([float(code @ p) for p in self.protos], np.float32)
        s = np.maximum(s, 0.0)
        e = np.exp((s - s.max()) * self.temperature)
        return (e / max(e.sum(), 1e-9)).astype(np.float32)

    def calibrate(self, images: Sequence[np.ndarray], accept: float = 0.7
                  ) -> float:
        """Set the "I don't know" threshold from the brain's own experience.

        Vigilance is put at the percentile of the confidence distribution over
        familiar input that leaves ``accept`` of it accepted. This uses no
        labels -- only how confident this brain typically is when looking at
        things it has seen before -- and it is what stops the threshold from
        being a number I picked."""
        conf = np.array([float(self._posterior(_unit(self.stream.code(im))).max())
                         for im in images], np.float32)
        self.vigilance = float(np.quantile(conf, 1.0 - accept))
        return self.vigilance

    def look(self, image: np.ndarray, row: int = 0, col: int = 0
             ) -> ObjectFile:
        """Perceive once and open an object file on what was seen."""
        f = ObjectFile(_unit(self.stream.code(image)), row, col)
        return self._interpret(f)

    def _interpret(self, f: ObjectFile) -> ObjectFile:
        """Code -> identity -> properties -> affordances -> causal expectations."""
        post = self._posterior(f.code)
        if not len(post):
            return f
        i = int(post.argmax())
        f.confidence = float(post[i])
        p = post[post > 1e-9]
        f.entropy = float(-(p * np.log(p)).sum() / np.log(max(len(post), 2)))
        # Below vigilance the honest answer is "I don't know", not a guess.
        f.identity = self.names[i] if f.confidence >= self.vigilance else None
        f.properties = set(self.props[i]) if f.identity else set()
        if f.identity and f.identity in self.graph.objects:
            f.properties |= self.graph.properties_of(f.identity)
        f.affordances = sorted({AFFORDANCE_OF[q] for q in f.properties
                                if q in AFFORDANCE_OF})
        f.expectations = {}
        if f.identity:
            for action in ("drop", "squeeze", "push"):
                try:
                    f.expectations[action] = list(
                        self.graph.expect(action, f.identity))
                except Exception:
                    pass
        return f

    # -- Phase 4: look again when the evidence is thin ---------------------
    def perceive_actively(self, glance_fn, max_glances: int = 4,
                          need: float = 0.55) -> ObjectFile:
        """Keep looking until confident enough, or until the budget runs out.

        ``glance_fn(n)`` returns the image for the n-th look. The loop is the
        point: the brain reports its own uncertainty, decides that one glance is
        not enough, takes another, fuses it into the same object file, and stops
        as soon as the evidence is sufficient -- so an easy object costs one
        glance and a hard one costs several."""
        f = self.look(glance_fn(0))
        while f.glances < max_glances and f.confidence < need:
            f.fuse(self.stream.code(glance_fn(f.glances)))
            f = self._interpret(f)
        return f

    def imagine(self, f: ObjectFile, action: str) -> str:
        """What would happen if I did this to the thing I am looking at?

        This is the far end of the chain: a percept made of spikes ends as a
        prediction about the world, and the brain can run it without acting."""
        if not f.identity:
            return "I cannot say -- I do not know what this is."
        eff = f.expectations.get(action) or list(
            self.graph.expect(action, f.identity))
        if not eff:
            return f"{action} {f.identity}: I expect nothing in particular."
        return f"{action} {f.identity} -> " + ", ".join(eff)


# ---------------------------------------------------------------------------
# The experiment
# ---------------------------------------------------------------------------
@dataclass
class LoopReport:
    """What the joined chain and active perception actually achieved."""

    one_glance: float = 0.0
    active: float = 0.0
    random_extra: float = 0.0
    random_sweep: float = 0.0
    mean_glances: float = 0.0
    dont_know_rate: float = 0.0
    property_accuracy: float = 0.0
    affordance_examples: List[str] = field(default_factory=list)
    imagination_examples: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (f"one glance (floor)           : {self.one_glance:.1%}\n"
                f"random glances, with repeats : {self.random_extra:.1%}\n"
                f"full sweep, no stopping rule : {self.random_sweep:.1%}\n"
                f"ACTIVE perception            : {self.active:.1%} "
                f"({self.mean_glances:.1f} glances on average)")


def _occlude(img: np.ndarray, part: int, rng: np.random.Generator
             ) -> np.ndarray:
    """Show only one quadrant-ish strip of the object.

    A single glance at a partly hidden thing is genuinely ambiguous, which is
    the condition under which looking again can help. Without it the first
    glance is already enough and the experiment would measure nothing."""
    out = np.zeros_like(np.asarray(img, np.float32))
    h, w = out.shape
    if part % 4 == 0:
        out[:h // 2] = img[:h // 2]
    elif part % 4 == 1:
        out[h // 2:] = img[h // 2:]
    elif part % 4 == 2:
        out[:, :w // 2] = img[:, :w // 2]
    else:
        out[:, w // 2:] = img[:, w // 2:]
    return out


def full_loop_experiment(n_v1: int = 512, n_v2: int = 128, n_train: int = 600,
                         n_test: int = 300, n_develop: int = 200,
                         max_glances: int = 4, seed: int = 0,
                         verbose: bool = False) -> LoopReport:
    """Run pixels -> spikes -> fragments -> object file -> affordances ->
    imagination, and test whether looking again beats guessing.

    The active condition is compared with a control that takes **the same number
    of extra glances at random parts**. Without that control the result would
    only show that more looks are better than fewer, which is not the claim."""
    from ..sensing.realworld import load_mnist
    from ..vision.widev1 import WideV1
    from ..learning.selforganize import randomise_filters, develop_v1
    from ..vision.v2binding import V2Binding, VentralV1V2, preferred_orientations
    from ..world.objects import build_causal_world

    def say(*a):
        if verbose:
            print(*a)

    trx, trY, tex, teY = load_mnist(n_train=n_train, n_test=n_test)

    say("V1 discovers its features; V2 discovers its conjunctions ...")
    v1 = randomise_filters(WideV1(n_cells=n_v1, seed=seed), seed=seed)
    develop_v1(v1, trx[:n_develop], epochs=2, seed=seed)
    orient = preferred_orientations(v1)
    v2 = V2Binding(n_units=n_v2, seed=seed)
    stream = VentralV1V2(v1, v2, orient)
    v2.train([stream.parts(im) for im in trx[:n_develop]], epochs=2)

    say("joining the stream to the causal world graph ...")
    graph = build_causal_world(verbose=False)
    mind = PerceptualObjectMind(stream, graph)

    # every digit is given a couple of properties so the chain has something to
    # carry all the way to affordances and imagination
    digit_props = {
        0: ("rollable", "rigid"), 1: ("rigid", "light"),
        2: ("soft", "light"), 3: ("soft", "elastic"),
        4: ("rigid", "heavy"), 5: ("fragile", "contains-liquid"),
        6: ("rollable", "soft"), 7: ("rigid", "fragile"),
        8: ("elastic", "soft"), 9: ("rollable", "light"),
    }
    for im, y in zip(trx, trY):
        mind.bind(im, str(int(y)), digit_props[int(y)])
    mind.calibrate(trx[:150], accept=0.7)

    say("perceiving occluded objects: one glance, random extra, active ...")
    rng = np.random.default_rng(seed + 3)
    rep = LoopReport()
    ok1 = ok_a = ok_r = ok_s = 0
    glances = 0
    unknown = 0
    prop_hits = prop_tot = 0

    for i in range(len(tex)):
        img, y = tex[i], str(int(teY[i]))
        order = rng.permutation(4)

        f1 = mind.look(_occlude(img, order[0], rng))
        ok1 += int(f1.identity == y)

        def seq(n, order=order, img=img):
            return _occlude(img, order[n % 4], rng)

        fa = mind.perceive_actively(seq, max_glances=max_glances)
        ok_a += int(fa.identity == y)
        glances += fa.glances
        unknown += int(fa.identity is None)

        # control A: same number of glances, sampled WITH replacement -- an eye
        # with no inhibition of return.
        rnd = rng.integers(0, 4, fa.glances)
        fr = mind.look(_occlude(img, int(rnd[0]), rng))
        for g in rnd[1:]:
            fr.fuse(stream.code(_occlude(img, int(g), rng)))
            fr = mind._interpret(fr)
        ok_r += int(fr.identity == y)

        # control B: the FULL sweep of distinct parts, no stopping rule -- the
        # same coverage the active condition can reach, but never stopping early
        fs = mind.look(_occlude(img, order[0], rng))
        for k in range(1, max_glances):
            fs.fuse(stream.code(_occlude(img, order[k % 4], rng)))
            fs = mind._interpret(fs)
        ok_s += int(fs.identity == y)

        if fa.identity is not None:
            truth = set(digit_props[int(teY[i])])
            prop_hits += len(truth & fa.properties)
            prop_tot += len(truth)

    n = len(tex)
    rep.one_glance = ok1 / n
    rep.active = ok_a / n
    rep.random_extra = ok_r / n
    rep.random_sweep = ok_s / n
    rep.mean_glances = glances / n
    rep.dont_know_rate = unknown / n
    rep.property_accuracy = prop_hits / max(prop_tot, 1)

    f = mind.perceive_actively(lambda k: _occlude(tex[0], k, rng))
    rep.affordance_examples = f.affordances
    rep.imagination_examples = [mind.imagine(f, a)
                                for a in ("drop", "squeeze", "push")]
    if verbose:
        print()
        print(rep.summary())
        print(f"   'I don't know' rate : {rep.dont_know_rate:.1%}")
        print(f"   properties recalled : {rep.property_accuracy:.1%}")
        print(f"\n   a percept carried all the way through:")
        print(f"     affordances : {rep.affordance_examples}")
        for line in rep.imagination_examples:
            print(f"     imagine     : {line}")
    return rep
