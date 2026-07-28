"""
builder.py
==========

Convenience factory that wires up a small but complete brain.

The layout follows the sensory -> association -> motor organisation of the
cortex, so activity naturally flows left to right through the visualizer:

    vision ┐
    sound  ├─> association ─┬─> memory ─┐
    text   ┘                └───────────┴─> motor (output)

Each sensory region is a thalamo-cortical style relay; association/memory are
mixed excitatory + inhibitory cortex; motor is the read-out. Every projection
is plastic (STDP) so the network can be taught by induction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .brain import Brain
from .region import (Region, ROLE_ASSOCIATION, ROLE_MEMORY, ROLE_MOTOR,
                     ROLE_SENSORY)
from .synapse import STDPConfig


@dataclass
class BrainSpec:
    """Sizes for a default brain (tweak for bigger/smaller networks)."""

    vision: int = 100
    sound: int = 80
    text: int = 120
    association: int = 200
    memory: int = 640
    motor: int = 60
    seed: int = 0
    noise: float = 0.6
    # grounding-path (sensory -> memory) connectivity + competitive target.
    # A larger vocabulary wants a bigger text region and slightly sparser,
    # stronger grounding (see build_knowledge_brain).
    ground_p: float = 0.50
    ground_target_in: float = 50.0


def build_default_brain(spec: Optional[BrainSpec] = None) -> Brain:
    """Construct and finalize a ready-to-teach brain.

    Returns a :class:`Brain` with sensory regions ``vision``/``sound``/``text``,
    an ``association`` hub, a ``memory`` store, and a ``motor`` output, all
    connected feed-forward with a little recurrence and feedback.
    """
    spec = spec or BrainSpec()
    b = Brain(seed=spec.seed, noise=spec.noise)

    # Sensory relays (thalamo-cortical cells fire readily and pass signals on).
    # These are purely excitatory: a sense organ *drives* the cortex, it does
    # not inhibit it, so every active input neuron pushes its targets on.
    b.add_region(Region("vision", spec.vision, role=ROLE_SENSORY,
                        excitatory_type="thalamo_cortical",
                        inhib_fraction=0.0, rng=b.rng))
    b.add_region(Region("sound", spec.sound, role=ROLE_SENSORY,
                        excitatory_type="thalamo_cortical",
                        inhib_fraction=0.0, rng=b.rng))
    b.add_region(Region("text", spec.text, role=ROLE_SENSORY,
                        excitatory_type="thalamo_cortical",
                        inhib_fraction=0.0, rng=b.rng))

    # Cortical processing (mixed excitatory pyramidal + fast-spiking inhibitory).
    b.add_region(Region("association", spec.association, role=ROLE_ASSOCIATION,
                        excitatory_type="regular_spiking",
                        inhibitory_type="fast_spiking", rng=b.rng))
    b.add_region(Region("memory", spec.memory, role=ROLE_MEMORY,
                        excitatory_type="intrinsic_bursting",
                        inhibitory_type="low_threshold", rng=b.rng))
    b.add_region(Region("motor", spec.motor, role=ROLE_MOTOR,
                        excitatory_type="chattering", rng=b.rng))

    # Feed-forward sensory -> association (the processing path that feeds motor
    # and drives the visualizer; plastic but not the binding site).
    for s in ("vision", "sound", "text"):
        b.connect(s, "association", p=0.15, w_scale=5.5, delay_range=(1, 4))

    # Direct sensory -> memory: the *grounding* path. A short, monosynaptic
    # route so a percept can be bound straight onto a concept assembly. It uses
    # competitive (normalised) STDP so each concept becomes selective for its
    # own input pattern -- this is what makes induction learn reliably.
    ground = STDPConfig(a_plus=0.10, a_minus=0.05,
                        w_max=spec.ground_target_in,
                        normalize=True, target_in=spec.ground_target_in)
    for s in ("vision", "sound", "text"):
        b.connect(s, "memory", p=spec.ground_p, w_scale=1.0, delay_range=(1, 3),
                  stdp=ground)

    # Association -> memory kept very weak and fixed so the shared hub cannot
    # flood the concept layer with generic drive and wash out the specific
    # bindings learned on the direct sensory path.
    b.connect("association", "memory", p=0.05, w_scale=0.15, delay_range=(1, 5),
              plastic=False)
    # Memory feeds back to association (so a woken concept can influence
    # processing and the motor read-out) but has no excitatory self-recurrence,
    # which would smear activity across unrelated assemblies.
    b.connect("memory", "association", p=0.06, w_scale=2.0, delay_range=(2, 6))
    b.connect("association", "association", p=0.04, w_scale=1.5,
              delay_range=(1, 3))

    # Read-out to motor (the brain's "output").
    b.connect("association", "motor", p=0.12, w_scale=3.5, delay_range=(1, 4))
    b.connect("memory", "motor", p=0.08, w_scale=2.5, delay_range=(1, 4))

    b.finalize(activation_threshold=0.18)
    return b


def build_knowledge_brain(seed: int = 0) -> Brain:
    """A brain tuned for a large foundational vocabulary (see knowledge.py).

    It uses a bigger text sense and concept store than the default so that
    ~150 words get distinct, low-collision codes, plus amortised normalisation
    so teaching that many concepts stays fast. The concept region still grows
    (neurogenesis) if the vocabulary outgrows it.
    """
    b = build_default_brain(BrainSpec(
        text=500, memory=1400, ground_p=0.5, ground_target_in=60, seed=seed))
    for bundle in b.projections:
        if bundle.dst_name == "memory" and bundle.stdp.normalize:
            bundle.stdp.normalize_every = 6
    return b


@dataclass
class LargeBrainSpec:
    """Sizes for a large-scale brain (millions of neurons).

    The defaults build ~10 million neurons and ~40 million synapses, which fit
    comfortably in a few GB of RAM thanks to the int32/float32 Structure-of-
    Arrays engine. A few million-neuron regions are wired sensory -> association
    <-> memory -> motor, exactly like the small brain, just scaled up.
    """

    vision: int = 1_000_000
    sound: int = 800_000
    text: int = 1_200_000
    association: int = 3_000_000
    memory: int = 3_000_000
    motor: int = 1_000_000
    seed: int = 0
    noise: float = 0.6
    plastic: bool = False   # off by default: at this size, run propagation; a
                            # trainable cognitive core is better kept moderate.


@dataclass
class ComprehensiveSpec:
    """A ~40-million-neuron substrate with the cognition-heavy areas enlarged.

    The pallium (associative memory), cerebrum (general cortex for reasoning /
    analysis) and world_model get the lion's share of neurons, because those are
    the areas a *comprehensive* mind leans on. Honest note: a big substrate is
    room to grow, not intelligence by itself -- the reasoning, memory and
    imagination are carried by the functional modules (psyche, analogy, mind);
    this just gives them cortical-scale headroom.
    """

    vision: int = 1_400_000
    sound: int = 1_000_000
    text: int = 1_600_000
    pallium: int = 12_000_000       # associative memory (reconstruction, recall)
    cerebrum: int = 13_000_000      # general cortex (reasoning, analysis)
    world_model: int = 7_000_000    # inner world / imagination substrate
    motor: int = 4_000_000
    seed: int = 0
    noise: float = 0.6


def build_comprehensive_substrate(spec: Optional[ComprehensiveSpec] = None,
                                  verbose: bool = True) -> Brain:
    """Build the ~40M-neuron substrate with enlarged pallium/cerebrum/world_model.

    Measured on a 15 GB machine: ~40M neurons + ~207M synapses, ~9 GB RAM
    (see the printed report; scale was raised from 30M in v0.20). Raw scale does
    not create cognition -- it is the headroom the functional modules run in.
    """
    import time
    spec = spec or ComprehensiveSpec()
    t0 = time.time()
    b = Brain(seed=spec.seed, noise=spec.noise)
    for name, size, role in (
            ("vision", spec.vision, ROLE_SENSORY),
            ("sound", spec.sound, ROLE_SENSORY),
            ("text", spec.text, ROLE_SENSORY),
            ("pallium", spec.pallium, ROLE_MEMORY),
            ("cerebrum", spec.cerebrum, ROLE_ASSOCIATION),
            ("world_model", spec.world_model, ROLE_ASSOCIATION),
            ("motor", spec.motor, ROLE_MOTOR)):
        b.add_region(Region(name, size, role=role,
                            inhib_fraction=0.0 if role == ROLE_SENSORY else 0.2,
                            rng=b.rng))
    if verbose:
        print(f"  {b.n_neurons:,} neurons ({time.time() - t0:.0f}s)")

    def deg(r, d):
        return int(b.region(r).size * d)
    for s in ("vision", "sound", "text"):
        b.connect(s, "cerebrum", n_edges=deg(s, 10), w_scale=12.0, delay_range=(1, 4))
        b.connect(s, "pallium", n_edges=deg(s, 6), w_scale=10.0, delay_range=(1, 3))
    b.connect("cerebrum", "pallium", n_edges=deg("cerebrum", 3), w_scale=6.0)
    b.connect("pallium", "cerebrum", n_edges=deg("pallium", 3), w_scale=5.0)
    b.connect("cerebrum", "world_model", n_edges=deg("cerebrum", 2), w_scale=3.0)
    b.connect("world_model", "cerebrum", n_edges=deg("world_model", 2), w_scale=3.0)
    b.connect("cerebrum", "motor", n_edges=deg("cerebrum", 2), w_scale=8.0)
    b.finalize(activation_threshold=0.18)
    if verbose:
        syn = sum(len(p) for p in b.projections)
        print(f"  {syn:,} synapses; built in {time.time() - t0:.0f}s total")
    return b


def build_large_brain(spec: Optional[LargeBrainSpec] = None,
                      verbose: bool = True) -> Brain:
    """Build a brain with millions of neurons and synapses.

    This exists to show the engine genuinely scales (the user asked for 10M+
    neurons and synapses). Note honestly: raw size alone does not create
    intelligence -- structure and learning do -- but the simulator handles the
    scale. Learning/teaching is best demonstrated on :func:`build_default_brain`;
    here plasticity defaults off so a large network can be stepped quickly.
    """
    import time
    spec = spec or LargeBrainSpec()
    t0 = time.time()
    b = Brain(seed=spec.seed, noise=spec.noise)

    for name, size in (("vision", spec.vision), ("sound", spec.sound),
                       ("text", spec.text)):
        b.add_region(Region(name, size, role=ROLE_SENSORY,
                            excitatory_type="thalamo_cortical",
                            inhib_fraction=0.0, rng=b.rng))
    b.add_region(Region("association", spec.association, role=ROLE_ASSOCIATION,
                        rng=b.rng))
    b.add_region(Region("memory", spec.memory, role=ROLE_MEMORY,
                        excitatory_type="intrinsic_bursting",
                        inhibitory_type="low_threshold", rng=b.rng))
    b.add_region(Region("motor", spec.motor, role=ROLE_MOTOR,
                        excitatory_type="chattering", rng=b.rng))

    if verbose:
        print(f"  regions built: {b.n_neurons:,} neurons "
              f"({time.time() - t0:.1f}s)")

    # Connectivity as a fixed out-degree per source neuron, so the synapse
    # count scales with the network size. The feed-forward path is denser and
    # stronger than the recurrent one so that, despite the low overall
    # synapse-per-neuron ratio a 15 GB machine allows, a stimulation wave still
    # visibly propagates input -> output.
    P = dict(plastic=spec.plastic)

    def deg(region_name: str, out_degree: float) -> int:
        return int(b.region(region_name).size * out_degree)

    for s in ("vision", "sound", "text"):
        b.connect(s, "association", n_edges=deg(s, 12), w_scale=14.0,
                  delay_range=(1, 4), **P)
        b.connect(s, "memory", n_edges=deg(s, 8), w_scale=12.0,
                  delay_range=(1, 3), **P)
    b.connect("association", "memory", n_edges=deg("association", 4),
              w_scale=6.0, **P)
    b.connect("memory", "association", n_edges=deg("memory", 3),
              w_scale=5.0, **P)
    b.connect("association", "association", n_edges=deg("association", 2),
              w_scale=2.0, **P)
    b.connect("association", "motor", n_edges=deg("association", 3),
              w_scale=9.0, **P)
    b.connect("memory", "motor", n_edges=deg("memory", 2), w_scale=8.0, **P)

    total_syn = sum(len(p) for p in b.projections)
    if verbose:
        print(f"  synapses built: {total_syn:,} ({time.time() - t0:.1f}s)")

    b.finalize(activation_threshold=0.18)
    if verbose:
        print(f"  finalized in {time.time() - t0:.1f}s total")
    return b
