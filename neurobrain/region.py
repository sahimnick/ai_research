"""
region.py
=========

A **region** is a named functional area of the brain: a population of neurons
that share a job and a location. Real brains are organised this way -- visual
cortex, auditory cortex, association areas, motor cortex -- and crucially the
*same kind of neuron* does different work depending on which region it sits in
and what it is wired to. That is the idea the user described:

    "مانند مغز انسان که سلول های اعصاب چند مدل هستند و مدل های مشابه
     میتوانند هر کاری را در ناحیه های خودشون انجام بدن"
    (like the human brain, where nerve cells come in a few models and similar
     models can do any job within their own regions)

A region here bundles:
    * a :class:`Population` of neurons (usually a mix of excitatory + inhibitory),
    * a role tag (``sensory`` / ``association`` / ``motor`` / ...),
    * a 2-D layout position used by the live visualizer,
    * local recurrent connectivity (added later by the Brain).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .neuron import Population, type_id


# Canonical roles. They are just labels, but they let the world model and the
# visualizer treat input/output regions specially.
ROLE_SENSORY = "sensory"          # receives external stimulation
ROLE_ASSOCIATION = "association"  # internal processing / concepts
ROLE_MOTOR = "motor"              # produces the brain's "output" / actions
ROLE_MEMORY = "memory"            # holds persistent assemblies


@dataclass
class Region:
    """One functional area of the brain.

    Parameters
    ----------
    name:
        Unique region name, e.g. ``"V1"`` or ``"auditory"``.
    size:
        Number of neurons.
    role:
        One of the ``ROLE_*`` constants.
    excitatory_type / inhibitory_type:
        Which neuron models fill the excitatory / inhibitory slots.
    inhib_fraction:
        Fraction of neurons that are inhibitory (~0.2 in real cortex).
    position:
        (x, y) centre used for graph layout. Auto-assigned if omitted.
    rng:
        Shared random generator for reproducibility.
    """

    name: str
    size: int
    role: str = ROLE_ASSOCIATION
    excitatory_type: str = "regular_spiking"
    inhibitory_type: str = "fast_spiking"
    inhib_fraction: float = 0.2
    position: Optional[Tuple[float, float]] = None
    rng: Optional[np.random.Generator] = None

    population: Population = field(init=False)
    # global neuron ids assigned by the Brain (for a single flat index space)
    gid_start: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.rng = self.rng if self.rng is not None else np.random.default_rng()
        if self.size <= 0:
            raise ValueError("Region size must be positive.")

        n_inhib = int(round(self.size * self.inhib_fraction))
        exc_id = type_id(self.excitatory_type)
        inh_id = type_id(self.inhibitory_type)

        # Build the per-neuron type-id array vectorised (no Python loop), so a
        # region can hold millions of neurons cheaply. Inhibitory cells are
        # spread evenly through the population.
        ids = np.full(self.size, exc_id, dtype=np.int16)
        if n_inhib:
            inhib_positions = np.linspace(
                0, self.size - 1, n_inhib, dtype=np.int64
            )
            ids[inhib_positions] = inh_id

        self.population = Population(self.size, ids, rng=self.rng, jitter=0.05)

    # -- convenience ------------------------------------------------------
    @property
    def n(self) -> int:
        return self.size

    @property
    def is_sensory(self) -> bool:
        return self.role == ROLE_SENSORY

    @property
    def is_motor(self) -> bool:
        return self.role == ROLE_MOTOR

    @property
    def excitatory_mask(self) -> np.ndarray:
        return ~self.population.inhibitory

    def activation(self) -> np.ndarray:
        """Per-neuron decaying activation (for the visualizer)."""
        return self.population.activation

    def spike_indices(self) -> np.ndarray:
        """Local indices of neurons that fired on the last step."""
        return np.nonzero(self.population.spiked)[0]

    def mean_rate(self) -> float:
        """Fraction of neurons firing on the current step (0..1)."""
        return float(self.population.spiked.mean())

    def reset(self) -> None:
        self.population.reset()

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<Region {self.name!r} role={self.role} n={self.size}>"


def default_layout(regions: Sequence[Region]) -> None:
    """Assign 2-D positions to regions that don't have one.

    Sensory regions are placed on the left, motor on the right, everything
    else spread through the middle -- so the live visualizer naturally shows
    activity flowing left (input) to right (output).
    """
    sensory = [r for r in regions if r.role == ROLE_SENSORY]
    motor = [r for r in regions if r.role == ROLE_MOTOR]
    middle = [r for r in regions if r.role not in (ROLE_SENSORY, ROLE_MOTOR)]

    def _column(items: Sequence[Region], x: float) -> None:
        for i, r in enumerate(items):
            if r.position is None:
                y = 0.5 if len(items) == 1 else i / (len(items) - 1)
                r.position = (x, float(y))

    _column(sensory, 0.0)
    _column(middle, 0.5)
    _column(motor, 1.0)
