"""
neuron.py
=========

Biologically-inspired neuron models and neuron *types*.

The core model is the **Izhikevich neuron** (E. M. Izhikevich, 2003):

    v' = 0.04 v^2 + 5 v + 140 - u + I
    u' = a (b v - u)
    if v >= 30 mV:   v <- c ,   u <- u + d

With only four parameters ``(a, b, c, d)`` this single equation can reproduce
almost every firing pattern seen in real cortical neurons (regular spiking,
fast spiking, bursting, chattering, ...). That is exactly the property the
brain uses: *the same kind of cell, tuned differently, does different jobs in
different regions*.

Design note (why it is fast **and** precise):
    Instead of creating one Python object per neuron (millions of objects,
    slow), we store the whole population as a *Structure of Arrays* inside
    :class:`Population`. Every neuron shares the same vectorised update, so
    stepping N neurons costs a handful of NumPy operations regardless of N.
    :class:`Neuron` is then a thin *handle* (a view) onto one row of those
    arrays, so you still get clean object access (``neuron.v``, ``neuron.spiked``)
    without paying the per-object cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Neuron types  --  "چند مدل نورون" / multiple neuron models
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NeuronType:
    """A named set of Izhikevich parameters describing one *kind* of neuron.

    Attributes
    ----------
    name:          human readable label (e.g. ``"regular_spiking"``).
    a, b, c, d:    the four Izhikevich parameters.
    inhibitory:    ``True`` for GABAergic (inhibitory) cells. Inhibitory cells
                   subtract from their targets instead of exciting them.
    color:         colour used by the live visualizer.
    description:   short note on where this cell is found / what it does.
    """

    name: str
    a: float
    b: float
    c: float
    d: float
    inhibitory: bool = False
    color: str = "#4C9BE8"
    description: str = ""


# The classic Izhikevich "gallery" of firing patterns. Each of these is a real
# cortical/thalamic cell class. `regular_spiking` and `bursting` are excitatory;
# `fast_spiking` and `low_threshold` are the two main inhibitory interneurons.
NEURON_TYPES: Dict[str, NeuronType] = {
    "regular_spiking": NeuronType(
        "regular_spiking", a=0.02, b=0.20, c=-65, d=8.0,
        inhibitory=False, color="#4C9BE8",
        description="Excitatory pyramidal cell. The workhorse of the cortex.",
    ),
    "intrinsic_bursting": NeuronType(
        "intrinsic_bursting", a=0.02, b=0.20, c=-55, d=4.0,
        inhibitory=False, color="#3A7BC8",
        description="Excitatory cell that fires a burst then settles.",
    ),
    "chattering": NeuronType(
        "chattering", a=0.02, b=0.20, c=-50, d=2.0,
        inhibitory=False, color="#2A5BA8",
        description="Excitatory cell firing fast rhythmic bursts (gamma).",
    ),
    "fast_spiking": NeuronType(
        "fast_spiking", a=0.10, b=0.20, c=-65, d=2.0,
        inhibitory=True, color="#E8734C",
        description="Inhibitory interneuron. Fast, keeps the network in check.",
    ),
    "low_threshold": NeuronType(
        "low_threshold", a=0.02, b=0.25, c=-65, d=2.0,
        inhibitory=True, color="#E85C8A",
        description="Inhibitory cell that fires easily; shapes timing.",
    ),
    "thalamo_cortical": NeuronType(
        "thalamo_cortical", a=0.02, b=0.25, c=-65, d=0.05,
        inhibitory=False, color="#6BCF6B",
        description="Relay cell of the thalamus. The brain's sensory gateway.",
    ),
    "resonator": NeuronType(
        "resonator", a=0.10, b=0.26, c=-65, d=2.0,
        inhibitory=False, color="#B06BCF",
        description="Prefers inputs at a particular frequency (resonance).",
    ),
}


def get_type(name: str) -> NeuronType:
    """Look up a :class:`NeuronType` by name, with a helpful error message."""
    try:
        return NEURON_TYPES[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(
            f"Unknown neuron type {name!r}. "
            f"Available: {sorted(NEURON_TYPES)}"
        ) from exc


# ---------------------------------------------------------------------------
# Type table  --  the key to scaling to millions of neurons
# ---------------------------------------------------------------------------
# Storing one Python NeuronType object per neuron does not scale (a list of ten
# million objects is huge and slow to build). Instead we keep the small list of
# distinct types here and give every neuron a single int8 *type id* into it. The
# a/b/c/d/inhibitory parameters of a whole population are then produced by a
# vectorised gather -- no Python loop over neurons, ever.
_TYPE_NAMES: List[str] = list(NEURON_TYPES)
_TYPE_INDEX: Dict[str, int] = {name: i for i, name in enumerate(_TYPE_NAMES)}
_TABLE_A = np.array([NEURON_TYPES[n].a for n in _TYPE_NAMES], dtype=np.float32)
_TABLE_B = np.array([NEURON_TYPES[n].b for n in _TYPE_NAMES], dtype=np.float32)
_TABLE_C = np.array([NEURON_TYPES[n].c for n in _TYPE_NAMES], dtype=np.float32)
_TABLE_D = np.array([NEURON_TYPES[n].d for n in _TYPE_NAMES], dtype=np.float32)
_TABLE_INH = np.array([NEURON_TYPES[n].inhibitory for n in _TYPE_NAMES],
                      dtype=bool)


def type_id(name: "str | NeuronType") -> int:
    """Return the integer id of a neuron type (for building type-id arrays)."""
    if isinstance(name, NeuronType):
        name = name.name
    return _TYPE_INDEX[name]


class _TypeView:
    """Lazy, memory-free accessor so ``population.types[i]`` still works.

    It reconstructs the :class:`NeuronType` for one neuron on demand from the
    population's compact int8 id array, instead of materialising a giant list.
    """

    __slots__ = ("_ids",)

    def __init__(self, ids: np.ndarray):
        self._ids = ids

    def __getitem__(self, i):
        tid = self._ids[i]
        if np.ndim(tid) == 0:
            return NEURON_TYPES[_TYPE_NAMES[int(tid)]]
        return [NEURON_TYPES[_TYPE_NAMES[int(t)]] for t in tid]

    def __len__(self):
        return len(self._ids)

    def __iter__(self):
        return (NEURON_TYPES[_TYPE_NAMES[int(t)]] for t in self._ids)


# ---------------------------------------------------------------------------
# Population  --  the vectorised (fast) core
# ---------------------------------------------------------------------------
class Population:
    """A group of Izhikevich neurons stored as a Structure of Arrays.

    All state (``v``, ``u``, the parameters, the spike flags) lives in NumPy
    arrays of length ``n``. One call to :meth:`step` advances *every* neuron.

    Parameters
    ----------
    n:
        Number of neurons.
    types:
        Either a single :class:`NeuronType`/name applied to all neurons, or a
        sequence of length ``n`` giving a type per neuron (so one population can
        mix excitatory and inhibitory cells, exactly like a real cortical area).
    rng:
        Optional NumPy random generator (for reproducible parameter jitter).
    jitter:
        Small multiplicative noise added to ``(a,b,c,d)`` so that no two
        neurons are perfectly identical -- real tissue is heterogeneous and the
        heterogeneity makes the dynamics richer.
    """

    SPIKE_PEAK = 30.0  # mV; membrane voltage at which a spike is declared
    _I_CLIP = np.float32(500.0)  # bound on input current (see step)
    _V_FLOOR = np.float32(-120.0)  # mV; below any reversal potential

    def __init__(
        self,
        n: int,
        types: "NeuronType | str | Sequence[NeuronType | str] | np.ndarray",
        rng: Optional[np.random.Generator] = None,
        jitter: float = 0.0,
    ) -> None:
        if n <= 0:
            raise ValueError("A population needs at least one neuron.")
        self.n = int(n)
        self.rng = rng if rng is not None else np.random.default_rng()

        # Resolve a compact int8 type-id array (no per-neuron Python objects).
        if isinstance(types, np.ndarray):
            ids = types.astype(np.int16, copy=False)
            if ids.shape != (n,):
                raise ValueError("type-id array must have shape (n,).")
        elif isinstance(types, (str, NeuronType)):
            ids = np.full(n, type_id(types), dtype=np.int16)
        else:
            resolved = list(types)
            if len(resolved) != n:
                raise ValueError("len(types) must equal n.")
            ids = np.fromiter((type_id(t) for t in resolved), dtype=np.int16,
                              count=n)
        self.type_id = ids
        self.types = _TypeView(ids)   # lazy accessor; costs no memory

        # Gather parameters by vectorised table lookup (float32 for scale).
        self.a = _TABLE_A[ids].copy()
        self.b = _TABLE_B[ids].copy()
        self.c = _TABLE_C[ids].copy()
        self.d = _TABLE_D[ids].copy()
        self.inhibitory = _TABLE_INH[ids]

        if jitter > 0:
            # Multiplicative jitter keeps parameters in a sensible range.
            self.a *= (1 + jitter * self.rng.standard_normal(n)).astype(np.float32)
            self.d *= (1 + jitter * self.rng.standard_normal(n)).astype(np.float32)

        # Dynamic state (float32 keeps a 10M-neuron brain in a few hundred MB).
        self.v = self.c.copy()            # membrane potential (mV)
        self.u = (self.b * self.v).astype(np.float32)   # recovery variable
        self.spiked = np.zeros(n, dtype=bool)

        # A per-neuron "activation" that decays over time -- used purely for
        # visualisation ("how recently / strongly did this neuron fire").
        self.activation = np.zeros(n, dtype=np.float32)

        # Reusable scratch buffer so :meth:`step` allocates almost nothing per
        # call (important when stepping millions of neurons many times).
        self._scratch = np.empty(n, dtype=np.float32)

    # -- simulation -------------------------------------------------------
    def step(self, I: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """Advance the whole population by ``dt`` milliseconds.

        Parameters
        ----------
        I:
            Total input current into each neuron (external + synaptic).
        dt:
            Time step in ms. For numerical stability the voltage update is
            split into ``dt / 0.5`` half-millisecond sub-steps, following
            Izhikevich's own reference implementation.

        Returns
        -------
        np.ndarray of bool
            ``spiked[i]`` is True if neuron *i* fired on this step.
        """
        I = np.asarray(I, dtype=np.float32)
        if I.shape != (self.n,):
            raise ValueError(f"I must have shape ({self.n},), got {I.shape}.")
        # A strongly inhibited cell can be driven so far below rest that the
        # quadratic term overflows float32 and the population fills with NaN --
        # seen for real when the inhibitory pool was made very strong. Real
        # tissue cannot be hyperpolarised without limit either: the driving
        # force on the synaptic conductance vanishes. Clipping the input current
        # is the cheap version of that and keeps the integrator in range.
        np.clip(I, -self._I_CLIP, self._I_CLIP, out=I)
        # and the voltage itself: a cell held far from rest between resets can
        # still drive the quadratic term to overflow. Real membranes are bounded
        # by their reversal potentials; this is that bound.
        np.clip(self.v, self._V_FLOOR, self.SPIKE_PEAK, out=self.v)

        v, u, dv = self.v, self.u, self._scratch
        peak = np.float32(self.SPIKE_PEAK)

        # Detect spikes from the *previous* step's peak, reset, then integrate.
        # In-place reset avoids reallocating the whole voltage/recovery arrays.
        reset = v >= peak
        if reset.any():
            v[reset] = self.c[reset]
            u[reset] += self.d[reset]

        # Sub-stepped voltage integration for stability, all in-place via one
        # scratch buffer:  dv = v*(0.04 v + 5) + 140 - u + I ;  v += h*dv
        substeps = max(1, int(round(dt / 0.5)))
        h = np.float32(dt / substeps)
        for _ in range(substeps):
            np.multiply(v, np.float32(0.04), out=dv)
            dv += np.float32(5.0)
            dv *= v
            dv += np.float32(140.0)
            dv -= u
            dv += I
            dv *= h
            v += dv
            # Clip INSIDE the loop. Clipping only on entry is not enough: a
            # single sub-step can carry v past the range on its own, and the
            # next sub-step then squares it. Seen for real as float32 overflow
            # once inhibition and after-depolarization were both strong.
            np.clip(v, self._V_FLOOR, peak, out=v)
        # u += dt * a * (b v - u)  -- reuse the scratch buffer
        np.multiply(self.b, v, out=dv)
        dv -= u
        dv *= self.a
        dv *= np.float32(dt)
        u += dv

        # Clamp the peak so the spike is well defined for the next step.
        self.spiked = v >= peak
        np.minimum(v, peak, out=v)

        # Decay the visual activation and light up the neurons that fired.
        self.activation *= np.float32(0.85)
        self.activation[self.spiked] = 1.0
        return self.spiked

    def reset(self) -> None:
        """Return the population to rest (start of a fresh trial)."""
        self.v = self.c.copy()
        self.u = self.b * self.v
        self.spiked[:] = False
        self.activation[:] = 0.0

    def grow(self, k: int, kind: "str | NeuronType | None" = None) -> None:
        """Add ``k`` new neurons to the population (neurogenesis).

        New cells are appended at the end (existing indices are untouched, which
        keeps every synapse and concept valid). They start at rest -- freshly
        born, silent until they are wired in and driven. ``kind`` chooses the
        new cells' type (defaults to the population's first type).
        """
        if k <= 0:
            return
        tid = type_id(kind) if kind is not None else int(self.type_id[0])
        new_ids = np.full(k, tid, dtype=np.int16)
        self.type_id = np.concatenate([self.type_id, new_ids])
        self.types = _TypeView(self.type_id)

        na, nb = _TABLE_A[new_ids].copy(), _TABLE_B[new_ids].copy()
        nc, nd = _TABLE_C[new_ids].copy(), _TABLE_D[new_ids].copy()
        self.a = np.concatenate([self.a, na])
        self.b = np.concatenate([self.b, nb])
        self.c = np.concatenate([self.c, nc])
        self.d = np.concatenate([self.d, nd])
        self.inhibitory = np.concatenate([self.inhibitory, _TABLE_INH[new_ids]])

        self.v = np.concatenate([self.v, nc.copy()])          # at rest (= c)
        self.u = np.concatenate([self.u, nb * nc])
        self.spiked = np.concatenate([self.spiked, np.zeros(k, dtype=bool)])
        self.activation = np.concatenate(
            [self.activation, np.zeros(k, dtype=np.float32)])
        self.n += k
        self._scratch = np.empty(self.n, dtype=np.float32)
        if hasattr(self, "_manual_I"):
            del self._manual_I

    # -- ergonomics -------------------------------------------------------
    def neuron(self, idx: int) -> "Neuron":
        """Return an object *handle* for neuron ``idx``."""
        return Neuron(self, idx)

    def __iter__(self) -> Iterable["Neuron"]:
        return (Neuron(self, i) for i in range(self.n))

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        kinds = sorted(_TYPE_NAMES[i] for i in np.unique(self.type_id))
        return f"<Population n={self.n} types={kinds}>"


# ---------------------------------------------------------------------------
# Neuron  --  the object handle (precise, ergonomic view onto one cell)
# ---------------------------------------------------------------------------
class Neuron:
    """A lightweight, object-oriented *view* onto one neuron of a population.

    It stores no state of its own (just a reference and an index), so creating
    millions of handles is cheap and they always reflect the live simulation.
    """

    __slots__ = ("pop", "idx")

    def __init__(self, pop: Population, idx: int) -> None:
        self.pop = pop
        self.idx = int(idx)

    @property
    def v(self) -> float:
        """Membrane potential (mV)."""
        return float(self.pop.v[self.idx])

    @property
    def u(self) -> float:
        """Recovery variable."""
        return float(self.pop.u[self.idx])

    @property
    def spiked(self) -> bool:
        """Did this neuron fire on the most recent step?"""
        return bool(self.pop.spiked[self.idx])

    @property
    def activation(self) -> float:
        """Decaying 0..1 'how recently did I fire' value (for visualisation)."""
        return float(self.pop.activation[self.idx])

    @property
    def type(self) -> NeuronType:
        return self.pop.types[self.idx]

    @property
    def inhibitory(self) -> bool:
        return bool(self.pop.inhibitory[self.idx])

    def inject(self, current: float) -> None:
        """Add a constant current to this neuron on the next step (helper)."""
        # Stored on the population's scratch buffer if present.
        buf = getattr(self.pop, "_manual_I", None)
        if buf is None:
            buf = np.zeros(self.pop.n)
            self.pop._manual_I = buf
        buf[self.idx] += current

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"<Neuron #{self.idx} {self.type.name} "
                f"v={self.v:.1f}mV{' spike!' if self.spiked else ''}>")
