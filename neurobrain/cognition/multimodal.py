"""
multimodal.py
=============

Where seeing and hearing meet. On top of the visual stream (which ends in IT
object cells) and the auditory stream (which ends in belt sound cells) sits a
**multisensory association area** -- the brain's convergence zone (superior
temporal sulcus / association cortex). It learns, by Hebbian co-occurrence, a
shared pool of **concept cells**, each of which fires for one audio-visual thing
(the *look* of a circle bound to the *sound* it makes).

Because the two senses meet on the same concept cells, the area does what a
convergence zone does: **cross-modal completion**. Show it only the picture and
it recalls the expected sound; play it only the sound and it recalls the
expected shape -- the neural basis of "hear a bell, picture a bell".

Nothing here is a lookup table: the binding lives in the synapses from each
modality onto the shared concept cells (``Wv`` and ``Wa``), learned the same
competitive-Hebbian way the sensory areas learned their features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x)
    return (x / n).astype(np.float32) if n > 1e-6 else x.astype(np.float32)


class AssociationArea:
    """A shared pool of concept cells wired to both senses.

    ``Wv`` are the synapses from the visual object code onto the concept cells,
    ``Wa`` from the sound code. ``bind`` presents both senses together and lets a
    concept cell win and tune both its visual and auditory synapses toward what
    it saw and heard -- so afterwards either sense alone can wake the concept and
    read out the other sense's expected pattern.
    """

    def __init__(self, n_vis: int, n_aud: int, n_concept: int,
                 lr: float = 0.15, vigilance: float = 0.80,
                 conscience: float = 1.0,
                 match_rule: str = "reliability",
                 novelty_rate: Optional[float] = 0.50,
                 vigilance_lr: float = 0.02, n_modes: int = 4,
                 mode_lr: float = 0.05, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.Wv = _rows_unit(rng.standard_normal((n_concept, n_vis)) * 0.1)
        self.Wa = _rows_unit(rng.standard_normal((n_concept, n_aud)) * 0.1)
        self.n_concept = n_concept
        self.lr = lr
        self.vigilance = float(vigilance)
        self.conscience = float(conscience)
        self.match_rule = str(match_rule)
        self.novelty_rate = (None if novelty_rate is None
                             else float(novelty_rate))
        self.vigilance_lr = float(vigilance_lr)
        self._rel_v = self._rel_a = 0.0   # running per-sense precision
        # the directions each concept VARIES in, so it can imagine a
        # member rather than only recall the average -- see _grow_subspace
        self.n_modes = int(n_modes)
        self.mode_lr = float(mode_lr)
        self.Pv = _rows_unit(rng.standard_normal(
            (n_concept * max(n_modes, 1), n_vis)).astype(np.float32)
            ).reshape(n_concept, max(n_modes, 1), n_vis)
        self.mode_var = np.zeros((n_concept, max(n_modes, 1)), np.float32)
        self.wins = np.zeros(n_concept)
        self._rng = rng
        self.v_mu = self.v_sd = self.a_mu = self.a_sd = None

    def set_stats(self, V: np.ndarray, A: np.ndarray) -> None:
        """Learn each modality's feature statistics so the concept cells compare
        codes on an equal footing (divisive normalisation -- what cortex does)."""
        self.v_mu, self.v_sd = V.mean(0), V.std(0) + 1e-6
        self.a_mu, self.a_sd = A.mean(0), A.std(0) + 1e-6

    def prep_v(self, v: np.ndarray) -> np.ndarray:
        return _unit(v if self.v_mu is None else (v - self.v_mu) / self.v_sd)

    def prep_a(self, a: np.ndarray) -> np.ndarray:
        return _unit(a if self.a_mu is None else (a - self.a_mu) / self.a_sd)

    def bind(self, v: np.ndarray, a: np.ndarray) -> int:
        """Co-present a visual code ``v`` and a sound code ``a``; a concept cell
        wins and learns both. Returns the winning concept cell.

        Two guards, and neither is optional -- without them this layer collapses.

        **Vigilance.** The old rule was pure argmax over the summed drive, and
        it had a runaway in it. The first cell to win tunes toward the data, so
        its drive rises; every subsequent pair then finds it the best match and
        tunes it further, while 31 other cells stay at their random
        initialisation and never win anything. Measured: binding 48 audio-visual
        pairs from 8 well-separated classes woke **1 concept cell out of 32**,
        and cross-modal recall sat at exactly chance (0.125) even though the
        sound codes themselves were 96% separable by nearest prototype. The
        association area was not failing to learn the pairing; it had one
        category and could not express a second.

        So a match below ``vigilance`` does not update the incumbent -- it
        recruits an uncommitted cell instead. This is the ART rule the rest of
        the project already runs on (:class:`GrowingCategoryMap`,
        :meth:`AssociativeCortex.consolidate`), applied here for the same
        reason: a new thing should become a new category rather than blurring
        an old one.

        **Conscience.** A frequency bias in the competition itself (DeSieno
        1988), scaled to the drive rather than the 0.1 the old rule used. A
        penalty of 0.1 cannot move an argmax whose winner leads by ~1.0, which
        is why the old homeostatic term was present and did nothing.

        The conscience default is measured (``benchmarks/concept_cells.py``) on
        cross-modal recall from sounds that were **never bound** -- 5 per class
        bound, 3 held out, 5 seeds. Querying with a code that *was* bound is a
        lookup and scores well even on shuffled labels; a held-out query does
        not. The layer as it was scored **exactly** chance (0.125) with one
        cell; with these guards it reaches 0.742 against a shuffled control at
        0.058, d=10.8, 5 of 5 seeds.

        ``vigilance`` is a *starting point*, not a setting -- see
        :meth:`_homeostasis`, which drives it to whatever value achieves
        ``novelty_rate``. Fixing it does not survive a change of data, and this
        layer failed at both ends of that (one cell per experience on real
        photographs; two cells total on the synthetic bank).

        Rule and rate together, on both worlds
        ---------------------------------------
        The two data sets disagree about which combination rule is right, which
        is why the reliability weighting exists. At ``novelty_rate=0.5``:

            rule           synthetic  |  real: cells/pair  s->label  s->vision
            mean (as was)     0.742   |       1.00           0.947     0.581
            mean               0.792   |       0.57           0.943     0.660
            max                0.458   |       0.48           0.942     0.681
            reliability        0.742   |       0.53           0.947     0.671

        ``mean`` is best on the synthetic bank and unusable on real data without
        homeostasis; ``max`` is the reverse. **reliability** is within a few
        points of the better of the two in each world without being told which
        world it is in -- and against the layer as shipped it holds the
        synthetic number *exactly* (0.742), holds ``s->label`` exactly (0.947),
        raises cross-modal retrieval 0.581 -> 0.671, and takes the layer from
        one cell per experience to roughly one per two. That last number is the
        point: below 1.0 there are concepts, at 1.0 there is only a list.
        """
        vn, an = self.prep_v(v), self.prep_a(a)
        match = self.match(vn, an)
        tot = max(float(self.wins.sum()), 1.0)
        bias = self.conscience * (self.wins / tot - 1.0 / self.n_concept)
        win = int(np.argmax(match - bias))
        recruited = 0
        if match[win] < self.vigilance:
            free = np.flatnonzero(self.wins == 0)
            if len(free):
                # an uncommitted cell takes the pair whole -- not a fraction of
                # the way toward it, since it has nothing worth preserving
                win = int(free[0])
                self.Wv[win], self.Wa[win] = vn.copy(), an.copy()
                self.wins[win] += 1
                recruited = 1
        if not recruited:
            self.Wv[win] = _unit(self.Wv[win] + self.lr * (vn - self.Wv[win]))
            self.Wa[win] = _unit(self.Wa[win] + self.lr * (an - self.Wa[win]))
            self.wins[win] += 1
            self._grow_subspace(win, vn)
        self._homeostasis(recruited)
        return win

    def unbind(self, v: np.ndarray, a: np.ndarray,
               lr: Optional[float] = None) -> int:
        """Push the winning concept **away** from a pairing that should not be.

        Everything else in this class treats what it is handed as an
        observation, and that is the whole reason imagination pays nothing back:
        measured, a night of recombinations bound as fact is indistinguishable
        from a night of random sights (`inner_world.py`, 72.3% against 72.5%),
        and it stays that way with a front end that clusters and a judge that
        discriminates at 0.818 (`heard_world.py`). The route in is a
        consolidator; it sharpens the average of what it is given, so an
        imagining is either absorbed as a false observation or is noise.

        An implausible crossing is neither. It is a **negative example** -- a
        combination the world does not present, generated for free and in
        unlimited supply by :meth:`imagine_factored`, and identifiable by
        :class:`FactorCompatibility` without any label. What is missing is only
        the ability to learn from one, and that is this method: the same
        competition, the same local rule, the sign reversed.

        This is the standard account of what a night is for. Crick & Mitchison
        (1983) proposed dreaming as **reverse learning** -- REM replay that
        weakens spurious attractors rather than strengthening real ones -- and
        the same two-phase structure is what makes contrastive Hebbian learning
        and wake-sleep work: a positive phase on data, a negative phase on what
        the model generates. Nothing here is a gradient; it is
        :meth:`bind`'s update with a minus sign, driven by a sample the mind
        made up.

        **A geometric no-op to know about, and it bites in practice.** The
        anti-instar step is ``w - lr*(x - w)``, which is exactly zero when
        ``w == x`` -- there is no direction in which to push a unit vector away
        from itself, and rescaling does not change a direction. So a cell that
        sits *on* the code being unlearned does not move. That is not rare here:
        vigilance recruits by copying, and a crossing is normally built from one
        real code and one foreign one, so the half taken from the cell's own
        memory is frozen while only the foreign half moves. Measured on a
        crossing, the auditory half went -0.284 -> -0.397 while the visual half
        stayed at 1.0000. `heard_world.py`'s `reverse` arm is therefore
        unlearning **half of each crossing**, which is the likeliest reason it
        damages (-0.0509) rather than helps, and it is a property of this rule
        rather than of reverse learning as such.
        """
        vn, an = self.prep_v(v), self.prep_a(a)
        win = int(np.argmax(self.match(vn, an)))
        lr = self.lr if lr is None else float(lr)
        self.Wv[win] = _unit(self.Wv[win] - lr * (vn - self.Wv[win]))
        self.Wa[win] = _unit(self.Wa[win] - lr * (an - self.Wa[win]))
        return win

    def bind_contrastive(self, v: np.ndarray, a: np.ndarray,
                         temperature: float = 1.0,
                         rng: Optional[np.random.Generator] = None
                         ) -> Tuple[int, int, float]:
        """Learn from the gap between what was predicted and what arrived.

        :meth:`bind` compares an input to a weight. It never compares a
        *prediction* to an *outcome*, and that is the mechanism seven arms
        across two modalities turned out to be missing: with a front end that
        clusters (0.788) and an unsupervised judge that discriminates (0.818),
        no night of any content and no sign of plasticity beat doing nothing,
        because consolidation can only sharpen the average of what it is handed.

        This is the two-phase rule that does compare them, and it is the reason
        to have an imagination at all:

        * **positive phase** -- the real pair arrives, a cell wins, and it moves
          toward the data. This is :meth:`bind`.
        * **negative phase** -- the layer is given *one* modality and
          **completes the other from its own model** (:meth:`imagine_from_sound`,
          sampled rather than averaged). A cell wins on that fantasy, and it
          moves *away* from it.

        When the completion matches what really arrived, the two phases land on
        the same cell and cancel: a well-predicted pair teaches nothing, which is
        the defining property of an error-driven rule and the one instar does not
        have. When they differ, the layer is pushed toward the world and away
        from its own fantasy, by exactly the amount it was wrong.

        That is Contrastive Hebbian Learning / the wake-sleep decomposition
        (Hinton et al. 1995), which is local, needs no gradients, and is the
        standard non-backprop account of error-driven cortical plasticity. The
        negative sample has to come from **the model's own generative
        distribution** -- this is why an earlier attempt with arbitrary
        recombinations (:meth:`unbind` on crossings) was the wrong negative
        phase and behaved like noise: a random crossing is not what the model
        believes, so unlearning it teaches nothing about the model.

        Returns ``(positive cell, negative cell, prediction error)``, the last
        being ``1 - cos`` between the completion and what arrived, so a caller
        can see how much was actually learned.
        """
        vn, an = self.prep_v(v), self.prep_a(a)
        win_p = self.bind(v, a)                       # positive phase

        v_hat = self.imagine_from_sound(a, temperature=temperature, rng=rng)
        err = float(1.0 - float(v_hat @ vn))
        win_n = int(np.argmax(self.match(v_hat, an)))
        if win_n != win_p:
            # only where the model disagrees with the world is there anything
            # to unlearn; agreement cancels, which is the point of the rule
            self.Wv[win_n] = _unit(self.Wv[win_n]
                                   - self.lr * (v_hat - self.Wv[win_n]))
            self.Wa[win_n] = _unit(self.Wa[win_n]
                                   - self.lr * (an - self.Wa[win_n]))
        return win_p, win_n, err

    def _homeostasis(self, recruited: int) -> None:
        """Drift ``vigilance`` toward a target rate of category creation.

        A fixed threshold on a similarity does not survive a change of data, and
        this layer broke at *both* ends of that. On real CIFAR/ESC-50 pairs the
        averaged match peaked at 0.740, so vigilance 0.80 was unreachable, every
        pair recruited, and the layer ended with one cell per experience. On the
        synthetic bank -- where both senses are strong -- the ``max`` rule put
        almost every match above 0.65, nothing ever recruited, and it collapsed
        to two cells. Same code, same parameter, opposite failures, because the
        similarity *scale* is a property of the data and an absolute threshold
        pretends it is not.

        So the parameter that is held fixed is not a similarity at all: it is
        **how much of experience becomes something new**. ``novelty_rate`` is
        that fraction, and vigilance is driven to achieve it by a plain integral
        controller -- threshold up when too little is being recruited, down when
        too much.

        This is the same homeostatic move the rest of the project already makes:
        :class:`PredictiveA1` holds a target sparsity by biasing its own drive,
        and the belt subtracts a running per-band floor. A cell that adjusts its
        threshold to keep its own activity near a set-point is one of the
        better-established pieces of cortical housekeeping (intrinsic
        plasticity), and it is what makes a single default work on data whose
        similarity distributions differ by half the scale.

        ``novelty_rate=None`` freezes vigilance at whatever it was set to.
        """
        if self.novelty_rate is None:
            return
        # Recruitment fires when the best match falls BELOW vigilance, so a
        # shortfall of new categories calls for a *higher* bar, not a lower one.
        # Written the other way round first, and it did not read as a bug: the
        # synthetic bank improved (16 cells -> 32) because the error term
        # happened to be positive there and drove vigilance up until the pool
        # ran out. That is a runaway wearing the costume of a result -- the
        # giveaway was the other rule, where the same sign drove vigilance down
        # and left the layer frozen at two cells.
        self.vigilance += self.vigilance_lr * (self.novelty_rate - recruited)
        self.vigilance = float(np.clip(self.vigilance, -1.0, 1.0))

    def _reliability(self, mv: np.ndarray, ma: np.ndarray) -> Tuple[float, float]:
        """How much each sense is worth listening to right now.

        A modality that returns nearly the same drive for every concept cell has
        told the layer nothing -- it cannot say *which* concept this is, only
        that something arrived. So reliability is measured as how **peaked** the
        drive is: the gap between the best-matching cell and the average one.

        This is reliability-weighted cue combination (Ernst & Banks 2002), which
        is the standard account of how the senses are actually combined --
        weights proportional to precision, so a blurred visual cue loses
        influence to a sharp haptic one without anything being switched off.
        Here it is what lets one rule cover both worlds: on the synthetic bank
        both senses are peaked and the weights come out near even, which is the
        ``mean`` behaviour that works there; on real photographs and field
        recordings the visual drive is nearly flat, its weight collapses, and
        the match is carried by hearing -- the ``max`` behaviour that works
        there. Neither is chosen by hand.

        Kept as a running average so it is a property of the mind's experience
        rather than of the current frame.
        """
        pv = float(mv.max() - mv.mean())
        pa = float(ma.max() - ma.mean())
        self._rel_v += 0.05 * (pv - self._rel_v)
        self._rel_a += 0.05 * (pa - self._rel_a)
        tot = self._rel_v + self._rel_a
        if tot < 1e-6:
            return 0.5, 0.5
        return self._rel_v / tot, self._rel_a / tot

    def match(self, vn: np.ndarray, an: np.ndarray) -> np.ndarray:
        """How well each concept cell explains this pair, in [-1, 1].

        ``match_rule="reliability"`` -- each sense weighted by how sharply it
        discriminates, learned online. The default; see :meth:`_reliability`.
        ``match_rule="max"`` -- a cell is as awake as its **best-driving** sense.
        ``match_rule="mean"`` -- it must be driven by both at once.

        The default is ``max`` and the reason is a hard failure of ``mean``.
        Averaging puts the scale of ``match`` at the mercy of the *weaker*
        modality: on real CIFAR/ESC-50 pairs, where the visual code's
        between-exemplar similarity tops out near 0 while audio reaches 0.95,
        the averaged best match over a whole day reached a median of 0.364 and a
        maximum of **0.740** -- so a vigilance of 0.80 was simply unreachable,
        every pair recruited an uncommitted cell, and the layer ended with
        exactly one cell per experience. Not a tuning problem: no vigilance in
        (0.74, 1] can ever be satisfied, and any lower value has to be
        re-chosen the moment either sense changes quality.

        Under ``max`` the same day spans 0.648 median to 0.954 maximum, which is
        a range a threshold can actually sit inside.

        It is also the better model. Multisensory neurons obey **inverse
        effectiveness** (Stein & Meredith): a strong unimodal input drives the
        cell on its own, and the superadditive gain from combining appears when
        each input alone is weak. A concept cell for *dog* should wake to a dog
        barking in the dark. Both synapse sets still learn on every bind, so the
        cell remains bimodal -- only the recognition test is permissive.
        """
        mv, ma = self.Wv @ vn, self.Wa @ an
        if self.match_rule == "max":
            return np.maximum(mv, ma)
        if self.match_rule == "mean":
            return 0.5 * (mv + ma)
        wv, wa = self._reliability(mv, ma)
        return wv * mv + wa * ma

    def concept_from_vision(self, v: np.ndarray) -> int:
        return int(np.argmax(self.Wv @ self.prep_v(v)))

    def concept_from_sound(self, a: np.ndarray) -> int:
        return int(np.argmax(self.Wa @ self.prep_a(a)))

    # -- the read-out the binding is *for* -----------------------------------
    def vision_from_sound(self, a: np.ndarray) -> np.ndarray:
        """Hear a sound; read out the visual pattern its concept expects.

        This is the half of binding that makes it cross-modal rather than a
        lookup: the sound wakes a concept cell through ``Wa``, and that cell's
        ``Wv`` row *is* the vision it has learned to expect. Nothing about the
        query is visual, and the answer is entirely visual."""
        return self.Wv[self.concept_from_sound(a)].copy()

    def sound_from_vision(self, v: np.ndarray) -> np.ndarray:
        """See a thing; read out the sound its concept expects."""
        return self.Wa[self.concept_from_vision(v)].copy()

    def novelty(self, v: np.ndarray, a: np.ndarray) -> float:
        """How poorly the current concept cells explain this audio-visual pair
        (1 = brand new, 0 = already a known concept)."""
        vn, an = self.prep_v(v), self.prep_a(a)
        best = 0.5 * (self.Wv @ vn).max() + 0.5 * (self.Wa @ an).max()
        return float(1.0 - max(best, 0.0))

    def grow_concept(self, v: np.ndarray, a: np.ndarray,
                     novelty_thresh: float = 0.4) -> int:
        """Online structural growth: if a pairing is novel enough, sprout a new
        concept cell tuned to it. Returns the new cell index, or -1 if the pair
        is already covered by an existing concept."""
        if self.novelty(v, a) < novelty_thresh:
            return -1
        self.Wv = np.vstack([self.Wv, self.prep_v(v)]).astype(np.float32)
        self.Wa = np.vstack([self.Wa, self.prep_a(a)]).astype(np.float32)
        self.wins = np.concatenate([self.wins, [1.0]])
        self.n_concept += 1
        return self.n_concept - 1

    # -- imagining, as opposed to recalling ---------------------------------
    def _grow_subspace(self, cell: int, vn: np.ndarray) -> None:
        """Track the directions a concept *varies* in, by Oja's rule.

        `Wv[cell]` is the mean of everything bound to a cell, so reading it back
        gives a category average and nothing else. Measured: what that produces
        is 3.2x closer to memory than a real unseen photograph is, sits closer
        to the class mean than any real member of the class, and has a total
        vocabulary of one output per cell. A mean is not an imagination.

        What is missing is the *spread*. A cell that also knows the few
        directions its members differ along can place a new point inside its own
        category instead of at its centre -- a cat it has not seen, rather than
        the average cat. That is a generative model, and the local rule for
        obtaining it is Oja's (1982): a Hebbian update with a decay that
        converges on principal components without anyone computing a covariance
        matrix or differentiating anything. Sanger's deflation makes the
        components distinct.

        Kept to ``n_modes`` directions per cell because this is a concept cell,
        not a density estimator: a handful of axes of variation is what a
        category has, and each one costs a vector.
        """
        if self.n_modes <= 0:
            return
        P = self.Pv[cell]
        r = vn - self.Wv[cell] * float(self.Wv[cell] @ vn)   # what the mean misses
        for k in range(self.n_modes):
            nr = float(np.linalg.norm(r))
            if nr < 1e-6:
                break
            if self.mode_var[cell, k] <= 0.0:
                # Cold start. Oja's rule is multiplicative in the current
                # correlation, and in 12288 dimensions a randomly initialised
                # direction has essentially none: measured, the modes never left
                # their initialisation, variance stayed at 0.003, and sampling
                # at temperature 16 moved fidelity by 0.002. So an unused mode
                # is *seeded* from the first residual it meets rather than
                # waiting for a correlation that never arrives -- which is also
                # the more biological start, a synapse shaped by early input.
                P[k] = (r / nr).astype(np.float32)
                self.mode_var[cell, k] = nr * nr
                break
            a = float(P[k] @ r)
            P[k] += self.mode_lr * a * (r - a * P[k])        # Oja
            n = float(np.linalg.norm(P[k]))
            if n > 1e-6:
                P[k] /= n
            self.mode_var[cell, k] += 0.05 * (a * a - self.mode_var[cell, k])
            r = r - a * P[k]                                 # Sanger deflation

    def imagine_vision(self, cell: int, temperature: float = 1.0,
                       rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """A sight this concept could have had, rather than the one it averages.

        ``mean + sum_k z_k * sigma_k * direction_k`` with ``z`` standard normal:
        a sample from the cell's own learned subspace of variation. At
        ``temperature=0`` this is exactly :meth:`vision_from_sound`'s answer, so
        the old behaviour is the zero-temperature limit of the new one and the
        two can be compared on the same axis.

        **Two cells this cannot help**, both measured in
        `benchmarks/composition.py` and locked in `tests/test_imagination_space.py`:

        * A cell that **won exactly once** holds its training pair verbatim
          (:meth:`bind` recruits by copying) and never reached
          :meth:`_grow_subspace`, so ``mode_var`` is exactly zero and this
          returns the stored photograph at *every* temperature. 59 of 112 cells
          on real data; 36.8% of held-out sounds wake one. This is not a defect
          to patch -- one observation carries no variation -- but it means the
          temperature sweep is a constant for a third of the population.
        * A cell that **won exactly twice** learns one mode, and that mode is
          necessarily the line joining its two members -- so sampling along it
          interpolates between two memorised photographs and at high temperature
          lands on one (verbatim 0.000 -> 0.044 as T goes 0 -> 4).

        :meth:`imagine_composite` is what reaches past both: a singleton has no
        spread, but the category it belongs to does.
        """
        v = self.Wv[cell].copy()
        if self.n_modes > 0 and temperature > 0:
            rng = rng or self._rng
            z = rng.standard_normal(self.n_modes).astype(np.float32)
            s = np.sqrt(np.maximum(self.mode_var[cell], 0.0))
            v = v + temperature * (z * s) @ self.Pv[cell]
        return _unit(v)

    def imagine_from_sound(self, a: np.ndarray, temperature: float = 1.0,
                           rng: Optional[np.random.Generator] = None
                           ) -> np.ndarray:
        """Hear a sound; imagine *a* sight it could go with, not *the* sight."""
        return self.imagine_vision(self.concept_from_sound(a), temperature, rng)

    def imagine_composite(self, cells: Sequence[int],
                          weights: Optional[Sequence[float]] = None,
                          temperature: float = 1.0,
                          anchor: Optional[np.ndarray] = None,
                          anchor_weight: float = 0.5,
                          rng: Optional[np.random.Generator] = None
                          ) -> np.ndarray:
        """Imagine from **several** concepts at once, optionally around a real one.

        :meth:`imagine_vision` perturbs a single cell inside its own subspace,
        which bounds what it can produce to a disc around one average. Two things
        widen that, and both are ordinary operations on the same weights:

        * **composition** -- mix several concepts' means *and* draw variation
          from each of their subspaces, so the sample lives in the span of all of
          them rather than one.
        * **an anchor** -- blend toward a real code that is actually present now.
          This is the "combine what is real with what is imagined" case: the
          imagining is pulled toward something the senses are delivering rather
          than floating free, which is the condition under which imagination was
          measured to help at all (a fully self-generated night is the worst arm
          in `cross_modal_dream.py`, -0.0301).

        ``anchor_weight`` is how much of the result is the real code. At 1.0 it
        is perception with no imagining; at 0.0 it is imagination with nothing
        real in it, and both extremes are worth having as the ends of one axis.

        Measured (`benchmarks/composition.py`), and both halves are worth
        knowing before relying on this:

        * it **ends verbatim recall**. Composing three concepts takes the
          fraction of outputs that are bit-identical to a stored photograph from
          0.397 to 0.001, which single-concept sampling cannot do at any
          temperature -- see :meth:`imagine_vision`.
        * the **novelty it creates is arithmetic, not conceptual**. Averaging
          *k* near-orthogonal unit codes sits 1/sqrt(k) from each of them for
          free; at k=3 that is 0.42, and the measured novelty is 0.389 against
          0.362 for three *stored photographs* averaged with no concept layer at
          all (paired +0.0276, d=2.76, 5/5 -- consistent and negligible).

        So this widens the output distribution and stops the layer parroting
        memories, and it does not make the mind inventive. Nothing assembled
        here exceeds novelty 0.389 where real held-out data sits at 0.797,
        because every operation in this method is a weighted sum of stored
        vectors.
        """
        cells = [int(c) for c in cells]
        if not cells:
            return np.zeros(self.Wv.shape[1], np.float32)
        w = (np.ones(len(cells), np.float32) if weights is None
             else np.asarray(weights, np.float32))
        w = w / max(float(w.sum()), 1e-9)
        v = (w[:, None] * self.Wv[cells]).sum(0)
        if self.n_modes > 0 and temperature > 0:
            rng = rng or self._rng
            for wi, c in zip(w, cells):
                z = rng.standard_normal(self.n_modes).astype(np.float32)
                s = np.sqrt(np.maximum(self.mode_var[c], 0.0))
                v = v + temperature * float(wi) * ((z * s) @ self.Pv[c])
        if anchor is not None and anchor_weight > 0:
            a = self.prep_v(anchor)
            v = (1.0 - anchor_weight) * _unit(v) + anchor_weight * a
        return _unit(v)

    def imagine_factored(self, cells: Sequence[int],
                         bounds: Sequence[Tuple[int, int]],
                         temperature: float = 0.0,
                         rng: Optional[np.random.Generator] = None
                         ) -> np.ndarray:
        """Take each **factor** of the code from a different concept.

        :meth:`imagine_composite` mixes whole codes, and that is why it cannot
        invent: every output is a weighted sum of stored vectors, so it lies in
        their span by construction and no temperature leaves it. Measured, the
        novelty it reaches is the arithmetic of averaging *k* near-orthogonal
        codes and nothing more.

        Recombining *parts* is a different operation, and the difference is
        algebraic rather than a matter of degree. With ``Wv[A] = [f_A, c_A]``
        and ``Wv[B] = [f_B, c_B]``, the recombination ``[f_A, c_B]`` satisfies

            alpha * [f_A, c_A] + beta * [f_B, c_B] = [f_A, c_B]

        only if ``alpha = 1, beta = 0`` from the first block and
        ``alpha = 0, beta = 1`` from the second -- a contradiction. **It is not
        in the span of the concepts it was built from**, and generically not in
        the span of the whole stored bank either. That is the one thing every
        mixing and sampling operation in this class provably cannot do.

        The factors have to be real ones for this to mean anything, and here
        they are: the eye emits ``[luminance | red-green | blue-yellow]``
        concatenated (see `benchmarks/real_binding.py:opponent`), so form and
        colour already occupy disjoint blocks of the code. Taking the form of a
        bus with the colour of something that was never bus-shaped is the
        "yellow bus" case -- a combination the world never presented.

        ``bounds[k]`` is the ``(start, stop)`` block that ``cells[k]`` supplies.
        Blocks should tile the code; anything left uncovered stays zero.
        """
        v = np.zeros(self.Wv.shape[1], np.float32)
        for c, (lo, hi) in zip(cells, bounds):
            c = int(c)
            seg = self.Wv[c, lo:hi].copy()
            if temperature > 0 and self.n_modes > 0:
                rng = rng or self._rng
                z = rng.standard_normal(self.n_modes).astype(np.float32)
                s = np.sqrt(np.maximum(self.mode_var[c], 0.0))
                seg = seg + temperature * ((z * s) @ self.Pv[c])[lo:hi]
            v[lo:hi] = seg
        return _unit(v)

    def _retire(self, c: int) -> None:
        """Return a merged-away cell to the uncommitted pool.

        Clearing ``wins`` is not enough and the difference is a silent one.
        :meth:`concept_from_vision` and :meth:`concept_from_sound` take an
        ``argmax`` over **every** row, so a retired cell still holding its old
        tuning keeps winning the read-out. Measured before this was fixed: after
        a merge pass that left 0% singletons, the cells the read-out actually
        returned had ``wins == 0`` and sat at cosine 1.0000 to a stored
        photograph -- the merger had worked and nothing downstream could see it.

        Re-randomising rather than zeroing, for the reason :meth:`consolidate`
        gives: a zero row has cosine 0 to everything, which makes it a uniform
        attractor the moment vigilance goes looking for a free cell.
        """
        c = int(c)
        self.wins[c] = 0.0
        self.mode_var[c] = 0.0
        self.Wv[c] = _unit(self._rng.standard_normal(self.Wv.shape[1]))
        self.Wa[c] = _unit(self._rng.standard_normal(self.Wa.shape[1]))

    def absorb_singletons(self, both_senses: bool = True) -> Dict[int, int]:
        """Force every cell that has seen exactly one thing into its best match.

        :meth:`consolidate_ranked` merges the most similar pairs wherever they
        are, and on real data the singleton fraction bottoms out near 31% under
        it however hard the population is compressed -- once the good pairs are
        used up the survivors are genuinely dissimilar, and merging them would
        be arbitrary. But a singleton is exactly the cell that cannot imagine:
        it holds one photograph verbatim, its ``mode_var`` is zero, and
        :meth:`imagine_vision` returns that photograph at every temperature.

        So this targets them by construction rather than by similarity: each
        one-win cell is absorbed into whichever committed cell explains it best,
        and afterwards **no cell holds a single experience**. That is a stronger
        intervention than ranked merging and a blunter one -- it will put some
        photographs in the wrong concept, which is why the benchmark reports
        purity and recall beside the singleton count rather than the count
        alone.
        """
        live = [int(c) for c in np.flatnonzero(self.wins > 0)]
        singles = [c for c in live if self.wins[c] == 1]
        hosts = [c for c in live if self.wins[c] > 1]
        if not singles or not hosts:
            return {}
        merged: Dict[int, int] = {}
        for c in singles:
            idx = np.array([h for h in hosts if h not in merged] or hosts)
            s = self.Wv[idx] @ self.Wv[c]
            if both_senses:
                s = 0.5 * (s + self.Wa[idx] @ self.Wa[c])
            h = int(idx[int(np.argmax(s))])
            while h in merged:
                h = merged[h]
            if h == c:
                continue
            wc, wh = self.wins[c], self.wins[h]
            self.Wv[h] = _unit((wh * self.Wv[h] + wc * self.Wv[c]) / (wh + wc))
            self.Wa[h] = _unit((wh * self.Wa[h] + wc * self.Wa[c]) / (wh + wc))
            self.wins[h] = wh + wc
            self._retire(c)
            merged[c] = h
        return merged

    def consolidate_ranked(self, keep: float = 0.7, min_cells: int = 8,
                           both_senses: bool = True) -> Dict[int, int]:
        """Merge the most similar cells by **rank**, until a target count.

        :meth:`consolidate` takes an absolute similarity threshold, and on real
        data that threshold cannot be set. Measured over 111 concept cells from
        216 CIFAR/ESC-50 pairs: the visual similarity between cells has a median
        of −0.009 and a 99th percentile of **0.206**, so a threshold of 0.85
        merges **zero** pairs and 0.75 merges zero, while dropping it far enough
        to merge anything (0.1) admits 189 pairs at once. There is no value that
        merges the right amount, because the scale is a property of the code --
        distinct photographs are nearly orthogonal in it -- and not of how alike
        two concepts are.

        The ordering, however, is fine: the **top 100 pairs by summed similarity
        are 82% same-category**. So this holds a *rate* rather than a
        similarity, which is the same correction the vigilance controller
        needed (see :meth:`_homeostasis`) -- merge the best-matching pairs in
        order until the population has shrunk to ``keep`` of its size, whatever
        the absolute numbers happen to be.

        ``keep=0.7`` retires 30% of the cells per pass. ``both_senses`` keeps
        :meth:`consolidate`'s requirement that the pair agree in vision *and*
        hearing, which is what stops a dog and a cat collapsing on spectral
        similarity alone; it is applied to the ordering rather than as a gate.

        Returns ``{old_cell: surviving_cell}``, as :meth:`consolidate` does.
        """
        live = [int(c) for c in np.flatnonzero(self.wins > 0)]
        target = max(int(round(len(live) * float(keep))), int(min_cells))
        if len(live) <= target or len(live) < 2:
            return {}
        idx = np.array(live)
        Sv = self.Wv[idx] @ self.Wv[idx].T
        Sa = self.Wa[idx] @ self.Wa[idx].T
        score = 0.5 * (Sv + Sa) if both_senses else Sv
        iu = np.triu_indices(len(idx), 1)
        order = np.argsort(-score[iu])
        merged: Dict[int, int] = {}
        n = len(live)
        for k in order:
            if n <= target:
                break
            c, d = int(idx[iu[0][k]]), int(idx[iu[1][k]])
            # follow either side to whatever absorbed it, so a chain of merges
            # accumulates into one survivor instead of being skipped
            while c in merged:
                c = merged[c]
            while d in merged:
                d = merged[d]
            if c == d:
                continue
            wc, wd = self.wins[c], self.wins[d]
            self.Wv[c] = _unit((wc * self.Wv[c] + wd * self.Wv[d]) / (wc + wd))
            self.Wa[c] = _unit((wc * self.Wa[c] + wd * self.Wa[d]) / (wc + wd))
            self.wins[c] = wc + wd
            self._retire(d)
            merged[d] = c
            n -= 1
        return merged

    def consolidate(self, threshold: float = 0.75) -> Dict[int, int]:
        """Merge concept cells that turned out to be the same thing.

        Vigilance recruits eagerly, and on real data it recruits *very* eagerly:
        76 training pairs of CIFAR photographs and ESC-50 recordings produced 77
        concept cells -- one per example. That is a good exemplar memory and not
        a set of concepts, and the difference matters for a mind that is meant
        to imagine from its concepts rather than replay its examples.

        Eager encoding followed by offline merging is how the biology is usually
        described: the hippocampus takes a separate trace for nearly every
        episode (pattern separation, which is what high vigilance is), and sleep
        is where those traces are integrated into cortical structure that
        generalises. So this belongs in a night rather than in the waking rule,
        and lowering vigilance instead would lose the separation that makes
        one-shot binding work at all.

        Two cells merge when they agree in **both** senses -- a cat photo and a
        cat recording must both be close, since agreeing on the sound alone is
        how a dog and a cat recording of similar spectral shape would collapse
        into one. The survivor is the win-weighted mean, so a cell that has seen
        forty pairs is not dragged by one that has seen one.

        Returns ``{old_cell: surviving_cell}`` for every cell that was absorbed,
        so a caller holding a vote map or a name table can fold it accordingly.
        """
        order = np.argsort(-self.wins)                # the best-evidenced lead
        alive = [int(c) for c in order if self.wins[c] > 0]
        merged: Dict[int, int] = {}
        for i, c in enumerate(alive):
            if c in merged:
                continue
            for d in alive[i + 1:]:
                if d in merged:
                    continue
                if (float(self.Wv[c] @ self.Wv[d]) >= threshold
                        and float(self.Wa[c] @ self.Wa[d]) >= threshold):
                    wc, wd = self.wins[c], self.wins[d]
                    self.Wv[c] = _unit((wc * self.Wv[c] + wd * self.Wv[d]) / (wc + wd))
                    self.Wa[c] = _unit((wc * self.Wa[c] + wd * self.Wa[d]) / (wc + wd))
                    self.wins[c] = wc + wd
                    # d becomes uncommitted, and is re-randomised rather than
                    # left holding its old tuning or zeroed. A zeroed row has
                    # cosine 0 to everything, which makes it a uniform attractor
                    # the moment vigilance looks for a free cell; a random row
                    # is what an unused cell actually looks like.
                    self.wins[d] = 0.0
                    self.Wv[d] = _unit(self._rng.standard_normal(self.Wv.shape[1]))
                    self.Wa[d] = _unit(self._rng.standard_normal(self.Wa.shape[1]))
                    merged[d] = c
        return merged

    def expect_sound(self, v: np.ndarray) -> np.ndarray:
        """See -> imagine: the sound code the vision-evoked concept expects."""
        return self.Wa[self.concept_from_vision(v)]

    def expect_vision(self, a: np.ndarray) -> np.ndarray:
        """Hear -> imagine: the visual code the sound-evoked concept expects."""
        return self.Wv[self.concept_from_sound(a)]


def _rows_unit(W: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(W, axis=1, keepdims=True)
    n[n < 1e-6] = 1.0
    return (W / n).astype(np.float32)


class FactorCompatibility:
    """Which combinations of parts the world actually presents.

    :meth:`AssociationArea.imagine_factored` can build any crossing of any two
    concepts, and that is exactly its problem: a bus with a frog's colour and a
    bus with a bird's colour are equally available and equally unjudged. Measured
    (`benchmarks/inner_world.py`), replaying unjudged crossings into the layer is
    indistinguishable from replaying *random* sights -- 72.3% against 72.5% on
    the probe the recombination is evidence for. The gap there was never novelty;
    it was that every route from an imagining back into the mind runs through
    :meth:`AssociationArea.bind`, which treats what it is handed as an
    observation. Something has to **judge** a crossing before the concepts absorb
    it.

    This is that judgement at its smallest: a Hebbian association between the
    two factors, learned from the pairs the world actually presented. Blocks
    that co-occurred drive each other; blocks that never did, do not. Nothing
    here is trained -- it is one outer product accumulated online, the same
    associative rule the rest of the project runs on, and it is read as a
    compatibility rather than as a recall.

    The outer product is taken in a **randomly projected** space (``rank``
    dimensions per factor) for the ordinary reason that the full one would be
    4096 x 8192 floats for a code this size. Random projection before an
    associative matrix is not a compromise borrowed from engineering either:
    sparse random connectivity feeding an associative layer is the cerebellar
    and mushroom-body plan, and it preserves inner products in expectation
    (Johnson-Lindenstrauss), which is all a compatibility score needs.
    """

    def __init__(self, bounds: Sequence[Tuple[int, int]], rank: int = 64,
                 seed: int = 0):
        if len(bounds) != 2:
            raise ValueError("compatibility is between exactly two factors")
        self.bounds = [(int(a), int(b)) for a, b in bounds]
        self.rank = int(rank)
        rng = np.random.default_rng(seed)
        self.P = [_rows_unit(rng.standard_normal((self.rank, hi - lo))
                             ).astype(np.float32) for lo, hi in self.bounds]
        self.M = np.zeros((self.rank, self.rank), np.float32)
        self.n = 0

    def _parts(self, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return tuple(_unit(self.P[k] @ v[lo:hi])
                     for k, (lo, hi) in enumerate(self.bounds))

    def observe(self, v: np.ndarray) -> None:
        """One real thing, seen whole: its parts learn that they go together."""
        f, c = self._parts(np.asarray(v, np.float32))
        self.M += np.outer(f, c)
        self.n += 1

    def score(self, v: np.ndarray) -> float:
        """How much this combination of parts looks like one the world presents.

        Not a probability and not calibrated -- a relative score, meaningful
        only against other candidates, which is all a selection rule needs."""
        if self.n == 0:
            return 0.0
        f, c = self._parts(np.asarray(v, np.float32))
        return float(f @ (self.M / self.n) @ c)

    def select(self, candidates: Sequence[np.ndarray], keep: int = 1
               ) -> List[int]:
        """The indices of the ``keep`` most plausible candidates, best first."""
        s = np.array([self.score(v) for v in candidates])
        return [int(i) for i in np.argsort(-s)[:max(keep, 0)]]


@dataclass
class MultisensoryBrain:
    """Vision + hearing + the association area that binds them."""

    vision: object            # VentralStream
    audio: object             # AuditoryStream
    assoc: AssociationArea
    concept_label: Optional[np.ndarray] = None   # concept cell -> pair index
    pair_names: Optional[List[str]] = None       # "circle+pure", ...

    def vision_code(self, image: np.ndarray) -> np.ndarray:
        return self.vision.it_code(image)

    def sound_code(self, sig: np.ndarray) -> np.ndarray:
        # the full auditory descriptor is the sound's identity (spectral +
        # modulation + belt) -- discriminative enough to name the sound
        return self.audio.descriptor(sig)

    def learn_pair(self, image: np.ndarray, sig: np.ndarray) -> int:
        return self.assoc.bind(self.vision_code(image), self.sound_code(sig))

    def see_then_hear(self, image: np.ndarray) -> str:
        """Given only a picture, name the sound the brain expects (cross-modal)."""
        c = self.assoc.concept_from_vision(self.vision_code(image))
        return self._sound_of(c)

    def hear_then_see(self, sig: np.ndarray) -> str:
        """Given only a sound, name the shape the brain pictures (cross-modal)."""
        c = self.assoc.concept_from_sound(self.sound_code(sig))
        return self._shape_of(c)

    def _shape_of(self, c: int) -> str:
        return self.pair_names[int(self.concept_label[c])].split("+")[0]

    def _sound_of(self, c: int) -> str:
        return self.pair_names[int(self.concept_label[c])].split("+")[1]


def build_multisensory_brain(pairs: Optional[List[Tuple[str, str]]] = None,
                             n_concept: int = 0, n_per_pair: int = 24,
                             verbose: bool = False) -> MultisensoryBrain:
    """Build both sensory streams, bind shape<->sound pairs, and measure
    cross-modal recall. Returns a :class:`MultisensoryBrain`."""
    from ..vision.ventral import build_ventral_stream, shape_images
    from ..audition.audio import build_auditory_stream, sound_dataset

    def say(*a):
        if verbose:
            print(*a)

    if pairs is None:
        pairs = [("circle", "pure"), ("star", "harmonic"), ("square", "up_chirp"),
                 ("triangle", "down_chirp"), ("bar", "am"), ("cross", "noise")]
    shapes = tuple(p[0] for p in pairs)
    pair_names = [f"{s}+{a}" for s, a in pairs]

    say("building the visual stream ...")
    vision = build_ventral_stream(verbose=verbose)
    say("building the auditory stream ...")
    audio = build_auditory_stream(verbose=verbose)

    # pre-compute per-pair visual object codes and sound codes
    imgs, ilab, _ = shape_images(n_per_pair, vision.size, seed=31,
                                 classes=shapes, noise=0.04)
    sigs, slab, snames = sound_dataset(n_per_pair, seed=32)
    say("association area: binding what is seen to what is heard ...")
    n_concept = n_concept or len(pairs)
    n_aud = len(audio.descriptor(sigs[0]))
    assoc = AssociationArea(vision.IT.n_units, n_aud, n_concept)
    brain = MultisensoryBrain(vision, audio, assoc, pair_names=pair_names)

    # each pair's examples are bound onto one concept cell -- repeated Hebbian
    # co-activation, whose fixed point is the mean audio-visual code
    v_by_pair, a_by_pair = {}, {}
    for p, (shape, snd) in enumerate(pairs):
        v_by_pair[p] = [brain.vision_code(im) for im in imgs[ilab == p]]
        a_by_pair[p] = [brain.sound_code(s) for s in
                        [sigs[i] for i in range(len(sigs))
                         if snames[slab[i]] == snd]]
    assoc.set_stats(np.array([v for vs in v_by_pair.values() for v in vs]),
                    np.array([a for as_ in a_by_pair.values() for a in as_]))
    for p in range(len(pairs)):
        assoc.Wv[p] = _unit(np.mean([assoc.prep_v(v) for v in v_by_pair[p]], 0))
        assoc.Wa[p] = _unit(np.mean([assoc.prep_a(a) for a in a_by_pair[p]], 0))
        assoc.wins[p] = len(v_by_pair[p])
    brain.concept_label = np.arange(len(pairs))

    # cross-modal accuracy on fresh examples: see->name sound, hear->name shape
    t_imgs, t_ilab, _ = shape_images(12, vision.size, seed=99,
                                     classes=shapes, noise=0.04)
    v2s = 0
    for im, lab in zip(t_imgs, t_ilab):
        v2s += brain.see_then_hear(im) == pairs[lab][1]
    brain.vision_to_sound_acc = v2s / len(t_imgs)

    t_sigs, t_slab, t_sn = sound_dataset(12, seed=98)
    snd_to_pair = {snd: p for p, (_, snd) in enumerate(pairs)}
    s2v = seen = 0
    for s, lab in zip(t_sigs, t_slab):
        snd = t_sn[lab]
        if snd not in snd_to_pair:           # this sound isn't a bound concept
            continue
        seen += 1
        s2v += brain.hear_then_see(s) == pairs[snd_to_pair[snd]][0]
    brain.sound_to_vision_acc = s2v / max(seen, 1)

    # binding consistency: do vision and sound of the same concept wake the
    # SAME concept cell? (the two senses meeting on one neuron)
    same = 0
    for p, (shape, snd) in enumerate(pairs):
        vi = brain.assoc.concept_from_vision(v_by_pair[p][0])
        ai = brain.assoc.concept_from_sound(a_by_pair[p][0])
        same += vi == ai == p
    brain.binding_consistency = same / len(pairs)
    say(f"   cross-modal: see->hear {brain.vision_to_sound_acc:.0%}, "
        f"hear->see {brain.sound_to_vision_acc:.0%}, "
        f"binding consistency {brain.binding_consistency:.0%}")
    return brain
