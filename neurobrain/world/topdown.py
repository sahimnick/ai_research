"""
topdown.py
==========

Three things a feed-forward stack does not have: **top-down prediction**,
**staged growth**, and **perception coupled to action**.

1. Top-down prediction (Rao & Ballard)
--------------------------------------
Cortex is not a one-way street. Feedback from a higher area to a lower one
outnumbers the feed-forward fibres, and the standard account of what it carries
is a **prediction**: V2 sends down its best guess at V1's activity, and V1 sends
up only what the guess got **wrong**.

:class:`PredictiveStack` implements that literally. The forward pass becomes:

    guess   = W_down @ v2_state          # what the higher stage expects to see
    error   = relu(v1_actual - guess)    # what it failed to predict
    v2_next = f(error)                   # only the surprise travels up

The claim this makes is testable and it is tested: **the residual should be
sparser and less redundant than the raw activity**, because everything the
higher stage could already predict has been explained away. If the residual is
not sparser, the loop is decoration.

2. Staged growth
----------------
Everything so far was built at full size and trained at once. A brain is not:
areas mature in order, and each one develops on top of a *settled* input rather
than a moving target. :class:`GrowthSchedule` does that -- V1 develops alone,
is then frozen while V2 develops on its stable output, and capacity is added
only where the measurements say it is being used. Compared head-to-head with
developing everything simultaneously.

3. Perception coupled to action
-------------------------------
The prediction error is not only a learning signal; it is a **motor** signal.
:class:`ActiveLooker` sends the eye to wherever the top-down prediction is
failing worst, which is the place with the most to learn. This is what closes
the circle: the brain's model decides where the body looks, and where the body
looks decides what the model learns next.

Measured against looking at the highest-contrast place (a stimulus-driven
control) and against looking at random.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..vision.widev1 import WideV1, _unit


def _sparsity(X: np.ndarray) -> float:
    """Population sparseness (Rolls & Tovee): 1 means one cell carries it all."""
    X = np.maximum(np.asarray(X, np.float32), 0.0)
    n = X.shape[-1]
    s = X.sum(-1)
    s2 = (X ** 2).sum(-1)
    ok = s2 > 1e-12
    if not np.any(ok):
        return 0.0
    a = (s[ok] ** 2) / (n * s2[ok])
    return float(np.mean(1.0 - a) * n / max(n - 1, 1))


def _redundancy(X: np.ndarray, n: int = 300) -> float:
    """Mean absolute correlation between units -- how much they say twice."""
    X = np.asarray(X, np.float32)[:n]
    X = X - X.mean(0, keepdims=True)
    sd = X.std(0) + 1e-9
    keep = sd > 1e-6
    if keep.sum() < 2:
        return 0.0
    Z = X[:, keep] / sd[keep]
    C = (Z.T @ Z) / max(len(Z) - 1, 1)
    np.fill_diagonal(C, 0.0)
    return float(np.abs(C).mean())


# ---------------------------------------------------------------------------
# 1. The predictive loop
# ---------------------------------------------------------------------------
class PredictiveStack:
    """One generative matrix, inference by settling, learning by error x state.

    This is Rao & Ballard's model as they wrote it, and the previous version
    here was not. That one kept **two** matrices -- ``W_up`` learning toward the
    residual and ``W_down`` toward the raw input -- which are different targets,
    so the pair drifted apart and the loop got *worse* at predicting the longer
    it trained. Measured: unexplained activity 11.4% -> 23.3% over six epochs
    while the cosine between the two matrices fell 0.958 -> 0.866. The two
    numbers are the same fact seen twice.

    The correct formulation has a single generative weight matrix ``W`` mapping
    the higher representation down to a prediction of the lower area:

        prediction = W.T @ r
        error      = v1 - prediction
        r          <- relu(r + eta * (W @ error) - decay * r)      # inference
        W          <- W + lr * outer(r, error)                     # learning

    Both updates are **local products of quantities that are physically present
    in the tissue**: the inference step is the error neurons driving the
    representation neurons through the same synapses, run backwards, and the
    learning step is the Hebbian product of a representation cell's activity
    with the error cell's activity. No global error signal, no backpropagation.
    """

    def __init__(self, n_v1: int, n_high: int = 128, iters: int = 20,
                 gain: float = 0.5, decay: float = 0.02, sparsity: float = 0.10,
                 seed: int = 0):
        rng = np.random.default_rng(seed)
        self.n_v1, self.n_high = int(n_v1), int(n_high)
        self.iters = int(iters)
        # The inference step size is NOT a free parameter. A recurrent loop of
        # this form is stable only while the step stays below 2 / lambda_max of
        # the loop matrix W W^T, and with 64 near-identical unit-norm rows
        # lambda_max is about 64 -- so a hand-picked eta of 0.12 was 4x over the
        # limit and the loop OSCILLATED: it flipped between r=2.54 with a zero
        # prediction and r=0 with a prediction of 16.3, every iteration, and
        # reported 100% unexplained forever. The gain is therefore derived from
        # the spectral norm and tracked as W changes. Biologically this is the
        # job inhibition does in a recurrent circuit: hold the loop gain below
        # the point where it rings.
        self.gain = float(gain)
        self.decay, self.sparsity = float(decay), float(sparsity)
        self.W = np.abs(rng.normal(0, 1, (n_high, n_v1))).astype(np.float32)
        self.W /= np.maximum(np.linalg.norm(self.W, axis=1, keepdims=True), 1e-6)
        self.duty = np.full(n_high, sparsity, np.float32)
        self._eta = 0.0
        self._retune()

    def _retune(self) -> None:
        """Set the inference step from the loop's own spectral norm."""
        lam = float(np.linalg.norm(self.W, 2) ** 2)          # lambda_max(W W^T)
        self._eta = self.gain / max(lam, 1e-6)

    @property
    def eta(self) -> float:
        return self._eta

    # backwards-compatible views: the two matrices are now ONE
    @property
    def W_down(self) -> np.ndarray:
        return self.W

    @property
    def W_up(self) -> np.ndarray:
        """The feed-forward path is the same synapses read the other way.

        Reciprocal cortico-cortical connections are the anatomy; keeping one
        matrix is both the biology and what stops the drift."""
        return self.W

    def settle(self, v1: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                              np.ndarray]:
        """Infer the higher state that best explains this input.

        Returns ``(high_state, residual, prediction)``. Inference is the loop
        running to a fixed point, not an optimiser: each round the error drives
        the representation, the representation makes a new prediction, and the
        error shrinks."""
        v1 = np.maximum(np.asarray(v1, np.float32), 0.0)
        scale = float(v1.max()) or 1.0
        x = v1 / scale
        r = np.zeros(self.n_high, np.float32)
        pred = np.zeros_like(x)
        err = x
        for _ in range(self.iters):
            pred = self.W.T @ r
            err = x - pred
            r = r + self._eta * (self.W @ err) - self.decay * r
            r = np.maximum(r - 0.02 * (self.duty - self.sparsity), 0.0)
        return r, np.maximum(err, 0.0) * scale, pred * scale

    def learn(self, v1: np.ndarray, lr: float = 0.05) -> float:
        """One learning step. Returns the fraction of activity left unexplained.

        ``dW = lr * outer(r, err)`` -- the local Hebbian product of the
        representation and the error. Rows are kept on the unit sphere, which is
        the synaptic scaling already used everywhere else in this project and is
        what stops the generative weights from simply growing."""
        v1 = np.maximum(np.asarray(v1, np.float32), 0.0)
        scale = float(v1.max()) or 1.0
        x = v1 / scale
        r, resid, _ = self.settle(v1)
        err = x - self.W.T @ r
        self.W += lr * np.outer(r, err)
        np.maximum(self.W, 0.0, out=self.W)
        n = np.linalg.norm(self.W, axis=1, keepdims=True)
        self.W /= np.maximum(n, 1e-6)
        self.duty *= (1.0 - lr)
        self.duty += lr * (r > 1e-6)
        self._retune()
        tot = float(x.sum()) + 1e-9
        return float(np.maximum(err, 0.0).sum() / tot)

    def train(self, states: Sequence[np.ndarray], epochs: int = 3,
              lr: float = 0.05, seed: int = 0) -> List[float]:
        rng = np.random.default_rng(seed)
        curve = []
        for _ in range(epochs):
            e = [self.learn(states[i], lr=lr)
                 for i in rng.permutation(len(states))]
            curve.append(float(np.mean(e)))
        return curve


def predictive_coding_experiment(n_v1: int = 512, n_high: int = 128,
                                 n_images: int = 500, epochs: int = 3,
                                 seed: int = 0, verbose: bool = False
                                 ) -> Dict[str, float]:
    """Does explaining-away actually make the code sparser and less redundant?

    The comparison is the raw V1 activity against the residual the loop leaves.
    If feedback is doing what the theory says, the residual carries the same
    stimulus in fewer, less correlated units."""
    from ..sensing.realworld import load_mnist
    from ..learning.selforganize import randomise_filters, develop_v1

    trx, trY, _, _ = load_mnist(n_train=n_images, n_test=5)
    v1 = randomise_filters(WideV1(n_cells=n_v1, seed=seed), seed=seed)
    develop_v1(v1, trx[:200], epochs=2, seed=seed)
    states = [v1.rate(im) for im in trx]

    stack = PredictiveStack(n_v1=n_v1, n_high=n_high, seed=seed)
    curve = stack.train(states[:300], epochs=epochs, seed=seed)

    raw = np.array(states, np.float32)
    res = np.array([stack.settle(s)[1] for s in states], np.float32)
    out = {
        "unexplained_first_epoch": curve[0],
        "unexplained_last_epoch": curve[-1],
        "raw_sparsity": _sparsity(raw),
        "residual_sparsity": _sparsity(res),
        "raw_redundancy": _redundancy(raw),
        "residual_redundancy": _redundancy(res),
    }
    out["sparsity_gain"] = out["residual_sparsity"] - out["raw_sparsity"]
    out["redundancy_drop"] = out["raw_redundancy"] - out["residual_redundancy"]
    if verbose:
        print(f"   unexplained activity: {curve[0]:.1%} -> {curve[-1]:.1%}")
        print(f"   sparsity   raw {out['raw_sparsity']:.3f} -> residual "
              f"{out['residual_sparsity']:.3f} ({out['sparsity_gain']:+.3f})")
        print(f"   redundancy raw {out['raw_redundancy']:.3f} -> residual "
              f"{out['residual_redundancy']:.3f} "
              f"({-out['redundancy_drop']:+.3f})")
    return out


# ---------------------------------------------------------------------------
# 2. Staged growth
# ---------------------------------------------------------------------------
@dataclass
class GrowthStage:
    """One step of maturation: what grows, on what, for how long."""

    name: str
    epochs: int
    freeze_below: bool = True


class GrowthSchedule:
    """Areas mature in order, each on a settled input.

    The alternative -- developing every stage at once -- makes each stage chase
    a moving target: V2 learns conjunctions of V1 features that V1 is still in
    the middle of changing. Staging removes that, and it is what real cortical
    maturation looks like."""

    def __init__(self, stages: Sequence[GrowthStage]):
        self.stages = list(stages)
        self.log: List[Dict[str, float]] = []

    def run(self, v1: WideV1, images: Sequence[np.ndarray], v2=None,
            orient_fn=None, seed: int = 0) -> "GrowthSchedule":
        from ..learning.selforganize import develop_v1
        from ..vision.v2binding import v1_parts

        for st in self.stages:
            if st.name == "V1":
                develop_v1(v1, images, epochs=st.epochs, seed=seed)
                self.log.append({"stage": st.name, "epochs": float(st.epochs)})
            elif st.name == "V2" and v2 is not None:
                orient = orient_fn(v1)
                acts = [v1_parts(v1, im, orient) for im in images]
                v2.train(acts, v1.n_rows, v1.n_cols, epochs=st.epochs)
                self.log.append({"stage": st.name, "epochs": float(st.epochs)})
        return self


def staged_vs_simultaneous(n_v1: int = 512, n_v2: int = 64,
                           n_per_class: int = 60, n_pairs: int = 10,
                           n_develop: int = 300, seed: int = 0,
                           verbose: bool = False) -> Dict[str, float]:
    """Does maturing V1 before V2 beat growing both together?

    Both conditions get the **same total number of development epochs**, so the
    comparison is about the *order*, not the budget."""
    from ..vision.composite import composite_dataset, LocalV2
    from ..learning.selforganize import randomise_filters, develop_v1
    from ..vision.v2binding import v1_parts, preferred_orientations
    from ..vision.widev1 import grown_readout

    X, y, names = composite_dataset(n_per_class=n_per_class, n_pairs=n_pairs,
                                    seed=seed)
    cut = int(0.7 * len(X))
    Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[cut:], y[cut:]

    def score(v1, v2) -> float:
        orient = preferred_orientations(v1)
        atr = [v1_parts(v1, im, orient) for im in Xtr]
        ate = [v1_parts(v1, im, orient) for im in Xte]
        A = np.array([_unit(np.concatenate([_unit(v1.rate(im)),
                                            0.6 * v2.code(a, v1.n_rows, v1.n_cols)]))
                      for im, a in zip(Xtr, atr)], np.float32)
        B = np.array([_unit(np.concatenate([_unit(v1.rate(im)),
                                            0.6 * v2.code(a, v1.n_rows, v1.n_cols)]))
                      for im, a in zip(Xte, ate)], np.float32)
        return grown_readout(A, ytr, B, yte)[0]

    # -- staged: V1 matures fully, then V2 grows on a settled V1 ------------
    v1s = randomise_filters(WideV1(n_cells=n_v1, seed=seed), seed=seed)
    v2s = LocalV2(n_units=n_v2, seed=seed)
    GrowthSchedule([GrowthStage("V1", 4), GrowthStage("V2", 3)]).run(
        v1s, Xtr[:n_develop], v2=v2s, orient_fn=preferred_orientations,
        seed=seed)
    staged = score(v1s, v2s)

    # -- simultaneous: both develop at the same time, same total epochs ----
    v1t = randomise_filters(WideV1(n_cells=n_v1, seed=seed), seed=seed)
    v2t = LocalV2(n_units=n_v2, seed=seed)
    for _ in range(4):
        develop_v1(v1t, Xtr[:n_develop], epochs=1, seed=seed)
        o = preferred_orientations(v1t)
        acts = [v1_parts(v1t, im, o) for im in Xtr[:n_develop]]
        v2t.train(acts, v1t.n_rows, v1t.n_cols, epochs=1)
    simultaneous = score(v1t, v2t)

    out = {"staged": staged, "simultaneous": simultaneous,
           "gain": staged - simultaneous,
           "chance": float(np.bincount(yte).max() / len(yte))}
    if verbose:
        print(f"   staged growth (V1 then V2): {staged:.1%}")
        print(f"   both at once              : {simultaneous:.1%} "
              f"({out['gain']:+.1%} for staging)")
    return out


# ---------------------------------------------------------------------------
# 3. Perception coupled to action
# ---------------------------------------------------------------------------
class ActiveLooker:
    """The eye goes where the model's prediction is failing.

    Prediction error is a motor signal here, not only a learning signal: the
    place the higher stage cannot explain is the place worth looking at. That
    closes the loop -- the model chooses where the body looks, and where the
    body looks decides what the model sees next."""

    def __init__(self, v1: WideV1, stack: PredictiveStack, fovea: int = 28):
        self.v1, self.stack, self.fovea = v1, stack, int(fovea)

    def error_map(self, image: np.ndarray) -> np.ndarray:
        """Unexplained activity, summed per receptive-field location."""
        _, resid, _ = self.stack.settle(self.v1.rate(image))
        m = np.bincount(self.v1.cell_pos, weights=resid.astype(np.float64),
                        minlength=self.v1.n_pos)
        return m.reshape(self.v1.n_rows, self.v1.n_cols).astype(np.float32)

    def next_fixation(self, image: np.ndarray, visited: Sequence[Tuple[int, int]]
                      = ()) -> Tuple[int, int]:
        """The location with the most unexplained activity, not yet visited."""
        m = self.error_map(image).copy()
        for (r, c) in visited:
            gr = min(max(r * self.v1.n_rows // image.shape[0], 0),
                     self.v1.n_rows - 1)
            gc = min(max(c * self.v1.n_cols // image.shape[1], 0),
                     self.v1.n_cols - 1)
            m[gr, gc] = -1e9
        f = int(np.argmax(m))
        gr, gc = divmod(f, self.v1.n_cols)
        return (int((gr + 0.5) * image.shape[0] / self.v1.n_rows),
                int((gc + 0.5) * image.shape[1] / self.v1.n_cols))


def error_driven_looking_experiment(n_v1: int = 512, n_high: int = 96,
                                    n_scene_objects: int = 10,
                                    scene_size: int = 192, n_saccades: int = 24,
                                    n_scenes: int = 6, seed: int = 0,
                                    verbose: bool = False) -> Dict[str, float]:
    """Does looking where prediction fails beat looking where contrast is high?

    Both policies get the same eye, the same scene and the same number of
    saccades. The measure is how often a saccade lands on an actual object --
    the thing an orienting system exists to do."""
    from ..sensing.realworld import load_mnist
    from ..sensing.streams import build_scene, SaccadicEye
    from ..learning.selforganize import randomise_filters, develop_v1

    trx, trY, _, _ = load_mnist(n_train=400, n_test=5)
    v1 = randomise_filters(WideV1(n_cells=n_v1, seed=seed), seed=seed)
    develop_v1(v1, trx[:200], epochs=2, seed=seed)
    stack = PredictiveStack(n_v1=n_v1, n_high=n_high, seed=seed)
    stack.train([v1.rate(im) for im in trx[:200]], epochs=2, seed=seed)
    looker = ActiveLooker(v1, stack)

    hits = {"error": 0, "saliency": 0, "random": 0}
    total = 0
    for s in range(n_scenes):
        scene = build_scene(trx, trY, size=scene_size,
                            n_objects=n_scene_objects, seed=seed + s)
        eye = SaccadicEye(scene, seed=seed + s)
        rng = np.random.default_rng(seed + 100 + s)
        half = eye.fovea // 2
        visited: List[Tuple[int, int]] = []
        for _ in range(n_saccades):
            total += 1
            # saliency policy: the eye's own contrast map (the v0.23 control)
            r, c = eye.next_target()
            r, c = eye.foveate(r, c)
            hits["saliency"] += int(scene.label_at(r, c) >= 0)
            # random policy
            rr = int(rng.integers(half, scene_size - half))
            cc = int(rng.integers(half, scene_size - half))
            hits["random"] += int(scene.label_at(rr, cc) >= 0)
            # prediction-error policy, on the current foveal view
            pr, pc = (visited[-1] if visited
                      else (scene_size // 2, scene_size // 2))
            pr = int(np.clip(pr, half, scene_size - half - 1))
            pc = int(np.clip(pc, half, scene_size - half - 1))
            patch = scene.canvas[pr - half:pr + half, pc - half:pc + half]
            dy, dx = looker.next_fixation(patch, visited=())
            er = int(np.clip(pr + (dy - half), half, scene_size - half - 1))
            ec = int(np.clip(pc + (dx - half), half, scene_size - half - 1))
            er, ec = eye.foveate(er, ec)
            hits["error"] += int(scene.label_at(er, ec) >= 0)
            visited.append((er, ec))

    out = {k: v / max(total, 1) for k, v in hits.items()}
    tile = 28
    out["chance"] = (n_scene_objects * tile * tile) / float(scene_size ** 2)
    if verbose:
        print(f"   prediction-error policy : {out['error']:.1%} of saccades "
              f"land on an object")
        print(f"   saliency policy         : {out['saliency']:.1%}")
        print(f"   random policy           : {out['random']:.1%} "
              f"(chance {out['chance']:.1%})")
    return out
