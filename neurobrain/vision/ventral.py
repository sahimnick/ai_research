"""
vision.py
=========

A **spiking convolutional ventral stream** -- the perceptual ladder
``edge -> corner/curve -> shape/object`` built from spiking neurons and
synapses, one cortical area per layer:

    retina/LGN  ->  V1  ->  V1 complex  ->  V2  ->  pool  ->  V4  ->  readout
                   (edges)  (phase-inv.)  (corners) (invar.)  (shapes)

What the rest of the project was missing for vision was *architecture*: random,
all-to-all wiring cannot see space. Every area here has **local receptive
fields** replicated across the image (convolution / retinotopy), and its filters
are learned **unsupervised** by competitive Hebbian ("STDP-style") learning, so:

  * **V1** discovers **oriented edge detectors** (simple cells), seeded from a
    Gabor bank and refined on edges -- exactly like real V1;
  * **V1 complex** cells take the *energy* of the simple cells and pool it, so
    they respond to an orientation regardless of its exact phase/position
    (Hubel & Wiesel's complex cells; the Adelson-Bergen energy model);
  * **V2** learns **combinations of orientations** -- corners, junctions,
    curved contour fragments -- from the complex-cell maps of corner/curve
    images;
  * **V4** learns **curvature / shape** units over pooled V2 features, with a
    larger effective receptive field.

Design goals the request asked for and this keeps:
  * **separated layers** -- each area is its own object with its own weights,
    state and ``layer.log``;
  * **per-layer input/output logging** -- every area records the exact maps that
    went in and came out, so each stage is inspectable.

Optimisation: the convolution now uses a fully vectorised ``im2col``
(``sliding_window_view``) instead of a Python loop, so the whole stream runs
fast enough to train all four areas in seconds.

Honesty: this is a *teaching-scale* stream -- small images, tens of maps. It
demonstrates that the same spiking neurons and synapses, wired retinotopically,
build a real perceptual hierarchy with measurable orientation-, corner- and
curvature-selectivity, and can recognise simple shapes with high accuracy. For
production recognition, standard deep CNNs still win.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# ---------------------------------------------------------------------------
# fast building blocks: vectorised im2col and pooling
# ---------------------------------------------------------------------------
def _im2col(x: np.ndarray, k: int, stride: int) -> Tuple[np.ndarray, int, int]:
    """(C, H, W) -> ((n_positions, C*k*k) patches, out_h, out_w).

    Vectorised with ``sliding_window_view`` -- no Python loop -- so a forward
    pass is a single matmul. Per-position layout is (channel, row, col), the
    same order the old loop produced, so learned weights stay compatible.
    """
    C, H, W = x.shape
    win = sliding_window_view(x, (k, k), axis=(1, 2))     # (C, H-k+1, W-k+1, k, k)
    win = win[:, ::stride, ::stride]                      # (C, oh, ow, k, k)
    oh, ow = win.shape[1], win.shape[2]
    cols = win.transpose(1, 2, 0, 3, 4).reshape(oh * ow, C * k * k)
    return np.ascontiguousarray(cols, dtype=np.float32), oh, ow


def _maxpool(x: np.ndarray, size: int) -> np.ndarray:
    """Spatial max-pool of (C, H, W) by an integer factor."""
    C, H, W = x.shape
    oh, ow = H // size, W // size
    if oh == 0 or ow == 0:
        return x.astype(np.float32)
    trimmed = x[:, :oh * size, :ow * size]
    return trimmed.reshape(C, oh, size, ow, size).max(axis=(2, 4)).astype(np.float32)


def gabor_kernel(size: int, theta: float, phase: float = 0.0,
                 freq: Optional[float] = None, sigma: Optional[float] = None
                 ) -> np.ndarray:
    """A single Gabor filter (size x size), oriented at ``theta`` radians.

    ``phase = pi/2`` gives an odd (edge) filter -- an ON lobe next to an OFF
    lobe -- which is what a V1 simple cell looks like. Used to seed V1 and to
    build the complex-cell quadrature energy model.
    """
    if freq is None:
        freq = 1.2 / size
    if sigma is None:
        sigma = size / 4.5
    c = np.linspace(-(size - 1) / 2.0, (size - 1) / 2.0, size)
    ys, xs = np.meshgrid(c, c, indexing="ij")
    xr = xs * np.cos(theta) + ys * np.sin(theta)          # across the edge
    yr = -xs * np.sin(theta) + ys * np.cos(theta)         # along the edge
    env = np.exp(-(xr ** 2 + yr ** 2) / (2.0 * sigma ** 2))
    g = (env * np.cos(2.0 * np.pi * freq * xr + phase)).astype(np.float32)
    g -= g.mean()
    n = np.linalg.norm(g)
    return g / n if n > 1e-6 else g


# ---------------------------------------------------------------------------
# V1 / V2 / V4: a convolutional area of spiking feature maps
# ---------------------------------------------------------------------------
class SpikingConvLayer:
    """One convolutional cortical area of leaky integrate-and-fire feature maps.

    Weights are shared local kernels (one per feature map). Inference runs the
    maps as spiking IF units with winner-take-all inhibition across maps at each
    location. Learning is competitive Hebbian on the winning map's kernel
    (frequency-balanced so every map specialises for a different feature) -- the
    rate-coded form of STDP that makes V1 discover oriented edges and V2/V4
    discover corners and curves.
    """

    def __init__(self, in_channels: int, n_maps: int, kernel: int,
                 stride: int = 1, name: str = "conv", threshold: float = 0.6,
                 lr: float = 0.02, seed: int = 0):
        self.in_channels = in_channels
        self.n_maps = n_maps
        self.k = kernel
        self.stride = stride
        self.name = name
        self.threshold = threshold
        self.lr = lr
        rng = np.random.default_rng(seed)
        # signed weights (real receptive fields have ON and OFF sub-regions)
        self.W = rng.standard_normal(
            (n_maps, in_channels * kernel * kernel)).astype(np.float32) * 0.1
        self._normalise()
        self.wins = np.zeros(n_maps)          # conscience: balance map usage
        self.log: Dict[str, np.ndarray] = {}

    def _normalise(self):
        n = np.linalg.norm(self.W, axis=1, keepdims=True)
        n[n < 1e-6] = 1.0
        self.W /= n

    # -- inference -------------------------------------------------------
    def response(self, x: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """Signed, graded linear drive of each map at each location (pre-spike).

        This is the membrane input a downstream energy model needs. Returns
        ``(drive (n_maps, oh, ow), oh, ow)``.
        """
        cols, oh, ow = _im2col(x, self.k, self.stride)
        cols = cols - cols.mean(axis=1, keepdims=True)    # respond to CONTRAST
        drive = cols @ self.W.T                           # (positions, n_maps)
        return drive.T.reshape(self.n_maps, oh, ow), oh, ow

    def forward(self, x: np.ndarray, T: int = 12) -> np.ndarray:
        """Run the area as spiking IF maps; return output spike-rate maps.

        ``x`` is (in_channels, H, W). Returns (n_maps, oh, ow) spike counts and
        logs the input and output spike maps in ``self.log``.
        """
        drive_maps, oh, ow = self.response(x)
        drive = np.maximum(drive_maps.reshape(self.n_maps, -1).T, 0.0)  # (P, M)
        mem = np.zeros_like(drive)
        out = np.zeros_like(drive)
        for _ in range(T):                          # leaky integrate-and-fire
            mem = 0.9 * mem + drive
            fired = mem >= self.threshold
            if fired.any():
                # winner-take-all across maps at a location (lateral inhibition)
                strongest = (mem == mem.max(axis=1, keepdims=True)) & fired
                out += strongest
                mem[fired] = 0.0
        out_maps = out.T.reshape(self.n_maps, oh, ow).astype(np.float32)
        self.log = {"input": x.copy(), "output": out_maps,
                    "drive": np.maximum(drive_maps, 0.0).astype(np.float32)}
        return out_maps

    # -- learning: competitive Hebbian on kernels ------------------------
    def learn_patch(self, patch: np.ndarray) -> int:
        """Move the best-matching map's kernel toward ``patch`` (returns winner).

        Frequency-balanced competition (a "conscience" bias) makes sure every
        map wins its share, so the maps cover different features instead of all
        collapsing onto one.
        """
        patch = patch.reshape(-1).astype(np.float32)
        patch = patch - patch.mean()             # learn CONTRAST, not brightness
        pn = np.linalg.norm(patch)
        if pn < 1e-6:
            return -1
        patch = patch / pn
        match = self.W @ patch - 0.15 * (self.wins / (self.wins.sum() + 1))
        win = int(np.argmax(match))
        self.W[win] += self.lr * (patch - self.W[win])   # signed, no clip
        self.wins[win] += 1
        self._normalise()
        return win

    def seed_gabor(self) -> None:
        """Seed each map with an edge Gabor at a different orientation.

        Single-channel (V1) only. Starting from a clean, evenly-spaced oriented
        bank gives well-tuned simple cells immediately; Hebbian learning then
        refines them on the data.
        """
        thetas = np.linspace(0, np.pi, self.n_maps, endpoint=False)
        W = np.empty_like(self.W)
        for i, th in enumerate(thetas):
            g = gabor_kernel(self.k, th, phase=np.pi / 2).reshape(-1)
            W[i] = np.tile(g, self.in_channels)
        self.W = W.astype(np.float32)
        self._normalise()

    def seed_from(self, patches: np.ndarray) -> None:
        """k-means++ style seeding: start each map from a spread-out patch.

        Used for multi-channel areas (V2/V4) where a Gabor bank makes no sense.
        Spreading the starting kernels stops a few maps from winning everything.
        """
        rng = np.random.default_rng(7)
        flat = patches.reshape(len(patches), -1).astype(np.float32)
        flat = flat - flat.mean(axis=1, keepdims=True)
        norm = np.linalg.norm(flat, axis=1, keepdims=True)
        flat = flat / np.where(norm < 1e-6, 1.0, norm)
        chosen = [int(rng.integers(len(flat)))]
        for _ in range(self.n_maps - 1):
            sim = np.max(flat @ flat[chosen].T, axis=1)   # closeness to chosen
            chosen.append(int(np.argmin(sim)))            # pick the most different
        self.W = flat[chosen].copy()
        self._normalise()

    def train(self, patches: np.ndarray, epochs: int = 4, seed: bool = True,
              init: str = "auto") -> "SpikingConvLayer":
        """Learn kernels by competitive Hebbian passes over ``patches``.

        ``init`` = ``"gabor"`` (V1), ``"kmeans"`` (V2/V4) or ``"auto"`` (Gabor
        for single-channel areas, k-means++ otherwise).
        """
        if seed:
            gabor = init == "gabor" or (init == "auto" and self.in_channels == 1)
            self.seed_gabor() if gabor else self.seed_from(patches)
        rng = np.random.default_rng(1)
        for _ in range(epochs):
            for i in rng.permutation(len(patches)):
                self.learn_patch(patches[i])
        return self

    def _prep(self, patches: np.ndarray) -> np.ndarray:
        flat = patches.reshape(len(patches), -1).astype(np.float32)
        flat = flat - flat.mean(axis=1, keepdims=True)
        n = np.linalg.norm(flat, axis=1, keepdims=True)
        return flat / np.where(n < 1e-6, 1.0, n)

    def coverage(self, patches: np.ndarray) -> np.ndarray:
        """How well each patch is explained by the current filters (best match,
        0..1). Low coverage means the area has no cell for that feature yet."""
        return np.maximum(self._prep(patches) @ self.W.T, 0.0).max(axis=1)

    def grow_maps(self, patches: np.ndarray, k: int = 2, thresh: float = 0.6,
                  epochs: int = 2) -> int:
        """Structural growth: add feature maps for stimuli the area cannot yet
        explain. Finds patches whose coverage is below ``thresh``, seeds ``k`` new
        maps from the most distinct of them (k-means++), appends them, and
        refines them by Hebbian learning -- an activity-dependent expansion of
        the cortical map, exactly what the sensory streams were missing. Returns
        how many maps were added.
        """
        flat = self._prep(patches)
        poor = np.flatnonzero(np.maximum(flat @ self.W.T, 0.0).max(1) < thresh)
        if len(poor) < k:
            return 0
        sub = flat[poor]
        rng = np.random.default_rng(len(self.W))
        chosen = [int(rng.integers(len(sub)))]
        for _ in range(k - 1):
            sim = np.max(sub @ sub[chosen].T, axis=1)
            chosen.append(int(np.argmin(sim)))
        self.W = np.vstack([self.W, sub[chosen]]).astype(np.float32)
        self.wins = np.concatenate([self.wins, np.zeros(k)])
        self.n_maps += k
        self._normalise()
        for _ in range(epochs):                       # refine the new maps
            for i in rng.permutation(poor):
                self.learn_patch(patches[i])
        return k

    def kernels(self) -> np.ndarray:
        """Learned kernels as (n_maps, in_channels, k, k) for viewing."""
        return self.W.reshape(self.n_maps, self.in_channels, self.k, self.k)


class ComplexCellLayer:
    """V1 complex cells: phase/position-invariant orientation channels.

    Wraps the trained V1 simple cells. For each orientation it takes the
    **energy** (squared response) of the simple cells tuned to that orientation
    -- squaring throws away contrast polarity and edge phase -- then pools over a
    small spatial window for position tolerance. The result is a small stack of
    orientation channels that fire for an oriented feature *wherever and however*
    it sits in the receptive field. This is Hubel & Wiesel's complex cell / the
    Adelson-Bergen motion-energy model, and it is what V2 reads from.
    """

    def __init__(self, simple: SpikingConvLayer, n_orient: int = 8,
                 pool: int = 2, name: str = "V1_complex"):
        self.simple = simple
        self.n_orient = n_orient
        self.pool = pool
        self.name = name
        self.log: Dict[str, np.ndarray] = {}
        # assign each learned simple cell to its nearest orientation bin
        prefs = np.array([preferred_orientation(k)
                          for k in simple.kernels()[:, 0]])
        self.bins = np.floor(prefs / np.pi * n_orient).astype(int) % n_orient
        # for each learned simple cell, build its QUADRATURE partner (90-deg
        # phase-shifted Gabor at the same preferred orientation). Energy across
        # a quadrature pair, cos^2 + sin^2, is what makes a complex cell fire for
        # an orientation regardless of the edge's exact phase/position.
        quad = np.stack([gabor_kernel(simple.k, p, phase=0.0).reshape(-1)
                         for p in prefs])
        quad = np.tile(quad, (1, simple.in_channels)).astype(np.float32)
        quad -= quad.mean(axis=1, keepdims=True)
        n = np.linalg.norm(quad, axis=1, keepdims=True)
        self.Wq = quad / np.where(n < 1e-6, 1.0, n)

    def forward(self, x: np.ndarray) -> np.ndarray:
        cols, oh, ow = _im2col(x, self.simple.k, self.simple.stride)
        cols = cols - cols.mean(axis=1, keepdims=True)
        even = cols @ self.simple.W.T                     # learned simple cell
        odd = cols @ self.Wq.T                            # quadrature partner
        energy = (even ** 2 + odd ** 2).T.reshape(self.simple.n_maps, oh, ow)
        out = np.zeros((self.n_orient, oh, ow), np.float32)
        for m, b in enumerate(self.bins):
            out[b] += energy[m]
        out = np.sqrt(out)
        pooled = _maxpool(out, self.pool) if self.pool > 1 else out
        peak = pooled.max()                               # comparable scales
        if peak > 1e-6:
            pooled = pooled / peak
        self.log = {"input": x.copy(),
                    "simple": energy.astype(np.float32), "output": pooled}
        return pooled


class ITLayer:
    """Inferotemporal object cells -- the top of the ventral stream.

    IT neurons have a near-whole-image receptive field and are tolerant to an
    object's position/size. Here each IT unit learns a **whole-object template**
    over the V4 shape-part code by competitive Hebbian learning (unsupervised),
    so units become object-selective -- the model's "grandmother cells". A soft
    k-winners step keeps the object code sparse. Logs its input and output.
    """

    def __init__(self, n_v4: int, n_units: int, name: str = "IT",
                 lr: float = 0.05, k_active: int = 4, seed: int = 0):
        self.n_units = n_units
        self.name = name
        self.lr = lr
        self.k_active = k_active
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((n_units, n_v4)).astype(np.float32) * 0.1
        self._normalise()
        self.wins = np.zeros(n_units)
        self.log: Dict[str, np.ndarray] = {}

    def _normalise(self):
        n = np.linalg.norm(self.W, axis=1, keepdims=True)
        n[n < 1e-6] = 1.0
        self.W /= n

    @staticmethod
    def _pool(v4: np.ndarray) -> np.ndarray:
        """Whole-field pool of V4 maps -> one position-tolerant vector."""
        x = v4.reshape(v4.shape[0], -1).mean(1)
        n = np.linalg.norm(x)
        return (x / n).astype(np.float32) if n > 1e-6 else x.astype(np.float32)

    def forward(self, v4: np.ndarray) -> np.ndarray:
        x = self._pool(v4)
        act = np.maximum(self.W @ x, 0.0)                  # object-cell drive
        if self.k_active < self.n_units:                   # k-winners sparsity
            thr = np.sort(act)[-self.k_active]
            act = np.where(act >= thr, act, 0.0)
        self.log = {"input": v4.copy(), "output": act.astype(np.float32)}
        return act.astype(np.float32)

    def learn(self, v4: np.ndarray) -> int:
        x = self._pool(v4)
        if np.linalg.norm(x) < 1e-6:
            return -1
        match = self.W @ x - 0.1 * (self.wins / (self.wins.sum() + 1))
        win = int(np.argmax(match))
        self.W[win] += self.lr * (x - self.W[win])
        self.wins[win] += 1
        self._normalise()
        return win

    def train(self, v4_maps: List[np.ndarray], epochs: int = 6
              ) -> "ITLayer":
        pooled = np.array([self._pool(m) for m in v4_maps], np.float32)
        # k-means++ spread so units cover different objects, not one
        rng = np.random.default_rng(3)
        chosen = [int(rng.integers(len(pooled)))]
        for _ in range(self.n_units - 1):
            sim = np.max(pooled @ pooled[chosen].T, axis=1)
            chosen.append(int(np.argmin(sim)))
        self.W = pooled[chosen].copy()
        self._normalise()
        for _ in range(epochs):
            for i in rng.permutation(len(v4_maps)):
                self.learn(v4_maps[i])
        return self


class SpikingPool:
    """Spatial max-pooling of spike maps -> local translation invariance."""

    def __init__(self, size: int = 2, name: str = "pool"):
        self.size = size
        self.name = name
        self.log: Dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = _maxpool(x, self.size)
        self.log = {"input": x.copy(), "output": out}
        return out


@dataclass
class VisionHierarchy:
    """A stack of spiking areas with per-area input/output logging."""

    layers: List[object] = field(default_factory=list)

    def add(self, layer) -> "VisionHierarchy":
        self.layers.append(layer)
        return self

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def logs(self) -> List[Dict]:
        """The input/output maps of every area from the last forward()."""
        out = []
        for layer in self.layers:
            log = dict(getattr(layer, "log", {}))
            out.append({"layer": getattr(layer, "name", type(layer).__name__),
                        "in_shape": None if "input" not in log
                        else tuple(log["input"].shape),
                        "out_shape": None if "output" not in log
                        else tuple(log["output"].shape),
                        "in_spikes": None if "input" not in log
                        else float(log["input"].sum()),
                        "out_spikes": None if "output" not in log
                        else float(log["output"].sum())})
        return out


# ---------------------------------------------------------------------------
# training data: edges (V1), corners+curves (V2), shapes (V4 / readout)
# ---------------------------------------------------------------------------
def oriented_edges(n: int = 800, size: int = 11, seed: int = 0
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """``n`` step-edge patches at random orientations. Returns
    ``(patches (n,1,size,size), angles)``. Feeds V1."""
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    ys -= size / 2 - 0.5
    xs -= size / 2 - 0.5
    patches = np.empty((n, 1, size, size), dtype=np.float32)
    angles = rng.uniform(0, np.pi, n)
    for i, th in enumerate(angles):
        proj = xs * np.cos(th) + ys * np.sin(th)
        offset = rng.uniform(-1.5, 1.5)
        patches[i, 0] = 1.0 / (1.0 + np.exp(-(proj - offset) * 2.5))
    return patches, angles


def _stroke(img: np.ndarray, pts: np.ndarray, width: float = 1.1) -> None:
    """Paint anti-aliased points onto a single-channel image in place."""
    H, W = img.shape
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    for py, px in pts:
        img += np.exp(-((ys - py) ** 2 + (xs - px) ** 2) / (2.0 * width ** 2))
    np.clip(img, 0.0, 1.0, out=img)


def corner_images(n: int = 300, size: int = 22, seed: int = 0
                  ) -> np.ndarray:
    """``n`` images of two line segments meeting at a random angle (corners /
    junctions). Feeds V2 after passing through V1+complex."""
    rng = np.random.default_rng(seed)
    imgs = np.zeros((n, 1, size, size), np.float32)
    for i in range(n):
        cy = rng.uniform(size * 0.35, size * 0.65)
        cx = rng.uniform(size * 0.35, size * 0.65)
        a0 = rng.uniform(0, 2 * np.pi)
        a1 = a0 + rng.uniform(np.pi / 4, 3 * np.pi / 4)   # a real bend
        length = rng.uniform(size * 0.3, size * 0.45)
        for a in (a0, a1):
            t = np.linspace(0, length, int(length) + 2)
            pts = np.stack([cy + t * np.sin(a), cx + t * np.cos(a)], axis=1)
            _stroke(imgs[i, 0], pts)
    return imgs


def curve_images(n: int = 300, size: int = 22, seed: int = 0) -> np.ndarray:
    """``n`` images of arcs with random curvature/position. Feeds V2/V4."""
    rng = np.random.default_rng(seed)
    imgs = np.zeros((n, 1, size, size), np.float32)
    for i in range(n):
        R = rng.uniform(size * 0.3, size * 0.9)
        cy = rng.uniform(0, size) - R * rng.uniform(-0.4, 0.4)
        cx = rng.uniform(0, size) - R * rng.uniform(-0.4, 0.4)
        a0 = rng.uniform(0, 2 * np.pi)
        span = rng.uniform(np.pi / 3, np.pi)
        a = np.linspace(a0, a0 + span, int(R * span) + 3)
        pts = np.stack([cy + R * np.sin(a), cx + R * np.cos(a)], axis=1)
        pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < size) &
                  (pts[:, 1] >= 0) & (pts[:, 1] < size)]
        if len(pts):
            _stroke(imgs[i, 0], pts)
    return imgs


SHAPE_CLASSES = ("bar", "cross", "square", "triangle", "circle",
                 "star", "hexagon", "arrow", "pentagon", "heart",
                 "crescent", "spiral")


def _draw_one(name: str, size: int, rng: np.random.Generator,
              pose: Optional[Tuple[float, float, float, float]] = None
              ) -> np.ndarray:
    """Render one outline shape with random position, scale and rotation.

    ``pose = (cy, cx, r, rot)`` renders an *explicit* view instead of a random
    one -- that is what lets us show the same object under a smooth, continuous
    transformation (see :func:`transformation_sequence`)."""
    if pose is None:
        cy = size / 2 + rng.uniform(-2, 2)
        cx = size / 2 + rng.uniform(-2, 2)
        r = rng.uniform(size * 0.26, size * 0.40)
        rot = rng.uniform(0, np.pi)
    else:
        cy, cx, r, rot = pose
    img = np.zeros((size, size), np.float32)
    c, s = np.cos(rot), np.sin(rot)
    R = np.array([[c, -s], [s, c]], np.float32)

    def poly(verts):
        v = np.asarray(verts, np.float32) @ R.T
        for j in range(len(v)):
            p0, p1 = v[j], v[(j + 1) % len(v)]
            t = np.linspace(0, 1, int(size))
            pts = np.stack([cy + p0[0] + t * (p1[0] - p0[0]),
                            cx + p0[1] + t * (p1[1] - p0[1])], axis=1)
            _stroke(img, pts)

    def line(angle):
        t = np.linspace(-r, r, int(size))
        _stroke(img, np.stack([cy + t * np.sin(angle), cx + t * np.cos(angle)], 1))

    if name == "bar":
        line(rot)
    elif name == "cross":
        line(rot); line(rot + np.pi / 2)
    elif name == "square":
        poly([[-r, -r], [-r, r], [r, r], [r, -r]])
    elif name == "triangle":
        a = np.array([0, 2 * np.pi / 3, 4 * np.pi / 3])
        poly(np.stack([r * np.sin(a), r * np.cos(a)], 1))
    elif name == "circle":
        a = np.linspace(0, 2 * np.pi, int(2 * np.pi * r) + 6)
        _stroke(img, np.stack([cy + r * np.sin(a), cx + r * np.cos(a)], 1))
    elif name == "star":
        pts = [[(r if k % 2 == 0 else 0.45 * r) * np.sin(k * np.pi / 5),
                (r if k % 2 == 0 else 0.45 * r) * np.cos(k * np.pi / 5)]
               for k in range(10)]
        poly(pts)
    elif name == "hexagon":
        poly([[r * np.sin(k * np.pi / 3), r * np.cos(k * np.pi / 3)]
              for k in range(6)])
    elif name == "arrow":
        poly([[-0.3 * r, -r], [-0.3 * r, 0.2 * r], [-0.6 * r, 0.2 * r],
              [0, r], [0.6 * r, 0.2 * r], [0.3 * r, 0.2 * r], [0.3 * r, -r]])
    elif name == "pentagon":
        poly([[r * np.sin(k * 2 * np.pi / 5), r * np.cos(k * 2 * np.pi / 5)]
              for k in range(5)])
    elif name == "heart":
        t = np.linspace(0, 2 * np.pi, 60)
        hx = 16 * np.sin(t) ** 3
        hy = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
        pts = np.stack([-hy / 16 * r, hx / 16 * r], 1) @ R.T
        _stroke(img, np.stack([cy + pts[:, 0], cx + pts[:, 1]], 1))
    elif name == "crescent":
        a = np.linspace(0, 2 * np.pi, int(2 * np.pi * r) + 8)
        outer = np.stack([r * np.sin(a), r * np.cos(a)], 1) @ R.T
        _stroke(img, np.stack([cy + outer[:, 0], cx + outer[:, 1]], 1))
        ir, off = 0.8 * r, 0.5 * r
        inner = (np.stack([ir * np.sin(a), ir * np.cos(a)], 1)
                 + np.array([0, off])) @ R.T
        _stroke(img, np.stack([cy + inner[:, 0], cx + inner[:, 1]], 1))
    elif name == "spiral":
        a = np.linspace(0, 4 * np.pi, 90)
        rad = r * a / (4 * np.pi)
        sp = np.stack([rad * np.sin(a), rad * np.cos(a)], 1) @ R.T
        _stroke(img, np.stack([cy + sp[:, 0], cx + sp[:, 1]], 1))
    return img


def shape_images(n_per_class: int = 40, size: int = 28, seed: int = 0,
                 classes: Tuple[str, ...] = ("bar", "cross", "square",
                                             "triangle", "circle"),
                 noise: float = 0.0, occlude: float = 0.0
                 ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Outline shapes for recognition tests, with jitter, scale and rotation.

    ``noise`` adds Gaussian pixel noise; ``occlude`` (0..1) blanks a random
    square of that relative size on ~70% of images -- together they make a
    harder test of the object-level (IT) tolerance. Returns
    ``(images (N,1,size,size), labels, class_names)``.
    """
    rng = np.random.default_rng(seed)
    names = list(classes)
    imgs, labels = [], []
    for cls, name in enumerate(names):
        for _ in range(n_per_class):
            img = _draw_one(name, size, rng)
            if noise > 0:
                img = np.clip(img + rng.normal(0, noise, img.shape),
                              0, 1).astype(np.float32)
            if occlude > 0 and rng.random() < 0.7:
                w = max(1, int(occlude * size))
                y = rng.integers(0, size - w + 1)
                x = rng.integers(0, size - w + 1)
                img[y:y + w, x:x + w] = 0.0
            imgs.append(img[None])
            labels.append(cls)
    return np.array(imgs, np.float32), np.array(labels), names


def hard_shape_images(n_per_class: int = 40, size: int = 56, seed: int = 0
                      ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """The complex-object test: all 12 shape classes (including pentagon, heart,
    crescent, spiral) with random rotation/scale/position and mild noise +
    occlusion."""
    return shape_images(n_per_class, size, seed, classes=SHAPE_CLASSES,
                        noise=0.04, occlude=0.06)


def two_object_image(name_a: str, name_b: str, size: int = 56, seed: int = 0
                     ) -> np.ndarray:
    """A wide image: full-scale object ``name_a`` on the left, ``name_b`` on the
    right -- two objects competing, for the top-down attention demo. The canvas
    is (size, 2*size); the size-agnostic conv/descriptor handle it directly."""
    rng = np.random.default_rng(seed)
    img = np.zeros((1, size, 2 * size), np.float32)
    img[0, :, :size] = _draw_one(name_a, size, rng)
    img[0, :, size:] = _draw_one(name_b, size, rng)
    return img


# ---------------------------------------------------------------------------
# selectivity metrics (evidence the areas learned what they should)
# ---------------------------------------------------------------------------
def preferred_orientation(kernel: np.ndarray) -> float:
    """Estimate a learned filter's preferred orientation (radians, 0..pi)."""
    gy, gx = np.gradient(kernel.astype(np.float32))
    ang = np.arctan2(gy.sum(), gx.sum())
    return float(ang % np.pi)


def orientation_selectivity(layer: SpikingConvLayer, n_angles: int = 16
                            ) -> np.ndarray:
    """Orientation-selectivity index (0..1) of every map in a V1-like area.

    Probes each map with edge Gabors at ``n_angles`` orientations and returns
    ``(R_pref - R_orth)/(R_pref + R_orth)`` -- 1 means perfectly orientation
    tuned, 0 means untuned.
    """
    thetas = np.linspace(0, np.pi, n_angles, endpoint=False)
    size = layer.k * 2 + 1
    c = np.linspace(-(size - 1) / 2.0, (size - 1) / 2.0, size)
    ys, xs = np.meshgrid(c, c, indexing="ij")
    resp = np.zeros((layer.n_maps, n_angles), np.float32)
    for j, th in enumerate(thetas):
        proj = xs * np.cos(th) + ys * np.sin(th)
        img = (1.0 / (1.0 + np.exp(-proj * 2.0)))[None].astype(np.float32)
        drive, _, _ = layer.response(img)      # best response over phase/position
        resp[:, j] = np.maximum(drive.reshape(layer.n_maps, -1).max(axis=1), 0)
    osi = np.zeros(layer.n_maps, np.float32)
    for m in range(layer.n_maps):
        r = resp[m]
        if r.max() <= 1e-6:
            continue
        best = int(r.argmax())
        orth = (best + n_angles // 2) % n_angles
        osi[m] = max((r[best] - r[orth]) / (r[best] + r[orth] + 1e-9), 0.0)
    return osi


def composition_order(layer: SpikingConvLayer, frac: float = 0.25) -> np.ndarray:
    """How many input channels each map genuinely combines.

    For every map, counts the input channels whose weight-energy exceeds
    ``frac`` of that map's strongest channel. V1 reads one channel (the image),
    so this is ~1; a V2 corner unit that fuses two orientations scores ~2+,
    which is the evidence that higher areas are **compositional**, not just
    bigger edge detectors.
    """
    W = layer.W.reshape(layer.n_maps, layer.in_channels, -1)
    energy = (W ** 2).sum(axis=2)                    # (n_maps, in_channels)
    peak = energy.max(axis=1, keepdims=True) + 1e-9
    return (energy >= frac * peak).sum(axis=1).astype(np.float32)


def phase_invariance(simple: SpikingConvLayer, complex_layer: "ComplexCellLayer",
                     n_phase: int = 16) -> Tuple[float, float]:
    """How much a complex cell beats a simple cell at ignoring edge phase.

    Presents a grating at a fixed orientation and *varies its spatial phase*,
    reading both cells at the same central location. A simple cell's response
    oscillates with phase (ON/OFF lobes falling in and out of register); a
    complex cell's stays flat. Returns ``(simple_cv, complex_cv)`` as the
    coefficient of variation across phase -- lower means more phase invariant.
    """
    size = simple.k * 2 + 1
    theta = 0.6
    freq = 1.2 / simple.k
    c = np.linspace(-(size - 1) / 2.0, (size - 1) / 2.0, size)
    ys, xs = np.meshgrid(c, c, indexing="ij")
    proj = xs * np.cos(theta) + ys * np.sin(theta)
    s_resp, c_resp = [], []
    for phi in np.linspace(0, 2 * np.pi, n_phase, endpoint=False):
        img = (0.5 + 0.5 * np.cos(2 * np.pi * freq * proj + phi)
               )[None].astype(np.float32)
        sd, oh, ow = simple.response(img)
        s_resp.append(max(sd[:, oh // 2, ow // 2].max(), 0.0))
        co = complex_layer.forward(img)
        c_resp.append(co[:, co.shape[1] // 2, co.shape[2] // 2].max())

    def cv(v):
        v = np.array(v)
        return float(v.std() / (v.mean() + 1e-9))
    return cv(s_resp), cv(c_resp)


# ---------------------------------------------------------------------------
# the whole ventral stream + a shape-recognition readout
# ---------------------------------------------------------------------------
def _junction_features(ori: np.ndarray) -> np.ndarray:
    """Corner/junction evidence from V1-complex orientation channels.

    A corner is where **two different orientations coincide** at one location; a
    straight edge or smooth curve has a single dominant orientation locally. We
    measure two things per location and pool them over the image:

      * **right-angle-corner** energy -- an orthogonal orientation pair firing
        together (bins ``b`` and ``b + n/2``); high for squares and crosses,
        near zero for a circle;
      * **general junction** energy -- the second-strongest orientation, which
        picks up the oblique corners of a triangle.

    This is what lets the readout tell a square (four right-angle corners) from a
    circle (all smooth curve) even at coarse resolution.
    """
    C = ori.shape[0]
    flat = ori.reshape(C, -1)
    peak = flat.max()
    if peak <= 1e-6:
        return np.zeros(4, np.float32)
    n = flat / peak
    hw = n.shape[1]
    corner90 = np.zeros(hw, np.float32)
    for b in range(C // 2):                    # orthogonal orientation pairs
        corner90 = np.maximum(corner90, np.minimum(n[b], n[(b + C // 2) % C]))
    srt = np.sort(n, axis=0)
    second = srt[-2]                            # 2nd-strongest orientation
    active = n.max(0) > 0.25
    return np.array([corner90.sum() / hw, float(corner90.max()),
                     float(second[active].mean()) if active.any() else 0.0,
                     float(active.mean())], np.float32)


def _pool_map(m2d: np.ndarray, g: int = 12) -> np.ndarray:
    """Downsample a 2-D map to a g x g grid (a coarse retinotopic sample)."""
    H, W = m2d.shape
    ys = np.linspace(0, H, g + 1).astype(int)
    xs = np.linspace(0, W, g + 1).astype(int)
    out = np.zeros((g, g), np.float32)
    for i in range(g):
        for j in range(g):
            blk = m2d[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            if blk.size:
                out[i, j] = blk.max()
    return out


def _centred_map(m2d: np.ndarray, g: int = 12) -> np.ndarray:
    """Crop to the object's bounding box, then sample it on a g x g grid.

    This is *not* a shape descriptor -- it removes only where the object is and
    how big it appears (which retinal/attentional centring does in the real
    visual system). It says nothing about rotation; that invariance has to be
    LEARNED (see :class:`TraceInvarianceLayer`).
    """
    peak = float(m2d.max())
    if peak <= 1e-6:
        return np.zeros((g, g), np.float32)
    ys, xs = np.nonzero(m2d > 0.15 * peak)
    if len(ys) < 5:
        return np.zeros((g, g), np.float32)
    crop = m2d[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return _pool_map(crop, g)


class TraceInvarianceLayer:
    """Cells that learn **transformation invariance from time**.

    The real visual system is not told that a rotated cup is still a cup. It
    sees objects *transform smoothly*, and it exploits a simple fact: identity
    changes slowly, appearance changes fast. Foldiak's **trace rule** (1991) --
    and Wallis & Rolls' account of view-invariant IT cells -- makes a cell bind
    together whatever it saw in quick succession, because learning is gated by a
    *temporal trace* of the cell's own recent activity rather than by its
    instantaneous response.

        trace <- (1 - eta) * trace + eta * y
        dW_i  <- lr * trace_i * (x - W_i)

    A cell that won on view 1 still carries trace when view 2 arrives, so it
    also learns view 2 -- and the two views end up on the same cell. Invariance
    is *acquired from experience*, not designed in.

    Feed-forward Hebbian learning alone is **not enough**, and getting this wrong
    is instructive: with only the trace rule, every cell drifts toward the same
    input and the code collapses -- one pattern for every object, a "perfectly
    invariant" representation that cannot tell a star from a circle. Foldiak's
    model therefore has three interacting parts, and all three are needed:

        y      = kWTA( W x - Q y - theta )        # lateral inhibition
        dW_i   = lr * trace_i * (x - W_i)         # trace-gated Hebbian
        dQ_ij  = -alpha * (y_i y_j - p^2)         # ANTI-Hebbian decorrelation
        dtheta = beta * (y - p)                   # threshold adaptation

    The anti-Hebbian lateral weights ``Q`` push apart cells that keep firing
    together, which is exactly the force that stops the collapse; the adaptive
    thresholds keep every cell working at the same average rate.
    """

    def __init__(self, n_in: int, n_cells: int = 128, k: int = 5,
                 eta: float = 0.5, lr: float = 0.05, alpha: float = 0.12,
                 beta: float = 0.05, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 1.0, (n_cells, n_in)).astype(np.float32)
        self.W /= np.maximum(np.linalg.norm(self.W, axis=1, keepdims=True), 1e-6)
        self.Q = np.zeros((n_cells, n_cells), np.float32)   # anti-Hebbian lateral
        self.theta = np.zeros(n_cells, np.float32)          # adaptive thresholds
        self.n_cells, self.k = n_cells, k
        self.eta, self.lr, self.alpha, self.beta = eta, lr, alpha, beta
        self.p = k / n_cells                                # target firing rate

    @staticmethod
    def _unit(x: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(x))
        return x.astype(np.float32) if n < 1e-9 else (x / n).astype(np.float32)

    def _sparse(self, d: np.ndarray) -> np.ndarray:
        y = np.zeros(self.n_cells, np.float32)
        idx = np.argpartition(d, -self.k)[-self.k:]
        y[idx] = np.maximum(d[idx], 1e-6)
        return y

    def forward(self, x: np.ndarray, settle: int = 3) -> np.ndarray:
        """The invariant sparse code for one view, after lateral competition."""
        ff = self.W @ self._unit(x) - self.theta
        y = self._sparse(ff)
        for _ in range(settle):                    # settle with lateral inhibition
            y = self._sparse(ff - self.Q @ self._unit(y))
        return self._unit(y)

    def learn_sequence(self, seq: Sequence[np.ndarray]) -> None:
        """Watch ONE object transform smoothly, and bind its views together --
        while the anti-Hebbian lateral weights keep different objects apart."""
        trace = np.zeros(self.n_cells, np.float32)
        for x in seq:
            y = self.forward(x)
            trace = (1.0 - self.eta) * trace + self.eta * y
            act = np.nonzero(trace > 1e-4)[0]
            if len(act):
                xu = self._unit(x)
                self.W[act] += self.lr * trace[act, None] * (xu[None, :] - self.W[act])
                self.W[act] /= np.maximum(
                    np.linalg.norm(self.W[act], axis=1, keepdims=True), 1e-6)
            # anti-Hebbian decorrelation: cells that co-fire inhibit each other
            yb = (y > 0).astype(np.float32)
            self.Q += self.alpha * (np.outer(yb, yb) - self.p ** 2)
            np.fill_diagonal(self.Q, 0.0)
            np.maximum(self.Q, 0.0, out=self.Q)     # lateral weights are inhibitory
            # threshold adaptation: hold every cell near the target rate
            self.theta += self.beta * (yb - self.p)

    def train(self, sequences: Sequence[Sequence[np.ndarray]], epochs: int = 6,
              seed: int = 0) -> "TraceInvarianceLayer":
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            for i in rng.permutation(len(sequences)):
                self.learn_sequence(sequences[i])
        return self

    def invariance_score(self, sequences: Sequence[Sequence[np.ndarray]]) -> float:
        """Similarity between the codes of the first and last view of each
        transformation sequence (1.0 = identical).

        **Never read this alone.** A collapsed code scores a perfect 1.0 while
        being useless; always read it together with :meth:`discriminability`.
        """
        s = [float(self.forward(seq[0]) @ self.forward(seq[-1])) for seq in sequences]
        return float(np.mean(s)) if s else 0.0

    def discriminability(self, groups: Sequence[Sequence[np.ndarray]]) -> float:
        """Within-group similarity MINUS between-group similarity, where each
        group holds views of one object. Zero means the code has collapsed."""
        codes = [[self.forward(x) for x in g] for g in groups]
        within, between = [], []
        for i, gi in enumerate(codes):
            for a in range(len(gi)):
                for b in range(a + 1, len(gi)):
                    within.append(float(gi[a] @ gi[b]))
            for j, gj in enumerate(codes):
                if j <= i:
                    continue
                for a in gi:
                    for b in gj:
                        between.append(float(a @ b))
        if not within or not between:
            return 0.0
        return float(np.mean(within) - np.mean(between))


def transformation_sequence(name: str, size: int = 28, n_views: int = 10,
                            seed: int = 0) -> np.ndarray:
    """The SAME object seen through a **smooth** rotation + scale + shift.

    This is the experience that teaches invariance in a real visual system:
    objects do not teleport between poses, they turn and approach continuously.
    """
    rng = np.random.default_rng(seed)
    cy0 = size / 2 + rng.uniform(-1.5, 1.5)
    cx0 = size / 2 + rng.uniform(-1.5, 1.5)
    r0 = rng.uniform(size * 0.27, size * 0.34)
    rot0 = rng.uniform(0, np.pi)
    drot = rng.uniform(0.5, 1.4) * (1 if rng.random() < 0.5 else -1)
    dr = rng.uniform(-0.07, 0.07) * size
    dy, dx = rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5)
    views = []
    for t in range(n_views):
        f = t / max(n_views - 1, 1)
        pose = (cy0 + dy * f, cx0 + dx * f,
                float(np.clip(r0 + dr * f, size * 0.22, size * 0.42)),
                rot0 + drot * f)
        views.append(_draw_one(name, size, rng, pose=pose))
    return np.array(views, np.float32)


def _radial_signature(m2d: np.ndarray, n_ang: int = 48, n_harm: int = 16
                      ) -> np.ndarray:
    """DEPRECATED (v0.18): a hand-designed rotation/scale-invariant signature.

    Kept only so the earlier results stay reproducible. It is **no longer used
    in the recognition path** -- it was an analytic Fourier descriptor that
    *gave* the network its invariance instead of letting it learn one, which is
    exactly the kind of engineering crutch this project is supposed to avoid.
    :class:`TraceInvarianceLayer` replaces it with invariance learned from
    watching objects transform.
    """
    peak = m2d.max()
    if peak <= 1e-6:
        return np.zeros(n_harm, np.float32)
    ys, xs = np.nonzero(m2d > 0.15 * peak)
    if len(ys) < 5:
        return np.zeros(n_harm, np.float32)
    cy, cx = ys.mean(), xs.mean()
    ang = np.arctan2(ys - cy, xs - cx)
    rad = np.hypot(ys - cy, xs - cx)
    b = ((ang + np.pi) / (2 * np.pi) * n_ang).astype(int) % n_ang
    r = np.zeros(n_ang)
    np.maximum.at(r, b, rad)                       # farthest contour per angle
    idx = np.arange(n_ang)
    good = r > 0
    if good.any() and (~good).any():
        r = np.interp(idx, idx[good], r[good], period=n_ang)
    r = r / (r.mean() + 1e-6)
    return np.abs(np.fft.rfft(r))[:n_harm].astype(np.float32)


def _grid_pool(maps: np.ndarray, g: int) -> np.ndarray:
    """Sum each map over a ``g x g`` spatial grid -> coarse 'where' layout."""
    C, H, W = maps.shape
    ys = np.linspace(0, H, g + 1).astype(int)
    xs = np.linspace(0, W, g + 1).astype(int)
    out = np.zeros((C, g, g), np.float32)
    for i in range(g):
        for j in range(g):
            out[:, i, j] = maps[:, ys[i]:ys[i + 1], xs[j]:xs[j + 1]].sum((1, 2))
    return out


def _sample_patches(images: np.ndarray, front: List[object], k: int,
                    per_image: int, seed: int, min_active: float = 0.15
                    ) -> np.ndarray:
    """Push images through ``front`` layers and sample *informative* k x k
    feature patches -- blank-background patches are rejected so no downstream
    map gets seeded on an empty field and stays dead."""
    rng = np.random.default_rng(seed)
    out = []
    for img in images:
        m = img
        for layer in front:
            m = layer.forward(m)
        C, H, W = m.shape
        if H < k or W < k:
            continue
        kept = 0
        for _ in range(per_image * 8):        # oversample, keep the lively ones
            y = rng.integers(0, H - k + 1)
            x = rng.integers(0, W - k + 1)
            patch = m[:, y:y + k, x:x + k]
            if patch.max() >= min_active:
                out.append(patch.copy())
                kept += 1
                if kept >= per_image:
                    break
    return np.array(out, np.float32)


@dataclass
class VentralStream:
    """A trained V1->complex->V2->pool->V4 stream plus a shape readout."""

    V1: SpikingConvLayer
    complex: ComplexCellLayer
    V2: SpikingConvLayer
    pool: SpikingPool
    V4: SpikingConvLayer
    hierarchy: VisionHierarchy
    size: int = 56                             # canvas the stream was built for
    trace: Optional["TraceInvarianceLayer"] = None   # LEARNED invariance cells
    invariance: float = 0.0                    # measured view-invariance
    discriminability: float = 0.0              # within- minus between-object
    IT: Optional[ITLayer] = None               # object cells (V4 -> IT)
    class_names: Optional[List[str]] = None
    it_names: Optional[List[str]] = None       # classes of the hard/object task
    it_class_proto: Optional[np.ndarray] = None  # mean IT code per object class
    it_class_v4proto: Optional[np.ndarray] = None  # mean V4 channel profile / class
    _mu: Optional[np.ndarray] = None
    _sd: Optional[np.ndarray] = None
    _W: Optional[np.ndarray] = None            # ridge readout weights
    _hmu: Optional[np.ndarray] = None
    _hsd: Optional[np.ndarray] = None
    _hW: Optional[np.ndarray] = None           # object-task readout

    def descriptor(self, image: np.ndarray) -> np.ndarray:
        """A biologically-tiered descriptor of one (1,H,W) image.

        Concatenates what each area sees: the V1-complex **orientation
        histogram** (how much of each orientation is present), the total drive
        of every **V2 corner/curve** unit, and the coarse spatial layout of the
        **V4 shape** units. This mirrors how the ventral stream builds an object
        code from parts, and it is what the readout classifies.
        """
        self.hierarchy.forward(image)
        return self._descriptor_from(self.complex.log["output"],
                                     self.V2.log["output"], self.V4.log["output"],
                                     image=image, trace=self.trace)

    @staticmethod
    def trace_input(ori: np.ndarray, image: Optional[np.ndarray]) -> np.ndarray:
        """What the invariance cells look at: the object centred on the retina,
        as contour energy and as raw form. Centring is retinal; *rotation*
        invariance is what the cells must learn."""
        # per-orientation centred maps keep the contour's local structure (what
        # tells a star from a pentagon); a summed map alone is too coarse.
        parts = [_centred_map(ori[c], 8).reshape(-1) for c in range(ori.shape[0])]
        parts.append(_centred_map(ori.sum(0), 14).reshape(-1))
        if image is not None:
            parts.append(_centred_map(image[0], 14).reshape(-1))
        else:
            parts.append(np.zeros(196, np.float32))
        return np.concatenate(parts).astype(np.float32)

    @staticmethod
    def _descriptor_from(ori: np.ndarray, v2: np.ndarray, v4: np.ndarray,
                         image: Optional[np.ndarray] = None,
                         trace: Optional["TraceInvarianceLayer"] = None
                         ) -> np.ndarray:
        parts = [_junction_features(ori)]                  # corners vs smooth curve
        for g in (2, 4):                                   # orientation layout
            parts.append(_grid_pool(ori, g).reshape(-1))
        for g in (3, 6):                                   # V2 corner/curve layout
            parts.append(_grid_pool(v2, g).reshape(-1))
        for g in (3, 5):                                   # V4 shape-part layout
            parts.append(_grid_pool(v4, g).reshape(-1))
        # LEARNED transformation-invariant code (trace rule), replacing the old
        # hand-designed Fourier signature.
        if trace is not None:
            parts.append(trace.forward(
                VentralStream.trace_input(ori, image)) * 3.0)
        d = np.concatenate(parts).astype(np.float32)
        n = np.linalg.norm(d)
        return d / n if n > 1e-6 else d

    def classify(self, image: np.ndarray) -> str:
        d = (self.descriptor(image) - self._mu) / self._sd
        scores = np.concatenate([d, [1.0]]) @ self._W
        return self.class_names[int(scores.argmax())]

    def it_code(self, image: np.ndarray) -> np.ndarray:
        """The IT object-cell response for one image (needs the V4 map first)."""
        self.hierarchy.forward(image)
        return self.IT.forward(self.V4.log["output"])

    def object_features(self, image: np.ndarray) -> np.ndarray:
        """The multi-area descriptor + the IT object code -- input to the readout."""
        d = self.descriptor(image)                         # runs the hierarchy
        it = self.IT.forward(self.V4.log["output"])
        return np.concatenate([d, it]).astype(np.float32)

    def classify_object(self, image: np.ndarray) -> str:
        f = (self.object_features(image) - self._hmu) / self._hsd
        scores = np.concatenate([f, [1.0]]) @ self._hW
        return self.it_names[int(scores.argmax())]

    def _attention_heatmap(self, v4: np.ndarray, concept: str) -> np.ndarray:
        """Where the cued object's V4 signature is strongest (feature-similarity
        gain): correlate each location's V4 channel vector with the object's
        prototype. This is the top-down feature template guiding *where* to look.
        """
        c = self.it_names.index(concept)
        proto = self.it_class_v4proto[c]
        proto = proto / (np.linalg.norm(proto) + 1e-6)
        h = np.maximum(np.tensordot(proto, v4, axes=(0, 0)), 0.0)   # (H4, W4)
        m = h.max()
        return h / m if m > 1e-6 else h

    @staticmethod
    def _upsample(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        yi = (np.linspace(0, mask.shape[0] - 1, shape[0])).round().astype(int)
        xi = (np.linspace(0, mask.shape[1] - 1, shape[1])).round().astype(int)
        return mask[np.ix_(yi, xi)]

    def attend(self, image: np.ndarray, cue: str, beta: float = 10.0,
               power: float = 3.0) -> str:
        """Goal-directed attention: look at ``image`` *for* the cued object.

        The cued object's V4 template sends a top-down feature-similarity gain
        down the hierarchy, enhancing the region whose features match the cue and
        suppressing the rest; IT then reads the attended maps and reports the
        nearest object prototype. With two objects competing in one image, cueing
        one vs the other gives a different answer from the *same input* -- the
        signature of top-down (biased-competition) attention, which no
        feed-forward pass can produce.
        """
        self.hierarchy.forward(image)
        v4 = self.V4.log["output"]
        g4 = (self._attention_heatmap(v4, cue) ** power)[None] * beta
        it = self.IT.forward(v4 * g4)
        scores = [float(it @ (p / (np.linalg.norm(p) + 1e-6)))
                  for p in self.it_class_proto]
        return self.it_names[int(np.argmax(scores))]


def _fit_readout(X: np.ndarray, labels: np.ndarray, n_classes: int,
                 lam: float = 1.0, seed: int = 0):
    """Ridge (least-squares) linear readout with a held-out test split.

    Returns ``(mu, sd, W, test_accuracy)``. The readout is a single linear map
    -- a stand-in for the downstream cortex that reads the object code; all the
    representation learning happened upstream and unsupervised.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    cut = int(0.6 * len(idx))
    tr, te = idx[:cut], idx[cut:]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xa = np.hstack([(X - mu) / sd, np.ones((len(X), 1), np.float32)])
    Y = np.eye(n_classes, dtype=np.float32)[labels]
    A = Xa[tr].T @ Xa[tr] + lam * np.eye(Xa.shape[1], dtype=np.float32)
    W = np.linalg.solve(A, Xa[tr].T @ Y[tr])
    acc = float((Xa[te] @ W).argmax(1).astype(int).__eq__(labels[te]).mean())
    return mu.astype(np.float32), sd.astype(np.float32), W.astype(np.float32), acc


def build_ventral_stream(n_v1: int = 12, n_v2: int = 36, n_v4: int = 44,
                         n_it: int = 72, verbose: bool = False) -> VentralStream:
    """Train the four areas in order, each on the previous area's output.

    Returns a :class:`VentralStream`. Also fits a nearest-centroid readout on
    shape images so the whole thing can be checked end-to-end.
    """
    def say(*a):
        if verbose:
            print(*a)

    SZ = 56                                    # working canvas; keeps maps large

    # V1: oriented simple cells, Gabor-seeded, refined on edges
    say("V1: learning oriented edge detectors ...")
    edges, _ = oriented_edges(n=1600, size=11, seed=0)
    V1 = SpikingConvLayer(1, n_v1, 11, stride=1, name="V1", lr=0.04, seed=1)
    V1.train(edges, epochs=6, init="gabor")
    complex_ = ComplexCellLayer(V1, n_orient=8, pool=2, name="V1_complex")

    # V2: combinations of orientations (corners, junctions, curve fragments)
    say("V2: learning corner/curve detectors from complex-cell maps ...")
    v2_imgs = np.concatenate([corner_images(320, SZ, seed=2),
                              curve_images(320, SZ, seed=3)])
    v2_patches = _sample_patches(v2_imgs, [complex_], k=5, per_image=8, seed=4)
    V2 = SpikingConvLayer(complex_.n_orient, n_v2, 5, name="V2", lr=0.03, seed=5)
    V2.train(v2_patches, epochs=5, init="kmeans")
    pool = SpikingPool(2, name="pool")

    # V4: curvature / shape parts over pooled V2 features
    say("V4: learning curvature/shape units over pooled V2 ...")
    v4_imgs = np.concatenate([curve_images(260, SZ, seed=6),
                              shape_images(40, SZ, seed=7)[0]])
    v4_patches = _sample_patches(v4_imgs, [complex_, V2, pool], k=3,
                                 per_image=8, seed=8)
    V4 = SpikingConvLayer(n_v2, n_v4, 3, name="V4", lr=0.03, seed=9)
    V4.train(v4_patches, epochs=5, init="kmeans")

    hierarchy = (VisionHierarchy().add(complex_).add(V2).add(pool).add(V4))
    stream = VentralStream(V1, complex_, V2, pool, V4, hierarchy, size=SZ)

    # LEARNED transformation invariance (trace rule): watch objects rotate and
    # change size continuously, and bind their views onto the same cells. This
    # replaces the old hand-designed rotation-invariant Fourier descriptor.
    say("invariance: learning from objects transforming in time (trace rule) ...")
    seqs = []
    for name in SHAPE_CLASSES:
        for s in range(26):
            views = transformation_sequence(name, SZ, n_views=10, seed=1000 + s)
            feats = []
            for v in views:
                im = v[None]
                hierarchy.forward(im)
                feats.append(VentralStream.trace_input(
                    complex_.log["output"], im))
            seqs.append(feats)
    stream.trace = TraceInvarianceLayer(n_in=len(seqs[0][0]), n_cells=160, k=5,
                                        seed=3).train(seqs, epochs=8)
    stream.invariance = stream.trace.invariance_score(seqs)
    # honest: invariance alone can be perfect on a COLLAPSED code, so measure
    # discriminability (within-object minus between-object similarity) too.
    by_obj = {}
    for name in SHAPE_CLASSES:
        by_obj[name] = []
    for i, name in enumerate(SHAPE_CLASSES):
        by_obj[name] = [seqs[i * 26 + s][v] for s in range(4) for v in (0, 5, 9)]
    stream.discriminability = stream.trace.discriminability(
        [by_obj[n] for n in SHAPE_CLASSES])
    say(f"   learned view-invariance {stream.invariance:.0%}, "
        f"discriminability {stream.discriminability:+.2f} "
        f"(0.00 would mean a collapsed code)")

    # fit + score the clean 5-shape readout (held-out test -> honest accuracy)
    say("readout: fitting shape classifier ...")
    imgs, labels, names = shape_images(n_per_class=80, size=SZ, seed=11)
    X = np.array([stream.descriptor(im) for im in imgs], np.float32)
    stream._mu, stream._sd, stream._W, stream.test_accuracy = _fit_readout(
        X, labels, len(names), lam=1.0, seed=0)
    stream.class_names = names
    say(f"   shape recognition accuracy: {stream.test_accuracy:.0%} "
        f"(5 classes, chance 20%)")

    # IT object cells over V4, trained on the harder set (8 classes, noisy +
    # occluded), then an object-task readout on the descriptor + IT code
    say("IT: learning object cells over V4 (harder: noise + occlusion) ...")
    h_imgs, h_lab, h_names = hard_shape_images(110, SZ, seed=21)
    descs, v4s = [], []
    for im in h_imgs:
        descs.append(stream.descriptor(im))
        v4s.append(stream.V4.log["output"].copy())
    IT = ITLayer(V4.n_maps, n_it, name="IT", seed=13).train(v4s, epochs=6)
    stream.IT = IT
    # object features = the multi-area descriptor + the IT object code
    Xo = np.array([np.concatenate([descs[i], IT.forward(v4s[i])])
                   for i in range(len(h_imgs))], np.float32)
    stream._hmu, stream._hsd, stream._hW, stream.hard_accuracy = _fit_readout(
        Xo, h_lab, len(h_names), lam=2.0, seed=1)
    stream.it_names = h_names
    # per-class IT prototype (mean object code) -- the top-down template used by
    # attention to enhance an object's own V4 evidence
    it_codes = np.array([IT.forward(v) for v in v4s])
    stream.it_class_proto = np.array(
        [it_codes[h_lab == c].mean(0) for c in range(len(h_names))], np.float32)
    v4_prof = np.array([v.reshape(v.shape[0], -1).mean(1) for v in v4s])
    stream.it_class_v4proto = np.array(
        [v4_prof[h_lab == c].mean(0) for c in range(len(h_names))], np.float32)
    # how object-selective the IT cells became (unsupervised): mean purity of
    # each cell's preferred class among the images it fires most for
    tops = np.array([IT.forward(v).argmax() for v in v4s])
    pur = []
    for u in range(n_it):
        lab_u = h_lab[tops == u]
        if len(lab_u):
            pur.append(np.bincount(lab_u).max() / len(lab_u))
    stream.it_purity = float(np.mean(pur)) if pur else 0.0
    say(f"   complex-object recognition: {stream.hard_accuracy:.0%} "
        f"({len(h_names)} classes, chance {100 // len(h_names)}%);  "
        f"IT cell purity {stream.it_purity:.0%}")
    return stream


def build_ventral_stream_on(images: np.ndarray, n_v1: int = 12, n_v2: int = 36,
                            n_v4: int = 44, size: int = 56,
                            verbose: bool = False) -> VentralStream:
    """The same four areas, grown on **real photographs** instead of drawings.

    :func:`build_ventral_stream` develops each area on its own hand-made
    stimulus set -- `oriented_edges` for V1, `corner_images` + `curve_images`
    for V2, `curve_images` + `shape_images` for V4. Line drawings. It reaches
    99% on shapes and 92% on complex objects, so the architecture works on what
    it was built for.

    On photographs it does not. Measured in EVALUATION.md §9.15, on CIFAR:

        stage         probe   participation   spikes per photograph
        V1 complex    0.354           20.8           254
        V2            0.358           66.8           588
        pool          0.288           35.0           214
        V4            0.246            7.9            12

    V4 -- the stage nearest object identity, the one the whole architecture
    points at -- clusters *below raw pixels* and raises twelve spikes. Its
    kernels are tuned to synthetic curvature and a photograph does not contain
    it.

    This builder changes exactly one thing: **every area is developed on the
    images it will actually be asked about**. Same architecture, same widths,
    same canvas, same competitive Hebbian rule, same patch sampler -- so a
    difference between the two can only be the training stimuli.

    The trace-invariance layer, the shape readout and the IT object cells are
    deliberately **not** fitted here. They are supervised or shape-specific and
    would confound the comparison; this returns a stream whose four feedforward
    areas are photograph-grown and nothing else.
    """
    def say(*a):
        if verbose:
            print(*a, flush=True)

    ims = np.asarray(images, np.float32)
    if ims.max() > 1.5:
        ims = ims / 255.0
    if ims.ndim == 3:
        ims = ims[:, None]                      # (n, 1, H, W)

    say(f"V1: oriented cells from {len(ims)} real photographs ...")
    v1_patches = []
    rng = np.random.default_rng(0)
    for im in ims:
        m = im
        for _ in range(1):
            pass
        C, H, W = m.shape
        for _ in range(12):
            y = rng.integers(0, H - 11 + 1)
            x = rng.integers(0, W - 11 + 1)
            p = m[:, y:y + 11, x:x + 11]
            if float(p.max()) - float(p.min()) > 0.05:   # reject flat sky
                v1_patches.append(p.copy())
    V1 = SpikingConvLayer(1, n_v1, 11, stride=1, name="V1", lr=0.04, seed=1)
    V1.train(np.stack(v1_patches), epochs=6, init="gabor")
    complex_ = ComplexCellLayer(V1, n_orient=8, pool=2, name="V1_complex")

    say("V2: from the complex-cell maps OF PHOTOGRAPHS ...")
    v2_patches = _sample_patches(ims, [complex_], k=5, per_image=8, seed=4)
    V2 = SpikingConvLayer(complex_.n_orient, n_v2, 5, name="V2", lr=0.03, seed=5)
    V2.train(v2_patches, epochs=5, init="kmeans")
    pool = SpikingPool(2, name="pool")

    say("V4: over pooled V2 OF PHOTOGRAPHS ...")
    v4_patches = _sample_patches(ims, [complex_, V2, pool], k=3, per_image=8,
                                 seed=8)
    V4 = SpikingConvLayer(n_v2, n_v4, 3, name="V4", lr=0.03, seed=9)
    V4.train(v4_patches, epochs=5, init="kmeans")

    hierarchy = VisionHierarchy().add(complex_).add(V2).add(pool).add(V4)
    say(f"   V1 {len(v1_patches)} patches, V2 {len(v2_patches)}, "
        f"V4 {len(v4_patches)}")
    return VentralStream(V1, complex_, V2, pool, V4, hierarchy, size=size)
