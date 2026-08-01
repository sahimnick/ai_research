"""A second visual pathway: local receptive fields in an object-centred frame.

Why a new pathway and not another repair
----------------------------------------
The eye's defect has one description that survives every test: **two
photographs of the same thing do not land near each other**. Cluster AUC sits at
0.578 against the ear's 0.788, and ten interventions have moved it over the
range 0.540-0.605 -- width, aperture, depth, whitening, resolution, capacity,
concept feedback under two rules, filter tying, fixation choice, second-order
spatial statistics. Every one of them operated on a representation that was
already inadequate, which is why `EVALUATION.md` records them as a convergent
negative rather than ten separate ones.

Two properties of the existing pathway look responsible, and they are properties
of its *construction*:

**It encodes globally.** One `WideV1` tiles the whole 32x32 frame and the code
is the concatenation of every cell at every position. Two photographs of a cat
share cat-shaped *parts* but almost never share which absolute pixel those parts
sit at, so the codes have little in common however good the filters are.

**Position is absolute.** `translation.py` measured a 5-px shift taking
concept-waking from 0.290 to **0.021**, below chance. The frame is the origin,
so moving the object moves everything about the code.

This module keeps neither. The image is cut into overlapping 8x8 patches, a
feature extractor runs on each one, and each local feature vector is tagged with
where it sits **relative to the object's own centre** rather than to the frame.
Translation then cancels exactly: shift the image and the centroid shifts with
it, leaving every relative coordinate unchanged.

That is an object-centred reference frame, which is what parietal cortex and IT
are usually described as building, and it is the one thing `relational_code`
approximated without ever locating an object: it summed over *all* relative
displacements, which is invariant but says nothing about where a part sits in
the thing it belongs to.

The hypotheses, written before the measurements
-----------------------------------------------
**H1 (invariance).** Object-centred local coding will hold same-object
similarity under a +-5 px shift: centred-minus-shifted gap below 0.05, against
0.254 for the global code.

**H2 (the one that decides it).** It will *cluster* better than the global code:
**cluster AUC above 0.605 by at least 0.02, with d >= 0.8 and >= 75% of seeds.**
This is the pre-registered rejection bar. Invariance alone is not a result --
plain pooling already achieves it (gap 0.029) and scores 0.540, and
`relational_code` achieves it better (0.005) and scores 0.560. A third invariant
code that fails to cluster would be the same negative a third time.

**H3 (attribution).** The gain, if any, comes from the **object-centred frame**,
not from patching. Ablation: identical patches with frame-absolute coordinates.

**H4 (attribution).** It comes from **local** receptive fields, not from the
position binning. Ablation: the global `WideV1` with the same binning applied.

**H5 (representation, not read-out).** It improves the *code*, so the gain must
appear in cluster AUC -- a quantity computed from raw codes with no learning
anywhere -- and not only in downstream accuracy. A gain that appears only
downstream means the concept layer adapted, not that the representation
improved, and requirement 11 rejects it.

Every component is switchable so each hypothesis has its own control:
``local`` (patches vs one global bank), ``centred`` (object-centred vs absolute),
``bins`` (position resolution; 1 collapses to a bag of features), and
``shared_bank`` (one filter bank at every patch vs an independent bank per
patch).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .widev1 import WideV1, _unit


class LocalSpatialEye:
    """Local features in an object-centred frame.

    The code is an outer product between **what** was seen and **where it sits
    in the object**, accumulated over patches:

        C[f, b] = sum over patches of  A[f, patch] * B[b, patch]

    ``A`` is the local feature activity of a patch and ``B`` is a soft binning
    of that patch's offset from the object's centroid. Summing over patches is
    what makes the code a fixed size regardless of how many patches fire, and
    the bin index is what stops it collapsing into a bag of features -- a nose
    above a mouth and a nose below one produce different codes.

    Nothing here is trained by this class. The feature bank is a
    :class:`WideV1`, developed by the project's existing competitive Hebbian
    rule if the caller wants; the binning is fixed geometry; the outer product
    is a product. Learning happens in the concept layer downstream, exactly as
    it does for the existing pathway, so the two can be compared on equal terms.

    Parameters
    ----------
    patch, stride:
        Local receptive field and its step. 8 and 4 on a 32x32 frame gives 7x7
        = 49 overlapping patches, each seeing 1/16 of the area.
    bins:
        Resolution of the relative-position code, per axis. ``bins=1`` removes
        position entirely and the code becomes a bag of local features, which is
        the ablation that shows what geometry is worth.
    centred:
        Measure each patch's offset from the object's centroid (True) or from
        the frame centre (False). The False case is the control for H3: the same
        local features, the same binning, and no object-centred frame.
    local:
        Run the bank on each patch (True) or once on the whole frame (False).
        The False case is the control for H4.
    shared_bank:
        One filter bank applied at every patch, or an independent bank per
        patch. Shared is the biological default -- a receptive field type is not
        re-invented at every retinal location -- and independent is what a
        literal reading of "an extractor per patch" gives, so both are here.
    resample:
        **Sample the patches in the object's frame**, not just label them in it.
        This distinction was caught by the first smoke test and it is the whole
        idea: with the patch grid nailed to the image, a shift smaller than
        ``stride`` moves different pixels into different patches, so the code
        changes even though every offset was measured from the centroid.
        Measured on a synthetic object, labelling alone gave cos 0.849 at 2 px,
        0.890 at 5 px and 0.996 at 8 px -- perfect only where the shift happened
        to be a whole number of strides. With ``resample`` the grid is laid out
        from the centroid, so the same pixels fall in the same patch whatever
        the object's position, and the invariance stops depending on alignment
        luck. Kept switchable because the difference between the two is exactly
        the claim.
    """

    def __init__(self, image_shape: Tuple[int, int] = (32, 32),
                 patch: int = 8, stride: int = 4, n_cells: int = 128,
                 rf: int = 3, bins: int = 3, centred: bool = True,
                 local: bool = True, shared_bank: bool = True,
                 resample: bool = True, kwta: float = 0.25,
                 spiking: bool = True, seed: int = 0):
        self.H, self.W = image_shape
        self.patch, self.stride = int(patch), int(stride)
        self.bins, self.centred, self.local = int(bins), bool(centred), bool(local)
        self.shared_bank = bool(shared_bank)
        self.resample = bool(resample)
        self.kwta = float(kwta)
        # `WideV1.rate` runs a 50 ms Izhikevich simulation per patch, and this
        # pathway calls it 121 times per image on a 48 px frame. The project has
        # already measured what that buys on STATIC identity -- see
        # `WideV1.rate`'s own docstring: 1-NN **0.322 against 0.323** for
        # `drive`, the same filters with the neuron model removed, at **18x the
        # wall clock**. Identical to the third decimal, eighteen times the cost.
        #
        # So `spiking=False` is not a shortcut around the project's spiking
        # commitment; it is that commitment applied where it was measured to
        # pay. It pays on movement (4-way direction 72.0% against a 28.5% static
        # control) and this is a still-image benchmark. Any arm that uses it
        # must be compared only against arms that also use it.
        self.spiking = bool(spiking)
        ys = list(range(0, self.H - self.patch + 1, self.stride))
        xs = list(range(0, self.W - self.patch + 1, self.stride))
        self.origins = np.array([(y, x) for y in ys for x in xs], int)
        self.n_patch = len(self.origins)
        #: patch centres in pixels, which is what offsets are measured from
        self.centres = self.origins + self.patch / 2.0

        if self.local:
            n = 1 if self.shared_bank else self.n_patch
            self.banks = [WideV1(image_shape=(self.patch, self.patch),
                                 n_cells=int(n_cells), rf=int(rf), stride=1,
                                 window_ms=50, motion_fraction=0.0,
                                 seed=seed + i) for i in range(n)]
        else:
            # The H4 control needs the SAME number of features per place as the
            # local arm, or it is handicapped rather than ablated. A global bank
            # at the patch stride has far more places than a patch does, so the
            # cell count must scale with them: 128 cells over 144 positions is
            # under one per hypercolumn and `develop_v1` refuses to run a
            # competition with nothing to compete. The stride is widened so the
            # number of places stays comparable to the patch grid, and the cell
            # count is n_feat per place.
            g_stride = max(self.stride, 8)
            probe = WideV1(image_shape=image_shape, n_cells=8, rf=int(rf),
                           stride=g_stride, window_ms=50,
                           motion_fraction=0.0, seed=seed)
            self.banks = [WideV1(image_shape=image_shape,
                                 n_cells=int(n_cells) * probe.n_pos,
                                 rf=int(rf), stride=g_stride, window_ms=50,
                                 motion_fraction=0.0, seed=seed)]
        self.n_feat = (self.banks[0].n_cells if self.local
                       else int(n_cells))
        self.dim = self.n_feat * (self.bins * self.bins)

    # -- the pieces, each testable on its own --------------------------------
    def image_centroid(self, image: np.ndarray) -> np.ndarray:
        """Where the object is, from local contrast in the pixels.

        Needed before any patch is cut, so it cannot come from the features --
        the features are what the patches produce. Local *contrast* rather than
        brightness, because a bright background would otherwise drag the origin
        off the object; contrast is what marks where there is something at all.
        """
        a = np.asarray(image, np.float32)
        gy, gx = np.gradient(a)
        w = np.abs(gy) + np.abs(gx)
        t = float(w.sum())
        if t < 1e-9:
            return np.array([self.H / 2.0, self.W / 2.0], np.float32)
        ys, xs = np.mgrid[0:a.shape[0], 0:a.shape[1]]
        return np.array([float((w * ys).sum() / t),
                         float((w * xs).sum() / t)], np.float32)

    def grid(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Patch origins and centres, laid out in whichever frame is in use."""
        if not (self.resample and self.centred):
            return self.origins, self.centres
        c = self.image_centroid(image)
        rel = self.centres - np.array([self.H / 2.0, self.W / 2.0], np.float32)
        centres = rel + c
        origins = np.rint(centres - self.patch / 2.0).astype(int)
        return origins, centres

    def patch_images(self, image: np.ndarray,
                     origins: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """Cut the patches, zero-padding wherever the grid leaves the frame.

        Padding rather than clipping: clipping would silently move a patch back
        inside the image and undo the object-centred sampling for exactly the
        objects nearest the edge, which are the ones the shift test creates.
        """
        a = np.asarray(image, np.float32)
        if origins is None:
            origins = self.origins
        h, w = a.shape
        out = []
        for y, x in origins:
            p = np.zeros((self.patch, self.patch), np.float32)
            y0, x0 = max(int(y), 0), max(int(x), 0)
            y1, x1 = min(int(y) + self.patch, h), min(int(x) + self.patch, w)
            if y1 > y0 and x1 > x0:
                p[y0 - int(y):y1 - int(y), x0 - int(x):x1 - int(x)] = \
                    a[y0:y1, x0:x1]
            out.append(p)
        return out

    def features(self, image: np.ndarray) -> np.ndarray:
        """(n_feat, n_patch) -- what each patch sees, sparsened.

        kWTA per patch is the competitive step and it is not decoration: without
        it every patch returns a dense response to whatever contrast it
        contains, the outer product below sums those, and the code becomes an
        image-energy map that says nothing about which features occurred.
        """
        origins, _ = self.grid(image)
        if self.local:
            out = np.empty((self.n_feat, self.n_patch), np.float32)
            for k, p in enumerate(self.patch_images(image, origins)):
                bank = self.banks[0] if self.shared_bank else self.banks[k]
                out[:, k] = bank.rate(p) if self.spiking else bank.drive(p)
        else:
            # the H4 control: one global bank, its cells regrouped by the
            # position they already look at, so the binning below sees the same
            # kind of (feature, place) table without any local extraction
            r = (self.banks[0].rate(image) if self.spiking
                 else self.banks[0].drive(image))
            per = max(self.banks[0].n_cells // self.banks[0].n_pos, 1)
            g = np.zeros((self.n_feat, self.n_patch), np.float32)
            pos = self.banks[0].anchors + self.banks[0].rf / 2.0
            for k, c in enumerate(self.centres):
                near = np.argmin(((pos - c) ** 2).sum(1))
                cells = np.flatnonzero(self.banks[0].cell_pos == near)
                g[:len(cells), k] = r[cells][:self.n_feat]
            out = g
        if self.kwta > 0:
            keep = max(int(round(self.kwta * out.shape[0])), 1)
            for k in range(out.shape[1]):
                col = out[:, k]
                if keep < len(col):
                    cut = np.partition(col, -keep)[-keep]
                    col[col < cut] = 0.0
        return out

    def centroid(self, A: np.ndarray) -> np.ndarray:
        """Where the object is, from the mind's own activity.

        Total feature energy per patch is the weight, so the origin follows what
        the eye is responding to rather than what a label says. With no activity
        anywhere it falls back to the frame centre, which keeps the code defined
        for a blank image instead of returning a division by zero.
        """
        if not self.centred:
            return np.array([self.H / 2.0, self.W / 2.0], np.float32)
        w = A.sum(0)
        t = float(w.sum())
        if t < 1e-9:
            return np.array([self.H / 2.0, self.W / 2.0], np.float32)
        return (w @ self.centres / t).astype(np.float32)

    def position_code(self, origin: np.ndarray) -> np.ndarray:
        """(bins*bins, n_patch) -- soft one-hot over each patch's offset.

        Offsets are scaled by the frame's half-size so the binning does not
        depend on the image dimensions, and bilinear weights keep the code
        continuous: a patch halfway between two bins contributes to both, so a
        one-pixel move does not flip a bin and jump the code.
        """
        if self.bins <= 1:
            return np.ones((1, self.n_patch), np.float32)
        d = (self.centres - origin) / (0.5 * max(self.H, self.W))
        u = np.clip((d + 1.0) * 0.5, 0.0, 1.0) * (self.bins - 1)
        out = np.zeros((self.bins * self.bins, self.n_patch), np.float32)
        for k in range(self.n_patch):
            fy, fx = u[k]
            y0, x0 = int(np.floor(fy)), int(np.floor(fx))
            ay, ax = fy - y0, fx - x0
            for dy, wy in ((0, 1 - ay), (1, ay)):
                for dx, wx in ((0, 1 - ax), (1, ax)):
                    yy, xx = min(y0 + dy, self.bins - 1), min(x0 + dx,
                                                              self.bins - 1)
                    out[yy * self.bins + xx, k] += wy * wx
        return out

    def code(self, image: np.ndarray) -> np.ndarray:
        """The pathway, end to end: what, and where in the object."""
        A = self.features(image)
        if self.resample and self.centred:
            # the grid was already laid out from the centroid, so each patch's
            # offset is its fixed place in the object and needs no re-measuring
            origin = np.array([self.H / 2.0, self.W / 2.0], np.float32)
        else:
            origin = self.centroid(A)
        B = self.position_code(origin)
        return _unit((A @ B.T).reshape(-1))

    def develop(self, images: Sequence[np.ndarray], epochs: int = 3,
                seed: int = 0) -> "LocalSpatialEye":
        """Grow the local filters from patches, by the project's own rule.

        A shared bank sees patches from every location, which is the whole point
        of sharing: a filter type is learned from far more examples than any one
        retinal position could provide.
        """
        from ..learning.selforganize import develop_v1
        if self.shared_bank or not self.local:
            pool: List[np.ndarray] = []
            for im in images:
                pool.extend(self.patch_images(im) if self.local else [im])
            develop_v1(self.banks[0], pool, epochs=epochs, tie=True, seed=seed)
        else:
            for k, bank in enumerate(self.banks):
                develop_v1(bank, [self.patch_images(im)[k] for im in images],
                           epochs=epochs, tie=True, seed=seed + k)
        return self
