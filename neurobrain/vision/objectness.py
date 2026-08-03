"""Regions worth looking at, and what the areas are tuned to -- without classes.

Three things the eye could not previously report, built from what it already
computes. Each is honest about what it is **not**:

`propose`
    Boxes on things the eye finds distinctive, whether or not they are tagged.
    This is **objectness, not detection**: it says "this region stands out from
    its surroundings", never "this is a person". There are no categories here
    and no labels to give -- naming a proposal would require a classifier
    trained on labelled objects, which this project does not have.

`segment`
    Regions of similar feature response, by clustering the area map's own cell
    vectors. This is **feature segmentation** -- closer to superpixels than to
    YOLO. YOLO's masks are *instance* masks from a network trained on labelled
    instances; these are unlabelled groupings of similar texture, and a single
    object will often be split across several of them.

`feature_report`
    What each area's channels are actually tuned to, measured by probing them
    with oriented gratings rather than asserted from the architecture, plus
    which channels a given frame drives hardest.

All three are computed from the same maps a single forward pass already
produced, so none of them costs another pass.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def distinctiveness(m: np.ndarray) -> np.ndarray:
    """How unlike the rest of the picture each location is.

    Cosine distance between a cell's channel vector and the map's mean vector,
    weighted by that cell's energy. A location that is merely bright is not
    interesting; a location whose *pattern* of features is unusual is.
    """
    m = np.asarray(m, np.float32)
    c, gh, gw = m.shape
    v = m.reshape(c, -1).T
    n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    u = v / n
    mu = u.mean(0)
    mu = mu / (np.linalg.norm(mu) + 1e-9)
    d = 1.0 - (u @ mu)
    e = n[:, 0]
    e = e / (e.max() + 1e-9)
    return (d * e).reshape(gh, gw).astype(np.float32)


def propose(m: np.ndarray, img_size: int, k: int = 6,
            box: float = 44.0, min_sep: float = 30.0) -> List[dict]:
    """Up to ``k`` distinctive regions, as pixel boxes with a score.

    Greedy peak-picking on :func:`distinctiveness` with a minimum separation, so
    the same blob is not returned six times. Scores are normalised to the
    strongest peak in this frame and are **not** comparable across frames.

    Returns ``[{"y","x","h","w","score"}]``. No labels: see the module note.
    """
    d = distinctiveness(m)
    gh, gw = d.shape
    work = d.copy()
    out = []
    rad_r = max(1, int(min_sep * gh / img_size))
    rad_c = max(1, int(min_sep * gw / img_size))
    peak = float(d.max()) + 1e-9
    for _ in range(int(k)):
        r, c = np.unravel_index(int(np.argmax(work)), work.shape)
        if work[r, c] <= 0:
            break
        out.append({"y": float((r + 0.5) * img_size / gh),
                    "x": float((c + 0.5) * img_size / gw),
                    "h": float(box), "w": float(box),
                    "score": float(d[r, c] / peak)})
        work[max(0, r - rad_r):r + rad_r + 1,
             max(0, c - rad_c):c + rad_c + 1] = -1.0
    return out


def segment(m: np.ndarray, k: int = 5, iters: int = 12,
            seed: int = 0) -> np.ndarray:
    """Group map cells by what they respond to. Returns a ``(gh, gw)`` labelling.

    k-means on unit-normalised channel vectors -- so cells are grouped by the
    *pattern* of features rather than by how strongly they fire. Not instance
    segmentation: one object may occupy several labels and one label may span
    several objects.
    """
    m = np.asarray(m, np.float32)
    c, gh, gw = m.shape
    v = m.reshape(c, -1).T
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    rng = np.random.default_rng(seed)
    cent = v[rng.choice(len(v), size=min(k, len(v)), replace=False)].copy()
    lab = np.zeros(len(v), np.int32)
    for _ in range(iters):
        sim = v @ cent.T
        new = np.argmax(sim, 1).astype(np.int32)
        if np.array_equal(new, lab):
            break
        lab = new
        for j in range(len(cent)):
            sel = v[lab == j]
            if len(sel):
                cent[j] = sel.mean(0) / (np.linalg.norm(sel.mean(0)) + 1e-9)
    return lab.reshape(gh, gw)


def _grating(size: int, theta: float, freq: float = 0.18) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    y -= size / 2
    x -= size / 2
    return (0.5 + 0.5 * np.cos(2 * np.pi * freq
                               * (x * np.cos(theta) + y * np.sin(theta))))


def feature_report(stream, areas: Sequence[str], img_size: int = 224,
                   n_orient: int = 8, top: int = 6,
                   frame: Optional[np.ndarray] = None) -> Dict[str, dict]:
    """What each area's channels are tuned to -- **probed**, not asserted.

    Every channel is shown oriented gratings at ``n_orient`` angles. Its
    preferred angle is the one it answers loudest to, and its *selectivity* is
    how much louder -- ``1 - min/max`` across angles, so 0 means the channel
    does not care about orientation at all and 1 means it answers to only one.

    Calling an area "edge detectors" because it is second in a stack is an
    assertion. This is the measurement, and it disagrees with the assertion
    often enough to be worth making.

    With ``frame``, also reports which channels that particular picture drives.
    """
    h = stream.hierarchy
    names = [getattr(l, "name", type(l).__name__) for l in h.layers]
    thetas = np.linspace(0, np.pi, n_orient, endpoint=False)
    resp = {a: [] for a in areas}
    for th in thetas:
        h.forward(_grating(img_size, float(th))[None])
        for a in areas:
            m = np.asarray(h.layers[names.index(a)].log["output"], np.float32)
            resp[a].append(m.reshape(m.shape[0], -1).mean(1))
    out = {}
    for a in areas:
        R = np.stack(resp[a])                       # (n_orient, channels)
        mx, mn = R.max(0), R.min(0)
        pref = np.degrees(thetas[np.argmax(R, 0)])
        sel = 1.0 - mn / (mx + 1e-9)
        order = np.argsort(-mx)[:top]
        out[a] = {
            "channels": int(R.shape[1]),
            "mean_selectivity": round(float(sel.mean()), 4),
            "oriented_fraction": round(float((sel > 0.5).mean()), 4),
            "top": [{"channel": int(i), "prefers_deg": round(float(pref[i]), 1),
                     "selectivity": round(float(sel[i]), 3),
                     "kind": ("oriented" if sel[i] > 0.5 else
                              "broadly tuned" if sel[i] > 0.2 else
                              "unselective")}
                    for i in order]}
    if frame is not None:
        h.forward(np.asarray(frame, np.float32)[None])
        for a in areas:
            m = np.asarray(h.layers[names.index(a)].log["output"], np.float32)
            drive = m.reshape(m.shape[0], -1).mean(1)
            idx = np.argsort(-drive)[:top]
            out[a]["driven_by_this_frame"] = [
                {"channel": int(i), "response": round(float(drive[i]), 4)}
                for i in idx]
    return out
