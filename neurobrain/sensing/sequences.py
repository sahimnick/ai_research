"""Consecutive views of one object -- the input a predictive rule needs.

Everything this project has trained a filter bank on was a **static, independent
image**. That is the condition under which
[Halvagal & Zenke, Nat Neuro 2023] report that Hebbian plasticity fails to
produce invariant object representations, and it is the condition under which
this project measured exactly that: eleven interventions over cluster AUC
0.540-0.630 against the ear's 0.788.

This module supplies the missing input. An object is followed as it travels,
turns and looms, and the crop around it is kept frame by frame -- so consecutive
entries are *the same thing seen differently*, which is the only fact a
predictive term needs and is available without any label.

The tracker is `MovingScene.nearest`, the same proximity pursuit
`benchmarks/report_vision.py` already uses for its smooth-pursuit panel. What is
under test here is the **learning rule**, not the tracker; pursuit is machinery
this project already had.
"""
from typing import List, Optional

import numpy as np

from .streams import MovingScene


def _crop(canvas: np.ndarray, centre, crop: int) -> np.ndarray:
    """A ``crop`` x ``crop`` window, zero-padded where it leaves the canvas.

    Padded rather than clipped back inside, for the reason
    `LocalSpatialEye.patch_images` gives: clipping would quietly re-centre the
    window for exactly the objects near an edge, which are the ones whose
    motion the sequence is trying to capture.
    """
    h, w = canvas.shape
    r, c = int(round(centre[0])), int(round(centre[1]))
    half = crop // 2
    out = np.zeros((crop, crop), np.float32)
    y0, x0 = max(r - half, 0), max(c - half, 0)
    y1, x1 = min(r - half + crop, h), min(c - half + crop, w)
    if y1 > y0 and x1 > x0:
        out[y0 - (r - half):y1 - (r - half),
            x0 - (c - half):x1 - (c - half)] = canvas[y0:y1, x0:x1]
    return out


def object_sequences(scene: MovingScene, n_seq: int = 64, len_seq: int = 6,
                     crop: int = 28, seed: int = 0) -> List[np.ndarray]:
    """``n_seq`` arrays of shape ``(len_seq, crop, crop)``.

    Each is one object followed across ``len_seq`` frames. **No label is read.**
    Which frames belong together comes from temporal adjacency alone, which is
    the claim being tested rather than an input to it.
    """
    rng = np.random.default_rng(seed)
    out: List[np.ndarray] = []
    for _ in range(int(n_seq)):
        here = scene.scene_now()
        if not here.positions:
            break
        tgt = here.positions[int(rng.integers(len(here.positions)))]
        frames = []
        for t in range(int(len_seq)):
            canvas = scene.render() if t == 0 else scene.step()
            near = scene.nearest(int(tgt[0]), int(tgt[1]))
            if near is not None:
                tgt = (int(round(near.row)), int(round(near.col)))
            frames.append(_crop(canvas, tgt, crop))
        out.append(np.stack(frames))
    return out


def shuffle_across(seqs: List[np.ndarray],
                   seed: int = 0) -> List[np.ndarray]:
    """The control that decides the experiment. Same frames, same count, same
    rule -- but "consecutive" no longer means "the same object".

    The obvious control is to shuffle *within* each sequence, and it is the
    wrong one: an object moves slowly, so frames 1 and 5 of one sequence are
    still the same object seen slightly differently. Shuffling within would
    leave the predictive term almost all of the signal it is being credited
    with, and it would pass whether or not temporal continuity mattered.

    So every frame from every sequence is pooled, shuffled globally, and dealt
    back into sequences of the same shape. Now a "consecutive" pair is two
    unrelated objects, and a predictive term that still produces invariance is
    not producing it from temporal continuity.
    """
    if not seqs:
        return []
    flat = np.concatenate([s for s in seqs], axis=0)
    idx = np.random.default_rng(seed).permutation(len(flat))
    flat = flat[idx]
    n, L = len(seqs), seqs[0].shape[0]
    return [flat[k * L:(k + 1) * L] for k in range(n)]


def sequence_stats(seqs: List[np.ndarray]) -> dict:
    """How much a sequence actually changes -- asserted, never assumed.

    A sequence whose frames are nearly identical tests nothing: the predictive
    term would have nothing to pull together and would trivially look like it
    worked. `phase10_2.py` and `limiting_factor.py` were both flattered by a
    transformation that silently did not happen, so the amount of motion is
    reported beside every result that depends on it.
    """
    if not seqs:
        return {"n": 0}
    adj, far = [], []
    for s in seqs:
        f = s.reshape(len(s), -1)
        f = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-9)
        adj.extend(float(f[t] @ f[t + 1]) for t in range(len(f) - 1))
        far.append(float(f[0] @ f[-1]))
    cross = []
    rng = np.random.default_rng(0)
    for _ in range(min(200, len(seqs) * 4)):
        i, j = int(rng.integers(len(seqs))), int(rng.integers(len(seqs)))
        if i == j:
            continue
        a = seqs[i][int(rng.integers(len(seqs[i])))].ravel()
        b = seqs[j][int(rng.integers(len(seqs[j])))].ravel()
        cross.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
    return {"n": len(seqs),
            "adjacent_cos": float(np.mean(adj)),
            "endpoints_cos": float(np.mean(far)),
            "across_objects_cos": float(np.mean(cross)) if cross else float("nan")}
