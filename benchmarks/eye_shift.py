"""Shift invariance on real frames -- the test where a retinotopic eye should win.

§9.19 found the live-camera identity test **saturated**: 23 traffic cameras
point at 23 different streets, so raw downsampled pixels re-identify every one
of them (1.000) exactly as the four-stage eye does (1.000). A test both arms
pass perfectly cannot separate them, and nothing in it supports the eye.

This is the same question asked where the two arms *must* come apart. Translate
the scene a few pixels and raw pixels fall off a cliff -- a pixel vector is not
shift invariant at all, it is the one thing it is worst at. A convolutional,
retinotopic hierarchy with two pooling stages is supposed to be exactly the
machine that survives this. That is what V1 complex cells and `SpikingPool` are
*for*.

So this is a falsifiable claim about the architecture rather than about the
data:

**H20.** The four-stage eye's code survives translation better than raw pixels
do, and the deeper stage (V4, two pools down) survives better than V2.

**H20 is falsified if the eye tracks the pixels down.** Then its pooling is not
buying invariance, and the four stages are an expensive way to compute
something a crop already computes -- which would be a structural finding about
this hierarchy, not a training-set complaint.

Real translation, not padding
-----------------------------
The window is cropped out of a **larger native frame at different offsets**, so
a shift genuinely moves the scene across the retina and new content enters at
the edge, exactly as it does when a camera pans. Zero-padding a shifted image
would instead measure how the eye responds to a black bar.

Scored two ways, because §9.19 showed one of them can saturate:

    similarity   cosine between the shifted code and the unshifted one
    identity     does the shifted frame still match its OWN camera among all
                 cameras? (chance 1/n)

Both arms go through one adaptation fitted on the unshifted frames and then
frozen -- the fix §9.19 had to make before any of these numbers meant anything.

The resolution control
----------------------
A first run had one pixel arm, subsampled to 32x32, against V2's 87x87 map and
V4's 41x41 map. The eye lost badly -- but a code held on a coarser grid is
*automatically* more shift tolerant, because the same 4px translation is a
smaller fraction of one cell. That confound would explain the whole result
without any statement about the hierarchy, so pixel arms are also built at each
eye stage's **own grid**. If matched-resolution pixels fall exactly as fast as
the eye, the eye's collapse is explained entirely by what resolution it holds
its code at -- and the finding sharpens from "the eye is bad" to "the eye never
leaves retinotopy", which is a specific structural claim about these four
stages.

Usage:  python3 benchmarks/eye_shift.py out_eye_shift.json [n_cam]
"""
import json
import sys

import numpy as np

from neurobrain.cognition.multimodal import _unit
from neurobrain.sensing.live import CctvCamera, NY511_LIVE_IDS, SensorUnavailable
from neurobrain.vision.ventral import (build_ventral_stream_on,
                                       code_participation)
from neurobrain.vision.widev1 import PopulationAdaptation

N_CAM = int(sys.argv[2]) if len(sys.argv) > 2 else 24
WINDOW = 192                       # the crop the eye actually sees
SHIFTS = (0, 4, 8, 16, 32, 48)     # pixels of real translation
STAGES = ("V2", "V4")


def grab_full(cam, n_want):
    """Whole native frames, big enough that a WINDOW crop can be moved."""
    out = []
    need_w = WINDOW + max(SHIFTS)
    for i in range(len(cam.urls)):
        if len(out) >= n_want:
            break
        try:
            r = cam.read(i)
        except SensorUnavailable:
            continue
        if not getattr(r, "live", False):
            continue
        a = np.asarray(r.data, np.float32)
        if a.ndim != 3 or a.shape[0] < WINDOW or a.shape[1] < need_w:
            continue
        out.append(a.transpose(2, 0, 1))
    return out


def crop(f, dx):
    """The WINDOW-sized view, moved dx pixels across the native frame."""
    _, h, w = f.shape
    y0 = (h - WINDOW) // 2
    x0 = (w - WINDOW) // 2 + dx - max(SHIFTS) // 2
    x0 = int(np.clip(x0, 0, w - WINDOW))
    return f[:, y0:y0 + WINDOW, x0:x0 + WINDOW]


def eye_raw(stream, views, stages):
    """Every requested area from **one** forward pass per view.

    Calling this once per stage would run the whole hierarchy twice to read two
    of its own layers -- the deeper stage is computed on the way to itself
    either way.
    """
    h = stream.hierarchy
    names = [getattr(l, "name", type(l).__name__) for l in h.layers]
    ks = [names.index(s) for s in stages]
    out = {s: [] for s in stages}
    grids = {}
    for v in views:
        h.forward(np.asarray(v, np.float32).mean(0)[None] / 255.0)
        for s, k in zip(stages, ks):
            a = np.asarray(h.layers[k].log["output"], np.float32)
            grids[s] = a.shape[-2:]            # the map this stage holds
            out[s].append(a.ravel())
    return {s: np.stack(out[s]) for s in stages}, grids


def pixel_grid(views, g):
    """Luminance area-averaged onto a g x g grid -- the same map the eye holds.

    Area averaging, not strided subsampling: a stride would alias, and the
    question here is what resolution costs, not what aliasing costs.
    """
    P = []
    for v in views:
        a = np.asarray(v, np.float32).mean(0) / 255.0
        h, w = a.shape
        ph, pw = h // g, w // g
        a = a[:g * ph, :g * pw].reshape(g, ph, g, pw).mean((1, 3))
        P.append(a.ravel())
    return np.stack(P)


def fit_adapt(M):
    ad = PopulationAdaptation(M.shape[1])
    for r in M:
        ad.observe(r)
    return ad


def apply_adapt(ad, M):
    return np.array([_unit(ad(r, learn=False)) for r in M], np.float32)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_eye_shift.json"
    cam = CctvCamera(ids=list(NY511_LIVE_IDS))
    print(f"reaching for {N_CAM} live cameras (need >= "
          f"{WINDOW + max(SHIFTS)}px wide) ...", flush=True)
    frames = grab_full(cam, N_CAM)
    if len(frames) < 8:
        print(f"only {len(frames)} usable cameras -- nothing reported.")
        json.dump({"error": "too few cameras", "n": len(frames)},
                  open(out_path, "w"), indent=1)
        return
    print(f"  {len(frames)} cameras, native {frames[0].shape}, "
          f"window {WINDOW}px, shifts {SHIFTS}", flush=True)

    base = [crop(f, 0) for f in frames]
    # the shift must actually move the scene -- a clipped crop would silently
    # compare a frame with itself, which is the trap §9.8's 64px frame fell in
    moved = float(np.mean([np.abs(crop(f, SHIFTS[-1]).astype(np.float32)
                                  - crop(f, 0)).mean() for f in frames]))
    print(f"  largest shift changes the pixels by {moved:.2f}/255", flush=True)
    if moved < 1.0:
        print("  the crop did not move -- REFUSING to report.")
        json.dump({"error": "crop did not move", "delta": moved},
                  open(out_path, "w"), indent=1)
        return

    print("\ngrowing the eye on the unshifted views ...", flush=True)
    lum = build_ventral_stream_on(np.stack([b.mean(0) for b in base]),
                                  size=WINDOW, verbose=False)

    # one pass over every shift, reading all arms out of the same forward pass
    R0, grids = eye_raw(lum, base, STAGES)
    # pixels at 32x32 (as first published) and at each eye stage's OWN grid
    pix_g = {"pixels-32": 32}
    for s in STAGES:
        pix_g[f"pixels@{s}grid"] = int(grids[s][0])
    for a, g in pix_g.items():
        R0[a] = pixel_grid(base, g)
    print("  eye grids " + ", ".join(f"{s} {grids[s][0]}x{grids[s][1]}"
                                     for s in STAGES), flush=True)

    ad = {k: fit_adapt(v) for k, v in R0.items()}
    V0 = {k: apply_adapt(ad[k], v) for k, v in R0.items()}
    names_arm = [f"eye-{s}" for s in STAGES] + list(pix_g)
    key = {f"eye-{s}": s for s in STAGES}
    key.update({a: a for a in pix_g})

    n = len(frames)
    sims = {a: [] for a in names_arm}
    ids = {a: [] for a in names_arm}
    for dx in SHIFTS:
        views = [crop(f, dx) for f in frames]
        R, _ = eye_raw(lum, views, STAGES)
        for a, g in pix_g.items():
            R[a] = pixel_grid(views, g)
        for a in names_arm:
            k = key[a]
            V = apply_adapt(ad[k], R[k])
            sims[a].append(float(np.mean(np.sum(V0[k] * V, 1))))
            S = V @ V0[k].T
            ids[a].append(float(np.mean(np.argmax(S, 1) == np.arange(n))))
        print(f"  +{dx:>3}px  " + "  ".join(
            f"{a} {sims[a][-1]:.3f}" for a in names_arm), flush=True)

    res = {"n_cameras": n, "window": WINDOW, "shifts": list(SHIFTS),
           "pixel_delta_at_max_shift": round(moved, 3), "arms": {}}
    for a in names_arm:
        k = key[a]
        res["arms"][a] = {"similarity": [round(s, 4) for s in sims[a]],
                          "identity": [round(i, 4) for i in ids[a]],
                          "participation": round(code_participation(V0[k]), 2),
                          "dim": int(R0[k].shape[1])}

    print(f"\n{'arm':<16}{'dim':>8}" + "".join(f"{'+' + str(d):>8}"
                                               for d in SHIFTS))
    for name in names_arm:
        a = res["arms"][name]
        print(f"{name:<16}{a['dim']:>8}"
              + "".join(f"{s:>8.3f}" for s in a["similarity"]))
    print(f"\nidentity (chance {1.0/n:.3f})")
    for name in names_arm:
        a = res["arms"][name]
        print(f"{name:<16}{'':>8}"
              + "".join(f"{s:>8.3f}" for s in a["identity"]))

    sim = {a: res["arms"][a]["similarity"] for a in names_arm}
    px = sim["pixels-32"][-1]
    v2, v4 = sim["eye-V2"][-1], sim["eye-V4"][-1]
    res["eye_beats_pixels"] = bool(max(v2, v4) > px + 0.05)
    # pre-registered, and evaluated at the point where BOTH stages have already
    # collapsed -- kept as declared, but re-read below where either is alive
    res["deeper_more_invariant"] = bool(v4 > v2 + 0.02)
    i1 = 1                                     # the smallest non-zero shift
    res["deeper_more_invariant_small_shift"] = bool(
        sim["eye-V4"][i1] > sim["eye-V2"][i1] + 0.02)
    res["at_max_shift"] = {"eye_V2": v2, "eye_V4": v4, "pixels_32": px}

    print(f"\n--- H20, at the largest shift (+{SHIFTS[-1]}px) ---")
    print(f"  pixels-32 {px:.3f}   eye V2 {v2:.3f}   eye V4 {v4:.3f}")
    print(f"  the eye survives translation better than pixels: "
          f"{res['eye_beats_pixels']}")
    print(f"  the deeper stage survives better, at max shift: "
          f"{res['deeper_more_invariant']}")
    print(f"  ... and at +{SHIFTS[i1]}px, where both are still above floor: "
          f"{res['deeper_more_invariant_small_shift']} "
          f"(V4 {sim['eye-V4'][i1]:.3f} vs V2 {sim['eye-V2'][i1]:.3f})")

    # is the eye's collapse just the resolution it holds its code at?
    print("\n--- the resolution control: pixels on the eye's own grid ---")
    gap = {}
    for s in STAGES:
        g = f"pixels@{s}grid"
        d = [sim[g][j] - sim[f"eye-{s}"][j] for j in range(len(SHIFTS))]
        gap[s] = float(np.mean(d[1:]))         # over the real shifts only
        print(f"  {s:<3} grid {grids[s][0]}x{grids[s][1]}:  "
              f"eye {sim[f'eye-{s}'][-1]:.3f}   pixels there "
              f"{sim[g][-1]:.3f}   mean gap over shifts {gap[s]:+.3f}")
    res["resolution_gap"] = {s: round(v, 4) for s, v in gap.items()}
    res["resolution_explains_it"] = bool(max(gap.values()) < 0.05)

    if not res["eye_beats_pixels"]:
        print("\n  H20 is FALSIFIED. Two pooling stages and a complex-cell "
              "layer do not buy invariance to the\n  simplest transformation "
              "there is.")
        if res["resolution_explains_it"]:
            print("  The resolution control says WHY: pixels held on the same "
                  "grid fall just as fast. Nothing in\n  these four stages "
                  "computes anything more position-tolerant than the map it is "
                  "written on --\n  the eye never leaves retinotopy. That is "
                  "structural, not a training-set complaint.")
        else:
            print("  And the eye is worse than pixels even at MATCHED "
                  "resolution, so coarseness does not\n  explain it: the "
                  "features themselves are less stable under translation than "
                  "the luminance\n  they are computed from.")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
