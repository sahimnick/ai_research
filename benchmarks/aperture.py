"""Is the eye's limit the learning rule, or how much of the world it gets?

Four independent attacks on the eye have now failed, and they exhaust the space
of "adapt the filters better":

    section 7.8   eleven mechanisms -- width, aperture, spike noise, integration
                  window, two second-stage rules, whitening, glance pooling
    phase 10.1    concept feedback through a delta rule, which could not carry a
                  teacher at all (8 cells share a position, all get collinear
                  updates)
    phase 10.1b   concept feedback through a reconstructive rule that *was*
                  verified to carry its teacher (own-vs-wrong separation 0.341)
                  -- correct, uninformative and misleading targets still agree
                  to within 0.003
    phase 10.3    choosing where to look, for agreement or for disagreement,
                  both negative at every k against not choosing

Every one of those changes **how the eye adapts**. None changes **what it
receives**. And there is a reason to suspect the input: the live-camera panel
shows a traffic scene reduced to unreadable mush at 32x32, which is the size
this eye has taken since it was built for MNIST digits.

That cannot be tested on CIFAR -- those photographs are natively 32x32, so
asking for more resolution returns interpolation, not information. The live
cameras are 352x240 and 720x576 native, and they have real detail to give.

The measurement without labels
------------------------------
Live cameras carry no class labels, so this uses the one ground truth the live
world does offer: **a camera is itself**. Two encodings of one camera should sit
closer than encodings of two different cameras, and the gap is a property of the
code rather than of the scene.

    identity AUC    same camera above different camera
    separation      the cosine gap behind that AUC, which is what the concept
                    layer downstream actually consumes
    noise floor     the same pixels encoded twice -- without it, a code that
                    simply got noisier would look like a code that got worse

If discrimination climbs with input size, the aperture is the constraint and the
next move is to change what the eye *is* rather than how it learns. If it is
flat, the input is not the limit either, and the failure is somewhere neither
adaptation nor resolution reaches.

The cell count is held **fixed** across sizes, so a larger input means each cell
covers proportionally less of the scene -- the question is about the field of
view and the detail in it, not about capacity. A capacity arm is included
separately so the two cannot be confused.

Usage:  python3 benchmarks/aperture.py out_aperture.json
"""
import json
import sys

import numpy as np

from neurobrain.learning.selforganize import develop_v1
from neurobrain.sensing.live import CctvCamera
from neurobrain.sensing.streams import StreamingBrain
from neurobrain.vision.widev1 import PopulationAdaptation, _unit

sys.path.insert(0, "benchmarks")
from real_binding import opponent                            # noqa: E402

SIZES = (32, 48, 64, 96)
CELLS = 4096
SEEDS = (0, 1, 2)


def to_frame(img, size):
    from PIL import Image
    h, w = img.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    im = Image.fromarray(img[y0:y0 + s, x0:x0 + s]).resize((size, size))
    return np.transpose(np.asarray(im, np.uint8), (2, 0, 1))


def encode(v1, ad, frames):
    R = np.array([np.concatenate([v1.rate(c) for c in opponent(f)])
                  for f in frames], np.float32)
    return np.array([_unit(ad(r)) for r in R], np.float32)


def evaluate(raw, size, cells, seed):
    frames = [to_frame(r, size) for r in raw]
    brain = StreamingBrain(seed=seed, image_shape=(size, size),
                           v1_cells=cells, rf=7, stride=2)
    develop_v1(brain.v1, [opponent(f)[0] for f in frames], epochs=3, seed=seed)
    ad = PopulationAdaptation(3 * brain.v1.n_cells)
    V, V2 = encode(brain.v1, ad, frames), encode(brain.v1, ad, frames)
    n = len(frames)
    rng = np.random.default_rng(seed)
    same = np.array([float(V[k] @ V2[k]) for k in range(n)])
    diff = np.array([float(V[k] @ V2[int(rng.choice(
        [j for j in range(n) if j != k]))]) for k in range(n)])
    return dict(identity_auc=float((same[:, None] > diff[None, :]).mean()),
                same=float(same.mean()), different=float(diff.mean()),
                separation=float(same.mean() - diff.mean()),
                positions=int(brain.v1.n_pos))


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_aperture.json"
    cam = CctvCamera()
    raw = [r.data for r in cam.survey() if r.live]
    if len(raw) < 6:
        print("too few live cameras to measure anything")
        return
    hs = [r.shape[0] for r in raw]
    print(f"{len(raw)} live cameras, native height {min(hs)}-{max(hs)}px, "
          f"{CELLS} cells held fixed across sizes\n", flush=True)

    res = {"n_cameras": len(raw), "cells": CELLS, "sizes": list(SIZES),
           "by_size": {}}
    print(f"{'input':<8}{'positions':>11}{'same':>9}{'different':>11}"
          f"{'separation':>12}{'identity AUC':>14}")
    for size in SIZES:
        rows = [evaluate(raw, size, CELLS, sd) for sd in SEEDS]
        m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        res["by_size"][str(size)] = {k: round(v, 4) for k, v in m.items()}
        print(f"{size}x{size:<4}{m['positions']:>11.0f}{m['same']:>9.3f}"
              f"{m['different']:>11.3f}{m['separation']:>12.3f}"
              f"{m['identity_auc']:>14.3f}", flush=True)

    # the arm that separates FIELD OF VIEW from CAPACITY: more cells at the
    # smallest input. If this matches the large-input arm, the gain was cells.
    rows = [evaluate(raw, SIZES[0], CELLS * 4, sd) for sd in SEEDS]
    m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    res["capacity_control"] = {k: round(v, 4) for k, v in m.items()}
    print(f"{SIZES[0]}x{SIZES[0]}, {CELLS*4} cells{m['positions']:>4.0f}"
          f"{m['same']:>9.3f}{m['different']:>11.3f}{m['separation']:>12.3f}"
          f"{m['identity_auc']:>14.3f}   <- capacity, not field of view")

    small = res["by_size"][str(SIZES[0])]
    big = res["by_size"][str(SIZES[-1])]
    cap = res["capacity_control"]
    d_sep = big["separation"] - small["separation"]
    d_cap = cap["separation"] - small["separation"]
    res["separation_gain_from_size"] = round(float(d_sep), 4)
    res["separation_gain_from_cells"] = round(float(d_cap), 4)

    print(f"\n=== is the eye's limit what it receives? ===")
    print(f"  {SIZES[0]}px -> {SIZES[-1]}px: separation "
          f"{small['separation']:.3f} -> {big['separation']:.3f} "
          f"({d_sep:+.3f}), identity AUC {small['identity_auc']:.3f} -> "
          f"{big['identity_auc']:.3f}")
    print(f"  4x the cells at {SIZES[0]}px: separation {d_cap:+.3f} "
          f"-- for comparison")
    res["aperture_is_the_limit"] = bool(d_sep > 0.05 and d_sep > 2 * d_cap)
    if res["aperture_is_the_limit"]:
        print(f"\n  Yes. The code discriminates better the more of the scene "
              f"it gets, and the gain is not capacity ({d_cap:+.3f} from four "
              f"times the cells).")
        print("  Four attempts to change how the eye ADAPTS have failed; this "
              "says the thing to change is what it RECEIVES.")
    elif d_sep > 0.05:
        print(f"\n  Resolution helps ({d_sep:+.3f}) but so does capacity "
              f"({d_cap:+.3f}), so this does not separate the two.")
    else:
        print(f"\n  No. More of the scene does not make the code more "
              f"discriminative ({d_sep:+.3f}). The input is not the limit "
              f"either --")
        print("  which rules out the last easy explanation, and means the "
              "failure is in neither the learning rule nor the aperture.")
        auc_range = (max(v["identity_auc"] for v in res["by_size"].values())
                     - min(v["identity_auc"] for v in res["by_size"].values()))
        res["identity_auc_saturated"] = bool(
            min(v["identity_auc"] for v in res["by_size"].values()) > 0.95)
        if res["identity_auc_saturated"]:
            print(f"\n  The power of this test is limited and the limit "
                  f"should be read with the result: identity AUC is at "
                  f"ceiling ({auc_range:.3f} of spread across a threefold")
            print("  change in linear resolution), so it could not have shown "
                  "an improvement even if one existed. `separation` is not "
                  "saturated and does fall, which is")
            print("  the part of this that carries information. And telling "
                  "two CAMERAS apart is an easier question than telling two "
                  "CATEGORIES apart -- the latter needs")
            print("  labels the live world does not provide, so this bounds "
                  "the aperture hypothesis rather than closing it.")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
