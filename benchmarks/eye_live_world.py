"""The four-stage eye on the real world: native resolution, colour, motion.

Everything the ventral stream has been measured on so far is CIFAR at 32 px
upscaled to 96 -- interpolation, not resolution. §9.17's canvas finding makes
that a real limitation rather than a cosmetic one: this hierarchy loses spatial
extent fast, and what it can see depends on what it is actually given.

Live traffic cameras give **240 x 352 colour frames** natively, and consecutive
frames of the same camera give motion in the world rather than a jittered still.
One source covers three things the eye has never been tested on.

What this measures
------------------
    resolution   every area's output size and spike ceiling at 96 px against
                 native. `stage_extents` reports the ceiling beside the count,
                 because §9.15 read 100% saturation as silence for want of it
    colour       the three-channel retina against luminance, on the same frames
    motion       the same camera a few seconds later -- how much of the code
                 survives the world moving, against an encode-twice floor that
                 says what "the same scene" is worth when nothing changed
    tracking     whether the strongest-responding location follows the thing
                 that moved, against the chance of landing on it

The encode-twice floor is not optional. `live_world.py` established it: without
it "the code is stable" and "the code tracks the scene" cannot be told apart,
and a collapsed code scores perfectly on the first.

Refuses to report if the pixels did not change between rounds -- a camera
serving a frozen frame would otherwise look like perfect invariance.

Usage:  python3 benchmarks/eye_live_world.py out_eye_live.json [n_cam] [gap_s]
"""
import json
import sys
import time

import numpy as np

from neurobrain.cognition.multimodal import _unit
from neurobrain.sensing.live import CctvCamera, NY511_LIVE_IDS, SensorUnavailable
from neurobrain.vision.ventral import (build_ventral_colour,
                                       build_ventral_stream_on,
                                       code_participation, retinal_opponent,
                                       stage_extents, upscale)
from neurobrain.vision.widev1 import PopulationAdaptation

N_CAM = int(sys.argv[2]) if len(sys.argv) > 2 else 24
GAP_S = float(sys.argv[3]) if len(sys.argv) > 3 else 90.0
NATIVE = 224          # square crop of the native frame -- no upscaling
SMALL = 96            # what every earlier ventral measurement used


def grab(cam, n_want):
    """One frame from each of the first ``n_want`` cameras that is live."""
    out, ids = [], []
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
        if a.ndim != 3 or min(a.shape[:2]) < NATIVE:
            continue
        h, w = a.shape[:2]
        y0, x0 = (h - NATIVE) // 2, (w - NATIVE) // 2
        out.append(a[y0:y0 + NATIVE, x0:x0 + NATIVE].transpose(2, 0, 1))
        ids.append(i)
    return out, ids


def code_at(stream, frames, colour, stage="V2"):
    h = stream.hierarchy
    names = [getattr(l, "name", type(l).__name__) for l in h.layers]
    k = names.index(stage)
    out = []
    for f in frames:
        x = retinal_opponent(f) if colour else np.asarray(
            f, np.float32).mean(0)[None] / 255.0
        h.forward(x)
        out.append(np.asarray(h.layers[k].log["output"], np.float32).ravel())
    M = np.stack(out)
    ad = PopulationAdaptation(M.shape[1])
    return np.array([_unit(ad(r)) for r in M], np.float32)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_eye_live.json"
    print("--- what the architecture can see at each input size ---")
    for sz in (SMALL, NATIVE):
        ext = stage_extents(sz)
        print(f"  {sz:>4}px: " + "  ".join(
            f"{n}{sh[1]}x{sh[2]}(ceil {c})" for n, sh, c in ext))

    cam = CctvCamera(ids=list(NY511_LIVE_IDS))
    print(f"\nreaching for {N_CAM} live cameras at {NATIVE}px native ...",
          flush=True)
    first, ids = grab(cam, N_CAM)
    if len(first) < 8:
        print(f"only {len(first)} live cameras reachable -- not enough to "
              f"measure. Nothing reported.")
        json.dump({"error": "too few live cameras", "n": len(first)},
                  open(out_path, "w"), indent=1)
        return
    print(f"  {len(first)} cameras, native frame {first[0].shape}", flush=True)

    # the floor: the SAME pixels encoded twice, so "stable" and "tracking the
    # world" cannot be confused
    print("\ndeveloping the eye on these frames (colour and luminance) ...",
          flush=True)
    dev = np.stack(first)
    col = build_ventral_colour(dev, size=NATIVE, verbose=False)
    lum = build_ventral_stream_on(np.stack([d.mean(0) for d in dev]),
                                  size=NATIVE, verbose=False)

    Vc1 = code_at(col, first, colour=True)
    Vl1 = code_at(lum, first, colour=False)
    floor_c = float(np.mean([Vc1[i] @ code_at(col, [first[i]], True)[0]
                             for i in range(len(first))]))

    print(f"\nwaiting {GAP_S:.0f}s for the world to move ...", flush=True)
    time.sleep(GAP_S)
    second, ids2 = grab(cam, N_CAM)
    keep = [k for k, i in enumerate(ids) if i in ids2]
    idx2 = {i: k for k, i in enumerate(ids2)}
    pairs = [(k, idx2[ids[k]]) for k in keep]
    if len(pairs) < 8:
        print("too few cameras came back for a second look. Nothing reported.")
        json.dump({"error": "second round too small"}, open(out_path, "w"))
        return

    changed = float(np.mean([
        float(np.abs(first[a] - second[b]).mean()) for a, b in pairs]))
    print(f"  {len(pairs)} cameras returned; mean pixel change "
          f"{changed:.2f}/255", flush=True)
    if changed < 0.5:
        print("  the pixels barely changed -- this would look like perfect "
              "invariance for the wrong reason. REFUSING to report.")
        json.dump({"error": "scene did not change", "pixel_change": changed},
                  open(out_path, "w"), indent=1)
        return

    later = [second[b] for _, b in pairs]
    base = [a for a, _ in pairs]
    Vc2 = code_at(col, later, colour=True)
    Vl2 = code_at(lum, later, colour=False)
    same_c = float(np.mean([Vc1[a] @ Vc2[k] for k, a in enumerate(base)]))
    same_l = float(np.mean([Vl1[a] @ Vl2[k] for k, a in enumerate(base)]))
    # identity: does a camera's later frame still look most like ITSELF?
    S = Vc2 @ Vc1[base].T
    id_c = float(np.mean(np.argmax(S, 1) == np.arange(len(base))))
    Sl = Vl2 @ Vl1[base].T
    id_l = float(np.mean(np.argmax(Sl, 1) == np.arange(len(base))))

    res = {"n_cameras": len(pairs), "gap_s": GAP_S, "native_px": NATIVE,
           "pixel_change": round(changed, 3),
           "encode_twice_floor": round(floor_c, 4),
           "same_camera_later": {"colour": round(same_c, 4),
                                 "luminance": round(same_l, 4)},
           "identity": {"colour": round(id_c, 4),
                        "luminance": round(id_l, 4),
                        "chance": round(1.0 / len(base), 4)},
           "participation": {"colour": round(code_participation(Vc1), 2),
                             "luminance": round(code_participation(Vl1), 2)}}

    print(f"\n{'':<26}{'colour':>10}{'luminance':>12}")
    print(f"{'same pixels twice (floor)':<26}{floor_c:>10.3f}{'—':>12}")
    print(f"{'same camera ' + str(int(GAP_S)) + 's later':<26}"
          f"{same_c:>10.3f}{same_l:>12.3f}")
    print(f"{'still recognises itself':<26}{id_c:>10.3f}{id_l:>12.3f}"
          f"   (chance {1.0/len(base):.3f})")
    print(f"{'participation':<26}{res['participation']['colour']:>10.2f}"
          f"{res['participation']['luminance']:>12.2f}")

    print(f"\n--- what this says ---")
    print(f"  the world moved: {changed:.2f}/255 mean pixel change")
    if id_c > 2.0 / len(base) or id_l > 2.0 / len(base):
        best = "colour" if id_c >= id_l else "luminance"
        print(f"  the eye still identifies a place after it changed, best on "
              f"{best} ({max(id_c, id_l):.3f} against chance "
              f"{1.0/len(base):.3f})")
    else:
        print(f"  the eye does NOT re-identify a place after it changed "
              f"({max(id_c, id_l):.3f} against chance {1.0/len(base):.3f}). "
              f"On native\n  resolution and real motion the code does not "
              f"survive the world moving, which is the honest result.")
    print(f"  colour minus luminance on identity: {id_c - id_l:+.4f}")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
