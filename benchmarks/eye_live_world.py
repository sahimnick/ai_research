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
    tracking     whether the location whose activity changed most is the
                 location where the *world* changed -- scored against a
                 shuffled-camera control, not against nominal chance, because
                 traffic cameras are centre-biased and both maps would peak
                 centrally for unrelated reasons

The encode-twice floor is not optional. `live_world.py` established it: without
it "the code is stable" and "the code tracks the scene" cannot be told apart,
and a collapsed code scores perfectly on the first.

Refuses to report if the pixels did not change between rounds -- a camera
serving a frozen frame would otherwise look like perfect invariance.

Correction, 2026-08-02
----------------------
The first run of this file reported an encode-twice floor of **exactly 0.000**,
and this project's own `_nearest_prototype` docstring names that signature: *not
an informative failure but a bug wearing a result's clothes*. Three defects, all
in the measurement rather than in the eye:

1. :class:`PopulationAdaptation` starts with ``n_seen = 0``, so its first update
   uses ``a = max(tau, 1/(n_seen+1)) = 1.0`` and the running mean becomes the
   first code **exactly**. Subtracting it returns zeros, so **row 0 of every
   batch was the zero vector**.
2. The floor called the encoder on a *single* frame, which is therefore row 0 --
   the zero vector -- so the floor was structurally 0.000 whatever the eye did.
   It could never have measured anything.
3. The two rounds were encoded through **independent** adaptation states, so
   ``Vc1`` and ``Vc2`` lived in different spaces. That is the same space
   mismatch §9.6 was caught making.

All three are fixed by fitting **one** adaptation per stream on the first round,
then freezing it (``learn=False``) for every later encode. The floor becomes a
real determinism check on the identical pixels.

A control is added that the first run had no answer to. Twenty-odd traffic
cameras are twenty-odd *different scenes*, so re-identifying which camera a
frame came from may be trivial from the pixels alone. Raw downsampled pixels are
scored through the identical pairing, and **the eye's identity score means
nothing unless it is read against that number.**

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


def raw_at(stream, frames, colour, stage="V2"):
    """Every frame's activity at one area -- **before** any adaptation."""
    h = stream.hierarchy
    names = [getattr(l, "name", type(l).__name__) for l in h.layers]
    k = names.index(stage)
    out = []
    for f in frames:
        x = retinal_opponent(f) if colour else np.asarray(
            f, np.float32).mean(0)[None] / 255.0
        h.forward(x)
        out.append(np.asarray(h.layers[k].log["output"], np.float32).ravel())
    return np.stack(out)


def fit_adapt(M):
    """One adaptation, fitted on the first round and then FROZEN.

    Fitting a fresh one per batch is what put the two rounds in different
    spaces, and its fast start (``a = 1/(n_seen+1)``) is what made the first row
    of every batch the zero vector. Both go away if the baseline is learned once
    and afterwards only applied.
    """
    ad = PopulationAdaptation(M.shape[1])
    for r in M:
        ad.observe(r)
    return ad


def apply_adapt(ad, M):
    return np.array([_unit(ad(r, learn=False)) for r in M], np.float32)


def change_maps(stream, f1, f2, colour, stage="V2"):
    """Where the eye's activity changed, and where the world changed.

    The docstring above has claimed a tracking measurement since this file was
    written and **there was none in it**. This is it, in the only form the data
    supports: not object tracking, but retinotopic correspondence -- does the
    eye's response change *where the scene changed*?

    Returns two same-shaped maps over the area's spatial grid, so they can be
    compared location by location.
    """
    h = stream.hierarchy
    names = [getattr(l, "name", type(l).__name__) for l in h.layers]
    k = names.index(stage)
    acts = []
    for f in (f1, f2):
        x = retinal_opponent(f) if colour else np.asarray(
            f, np.float32).mean(0)[None] / 255.0
        h.forward(x)
        acts.append(np.asarray(h.layers[k].log["output"], np.float32))
    eye = np.abs(acts[1] - acts[0]).sum(0)             # (H, W) over features

    world = np.abs(np.asarray(f2, np.float32)
                   - np.asarray(f1, np.float32)).mean(0)
    # block-average the frame down onto the area's grid, so a location in one
    # map is the same patch of the world as the same location in the other
    gh, gw = eye.shape
    ph, pw = world.shape[0] // gh, world.shape[1] // gw
    world = world[:gh * ph, :gw * pw].reshape(gh, ph, gw, pw).mean((1, 3))
    return eye, world


def hit_rate(eye, world, q=0.10):
    """Does the eye's peak land in the world's most-changed ``q`` of locations?"""
    thr = np.quantile(world, 1.0 - q)
    r, c = np.unravel_index(int(np.argmax(eye)), eye.shape)
    return float(world[r, c] >= thr)


def pixel_raw(frames, colour):
    """The control: the frames themselves, downsampled. No eye involved.

    If this re-identifies a camera as well as the eye does, the eye's score is
    reporting that traffic cameras point at different streets, not that a
    spiking hierarchy perceives anything. Returned unadapted so it goes through
    the *same* fit-once-then-freeze treatment as the eye's code -- a control
    scored in a different space would not be a control.
    """
    P = []
    for f in frames:
        a = np.asarray(f, np.float32) / 255.0
        if not colour:
            a = a.mean(0)[None]
        s = max(1, a.shape[-1] // 32)
        P.append(a[:, ::s, ::s].ravel())
    return np.stack(P)


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

    Rc1 = raw_at(col, first, colour=True)
    Rl1 = raw_at(lum, first, colour=False)
    Pc1 = pixel_raw(first, colour=True)
    adc, adl, adp = fit_adapt(Rc1), fit_adapt(Rl1), fit_adapt(Pc1)
    Vc1, Vl1, Vp1 = (apply_adapt(adc, Rc1), apply_adapt(adl, Rl1),
                     apply_adapt(adp, Pc1))

    # the floor: the SAME pixels through the SAME frozen baseline. Anything
    # below 1.000 is the encoder itself being stochastic, and is the ceiling on
    # what "the code survived the world moving" can possibly mean.
    floor_c = float(np.mean(np.sum(
        Vc1 * apply_adapt(adc, raw_at(col, first, colour=True)), 1)))
    dead = int((np.linalg.norm(Vc1, axis=1) < 1e-6).sum())
    if dead:
        print(f"  {dead} of {len(Vc1)} codes are the zero vector -- the "
              f"adaptation bug is back. REFUSING to report.")
        json.dump({"error": "zero codes", "n_dead": dead},
                  open(out_path, "w"), indent=1)
        return

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

    # every later frame through the SAME frozen baseline as round one
    Vc2 = apply_adapt(adc, raw_at(col, later, colour=True))
    Vl2 = apply_adapt(adl, raw_at(lum, later, colour=False))
    Vp2 = apply_adapt(adp, pixel_raw(later, colour=True))

    def score(A, B):
        """Mean self-similarity, and how often a later frame's nearest
        neighbour among all first-round frames is its own camera."""
        same = float(np.mean([A[a] @ B[k] for k, a in enumerate(base)]))
        S = B @ A[base].T
        return same, float(np.mean(np.argmax(S, 1) == np.arange(len(base))))

    same_c, id_c = score(Vc1, Vc2)
    same_l, id_l = score(Vl1, Vl2)
    same_p, id_p = score(Vp1, Vp2)
    chance = 1.0 / len(base)

    res = {"n_cameras": len(pairs), "gap_s": GAP_S, "native_px": NATIVE,
           "pixel_change": round(changed, 3),
           "encode_twice_floor": round(floor_c, 4),
           "same_camera_later": {"colour": round(same_c, 4),
                                 "luminance": round(same_l, 4),
                                 "raw_pixels": round(same_p, 4)},
           "identity": {"colour": round(id_c, 4),
                        "luminance": round(id_l, 4),
                        "raw_pixels": round(id_p, 4),
                        "chance": round(chance, 4)},
           "participation": {"colour": round(code_participation(Vc1), 2),
                             "luminance": round(code_participation(Vl1), 2),
                             "raw_pixels": round(code_participation(Vp1), 2)}}

    print(f"\n{'':<26}{'colour':>10}{'luminance':>12}{'RAW PIXELS':>13}")
    print(f"{'same pixels twice (floor)':<26}{floor_c:>10.3f}"
          f"{'—':>12}{'—':>13}")
    print(f"{'same camera ' + str(int(GAP_S)) + 's later':<26}"
          f"{same_c:>10.3f}{same_l:>12.3f}{same_p:>13.3f}")
    print(f"{'still recognises itself':<26}{id_c:>10.3f}{id_l:>12.3f}"
          f"{id_p:>13.3f}   (chance {chance:.3f})")
    print(f"{'participation':<26}{res['participation']['colour']:>10.2f}"
          f"{res['participation']['luminance']:>12.2f}"
          f"{res['participation']['raw_pixels']:>13.2f}")

    best_eye = max(id_c, id_l)
    res["eye_minus_pixels"] = round(float(best_eye - id_p), 4)
    res["colour_minus_luminance"] = round(float(id_c - id_l), 4)

    # --- tracking: does the eye's activity change WHERE the world changed? ---
    Q = 0.10
    eyes, worlds = [], []
    for a, b in pairs:
        e, w = change_maps(lum, first[a], second[b], colour=False)
        eyes.append(e)
        worlds.append(w)
    hit = float(np.mean([hit_rate(e, w, Q) for e, w in zip(eyes, worlds)]))
    # The control that decides it. Traffic cameras are centre-biased, and both
    # maps can peak centrally for unrelated reasons -- which is exactly the
    # artefact §9.8's oracle arm was built on. Pairing each camera's EYE map
    # with a DIFFERENT camera's world map keeps every spatial bias and destroys
    # only the correspondence, so it is the real chance level.
    rng = np.random.default_rng(0)
    shuf = float(np.mean([
        np.mean([hit_rate(eyes[i], worlds[j], Q)
                 for i, j in enumerate(rng.permutation(len(eyes)))])
        for _ in range(20)]))
    res["tracking"] = {"hit_rate": round(hit, 4),
                       "shuffled_control": round(shuf, 4),
                       "nominal_chance": Q,
                       "grid": list(eyes[0].shape)}

    print(f"\n--- tracking: does the eye's peak change land where the world "
          f"changed? ---")
    print(f"  V2 grid {eyes[0].shape[0]}x{eyes[0].shape[1]}, top {Q:.0%} of "
          f"locations counted as a hit")
    print(f"  hit rate            {hit:.3f}")
    print(f"  shuffled cameras    {shuf:.3f}   <- the real chance level")
    print(f"  nominal chance      {Q:.3f}")
    res["tracking_beats_shuffle"] = bool(hit > shuf + 0.10)

    print(f"\n--- what this says ---")
    print(f"  the world moved: {changed:.2f}/255 mean pixel change")
    print(f"  the encoder is deterministic on identical pixels: "
          f"{floor_c:.4f} (1.000 = perfectly)")
    if best_eye <= 2.0 * chance:
        print(f"  the eye does NOT re-identify a place after it changed "
              f"({best_eye:.3f} against chance {chance:.3f}). On native\n  "
              f"resolution and real motion the code does not survive the world "
              f"moving, which is the honest result.")
    elif best_eye <= id_p + 1e-9:
        print(f"  the eye re-identifies a place ({best_eye:.3f}) -- but SO DO "
              f"RAW PIXELS ({id_p:.3f}). Twenty-odd traffic\n  cameras point at "
              f"twenty-odd different streets, and this number is reporting "
              f"that, not perception.\n  The eye adds "
              f"{best_eye - id_p:+.3f} over doing nothing at all.")
    else:
        print(f"  the eye re-identifies a place ({best_eye:.3f}) and beats raw "
              f"pixels ({id_p:.3f}) by {best_eye - id_p:+.3f} --\n  the "
              f"hierarchy is carrying something the pixels are not.")
    print(f"  colour minus luminance on identity: {id_c - id_l:+.4f}")
    if res["tracking_beats_shuffle"]:
        print(f"  the eye localises change: {hit:.3f} against a shuffled "
              f"{shuf:.3f}. Where it responds differently is where\n  the world "
              f"is different -- retinotopy survives to V2 on real motion.")
    else:
        print(f"  the eye does NOT localise change: {hit:.3f} against a "
              f"shuffled {shuf:.3f}. Its peak response moves, but not\n  where "
              f"the world moved, so nothing here supports tracking.")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
