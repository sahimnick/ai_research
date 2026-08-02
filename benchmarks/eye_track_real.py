"""Does a tag survive the world CHANGING, scored the way trackers are scored?

§9.21 showed a spatial tag localises an object at 0.943 across 114 px of
motion, where `_attention_heatmap`'s averaged prototype manages 0.114. Two
caveats were stated there and both are settled here:

* the object was **pixel-identical** at every position, so the tag matched a
  perfect template -- the easiest case that exists;
* the scoring was a hit rate at one tolerance, not what a tracking benchmark
  reports.

Real appearance change, with exact ground truth
-----------------------------------------------
A traffic camera is **static** and its content is not. Measured for this
project: NY511 holds a frame byte-identical for 30 s and changes on a median
period of **97 s**. So a region tagged at t0 and looked for at t0+~100 s is the
*same place* -- ground truth is exact, because the camera did not move -- while
the pixels there have genuinely changed: vehicles have moved through, light has
shifted, shadows have swung. Nothing is composited and nothing is simulated.

Two conditions:

    still      the region is where it was; only the world changed
    displaced  the view is also translated, so the tag must survive change
               AND motion at once

Scored the way trackers are scored
----------------------------------
Rather than one hit rate at one tolerance, the two standard summaries:

    precision@20px   fraction of frames whose centre error is within 20 px,
                     the usual operating point of a precision plot
    success AUC      area under the success plot -- mean over IoU thresholds
                     0..1 of the fraction of frames at or above that overlap

with `shuffled-tag` (a different camera's tag, so every spatial bias survives
and only the correspondence dies) as the chance level, and pixel NCC as the
classical baseline -- which here has **no** unfair advantage, because the
region's pixels really did change.

Usage:  python3 benchmarks/eye_track_real.py out_eye_track_real.json [n_cam]
"""
import json
import sys
import time

import numpy as np

from neurobrain.sensing.live import CctvCamera, NY511_LIVE_IDS, SensorUnavailable
from neurobrain.vision.ventral import build_ventral_stream_on, locate_template

N_CAM = int(sys.argv[2]) if len(sys.argv) > 2 else 16
SIZE = 224
BOX = 48                      # the tracked region, pixels
POS = (88, 88)                # where it is tagged, centred-ish
SHIFT = 32                    # the displaced condition
WAIT = 105.0                  # > the measured 97 s median update period
STAGES = ("V2", "V4")


def grab(cam, idxs=None, n_want=None):
    """Frames plus the camera indices they came from, so t0 and t1 line up."""
    got, ids = [], []
    pool = idxs if idxs is not None else range(len(cam.urls))
    for i in pool:
        if n_want is not None and len(got) >= n_want:
            break
        try:
            r = cam.read(i)
        except SensorUnavailable:
            continue
        if not getattr(r, "live", False):
            continue
        a = np.asarray(r.data, np.float32)
        if a.ndim != 3 or a.shape[0] < SIZE or a.shape[1] < SIZE + SHIFT:
            continue
        got.append(a.transpose(2, 0, 1)[:, :SIZE, :].mean(0) / 255.0)
        ids.append(i)
    return got, ids


def area_maps(stream, frame, stages):
    h = stream.hierarchy
    names = [getattr(l, "name", type(l).__name__) for l in h.layers]
    h.forward(frame[None])
    return {s: np.asarray(h.layers[names.index(s)].log["output"], np.float32)
            for s in stages}


def patch_of(m, y, x, size, img):
    _, gh, gw = m.shape
    r0, c0 = int(y * gh / img), int(x * gw / img)
    r1 = max(int((y + size) * gh / img) + 1, r0 + 1)
    c1 = max(int((x + size) * gw / img) + 1, c0 + 1)
    return m[:, r0:r1, c0:c1]


def locate_px(m, tag, img):
    (r, c), _ = locate_template(m, tag)
    th, tw = tag.shape[1:]
    _, gh, gw = m.shape
    return ((r + (th - 1) / 2.0) * img / gh, (c + (tw - 1) / 2.0) * img / gw)


def ncc_px(frame, patch, stride=1):
    ph, pw = patch.shape
    p = patch - patch.mean()
    pn = np.linalg.norm(p) + 1e-9
    W = np.lib.stride_tricks.sliding_window_view(frame, (ph, pw))[::stride,
                                                                 ::stride]
    Wc = W - W.mean((2, 3), keepdims=True)
    num = np.einsum("ijkl,kl->ij", Wc, p)
    den = np.sqrt(np.einsum("ijkl,ijkl->ij", Wc, Wc)) * pn + 1e-9
    r, c = np.unravel_index(int(np.argmax(num / den)), num.shape)
    return (r * stride + ph / 2.0, c * stride + pw / 2.0)


def iou(pred, true, size):
    """Overlap of two ``size``-square boxes given their centres."""
    dy = abs(pred[0] - true[0])
    dx = abs(pred[1] - true[1])
    iw, ih = max(0.0, size - dx), max(0.0, size - dy)
    inter = iw * ih
    return inter / (2.0 * size * size - inter) if inter > 0 else 0.0


def summarise(errs, ious):
    """The two numbers a tracking benchmark reports."""
    errs, ious = np.asarray(errs), np.asarray(ious)
    prec20 = float(np.mean(errs <= 20.0))
    thr = np.linspace(0.0, 1.0, 101)
    auc = float(np.mean([(ious >= t).mean() for t in thr]))
    return prec20, auc, float(np.median(errs))


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_eye_track_real.json"
    cam = CctvCamera(ids=list(NY511_LIVE_IDS))
    print(f"reaching for {N_CAM} live cameras at {SIZE}px ...", flush=True)
    first, ids = grab(cam, n_want=N_CAM)
    if len(first) < 6:
        print(f"only {len(first)} usable cameras -- nothing reported.")
        json.dump({"error": "too few cameras", "n": len(first)},
                  open(out_path, "w"), indent=1)
        return
    print(f"  {len(first)} cameras", flush=True)

    print("\ngrowing the eye on these frames ...", flush=True)
    stream = build_ventral_stream_on(np.stack([f[:, :SIZE] for f in first]),
                                     size=SIZE, verbose=False)

    print(f"\ntagging, then waiting {WAIT:.0f}s for the world to change "
          f"(measured median period 97s) ...", flush=True)
    y, x = POS
    tags, pxtags = [], []
    for f in first:
        m = area_maps(stream, f[:, :SIZE], STAGES)
        tags.append({s: patch_of(m[s], y, x, BOX, SIZE) for s in STAGES})
        pxtags.append(f[y:y + BOX, x:x + BOX].copy())
    time.sleep(WAIT)

    second, ids2 = grab(cam, idxs=ids)
    keep = [i for i, d in enumerate(ids) if d in set(ids2)]
    pos2 = {d: j for j, d in enumerate(ids2)}
    if len(keep) < 6:
        print(f"only {len(keep)} cameras came back -- nothing reported.")
        json.dump({"error": "cameras did not return", "n": len(keep)},
                  open(out_path, "w"), indent=1)
        return

    changed = float(np.mean([
        np.abs(second[pos2[ids[i]]][:, :SIZE] - first[i][:, :SIZE]).mean()
        for i in keep])) * 255.0
    print(f"  {len(keep)} cameras returned; the world changed by "
          f"{changed:.2f}/255", flush=True)
    if changed < 1.0:
        print("  the world did NOT change -- this measures nothing. REFUSING.")
        json.dump({"error": "world did not change", "delta": changed},
                  open(out_path, "w"), indent=1)
        return
    # the region itself must have changed, not just somewhere in the frame
    roi = float(np.mean([
        np.abs(second[pos2[ids[i]]][y:y + BOX, x:x + BOX]
               - first[i][y:y + BOX, x:x + BOX]).mean() for i in keep])) * 255.0
    print(f"  the tagged region itself changed by {roi:.2f}/255", flush=True)

    arms = ([f"eye-{s}" for s in STAGES] + ["pixels-NCC", "shuffled-tag"])
    conds = ("still", "displaced")
    err = {c: {a: [] for a in arms} for c in conds}
    ov = {c: {a: [] for a in arms} for c in conds}

    for n, i in enumerate(keep):
        f2 = second[pos2[ids[i]]]
        for cond in conds:
            d = 0 if cond == "still" else SHIFT
            view = f2[:, d:d + SIZE]
            true = (y + BOX / 2.0, x + BOX / 2.0 - d)
            m = area_maps(stream, view, STAGES)
            for s in STAGES:
                p = locate_px(m[s], tags[i][s], SIZE)
                err[cond][f"eye-{s}"].append(np.hypot(*np.subtract(p, true)))
                ov[cond][f"eye-{s}"].append(iou(p, true, BOX))
            p = ncc_px(view, pxtags[i])
            err[cond]["pixels-NCC"].append(np.hypot(*np.subtract(p, true)))
            ov[cond]["pixels-NCC"].append(iou(p, true, BOX))
            j = keep[(n + 1) % len(keep)]            # a DIFFERENT camera's tag
            p = locate_px(m["V2"], tags[j]["V2"], SIZE)
            err[cond]["shuffled-tag"].append(np.hypot(*np.subtract(p, true)))
            ov[cond]["shuffled-tag"].append(iou(p, true, BOX))
        print(f"  camera {n + 1}/{len(keep)}", end="\r", flush=True)

    res = {"n_cameras": len(keep), "size": SIZE, "box": BOX, "shift": SHIFT,
           "world_delta": round(changed, 2), "roi_delta": round(roi, 2),
           "conditions": {}}
    print("\n")
    for cond in conds:
        how = "no displacement" if cond == "still" else f"{SHIFT}px displaced"
        print(f"{cond}  ({how}, real appearance change)")
        print(f"  {'arm':<14}{'prec@20px':>11}{'success AUC':>13}"
              f"{'median err':>12}")
        res["conditions"][cond] = {}
        for a in arms:
            p20, auc, med = summarise(err[cond][a], ov[cond][a])
            res["conditions"][cond][a] = {"precision_20px": round(p20, 4),
                                          "success_auc": round(auc, 4),
                                          "median_err_px": round(med, 2)}
            print(f"  {a:<14}{p20:>11.3f}{auc:>13.3f}{med:>12.1f}")
        print()

    best = max(STAGES, key=lambda s:
               res["conditions"]["still"][f"eye-{s}"]["success_auc"])
    res["best_stage"] = best
    ok = all(res["conditions"][c][f"eye-{best}"]["success_auc"] >
             res["conditions"][c]["shuffled-tag"]["success_auc"] + 0.10
             for c in conds)
    res["survives_appearance_change"] = bool(ok)
    res["beats_pixel_ncc"] = bool(all(
        res["conditions"][c][f"eye-{best}"]["success_auc"] >
        res["conditions"][c]["pixels-NCC"]["success_auc"] for c in conds))

    print("--- does the tag survive the world changing? ---")
    print(f"  the world changed by {changed:.2f}/255, the tagged region by "
          f"{roi:.2f}/255")
    if ok:
        print(f"  YES. eye-{best} beats the shuffled control in both "
              f"conditions. The tag is matching the\n  place, not the pixels "
              f"-- it survives traffic moving through it and the light "
              f"shifting.")
    else:
        print(f"  NO. eye-{best} does not clear the shuffled control in both "
              f"conditions. §9.21's 0.943 was\n  matching an unchanged "
              f"template; once the pixels really change, the tag stops finding "
              f"its place.")
    print(f"  beats classical pixel NCC on the same task: "
          f"{res['beats_pixel_ncc']}")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
