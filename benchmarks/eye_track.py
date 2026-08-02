"""Can a tag find its object, and follow it when it moves?

§9.20 found the four-stage eye badly *shift sensitive*: 4 px of translation
costs 63% of its code, and pixels on the eye's own grid survive ten times
better. The limiting factor was named as "the eye never leaves retinotopy".

That is a statement about the **flattened** code, and it has a corollary that
points the other way. A convolutional map does not destroy a translated object
-- it *moves* it. The flattened vector changes completely while the content is
merely relocated. So the same retinotopy that costs invariant recognition
should **buy localisation**: a stored feature template should still find its
object, at the object's new place.

    **H21.** The eye's own top-down mechanism -- correlate a stored channel
    vector ("the tag") with every location, which is exactly what
    `VentralStream._attention_heatmap` already does -- localises a real object
    in a real scene, and keeps localising it as the object moves.

This is the goal's تعقیب و تمرکز بر اساس تگ و توجه بالا به پایین measured
rather than asserted. `_attention_heatmap` has existed since the stream was
written, is used by `attend()` to *classify*, and its localisation has never
been measured at all -- the same gap §9.19 found in this file's own tracking
claim.

Ground truth, which live cameras alone cannot give
--------------------------------------------------
A real object patch (a crop of one live camera) is composited into a different
live camera's frame at **known** positions along a trajectory. Background and
object are both real-world imagery at native 224 px; the position is exact
because the benchmark put it there. Nothing else gives both at once.

Arms, and why each is there:

    eye-V2, eye-V4   the implemented mechanism, at two depths
    pixels-NCC       normalised cross-correlation -- the strong classical
                     baseline, and deliberately given the ADVANTAGE of a full
                     spatial patch where the eye arms get only a single mean
                     channel vector, because that is the mechanism that exists
    shuffled-tag     the same eye machinery cued with a DIFFERENT object's tag:
                     keeps every spatial bias of the scene and the map, and
                     destroys only the correspondence. This is the real chance
                     level, not the uniform-random one

The verdict is keyed on **eye vs shuffled-tag**, not on eye vs pixels, and the
reason is a limitation of the design that has to be said before the numbers
rather than after: the composited object is *pixel-identical* at every position
-- no lighting change, no scale change, no deformation -- so `pixels-NCC` is
matching a perfect template and should be near ceiling. Beating it here would be
remarkable; losing to it means very little. What this benchmark can decide is
whether the stored tag localises **at all**, which is the thing `attend()`
assumes and nothing has ever checked.

The instrument check that has to pass first
-------------------------------------------
Frame 0 asks each arm to find the tag in **the very frame the tag was taken
from**. An arm that cannot do that is not measuring localisation and the run
refuses to report -- the lesson of §9.19's floor, applied before the fact
rather than after.

Usage:  python3 benchmarks/eye_track.py out_eye_track.json [n_cam]
"""
import json
import sys

import numpy as np

from neurobrain.sensing.live import CctvCamera, NY511_LIVE_IDS, SensorUnavailable
from neurobrain.vision.ventral import build_ventral_stream_on

N_CAM = int(sys.argv[2]) if len(sys.argv) > 2 else 16
SIZE = 224                      # native, per the goal's real dimensions
OBJ = 40                        # the object's side in pixels
# a trajectory: down-and-right, so both axes are exercised
START = (72, 32)
STEP = (6, 22)
N_STEP = 6
STAGES = ("V2", "V4")


def grab(cam, n_want):
    out = []
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
        if a.ndim != 3 or a.shape[0] < SIZE or a.shape[1] < SIZE:
            continue
        out.append(a.transpose(2, 0, 1)[:, :SIZE, :SIZE].mean(0) / 255.0)
    return out


def paste(bg, obj, y, x):
    f = bg.copy()
    f[y:y + obj.shape[0], x:x + obj.shape[1]] = obj
    return f


def maps(stream, frame, stages):
    """Every requested area's (C, H, W) map from one forward pass."""
    h = stream.hierarchy
    names = [getattr(l, "name", type(l).__name__) for l in h.layers]
    h.forward(frame[None])
    return {s: np.asarray(h.layers[names.index(s)].log["output"], np.float32)
            for s in stages}


def footprint(gh, gw, y, x, size, img_size):
    """The object's extent on an area's grid.

    Every map coordinate in this file goes through here, including the *truth*
    it is scored against, so that the map-to-pixel offset cancels in the
    difference. It does not otherwise: a convolutional map does not span its
    input uniformly -- V2's cell 0 sits about 9 px in at 224 px -- and scoring a
    peak cell against a naively rescaled true pixel would charge the eye a
    position-dependent error of up to ~13 px against a 20 px tolerance, while
    leaving the pixel arm (which never leaves pixel space) untouched.
    """
    r0 = int(y * gh / img_size)
    c0 = int(x * gw / img_size)
    r1 = max(int((y + size) * gh / img_size) + 1, r0 + 1)
    c1 = max(int((x + size) * gw / img_size) + 1, c0 + 1)
    return r0, r1, c0, c1


def tag_of(m, y, x, size, img_size):
    """The object's mean channel vector on this area's grid -- the tag.

    This is what `VentralStream.it_class_v4proto` holds and what
    `_attention_heatmap` correlates: a single channel vector, no spatial layout.
    """
    _, gh, gw = m.shape
    r0, r1, c0, c1 = footprint(gh, gw, y, x, size, img_size)
    v = m[:, r0:r1, c0:c1].mean((1, 2))
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def map_error(m, tag, y, x, size, img_size):
    """`_attention_heatmap`'s computation, read for WHERE rather than what.

    Returns the distance from the heatmap's peak to the object's true footprint
    centre, both in cells, converted to a nominal pixel figure by the grid
    scale. Because both terms come from `footprint`, the border offset cancels.
    """
    _, gh, gw = m.shape
    norm = np.linalg.norm(m, axis=0) + 1e-9
    heat = np.tensordot(tag, m, axes=(0, 0)) / norm          # (gh, gw) cosine
    r, c = np.unravel_index(int(np.argmax(heat)), heat.shape)
    r0, r1, c0, c1 = footprint(gh, gw, y, x, size, img_size)
    d = np.hypot((r - (r0 + r1 - 1) / 2.0) * img_size / gh,
                 (c - (c0 + c1 - 1) / 2.0) * img_size / gw)
    return float(d)


def ncc_peak(frame, patch, stride=2):
    """Normalised cross-correlation: the strong classical baseline.

    Given the full spatial patch, where the eye arms get one mean vector.
    """
    ph, pw = patch.shape
    p = patch - patch.mean()
    pn = np.linalg.norm(p) + 1e-9
    W = np.lib.stride_tricks.sliding_window_view(frame, (ph, pw))[::stride,
                                                                 ::stride]
    Wm = W.mean((2, 3), keepdims=True)
    Wc = W - Wm
    num = np.einsum("ijkl,kl->ij", Wc, p)
    den = np.sqrt(np.einsum("ijkl,ijkl->ij", Wc, Wc)) * pn + 1e-9
    r, c = np.unravel_index(int(np.argmax(num / den)), num.shape)
    return (r * stride + ph / 2.0, c * stride + pw / 2.0)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_eye_track.json"
    cam = CctvCamera(ids=list(NY511_LIVE_IDS))
    print(f"reaching for {N_CAM + 2} live cameras at {SIZE}px ...", flush=True)
    frames = grab(cam, N_CAM + 2)
    if len(frames) < 6:
        print(f"only {len(frames)} usable cameras -- nothing reported.")
        json.dump({"error": "too few cameras", "n": len(frames)},
                  open(out_path, "w"), indent=1)
        return

    # the objects are real: a crop of one camera composited into another
    objs = [frames[-1][80:80 + OBJ, 80:80 + OBJ],
            frames[-2][60:60 + OBJ, 100:100 + OBJ]]
    bgs = frames[:-2]
    n = len(bgs)
    traj = [(START[0] + i * STEP[0], START[1] + i * STEP[1])
            for i in range(N_STEP)]
    assert traj[-1][0] + OBJ <= SIZE and traj[-1][1] + OBJ <= SIZE, "off canvas"
    moved = float(np.hypot(traj[-1][0] - traj[0][0], traj[-1][1] - traj[0][1]))
    print(f"  {n} background cameras, object {OBJ}px, trajectory {traj[0]} -> "
          f"{traj[-1]} ({moved:.0f}px)", flush=True)

    print("\ngrowing the eye on these scenes ...", flush=True)
    stream = build_ventral_stream_on(np.stack([paste(b, objs[0], *traj[0])
                                               for b in bgs]),
                                     size=SIZE, verbose=False)

    arms = [f"eye-{s}" for s in STAGES] + ["pixels-NCC", "shuffled-tag"]
    err = {a: [[] for _ in traj] for a in arms}

    for bi, bg in enumerate(bgs):
        f0 = paste(bg, objs[0], *traj[0])
        m0 = maps(stream, f0, STAGES)
        tags = {s: tag_of(m0[s], traj[0][0], traj[0][1], OBJ, SIZE)
                for s in STAGES}
        # the shuffled tag: a DIFFERENT object, pasted in a DIFFERENT scene
        alt = paste(bgs[(bi + 1) % n], objs[1], *traj[0])
        bad = tag_of(maps(stream, alt, STAGES)["V2"],
                     traj[0][0], traj[0][1], OBJ, SIZE)

        for ti, (y, x) in enumerate(traj):
            f = paste(bg, objs[0], y, x)
            true = (y + OBJ / 2.0, x + OBJ / 2.0)
            m = maps(stream, f, STAGES)
            for s in STAGES:
                err[f"eye-{s}"][ti].append(
                    map_error(m[s], tags[s], y, x, OBJ, SIZE))
            p = ncc_peak(f, objs[0])
            err["pixels-NCC"][ti].append(np.hypot(*np.subtract(p, true)))
            err["shuffled-tag"][ti].append(
                map_error(m["V2"], bad, y, x, OBJ, SIZE))
        print(f"  camera {bi + 1}/{n}", end="\r", flush=True)

    med = {a: [float(np.median(e)) for e in err[a]] for a in arms}
    hit = {a: [float(np.mean(np.asarray(e) <= OBJ / 2.0)) for e in err[a]]
           for a in arms}

    print("\n\nmedian localisation error, pixels (object is "
          f"{OBJ}px, so <={OBJ // 2} is on it)")
    print(f"{'arm':<14}" + "".join(f"{'t' + str(i):>8}"
                                   for i in range(len(traj))))
    for a in arms:
        print(f"{a:<14}" + "".join(f"{v:>8.1f}" for v in med[a]))
    print(f"\nhit rate (peak lands on the object)")
    for a in arms:
        print(f"{a:<14}" + "".join(f"{v:>8.3f}" for v in hit[a]))

    res = {"n_cameras": n, "size": SIZE, "object_px": OBJ,
           "trajectory": [list(t) for t in traj], "moved_px": round(moved, 1),
           "median_error": {a: [round(v, 2) for v in med[a]] for a in arms},
           "hit_rate": {a: [round(v, 4) for v in hit[a]] for a in arms}}

    # --- the instrument check: frame 0 is the tag's own frame ---
    res["instrument_ok"] = bool(hit["eye-V2"][0] >= 0.5 and
                                hit["pixels-NCC"][0] >= 0.5)
    print(f"\n--- instrument check (t0 is the frame the tag came from) ---")
    print(f"  eye-V2 {hit['eye-V2'][0]:.3f}   pixels-NCC "
          f"{hit['pixels-NCC'][0]:.3f}   shuffled "
          f"{hit['shuffled-tag'][0]:.3f}")
    if not res["instrument_ok"]:
        print("  An arm cannot find the tag in the frame the tag was taken "
              "from. That is not a\n  localisation result, it is a broken "
              "instrument. REFUSING to report the rest.")
        json.dump(res, open(out_path, "w"), indent=1)
        return

    # --- H21, over the moving part of the trajectory ---
    mv = slice(1, None)
    best = max(STAGES, key=lambda s: float(np.mean(hit[f"eye-{s}"][mv])))
    h_eye = float(np.mean(hit[f"eye-{best}"][mv]))
    h_shuf = float(np.mean(hit["shuffled-tag"][mv]))
    h_ncc = float(np.mean(hit["pixels-NCC"][mv]))
    res["moving"] = {"eye_best_stage": best, "eye": round(h_eye, 4),
                     "shuffled_tag": round(h_shuf, 4),
                     "pixels_ncc": round(h_ncc, 4)}
    res["tag_localises"] = bool(h_eye > h_shuf + 0.10)
    res["eye_beats_pixels"] = bool(h_eye > h_ncc + 0.05)

    print(f"\n--- H21, while the object is moving (t1..t{len(traj) - 1}) ---")
    print(f"  eye-{best} {h_eye:.3f}   shuffled tag {h_shuf:.3f}   "
          f"pixels-NCC {h_ncc:.3f}")
    print(f"  the tag finds its object: {res['tag_localises']}")
    print(f"  the eye beats pixel template matching: "
          f"{res['eye_beats_pixels']}")
    if res["tag_localises"]:
        print(f"  H21 SUPPORTED. The retinotopy that cost §9.20 its "
              f"translation invariance is what makes\n  this work: the map "
              f"moves the object rather than destroying it, so a stored tag "
              f"still finds it.\n  This is localisation driven by a stored "
              f"feature template -- tag-based top-down attention.")
    else:
        print(f"  H21 FALSIFIED. The stored tag does not find its object "
              f"above the shuffled control, so\n  the top-down heatmap "
              f"`attend()` relies on is not localising anything.")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
