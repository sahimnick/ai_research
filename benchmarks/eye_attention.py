"""Top-down attention as GATING: same image, different cue, different answer.

§9.21–§9.23 measured the "where" half of attention — a spatial tag finds its
object. The half repeatedly named as missing is **selection**: the cue biasing
what the system picks out of a scene that contains more than one thing. That is
the goal's **توجه بالا به پایین**, and it is the part `attend()` claims and
cannot deliver, because it is built on the averaged prototype §9.21 showed
scores 0.000 at finding a tag in the tag's own frame.

The signature, and why it cannot be faked feed-forward
------------------------------------------------------
Two objects are composited into one real scene. The **image is identical** for
both trials; only the cue changes. So:

    cue A  ->  selects A
    cue B  ->  selects B          from the SAME input

No feed-forward pass can do this. Whatever a stimulus-driven saliency mechanism
picks, it picks the same thing both times — its switch rate is **0 by
construction**, which is why it is the arm to beat rather than a number to
gloss. Biased competition is exactly the claim that the answer moves when only
the goal moves.

Two measures, because one of them can be passed by accident:

    cue-following   is the selected object the cued one? (chance 0.5)
    switch rate     does cueing A vs B change the answer on the SAME image?
                    (feed-forward saliency = 0.000, by construction)

Controls:

    saliency        pick the highest-energy location -- the feed-forward arm
    shuffled cue    a THIRD object's tag, never present in the image: keeps the
                    machinery and destroys only the correspondence

The tag is taken on a DIFFERENT background
------------------------------------------
Each object is tagged from a reference frame whose background is a different
scene from the test frame. Tagging and testing on the same background would let
a tag match the surround rather than the object, and the benchmark would report
attention when it had measured wallpaper.

Which object sits at which position is randomised per trial, so identity is not
confounded with location — otherwise "cue A" could be answered by a fixed
positional bias.

Usage:  python3 benchmarks/eye_attention.py out_eye_attention.json [vot_dir] [n]
"""
import json
import os
import sys

import numpy as np

from neurobrain.vision.ventral import build_ventral_stream_on, locate_template

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eye_breadth import seq_frames                            # noqa: E402

VOT_DIR = sys.argv[2] if len(sys.argv) > 2 else "vot"
N_TRIAL = int(sys.argv[3]) if len(sys.argv) > 3 else 48
SIZE = 224
OBJ = 44
SLOTS = ((52, 34), (128, 146))          # two well-separated placements
AREA = "V2"


def paste(bg, obj, y, x):
    f = bg.copy()
    f[y:y + obj.shape[0], x:x + obj.shape[1]] = obj
    return f


def amap(stream, frame):
    h = stream.hierarchy
    names = [getattr(l, "name", type(l).__name__) for l in h.layers]
    h.forward(frame[None])
    return np.asarray(h.layers[names.index(AREA)].log["output"], np.float32)


def cell_box(gh, gw, y, x, size, img):
    r0, c0 = int(y * gh / img), int(x * gw / img)
    r1 = max(int((y + size) * gh / img) + 1, r0 + 1)
    c1 = max(int((x + size) * gw / img) + 1, c0 + 1)
    return r0, r1, c0, c1


def which_slot(py, px):
    """Which of the two placements is this peak nearest?"""
    d = [np.hypot(py - (sy + OBJ / 2), px - (sx + OBJ / 2)) for sy, sx in SLOTS]
    return int(np.argmin(d))


def peak_px(m, tag, img):
    (r, c), _ = locate_template(m, tag)
    th, tw = tag.shape[1:]
    _, gh, gw = m.shape
    return ((r + (th - 1) / 2.0) * img / gh, (c + (tw - 1) / 2.0) * img / gw)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_eye_attention.json"
    if not os.path.isdir(VOT_DIR):
        print(f"no frame directory at {VOT_DIR!r} -- nothing reported.")
        json.dump({"error": "no frames"}, open(out_path, "w"), indent=1)
        return
    seqs = sorted(d for d in os.listdir(VOT_DIR)
                  if os.path.isdir(os.path.join(VOT_DIR, d)))
    frames = []
    for s in seqs:
        frames += seq_frames(VOT_DIR, s, 3)
        if len(frames) >= N_TRIAL + 12:
            break
    if len(frames) < 20:
        print(f"only {len(frames)} frames -- nothing reported.")
        json.dump({"error": "too few frames", "n": len(frames)},
                  open(out_path, "w"), indent=1)
        return

    rng = np.random.default_rng(0)
    # three real objects: two competing in the image, one never present
    src = frames[-6:]
    objs = [src[0][70:70 + OBJ, 70:70 + OBJ], src[1][90:90 + OBJ, 60:60 + OBJ],
            src[2][50:50 + OBJ, 110:110 + OBJ]]
    ref_bg = frames[-6 + 3:]                      # backgrounds for TAGGING only
    bgs = frames[:N_TRIAL]
    print(f"{len(bgs)} trial scenes, objects {OBJ}px, slots {SLOTS}",
          flush=True)

    print("\ngrowing the eye ...", flush=True)
    stream = build_ventral_stream_on(
        np.stack([paste(b, objs[0], *SLOTS[0]) for b in bgs[:24]]),
        size=SIZE, verbose=False)

    # tags come from a DIFFERENT background than any trial scene
    tags = []
    for i, o in enumerate(objs):
        f = paste(ref_bg[i % len(ref_bg)], o, *SLOTS[0])
        m = amap(stream, f)
        _, gh, gw = m.shape
        r0, r1, c0, c1 = cell_box(gh, gw, SLOTS[0][0], SLOTS[0][1], OBJ, SIZE)
        tags.append(m[:, r0:r1, c0:c1])
    print(f"  tagged {len(tags)} objects on held-out backgrounds", flush=True)

    hit = {"cued": [], "shuffled": []}
    switch = {"cued": [], "shuffled": [], "saliency": []}
    for t, bg in enumerate(bgs):
        order = [0, 1] if rng.random() < 0.5 else [1, 0]   # which object where
        f = paste(paste(bg, objs[order[0]], *SLOTS[0]),
                  objs[order[1]], *SLOTS[1])
        m = amap(stream, f)
        # feed-forward saliency: highest-energy location, cue-blind
        e = np.linalg.norm(m, axis=0)
        _, gh, gw = m.shape
        sr, sc = np.unravel_index(int(np.argmax(e)), e.shape)
        sal = which_slot((sr + 0.5) * SIZE / gh, (sc + 0.5) * SIZE / gw)
        switch["saliency"].append(0.0)      # cue-blind: identical both trials

        picks, picks_sh = [], []
        for cue in (0, 1):
            truth = order.index(cue)                   # slot holding the cue
            picks.append(which_slot(*peak_px(m, tags[cue], SIZE)))
            hit["cued"].append(float(picks[-1] == truth))
            picks_sh.append(which_slot(*peak_px(m, tags[2], SIZE)))
            hit["shuffled"].append(float(picks_sh[-1] == truth))
        switch["cued"].append(float(picks[0] != picks[1]))
        switch["shuffled"].append(float(picks_sh[0] != picks_sh[1]))
        if (t + 1) % 10 == 0:
            print(f"  {t + 1}/{len(bgs)}", end="\r", flush=True)

    res = {"n_trials": len(bgs), "object_px": OBJ, "slots": [list(s)
                                                             for s in SLOTS],
           "area": AREA,
           "cue_following": {k: round(float(np.mean(v)), 4)
                             for k, v in hit.items()},
           "switch_rate": {k: round(float(np.mean(v)), 4)
                           for k, v in switch.items()},
           "saliency_slot_bias": round(float(sal), 4)}

    print(f"\n\n{'arm':<16}{'cue-following':>15}{'switch rate':>14}")
    print(f"{'cued tag':<16}{res['cue_following']['cued']:>15.3f}"
          f"{res['switch_rate']['cued']:>14.3f}")
    print(f"{'shuffled cue':<16}{res['cue_following']['shuffled']:>15.3f}"
          f"{res['switch_rate']['shuffled']:>14.3f}")
    print(f"{'saliency (ff)':<16}{'--':>15}"
          f"{res['switch_rate']['saliency']:>14.3f}")
    print(f"{'chance':<16}{0.5:>15.3f}{'0.000 (ff)':>14}")

    cf = res["cue_following"]["cued"]
    sw = res["switch_rate"]["cued"]
    res["follows_cue"] = bool(cf > 0.65)
    res["biased_competition"] = bool(sw > 0.5 and
                                     cf > res["cue_following"]["shuffled"]
                                     + 0.15)

    print(f"\n--- top-down attention as gating ---")
    if res["biased_competition"]:
        print(f"  YES. The same image gives a different answer when the cue "
              f"changes: switch rate {sw:.3f}\n  against a feed-forward "
              f"saliency arm that is 0.000 by construction, and the cued "
              f"object is\n  selected {cf:.3f} of the time against a shuffled "
              f"cue's {res['cue_following']['shuffled']:.3f}. This is biased "
              f"competition -- goal-\n  directed selection that no feed-forward "
              f"pass can produce.")
    elif res["follows_cue"]:
        print(f"  PARTLY. The cued object is selected {cf:.3f} of the time, "
              f"but the switch rate is only\n  {sw:.3f} -- the cue is followed "
              f"without the answer reliably moving when the cue moves, which "
              f"is\n  weaker than biased competition.")
    else:
        print(f"  NO. Cue-following {cf:.3f} against a shuffled cue's "
              f"{res['cue_following']['shuffled']:.3f} and chance 0.500. The "
              f"tag does not\n  select its object out of a two-object scene, "
              f"so the localisation of §9.23 does not extend to\n  selection "
              f"under competition.")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
