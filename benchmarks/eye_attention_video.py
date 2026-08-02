"""Does top-down selection survive on video? Testing §9.28's stated expectation.

§9.28 demonstrated biased competition -- same image, different cue, different
answer -- at 0.917 cue-following and 0.875 switch rate. It then said, without
running the test:

    "This does NOT establish that attention would survive on video, and the
    honest expectation from §9.24 is that it would degrade the same way."

An expectation asserted is not a result. This runs it.

The design isolates appearance change as the cause
--------------------------------------------------
Each trial scene contains **two competitors**, and they differ in exactly one
way:

    the TARGET      a real VOT object, tagged at frame 0, whose appearance then
                    genuinely changes -- it translates, rotates, changes scale,
                    deforms and relights as the video runs
    the DISTRACTOR  a composited patch, pixel-identical at every frame, tagged
                    on a different background

Both are cued through the identical machinery and scored the same way, against
frame lag. If cue-following for the *target* decays while the *distractor*
stays flat, appearance change is the cause and the mechanism is otherwise
intact. If both decay, the problem is the growing scene difference rather than
the target's own change. If neither decays, §9.28's expectation was wrong.

That contrast is the point: an overall decay number could not distinguish these,
and §9.24 could not either, having no unchanging competitor in it.

The search field is fixed per sequence -- a 224 px window on the initial target,
with frames dropped when ground truth leaves it -- so "the target got harder to
find" cannot be "the target left the picture".

Usage:  python3 benchmarks/eye_attention_video.py out.json [vot_dir] [n_seq]
"""
import json
import os
import sys

import numpy as np

from neurobrain.vision.ventral import build_ventral_stream_on, locate_template

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eye_breadth import seq_frames                            # noqa: E402
from eye_vot import load_gt, resize                           # noqa: E402

VOT_DIR = sys.argv[2] if len(sys.argv) > 2 else "vot"
N_SEQ = int(sys.argv[3]) if len(sys.argv) > 3 else 14
SIZE = 224
OBJ = 44
TARGET_PX = 44
LAGS = (0, 4, 8, 16, 32)
AREA = "V2"


def load_seq(d, name):
    """Rescaled frames + target boxes, in a fixed 224px field on frame 0."""
    from PIL import Image
    p = os.path.join(d, name)
    jpgs = sorted(f for f in os.listdir(p) if f.endswith(".jpg"))
    gtp = os.path.join(p, "groundtruth.txt")
    if not jpgs or not os.path.exists(gtp):
        return None
    gt = load_gt(gtp)
    if len(gt) < max(LAGS) + 1 or gt[0] is None:
        return None
    g0 = gt[0]
    if min(g0[2], g0[3]) < 4:
        return None
    s = TARGET_PX / max(g0[2], g0[3])
    if not (0.02 < s < 4.0):
        return None
    cy, cx = (g0[0] + g0[2] / 2) * s, (g0[1] + g0[3] / 2) * s
    out = {}
    for lag in LAGS:
        if lag >= len(gt) or gt[lag] is None:
            return None
        a = np.asarray(Image.open(os.path.join(p, jpgs[lag])).convert("L"),
                       np.float32) / 255.0
        f = resize(a, s, s)
        y0 = int(np.clip(round(cy - SIZE / 2), 0, max(0, f.shape[0] - SIZE)))
        x0 = int(np.clip(round(cx - SIZE / 2), 0, max(0, f.shape[1] - SIZE)))
        v = f[y0:y0 + SIZE, x0:x0 + SIZE]
        if v.shape != (SIZE, SIZE):
            return None
        gy, gx, gh, gw = gt[lag]
        ty, tx = (gy + gh / 2) * s - y0, (gx + gw / 2) * s - x0
        if not (OBJ < ty < SIZE - OBJ and OBJ < tx < SIZE - OBJ):
            return None                    # target left the fixed field
        out[lag] = (v, (ty, tx))
    return out


def paste(bg, obj, y, x):
    f = bg.copy()
    f[y:y + obj.shape[0], x:x + obj.shape[1]] = obj
    return f


def amap(stream, frame):
    h = stream.hierarchy
    names = [getattr(l, "name", type(l).__name__) for l in h.layers]
    h.forward(frame[None])
    return np.asarray(h.layers[names.index(AREA)].log["output"], np.float32)


def patch(m, y, x, size, img):
    _, gh, gw = m.shape
    r0 = int(np.clip(y * gh / img, 0, gh - 2))
    c0 = int(np.clip(x * gw / img, 0, gw - 2))
    r1 = int(np.clip(np.ceil((y + size) * gh / img), r0 + 1, gh))
    c1 = int(np.clip(np.ceil((x + size) * gw / img), c0 + 1, gw))
    return m[:, r0:r1, c0:c1]


def peak_px(m, tag, img):
    (r, c), _ = locate_template(m, tag)
    th, tw = tag.shape[1:]
    _, gh, gw = m.shape
    return ((r + (th - 1) / 2.0) * img / gh, (c + (tw - 1) / 2.0) * img / gw)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_attn_video.json"
    if not os.path.isdir(VOT_DIR):
        print(f"no frame directory at {VOT_DIR!r} -- nothing reported.")
        json.dump({"error": "no frames"}, open(out_path, "w"), indent=1)
        return
    names = sorted(d for d in os.listdir(VOT_DIR)
                   if os.path.isdir(os.path.join(VOT_DIR, d)))
    seqs = {}
    for n in names:
        s = load_seq(VOT_DIR, n)
        if s is not None:
            seqs[n] = s
        if len(seqs) >= N_SEQ:
            break
    if len(seqs) < 5:
        print(f"only {len(seqs)} usable sequences -- nothing reported.")
        json.dump({"error": "too few sequences", "n": len(seqs)},
                  open(out_path, "w"), indent=1)
        return
    print(f"{len(seqs)} sequences: {', '.join(seqs)}", flush=True)

    # distractor + a never-present third object, from frames of other videos
    extra = []
    for n in names[::-1]:
        extra += seq_frames(VOT_DIR, n, 1)
        if len(extra) >= 4:
            break
    dis = extra[0][60:60 + OBJ, 60:60 + OBJ]
    third = extra[1][40:40 + OBJ, 90:90 + OBJ]

    print("\ngrowing the eye ...", flush=True)
    firsts = [s[0][0] for s in seqs.values()]
    stream = build_ventral_stream_on(np.stack(firsts), size=SIZE, verbose=False)

    def slot_for(ty, tx):
        """A distractor corner far from the target, in the fixed field."""
        best, bd = (12, 12), -1.0
        for sy in (12, SIZE - OBJ - 12):
            for sx in (12, SIZE - OBJ - 12):
                d = np.hypot(sy + OBJ / 2 - ty, sx + OBJ / 2 - tx)
                if d > bd:
                    best, bd = (sy, sx), d
        return best

    per_lag = {lag: {"target": [], "distractor": [], "shuffled": [],
                     "switch": []} for lag in LAGS}
    for si, (name, s) in enumerate(seqs.items()):
        v0, (ty0, tx0) = s[0]
        sy, sx = slot_for(ty0, tx0)
        # tag the distractor on ANOTHER sequence's background
        other = firsts[(si + 1) % len(firsts)]
        md = amap(stream, paste(other, dis, sy, sx))
        tag_dis = patch(md, sy, sx, OBJ, SIZE)
        # The never-present control is tagged at the TARGET's position, not the
        # distractor's. Tagged at the distractor slot it carries that corner's
        # background statistics and is pushed away from the target, which
        # flatters the real tag -- a control must not be biased toward the
        # hypothesis. Here the bias runs the other way, against it.
        ty0i, tx0i = int(np.clip(ty0 - OBJ / 2, 0, SIZE - OBJ)), \
            int(np.clip(tx0 - OBJ / 2, 0, SIZE - OBJ))
        mt = amap(stream, paste(other, third, ty0i, tx0i))
        tag_third = patch(mt, ty0i, tx0i, OBJ, SIZE)
        # tag the real target at frame 0, in its own scene
        m0 = amap(stream, paste(v0, dis, sy, sx))
        tag_tgt = patch(m0, ty0 - OBJ / 2, tx0 - OBJ / 2, OBJ, SIZE)

        for lag in LAGS:
            v, (ty, tx) = s[lag]
            f = paste(v, dis, sy, sx)
            m = amap(stream, f)
            slots = [(ty, tx), (sy + OBJ / 2, sx + OBJ / 2)]

            def pick(tag):
                py, px = peak_px(m, tag, SIZE)
                return int(np.argmin([np.hypot(py - a, px - b)
                                      for a, b in slots]))
            pt, pd = pick(tag_tgt), pick(tag_dis)
            per_lag[lag]["target"].append(float(pt == 0))
            per_lag[lag]["distractor"].append(float(pd == 1))
            per_lag[lag]["shuffled"].append(float(pick(tag_third) == 0))
            per_lag[lag]["switch"].append(float(pt != pd))
        print(f"  {si + 1}/{len(seqs)} {name}", end="\r", flush=True)

    res = {"n_sequences": len(seqs), "lags": list(LAGS), "size": SIZE,
           "object_px": OBJ, "by_lag": {}}
    print(f"\n\n{'lag':>6}{'target':>10}{'distractor':>13}{'shuffled':>11}"
          f"{'switch':>9}")
    for lag in LAGS:
        d = {k: round(float(np.mean(v)), 4) for k, v in per_lag[lag].items()}
        res["by_lag"][str(lag)] = d
        print(f"{lag:>6}{d['target']:>10.3f}{d['distractor']:>13.3f}"
              f"{d['shuffled']:>11.3f}{d['switch']:>9.3f}")

    t0, tN = res["by_lag"]["0"]["target"], res["by_lag"][str(LAGS[-1])]["target"]
    d0, dN = (res["by_lag"]["0"]["distractor"],
              res["by_lag"][str(LAGS[-1])]["distractor"])
    res["target_decay"] = round(t0 - tN, 4)
    res["distractor_decay"] = round(d0 - dN, 4)
    res["selection_survives_video"] = bool(tN > 0.65)
    res["decay_is_appearance"] = bool(res["target_decay"] >
                                      res["distractor_decay"] + 0.15)

    print(f"\n--- does top-down selection survive on video? ---")
    print(f"  target      {t0:.3f} -> {tN:.3f}  (decay "
          f"{res['target_decay']:+.3f})   appearance really changes")
    print(f"  distractor  {d0:.3f} -> {dN:.3f}  (decay "
          f"{res['distractor_decay']:+.3f})   pixel-identical throughout")
    if res["selection_survives_video"]:
        print(f"  §9.28's expectation was WRONG: selection of the real, "
              f"changing target holds at {tN:.3f}\n  after {LAGS[-1]} frames. "
              f"Attention survives on video.")
    else:
        print(f"  §9.28's expectation is CONFIRMED: selection of the real "
              f"target falls to {tN:.3f} by frame\n  {LAGS[-1]}. Top-down "
              f"attention does not survive real appearance change, and the "
              f"0.917 of §9.28\n  belonged to its unchanged-object condition.")
    if res["decay_is_appearance"]:
        print(f"  And the cause is located: the pixel-identical distractor, "
              f"cued through the SAME machinery\n  in the SAME frames, decays "
              f"only {res['distractor_decay']:+.3f} against the target's "
              f"{res['target_decay']:+.3f}. The mechanism is\n  intact; what "
              f"it cannot survive is the object changing.")
    else:
        print(f"  The distractor decays comparably "
              f"({res['distractor_decay']:+.3f} vs {res['target_decay']:+.3f}),"
              f" so this is not\n  specifically about the target's appearance "
              f"-- the whole scene drifting is enough.")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
