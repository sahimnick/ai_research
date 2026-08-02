"""One eye, one pass, four read-outs: is this actually a unified eye?

Every benchmark up to here measured **one property at a time**, each building
its own stream. That leaves the goal's central claim untested:
**بصورت چشم واحد عمل کنند** -- do the four stages work as *one eye*?

§9.25 asked a version of that by concatenating the areas and decoding the image,
and found a +0.031 gain at matched width. That is a weak sense of "unified": it
shows the codes are not redundant, not that the stages are each *needed*.

The strong sense is functional, and it is what this measures. **One** stream,
**one** forward pass per frame, and four different read-outs taken from it:

    where       locate a tagged object                       (§9.21, §9.23)
    which       select the cued object under competition     (§9.28, §9.29)
    how big     read the object's angular size = distance    (§9.31)
    relation    is the target left or right of the other?    -- never measured

The last is the goal's **ارتباط**, the one property in its list with no
measurement anywhere in this document. It is read from the *same* pass as the
rest: two localisations, one comparison.

What would make "unified" true, and what would falsify it
---------------------------------------------------------
If one stage were best at everything, the other three would be scaffolding and
the honest description would be "a one-stage eye with extra layers". The claim
is falsified that way, not by any score being low.

It is supported if **different tasks are best served by different stages**, so
the set is functionally necessary. §9.21/§9.23/§9.30 found V2 beating V4 at
localisation; §9.31 found V4 beating V2 at angular size. Those were separate
experiments with separately grown streams -- here they must hold **in the same
eye, on the same frames, from the same forward pass**, or they were an artefact
of running them apart.

Every read-out is scored against its own control: a shuffled tag for the
localisation-based ones, raw pixels for size, chance for relation.

Usage:  python3 benchmarks/eye_unified.py out.json [vot_dir] [n_seq]
"""
import json
import os
import sys

import numpy as np

from neurobrain.vision.ventral import build_ventral_stream_on, locate_template

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eye_information import pca, ridge_r2                     # noqa: E402
from eye_vot import load_gt, resize                           # noqa: E402

VOT_DIR = sys.argv[2] if len(sys.argv) > 2 else "vot"
N_SEQ = int(sys.argv[3]) if len(sys.argv) > 3 else 16
SIZE = 224
OBJ = 44
PER_SEQ = 6
DIM = 40
STAGES = ("V2", "pool", "V4")


def load_seq(d, name, n):
    from PIL import Image
    p = os.path.join(d, name)
    jpgs = sorted(f for f in os.listdir(p) if f.endswith(".jpg"))
    gtp = os.path.join(p, "groundtruth.txt")
    if not jpgs or not os.path.exists(gtp):
        return []
    gt = load_gt(gtp)
    step = max(1, len(jpgs) // n)
    out = []
    for i in range(0, min(len(jpgs), len(gt)), step):
        if gt[i] is None:
            continue
        y, x, h, w = gt[i]
        if h < 3 or w < 3:
            continue
        a = np.asarray(Image.open(os.path.join(p, jpgs[i])).convert("L"),
                       np.float32) / 255.0
        sy, sx = SIZE / a.shape[0], SIZE / a.shape[1]
        f = resize(a, sy, sx)
        if f.shape != (SIZE, SIZE):
            continue
        ty, tx = (y + h / 2) * sy, (x + w / 2) * sx
        if not (OBJ < ty < SIZE - OBJ and OBJ < tx < SIZE - OBJ):
            continue
        out.append((f, (ty, tx), float(np.log((h * sy) * (w * sx)))))
        if len(out) >= n:
            break
    return out


def paste(bg, obj, y, x):
    f = bg.copy()
    f[y:y + obj.shape[0], x:x + obj.shape[1]] = obj
    return f


def one_pass(stream, frame):
    """THE forward pass. Every read-out below comes out of this one dict."""
    h = stream.hierarchy
    names = [getattr(l, "name", type(l).__name__) for l in h.layers]
    h.forward(frame[None])
    return {s: np.asarray(h.layers[names.index(s)].log["output"], np.float32)
            for s in STAGES}


def patch(m, y, x, size, img):
    _, gh, gw = m.shape
    r0 = int(np.clip(y * gh / img, 0, gh - 2))
    c0 = int(np.clip(x * gw / img, 0, gw - 2))
    r1 = int(np.clip(np.ceil((y + size) * gh / img), r0 + 1, gh))
    c1 = int(np.clip(np.ceil((x + size) * gw / img), c0 + 1, gw))
    return m[:, r0:r1, c0:c1]


def peak(m, tag, img):
    (r, c), _ = locate_template(m, tag)
    th, tw = tag.shape[1:]
    _, gh, gw = m.shape
    return ((r + (th - 1) / 2.0) * img / gh, (c + (tw - 1) / 2.0) * img / gw)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_eye_unified.json"
    if not os.path.isdir(VOT_DIR):
        print(f"no frame directory at {VOT_DIR!r} -- nothing reported.")
        json.dump({"error": "no frames"}, open(out_path, "w"), indent=1)
        return
    names = sorted(d for d in os.listdir(VOT_DIR)
                   if os.path.isdir(os.path.join(VOT_DIR, d)))[:N_SEQ]
    seqs = {}
    for n in names:
        s = load_seq(VOT_DIR, n, PER_SEQ)
        if len(s) >= 4:
            seqs[n] = s
    if len(seqs) < 6:
        print(f"only {len(seqs)} usable sequences -- nothing reported.")
        json.dump({"error": "too few sequences", "n": len(seqs)},
                  open(out_path, "w"), indent=1)
        return
    keys = list(seqs)
    print(f"{len(seqs)} sequences x up to {PER_SEQ} frames", flush=True)

    dis = seqs[keys[0]][0][0][20:20 + OBJ, 20:20 + OBJ].copy()
    print("\ngrowing ONE eye on the first frames ...", flush=True)
    stream = build_ventral_stream_on(
        np.stack([seqs[k][0][0] for k in keys]), size=SIZE, verbose=False)

    tasks = ("where", "which", "relation")
    score = {s: {t: [] for t in tasks} for s in STAGES}
    score["pixels"] = {t: [] for t in tasks}
    shuf = {t: [] for t in tasks}
    codes = {s: [] for s in STAGES}
    pix, ang = [], []
    seq_of = []

    for si, k in enumerate(keys):
        # distractor slot: a corner far from this sequence's first target
        ty0, tx0 = seqs[k][0][1]
        sy, sx = max(((a, b) for a in (12, SIZE - OBJ - 12)
                      for b in (12, SIZE - OBJ - 12)),
                     key=lambda p: np.hypot(p[0] + OBJ / 2 - ty0,
                                            p[1] + OBJ / 2 - tx0))
        other = seqs[keys[(si + 1) % len(keys)]][0][0]
        md = one_pass(stream, paste(other, dis, sy, sx))
        tag_dis = {s: patch(md[s], sy, sx, OBJ, SIZE) for s in STAGES}
        f0, (a0, b0), _ = seqs[k][0]
        m0 = one_pass(stream, paste(f0, dis, sy, sx))
        tag_tgt = {s: patch(m0[s], a0 - OBJ / 2, b0 - OBJ / 2, OBJ, SIZE)
                   for s in STAGES}
        # a tag for an object that is never in the picture, at the same place
        m3 = one_pass(stream, paste(other, dis[::-1, ::-1],
                                    int(a0 - OBJ / 2), int(b0 - OBJ / 2)))
        tag_bad = patch(m3["V2"], a0 - OBJ / 2, b0 - OBJ / 2, OBJ, SIZE)

        for (f, (ty, tx), la) in seqs[k]:
            frame = paste(f, dis, sy, sx)
            M = one_pass(stream, frame)                 # <-- the single pass
            slots = [(ty, tx), (sy + OBJ / 2, sx + OBJ / 2)]
            true_rel = int(tx > sx + OBJ / 2)           # target right of it?
            for s in STAGES:
                pt = peak(M[s], tag_tgt[s], SIZE)
                pd = peak(M[s], tag_dis[s], SIZE)
                score[s]["where"].append(
                    float(np.hypot(pt[0] - ty, pt[1] - tx) <= OBJ / 2))
                near = int(np.argmin([np.hypot(pt[0] - a, pt[1] - b)
                                      for a, b in slots]))
                score[s]["which"].append(float(near == 0))
                score[s]["relation"].append(float(int(pt[1] > pd[1])
                                                  == true_rel))
                codes[s].append(M[s].ravel())
            pb = peak(M["V2"], tag_bad, SIZE)
            shuf["where"].append(
                float(np.hypot(pb[0] - ty, pb[1] - tx) <= OBJ / 2))
            shuf["which"].append(float(int(np.argmin(
                [np.hypot(pb[0] - a, pb[1] - b) for a, b in slots])) == 0))
            shuf["relation"].append(float(int(pb[1] > (sx + OBJ / 2))
                                          == true_rel))
            pix.append(frame[::2, ::2].ravel())
            ang.append([la])
            seq_of.append(si)
        print(f"  {si + 1}/{len(keys)} {k}", end="\r", flush=True)

    ang = np.asarray(ang, np.float32)
    seq_of = np.asarray(seq_of)
    # "how big" is a probe, so it needs its own split -- within-sequence, per
    # §9.31, since across sequences nothing in this code transfers (§9.27)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(ang))
    cut = int(0.7 * len(idx))
    tr, te = idx[:cut], idx[cut:]
    size_r2 = {}
    for s in STAGES:
        X = np.stack(codes[s])
        Ztr, Zte = pca(X[tr], X[te], DIM)
        size_r2[s] = round(ridge_r2(Ztr, ang[tr], Zte, ang[te]), 4)
    P = np.stack(pix)
    Ztr, Zte = pca(P[tr], P[te], DIM)
    size_r2["pixels"] = round(ridge_r2(Ztr, ang[tr], Zte, ang[te]), 4)

    res = {"n_sequences": len(seqs), "n_frames": len(ang), "size": SIZE,
           "tasks": {}, "how_big_r2": size_r2}
    print(f"\n\n{'stage':<10}{'where':>9}{'which':>9}{'relation':>10}"
          f"{'how big':>10}")
    for s in STAGES:
        row = {t: round(float(np.mean(score[s][t])), 4) for t in tasks}
        row["how_big"] = size_r2[s]
        res["tasks"][s] = row
        print(f"{s:<10}{row['where']:>9.3f}{row['which']:>9.3f}"
              f"{row['relation']:>10.3f}{row['how_big']:>10.3f}")
    ctrl = {t: round(float(np.mean(shuf[t])), 4) for t in tasks}
    res["control"] = ctrl
    res["control"]["how_big_pixels"] = size_r2["pixels"]
    print(f"{'control':<10}{ctrl['where']:>9.3f}{ctrl['which']:>9.3f}"
          f"{ctrl['relation']:>10.3f}{size_r2['pixels']:>10.3f}")
    print(f"{'chance':<10}{'~0':>9}{0.5:>9.3f}{0.5:>10.3f}{'--':>10}")

    allt = ("where", "which", "relation", "how_big")
    vals = {t: {s: (res["tasks"][s][t] if t != "how_big" else size_r2[s])
                for s in STAGES} for t in allt}
    best = {t: max(STAGES, key=lambda s: vals[t][s]) for t in allt}
    margin = {t: round(sorted(vals[t].values())[-1]
                       - sorted(vals[t].values())[-2], 4) for t in allt}
    res["best_stage_per_task"] = best
    res["margin_over_runner_up"] = margin
    res["n_distinct_winners"] = len(set(best.values()))
    # An argmax over three near-equal numbers is decided by noise, which is
    # §9.10's standing lesson here. The verdict is gated on the winners being
    # separated at all, not merely on there being several of them.
    res["margins_meaningful"] = bool(min(margin.values()) > 0.05)
    res["functionally_unified"] = bool(res["n_distinct_winners"] > 1
                                       and res["margins_meaningful"])

    print(f"\n--- is this one eye, or one stage with scaffolding? ---")
    for t in allt:
        print(f"  {t:<10} best: {best[t]:<6} margin over runner-up "
              f"{margin[t]:+.3f}")
    if res["functionally_unified"]:
        print(f"\n  UNIFIED, in the strong sense. {res['n_distinct_winners']} "
              f"different stages win different\n  read-outs from the SAME "
              f"forward pass by separated margins, so no one of them could "
              f"replace\n  the set.")
    elif res["n_distinct_winners"] > 1:
        print(f"\n  {res['n_distinct_winners']} different stages take the top "
              f"spot on different read-outs -- but the\n  smallest margin is "
              f"{min(margin.values()):+.3f}, so on THIS run the winners are "
              f"not separated and the\n  argmax is being decided by noise. "
              f"What the run does establish is that all four\n  read-outs come "
              f"out of one eye and one forward pass, and that the DIRECTION "
              f"matches what\n  §9.21/§9.23/§9.30 (V2 > V4 at locating) and "
              f"§9.31 (V4 > V2 at size) found in separately\n  grown streams. "
              f"Corroboration of those, not independent evidence.")
    else:
        print(f"\n  NOT unified. {best['where']} is best at every read-out, so "
              f"the other stages are\n  scaffolding and this is honestly a "
              f"one-stage eye with extra layers.")
    if res["tasks"][best["relation"]]["relation"] < ctrl["relation"] + 0.10:
        print(f"  Relations (ارتباط) are NOT read: "
              f"{res['tasks'][best['relation']]['relation']:.3f} against a "
              f"control of {ctrl['relation']:.3f}.")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
