"""When it answers, is it right? Abstention, and the price of being sure.

The visual report drew a box on every frame, including frames where the match
score was 0.19 and the box was nowhere near the object. A system that always
answers is not a system with poor accuracy -- it is a system that cannot say
"I don't know", and every weak frame becomes a false detection.

This measures the alternative. `locate_template` returns a correlation peak;
that peak is a **confidence**. Withhold the answer when it is low and two
numbers move in opposite directions:

    precision   of the frames it DID answer, how many were right
    coverage    what fraction of frames it answered at all

The question is whether a threshold exists where precision clears **0.90** at
coverage worth having. If it does, the honest product is a tracker that
abstains; if it does not, no threshold makes this reliable and that is the
finding.

The threshold is CHOSEN and REPORTED on different sequences
-----------------------------------------------------------
The first version of this picked "the lowest threshold whose precision >= 0.90"
and then reported that precision -- on the same frames. That is selection on the
test set, and it makes any threshold look like it works: with enough thresholds
to try, one of them clears the bar by luck.

So the sequences are split. The threshold is chosen on the **calibration** half
and the precision at that threshold is reported on the **held-out** half, which
the choice never saw. The gap between the two is printed, because that gap IS
the overfitting, and it is the only honest way to quote a number that a
threshold search produced.

What "right" means here
-----------------------
Centre error within half the tag's own size -- the same criterion §9.21/§9.23
used, so these numbers sit beside those rather than replacing them.

Controls
--------
    shuffled-tag   a tag for an object that is NOT in the picture. This is the
                   one that matters: if a never-present tag scores the same
                   confidence as a real one, the confidence is not evidence and
                   abstention cannot work. It is the difference between the two
                   distributions that a threshold exploits.
    chance         the hit rate of a uniformly random guess in the frame

Usage:  python3 benchmarks/eye_confidence.py out.json [vot_dir] [n_seq]
"""
import json
import os
import sys

import numpy as np

from neurobrain.vision.unified_eye import UnifiedEye

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eye_vot import load_gt, resize                           # noqa: E402

VOT_DIR = sys.argv[2] if len(sys.argv) > 2 else "vot"
N_SEQ = int(sys.argv[3]) if len(sys.argv) > 3 else 24
SIZE = 224
OBJ = 44
REGIMES = {"consecutive": 1, "every-10th": 10}
PER_SEQ = 12


def load_seq(d, name, n, stride=1):
    from PIL import Image
    p = os.path.join(d, name)
    jpgs = sorted(f for f in os.listdir(p) if f.endswith(".jpg"))
    gtp = os.path.join(p, "groundtruth.txt")
    if not jpgs or not os.path.exists(gtp):
        return []
    gt = load_gt(gtp)
    out = []
    for i in range(0, min(len(jpgs), len(gt)), stride):
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
        if not (OBJ / 2 < ty < SIZE - OBJ / 2 and OBJ / 2 < tx < SIZE - OBJ / 2):
            continue
        out.append((f, (ty, tx)))
        if len(out) >= n:
            break
    return out


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_eye_confidence.json"
    if not os.path.isdir(VOT_DIR):
        print(f"no video at {VOT_DIR!r} -- nothing reported.")
        json.dump({"error": "no frames"}, open(out_path, "w"), indent=1)
        return
    names = sorted(d for d in os.listdir(VOT_DIR)
                   if os.path.isdir(os.path.join(VOT_DIR, d)))[:N_SEQ]
    stride = REGIMES[os.environ.get("REGIME", "consecutive")]
    print(f"regime: every {stride} frame(s)")
    seqs = {}
    for n in names:
        s = load_seq(VOT_DIR, n, PER_SEQ, stride)
        if len(s) >= 5:
            seqs[n] = s
    if len(seqs) < 6:
        print(f"only {len(seqs)} usable sequences -- nothing reported.")
        json.dump({"error": "too few sequences", "n": len(seqs)},
                  open(out_path, "w"), indent=1)
        return
    keys = list(seqs)
    print(f"{len(seqs)} sequences x {PER_SEQ} frames", flush=True)

    # gate off here: this benchmark IS the measurement of where to put it
    eye = UnifiedEye(size=SIZE, min_confidence=0.0)
    eye.develop([seqs[k][0][0] for k in keys])
    res_regime = stride

    conf, ok, arm, seq_of = [], [], [], []
    for si, k in enumerate(keys):
        f0, (ty0, tx0) = seqs[k][0]
        eye.tags.clear()
        eye.add_tag("real", f0, (ty0 - OBJ / 2, tx0 - OBJ / 2, OBJ, OBJ))
        # a tag for something that is NOT in this footage: taken from another
        # sequence and mirrored, so it matches nothing here by construction
        other, _ = seqs[keys[(si + 1) % len(keys)]][0]
        eye.add_tag("absent", other[::-1, ::-1].copy(),
                    (ty0 - OBJ / 2, tx0 - OBJ / 2, OBJ, OBJ))
        for f, (ty, tx) in seqs[k][1:]:
            p = eye.look(f, relations=False)
            for name in ("real", "absent"):
                if name not in p.where:
                    continue
                py, px = p.where[name]
                conf.append(p.confidence[name])
                ok.append(float(np.hypot(py - ty, px - tx) <= OBJ / 2))
                arm.append(name)
                seq_of.append(si)
        print(f"  {si + 1}/{len(keys)} {k}", end="\r", flush=True)

    conf = np.asarray(conf, np.float32)
    ok = np.asarray(ok, np.float32)
    arm = np.asarray(arm)
    seq_of = np.asarray(seq_of)
    # split by SEQUENCE, not by frame: frames from one video are near-copies of
    # each other, so a frame-level split would leak the answer across it
    half = set(range(0, len(keys), 2))
    calib = np.array([s in half for s in seq_of])
    real = arm == "real"
    absent = arm == "absent"
    # chance: a uniformly placed guess lands within OBJ/2 of the target
    chance = float(np.pi * (OBJ / 2) ** 2 / (SIZE * SIZE))

    print(f"\n\nconfidence, real tag   {conf[real].mean():.3f} "
          f"+/- {conf[real].std():.3f}")
    print(f"confidence, absent tag {conf[absent].mean():.3f} "
          f"+/- {conf[absent].std():.3f}")
    sep = float(conf[real].mean() - conf[absent].mean())
    print(f"separation             {sep:+.3f}   <- a threshold can only work "
          f"if this is real")

    res = {"n_sequences": len(seqs), "frames": int(real.sum()),
           "frame_stride": res_regime,
           "chance_hit": round(chance, 4),
           "conf_real": round(float(conf[real].mean()), 4),
           "conf_absent": round(float(conf[absent].mean()), 4),
           "separation": round(sep, 4), "curve": []}

    print(f"\n{'threshold':>10}{'coverage':>11}{'precision':>11}"
          f"{'answered':>10}")
    best = None
    for t in np.arange(0.0, 0.96, 0.05):
        m = real & (conf >= t)
        n = int(m.sum())
        if n == 0:
            continue
        cov = n / float(real.sum())
        prec = float(ok[m].mean())
        res["curve"].append({"threshold": round(float(t), 2),
                             "coverage": round(cov, 4),
                             "precision": round(prec, 4), "n": n})
        print(f"{t:>10.2f}{cov:>11.3f}{prec:>11.3f}{n:>10}")
        if prec >= 0.90 and best is None and cov >= 0.05:
            best = (float(t), cov, prec, n)

    # how often does a NEVER-PRESENT tag get through the threshold?
    print(f"\n{'threshold':>10}{'absent tag passes':>20}"
          f"   <- false alarms a threshold lets through")
    for t in (0.3, 0.5, 0.7, 0.8, 0.9):
        fa = float((conf[absent] >= t).mean())
        res.setdefault("false_alarm", {})[str(t)] = round(fa, 4)
        print(f"{t:>10.2f}{fa:>20.3f}")

    # --- the selection-free answer: the curve on HELD-OUT sequences only ----
    # Picking a threshold and quoting its precision is a search over thresholds,
    # and the search itself inflates the number (measured below at +0.12). The
    # curve avoids the question: it selects nothing, and every point on it is
    # computed on sequences no choice was made from.
    print(f"\n--- held-out curve (no threshold was chosen from this data) ---")
    print(f"{'threshold':>10}{'coverage':>11}{'precision':>11}{'n':>7}")
    res["heldout_curve"] = []
    for t in np.arange(0.0, 0.71, 0.05):
        m = real & (~calib) & (conf >= t)
        n = int(m.sum())
        if n < 15:
            continue
        row = {"threshold": round(float(t), 2),
               "coverage": round(n / float((real & ~calib).sum()), 4),
               "precision": round(float(ok[m].mean()), 4), "n": n}
        res["heldout_curve"].append(row)
        print(f"{t:>10.2f}{row['coverage']:>11.3f}{row['precision']:>11.3f}"
              f"{n:>7}")
    hi = [r for r in res["heldout_curve"] if r["precision"] >= 0.90]
    res["heldout_reaches_90"] = bool(hi)
    if hi:
        b = max(hi, key=lambda r: r["coverage"])
        res["heldout_best"] = b
        print(f"\n  >=0.90 on held-out data at threshold {b['threshold']:.2f}: "
              f"precision {b['precision']:.3f}, answering "
              f"{b['coverage']:.0%}\n  of frames ({b['n']}). Nothing was "
              f"tuned to this half.")
    else:
        print(f"\n  No point on the held-out curve reaches 0.90.")

    # --- and the same thing done the tempting way, to show what it costs ----
    pick, pick_prec = None, None
    for t in np.arange(0.0, 0.96, 0.05):
        m = real & calib & (conf >= t)
        if m.sum() >= 10 and float(ok[m].mean()) >= 0.90:
            pick, pick_prec = float(t), float(ok[m].mean())
            break
    res["reaches_90"] = False
    if pick is not None:
        te = real & (~calib) & (conf >= pick)
        n_te = int(te.sum())
        prec_te = float(ok[te].mean()) if n_te else float("nan")
        cov_te = n_te / float((real & ~calib).sum())
        res["calibrated"] = {
            "threshold": pick,
            "precision_calibration": round(pick_prec, 4),
            "precision_heldout": round(prec_te, 4),
            "coverage_heldout": round(cov_te, 4), "n_heldout": n_te,
            "optimism": round(pick_prec - prec_te, 4)}
        res["reaches_90"] = bool(prec_te >= 0.90)
        print(f"\n--- threshold chosen on one half, reported on the other ---")
        print(f"  chosen threshold          {pick:.2f}")
        print(f"  precision on calibration  {pick_prec:.3f}  <- what a "
              f"same-data report would have claimed")
        print(f"  precision on HELD-OUT     {prec_te:.3f}  ({n_te} frames, "
              f"{cov_te:.0%} coverage)")
        print(f"  optimism from choosing    "
              f"{pick_prec - prec_te:+.3f}")
    if best:
        res["operating_point_same_data"] = {
            "threshold": best[0], "coverage": best[1],
            "precision": best[2], "n": best[3],
            "note": "chosen and reported on the same frames -- optimistic"}
    print(f"\n--- can it be right more than 90% of the time it answers? ---")
    c = res.get("calibrated")
    if c and res["reaches_90"]:
        print(f"  YES, and on data the threshold never saw. Gate at "
              f"{c['threshold']:.2f}: answers {c['coverage_heldout']:.0%} of "
              f"held-out\n  frames and is right {c['precision_heldout']:.1%} "
              f"of those. Below the gate it says LOST rather than guessing.")
    elif c:
        print(f"  NO. A threshold clears 0.90 on the half it was chosen from "
              f"({c['precision_calibration']:.3f}) and\n  only reaches "
              f"{c['precision_heldout']:.3f} on the half it did not "
              f"({c['optimism']:+.3f} optimism). The threshold was fitted to "
              f"noise.")
    else:
        top = max(res["curve"], key=lambda c: c["precision"])
        print(f"  NO. The best precision at any threshold is "
              f"{top['precision']:.3f} (at >= {top['threshold']:.2f}, "
              f"covering\n  {top['coverage']:.0%}). No confidence cut makes "
              f"this reliable, so abstention is not the fix and the\n  "
              f"limit is the representation, not the reporting.")
    print(f"  Chance, for scale: a random guess is right {chance:.3f} of the "
          f"time.")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
