"""Why does tracking collapse -- drift, or per-frame precision?

§9.24 ran one-pass evaluation on VOT2019 and the eye scored 0.094 success AUC
against raw pixels' 0.172. §9.29 then showed the same tag selecting a real,
changing target at 0.818 in a two-alternative choice. Both are true, and §9.29
attributed the gap to two differences at once:

    accumulation   OPE follows the tracker's OWN prediction, so one drift is
                   unrecoverable and errors compound over ~50 frames
    precision      OPE scores continuous localisation by IoU, not a choice
                   between two well-separated slots

Naming two causes together does not identify either. They come apart cleanly:

    OPE            search window centred on the previous PREDICTION  (§9.24)
    re-anchored    search window centred on the previous GROUND TRUTH

Re-anchoring removes accumulation and changes nothing else -- same frames, same
template, same search, same scoring, same per-frame difficulty. So:

    re-anchored HIGH, OPE low   -> the collapse is drift. Per-frame localisation
                                   is sound and the tracker loses the target and
                                   never recovers, which is a tracker problem.
    both LOW                    -> the representation cannot localise precisely
                                   on video at all, and §9.29's 0.818 is what a
                                   two-alternative choice buys over a continuous
                                   one. That is a representation problem.

Raw-pixel NCC runs through both protocols too, since it is the arm that beat the
eye in §9.24 and the comparison is only meaningful if both are measured the same
way.

Note this is **not** a proposal to fix tracking: re-anchoring uses ground truth
at every frame and is not a tracker anyone could run. It is an instrument for
splitting one number into its two causes.

Usage:  python3 benchmarks/eye_track_drift.py out.json [vot_dir] [n_seq]
"""
import json
import os
import sys

import numpy as np

from neurobrain.vision.ventral import build_ventral_stream_on, locate_template

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eye_vot import (SEARCH, STRIDE, TARGET, area_map, iou,   # noqa: E402
                     load_seq, ncc, patch_of, summarise, window)

VOT_DIR = sys.argv[2] if len(sys.argv) > 2 else "vot"
N_SEQ = int(sys.argv[3]) if len(sys.argv) > 3 else 12
STAGES = ("V2",)


def run(stream, frames, boxes, mode, anchor):
    """``anchor`` is 'pred' (OPE, §9.24) or 'gt' (re-anchored, no accumulation)."""
    y0, x0, bh, bw = boxes[0]
    v0, wy, wx = window(frames[0], y0 + bh / 2, x0 + bw / 2, SEARCH)
    if mode == "pixels":
        yy = int(np.clip(round(y0 - wy), 0, max(0, v0.shape[0] - 2)))
        xx = int(np.clip(round(x0 - wx), 0, max(0, v0.shape[1] - 2)))
        tmpl = v0[yy:yy + max(2, int(round(bh))),
                  xx:xx + max(2, int(round(bw)))].copy()
    else:
        tmpl = patch_of(area_map(stream, v0, mode), y0 - wy, x0 - wx,
                        bh, bw, SEARCH)

    th, tw = boxes[0][2], boxes[0][3]
    cy, cx = boxes[0][0] + th / 2, boxes[0][1] + tw / 2
    errs, ious = [], []
    for t in range(1, len(frames)):
        view, wy, wx = window(frames[t], cy, cx, SEARCH)
        if min(view.shape) < 8:
            break
        if mode == "pixels":
            r = ncc(view, tmpl)
            if r is None:
                break
            py, px = r
        else:
            m = area_map(stream, view, mode)
            if m.shape[1] < tmpl.shape[1] or m.shape[2] < tmpl.shape[2]:
                break
            (rr, cc), _ = locate_template(m, tmpl)
            py = (rr + (tmpl.shape[1] - 1) / 2.0) * view.shape[0] / m.shape[1]
            px = (cc + (tmpl.shape[2] - 1) / 2.0) * view.shape[1] / m.shape[2]
        pred = (wy + py, wx + px)
        gy, gx, gh_, gw_ = boxes[t]
        errs.append(float(np.hypot(pred[0] - (gy + gh_ / 2),
                                   pred[1] - (gx + gw_ / 2))))
        ious.append(iou((pred[0] - th / 2, pred[1] - tw / 2, th, tw),
                        boxes[t]))
        # where the NEXT frame's window is centred -- the only thing that varies
        if anchor == "pred":
            cy, cx = pred
        else:
            cy, cx = gy + gh_ / 2, gx + gw_ / 2
    return errs, ious


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_track_drift.json"
    if not os.path.isdir(VOT_DIR):
        print(f"no VOT directory at {VOT_DIR!r} -- nothing reported.")
        json.dump({"error": "no vot dir"}, open(out_path, "w"), indent=1)
        return
    names = sorted(d for d in os.listdir(VOT_DIR)
                   if os.path.isdir(os.path.join(VOT_DIR, d)))
    seqs = {}
    for n in names:
        s = load_seq(os.path.join(VOT_DIR, n), STRIDE)
        if s is not None:
            seqs[n] = s
        if len(seqs) >= N_SEQ:
            break
    if len(seqs) < 5:
        print(f"only {len(seqs)} usable sequences -- nothing reported.")
        json.dump({"error": "too few sequences", "n": len(seqs)},
                  open(out_path, "w"), indent=1)
        return
    print(f"{len(seqs)} sequences, target ~{TARGET}px, search {SEARCH}px",
          flush=True)

    firsts = []
    for fr, bx in seqs.values():
        v, _, _ = window(fr[0], bx[0][0] + bx[0][2] / 2,
                         bx[0][1] + bx[0][3] / 2, SEARCH)
        if v.shape == (SEARCH, SEARCH):
            firsts.append(v)
    print("\ngrowing the eye ...", flush=True)
    stream = build_ventral_stream_on(np.stack(firsts), size=SEARCH,
                                     verbose=False)

    arms = [(m, a) for m in list(STAGES) + ["pixels"]
            for a in ("pred", "gt")]
    got = {f"{m}/{a}": [] for m, a in arms}
    for i, (n, (fr, bx)) in enumerate(seqs.items()):
        for m, a in arms:
            e, o = run(stream, fr, bx, m, a)
            got[f"{m}/{a}"].append(summarise(e, o) if e
                                   else (0.0, 0.0, float("nan")))
        print(f"  {i + 1}/{len(seqs)} {n}", end="\r", flush=True)

    res = {"n_sequences": len(seqs), "search_px": SEARCH,
           "sequences": list(seqs), "arms": {},
           # per-sequence, so a PAIRED comparison is possible. The aggregates
           # alone cannot say whether two arms differ: §9.24 reported a 2x gap
           # on 12 sequences that reversed on a different 12, which is exactly
           # what a paired test over sequences exists to catch.
           "per_sequence": {k: [[round(x, 4) for x in s] for s in v]
                            for k, v in got.items()}}
    print(f"\n\n{'arm':<18}{'prec@20px':>11}{'success AUC':>13}"
          f"{'median err':>12}")
    for k, v in got.items():
        p = float(np.mean([x[0] for x in v]))
        u = float(np.mean([x[1] for x in v]))
        md = float(np.nanmedian([x[2] for x in v]))
        res["arms"][k] = {"precision_20px": round(p, 4),
                          "success_auc": round(u, 4),
                          "median_err_px": round(md, 2)}
        label = k.replace("/pred", "  (OPE)").replace("/gt", "  (re-anchored)")
        print(f"{label:<18}{p:>11.3f}{u:>13.3f}{md:>12.1f}")

    s = STAGES[0]
    ope = res["arms"][f"{s}/pred"]["success_auc"]
    anc = res["arms"][f"{s}/gt"]["success_auc"]
    res["drift_cost"] = round(anc - ope, 4)
    res["collapse_is_drift"] = bool(anc > ope + 0.10 and anc > 0.35)

    print(f"\n--- what breaks tracking? ---")
    print(f"  eye {s}: OPE {ope:.3f} -> re-anchored {anc:.3f}   "
          f"(drift costs {res['drift_cost']:+.3f})")
    px_o = res["arms"]["pixels/pred"]["success_auc"]
    px_a = res["arms"]["pixels/gt"]["success_auc"]
    print(f"  pixels : OPE {px_o:.3f} -> re-anchored {px_a:.3f}   "
          f"(drift costs {px_a - px_o:+.3f})")
    if res["collapse_is_drift"]:
        print(f"  DRIFT. Remove accumulation and the eye localises well "
              f"({anc:.3f}). Per-frame localisation is\n  sound; §9.24's 0.094 "
              f"is a tracker losing its target and never recovering, not a "
              f"representation\n  that cannot find things.")
    else:
        print(f"  PRECISION. Even with accumulation removed the eye reaches "
              f"only {anc:.3f}. The representation\n  cannot localise "
              f"precisely on video frame by frame, and §9.29's 0.818 is what a "
              f"two-alternative\n  choice buys over a continuous one -- a "
              f"representation limit, not a tracker one.")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
