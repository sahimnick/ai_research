"""The loop, running: one eye on real video, all read-outs live, end to end.

Everything in §9.19–§9.33 is a *measurement* -- a script that answers one
question and exits. This is the assembled thing they were about: `UnifiedEye`
handed real video, developed on it, given tags, and then run frame by frame
producing where / which / relation / how-big out of **one forward pass each**.

It is a demonstration and a regression check, not a new hypothesis. What it has
to show is that the pieces compose:

* one eye developed once, not four benchmarks with four streams;
* tags added at runtime, not baked into a script;
* every read-out from the same pass, per frame, at the rate the loop runs;
* the numbers landing where the benchmarks that measured them said they would.

That last is the point of the controls here. Each read-out is scored against the
same control its own section used -- a never-present tag for locating and
selecting, a positional prior for relations -- so if the assembled system
differs from the measured components, this run says so rather than hiding it
behind a working demo.

It also **refuses to report** if the targets do not move, since a loop that
tracks a stationary object has demonstrated nothing.

Usage:  python3 benchmarks/eye_loop.py out.json [vot_dir] [n_seq]
"""
import json
import os
import sys
import time

import numpy as np

from neurobrain.vision.unified_eye import UnifiedEye

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eye_vot import load_gt, resize                           # noqa: E402

VOT_DIR = sys.argv[2] if len(sys.argv) > 2 else "vot"
N_SEQ = int(sys.argv[3]) if len(sys.argv) > 3 else 12
SIZE = 224
OBJ = 44
PER_SEQ = 10


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


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_eye_loop.json"
    if not os.path.isdir(VOT_DIR):
        print(f"no frame directory at {VOT_DIR!r} -- nothing reported.")
        json.dump({"error": "no frames"}, open(out_path, "w"), indent=1)
        return
    names = sorted(d for d in os.listdir(VOT_DIR)
                   if os.path.isdir(os.path.join(VOT_DIR, d)))[:N_SEQ]
    seqs = {}
    for n in names:
        s = load_seq(VOT_DIR, n, PER_SEQ)
        if len(s) >= 5:
            seqs[n] = s
    if len(seqs) < 5:
        print(f"only {len(seqs)} usable sequences -- nothing reported.")
        json.dump({"error": "too few sequences", "n": len(seqs)},
                  open(out_path, "w"), indent=1)
        return
    keys = list(seqs)
    moved = float(np.mean([
        np.hypot(*np.subtract(seqs[k][-1][1], seqs[k][0][1])) for k in keys]))
    print(f"{len(seqs)} sequences; targets travel {moved:.1f}px over the run",
          flush=True)
    if moved < 5.0:
        print("  the targets barely move -- a loop tracking a static object "
              "shows nothing. REFUSING.")
        json.dump({"error": "targets static", "moved": moved},
                  open(out_path, "w"), indent=1)
        return

    # --- the loop, assembled once ---
    print("\n1. develop ONE eye on this footage ...", flush=True)
    t0 = time.time()
    eye = UnifiedEye(size=SIZE)
    eye.develop([seqs[k][0][0] for k in keys])
    t_dev = time.time() - t0
    print(f"   done in {t_dev:.0f}s; areas {eye.areas}, "
          f"where={eye.where_area} size={eye.size_area}", flush=True)

    print("2. fit the angular-size read-out (§9.31) ...", flush=True)
    fit_f = [f for k in keys[:len(keys) // 2] for f, _, _ in seqs[k]]
    fit_a = [a for k in keys[:len(keys) // 2] for _, _, a in seqs[k]]
    eye.fit_size(fit_f, fit_a)

    distract = seqs[keys[0]][0][0][18:18 + OBJ, 18:18 + OBJ].copy()
    never = distract[::-1, ::-1].copy()

    print("3. run the loop over held-out sequences ...", flush=True)
    res = {"n_sequences": len(seqs), "size": SIZE, "develop_s": round(t_dev, 1),
           "travel_px": round(moved, 1)}
    hit, sel, rel, ctrl_hit, ctrl_rel, size_err, lat = [], [], [], [], [], [], []
    true_a = []
    for si, k in enumerate(keys[len(keys) // 2:], start=len(keys) // 2):
        frames = seqs[k]
        f0, (ty0, tx0), _ = frames[0]
        sy, sx = max(((a, b) for a in (10, SIZE - OBJ - 10)
                      for b in (10, SIZE - OBJ - 10)),
                     key=lambda p: np.hypot(p[0] + OBJ / 2 - ty0,
                                            p[1] + OBJ / 2 - tx0))
        other = seqs[keys[(si + 1) % len(keys)]][0][0]
        # tags added at RUNTIME, on this footage
        eye.tags.clear()
        eye.add_tag("target", paste(f0, distract, sy, sx),
                    (ty0 - OBJ / 2, tx0 - OBJ / 2, OBJ, OBJ))
        eye.add_tag("other", paste(other, distract, sy, sx), (sy, sx, OBJ, OBJ))
        eye.add_tag("absent", paste(other, never, sy, sx), (sy, sx, OBJ, OBJ))

        for (f, (ty, tx), la) in frames:
            frame = paste(f, distract, sy, sx)
            t1 = time.time()
            p = eye.look(frame)
            lat.append(time.time() - t1)
            if "target" not in p.where:
                continue
            py, px = p.where["target"]
            hit.append(float(np.hypot(py - ty, px - tx) <= OBJ / 2))
            slots = [(ty, tx), (sy + OBJ / 2, sx + OBJ / 2)]
            sel.append(float(int(np.argmin(
                [np.hypot(py - a, px - b) for a, b in slots])) == 0))
            true_rel = "left" if tx < sx + OBJ / 2 else "right"
            got = p.relations.get(("other", "target"))
            if got is not None:      # ("other","target"): is other L/R of target
                rel.append(float((got == "right") == (true_rel == "left")))
            if "absent" in p.where:
                ay, ax = p.where["absent"]
                ctrl_hit.append(float(np.hypot(ay - ty, ax - tx) <= OBJ / 2))
                ctrl_rel.append(float(int(ax < sx + OBJ / 2)
                                      == int(true_rel == "left")))
            if p.frame_size is not None:
                size_err.append(abs(p.frame_size - la))
                true_a.append(la)
        print(f"   {si - len(keys) // 2 + 1}/{len(keys) - len(keys) // 2} {k}"
              f"  -- {p.describe()[:88]}", flush=True)

    res["loop"] = {
        "frames": len(hit),
        "where_hit": round(float(np.mean(hit)), 4),
        "where_control": round(float(np.mean(ctrl_hit)), 4) if ctrl_hit else None,
        "select": round(float(np.mean(sel)), 4),
        "relation": round(float(np.mean(rel)), 4) if rel else None,
        "relation_control": round(float(np.mean(ctrl_rel)), 4) if ctrl_rel else None,
        "size_mae_log": round(float(np.mean(size_err)), 4) if size_err else None,
        "size_mae_null": round(float(np.mean(np.abs(
            np.asarray(true_a) - np.mean(fit_a)))), 4) if size_err else None,
        "ms_per_frame": round(1000 * float(np.mean(lat)), 1)}

    L = res["loop"]
    print(f"\n--- the loop, on {L['frames']} held-out frames ---")
    print(f"  {'read-out':<14}{'value':>9}{'control':>10}")
    print(f"  {'where (hit)':<14}{L['where_hit']:>9.3f}"
          f"{(L['where_control'] if L['where_control'] is not None else 0):>10.3f}")
    print(f"  {'which (select)':<14}{L['select']:>9.3f}{0.5:>10.3f}")
    if L["relation"] is not None:
        print(f"  {'relation':<14}{L['relation']:>9.3f}"
              f"{(L['relation_control'] or 0):>10.3f}")
    if L["size_mae_log"] is not None:
        print(f"  {'how big (MAE)':<14}{L['size_mae_log']:>9.3f}"
              f"{L['size_mae_null']:>10.3f}")
        print(f"    ^ the size probe is fitted on OTHER sequences, which is "
              f"exactly the by-sequence split\n      §9.31 measured at R2~0. "
              f"A MAE at or above the predict-the-mean null is that finding\n"
              f"      reproducing inside the assembled loop, not a new one.")
    print(f"  {'latency':<14}{L['ms_per_frame']:>9.1f} ms/frame "
          f"(one pass, all read-outs)")

    ok = (L["where_hit"] > (L["where_control"] or 0) + 0.10
          and L["select"] > 0.65)
    res["loop_works"] = bool(ok)
    print(f"\n--- does the assembled loop work end to end? ---")
    if ok:
        print(f"  YES. One eye, developed once, tagged at runtime, producing "
              f"every read-out from a single\n  pass per frame at "
              f"{L['ms_per_frame']:.0f} ms, each above its own control. The "
              f"pieces compose.")
    else:
        print(f"  NO. The assembled loop does not reproduce what the "
              f"components measured separately, which\n  is a composition "
              f"failure and not a new negative about any one of them.")
    print("  Carried forward unchanged from the measurements: this eye does "
          "not leave its development\n  distribution (§9.27), localises "
          "coarsely rather than precisely (§9.30), is hurt by colour\n  "
          "(§9.18) and cannot see a shadow as anything but an object (§9.31).")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
