"""Where does the temporal signal get lost? A capacity sweep.

`temporal_invariance.py` falsified H14: the temporal rule neither cleared its
pre-registered bar nor beat its own shuffled-time control (+0.0048, d=0.19,
2/4). That result is not a failure of power -- the predictive term receives a
**2.4x different signal** on ordered against shuffled input (mean
|z_t - z_{t-1}| = 0.0333 against 0.0796) and still produces the same
representation. The difference goes in and does not come out.

This asks the one structural question that difference raises, and asks it as a
measurement rather than an argument. `pathways.py`'s configuration is
``n_cells=4096`` over ``n_pos=441``, so a **tied** bank has

    4096 // 441 = 9

distinct filters. LPL in Halvagal & Zenke is a deep network with thousands of
features. Nine filters shared across 441 positions may simply have no room for
a temporal rule to shape anything, in which case the falsification above is
about this architecture's capacity and not about the rule.

Hypothesis
----------
**H16 (capacity is what binds).** If nine filters is the constraint, the gap
between `temporal` and `shuffled-time` should **grow with the number of
distinct filters**. If the gap stays flat while capacity rises severalfold,
capacity is not what limits it, and the temporal rule genuinely buys nothing
here.

Only `n_cells` changes; the rule, the data, the sequences and the evaluation
are identical to `temporal_invariance.py`, which is what makes this a diagnostic
rather than a second architecture.

Usage:  python3 benchmarks/temporal_capacity.py out_temporal_capacity.json
"""
import json
import sys

import numpy as np

from neurobrain.cognition.multimodal import _unit
from neurobrain.learning.selforganize import develop_v1, develop_v1_temporal
from neurobrain.sensing.natural import load_audiovisual
from neurobrain.sensing.sequences import shuffle_across
from neurobrain.vision.widev1 import PopulationAdaptation, WideV1

sys.path.insert(0, "benchmarks")
from pathways import FRAME, cluster_auc, place                 # noqa: E402
from real_binding import split                                 # noqa: E402
from temporal_invariance import (EPOCHS, N_IMAGES,             # noqa: E402
                                 build_sequences)

SEEDS = (0, 1, 2)
#: n_cells -> distinct tied filters over the 441 positions of a 48px frame
CAPACITY = (4096, 12288, 24576)
ARMS = ("static", "temporal", "shuffled-time")


def grow(arm, seqs, cells, seed):
    v1 = WideV1(n_cells=cells, window_ms=50, rf=7, stride=2,
                image_shape=(FRAME, FRAME), seed=seed)
    if arm == "static":
        develop_v1(v1, [f for s in seqs for f in s], epochs=EPOCHS,
                   tie=True, seed=seed)
    elif arm == "shuffled-time":
        develop_v1_temporal(v1, shuffle_across(seqs, seed=seed),
                            epochs=EPOCHS, seed=seed)
    else:
        develop_v1_temporal(v1, seqs, epochs=EPOCHS, seed=seed)
    return v1


def main():
    out_path = (sys.argv[1] if len(sys.argv) > 1
                else "out_temporal_capacity.json")
    images, _w, y, _n = load_audiovisual(n_per_class=N_IMAGES // 6, seed=0,
                                         grayscale=True, size=32)
    y = np.asarray(y, int)
    frames = [place(im) for im in images]
    probe = WideV1(n_cells=4096, rf=7, stride=2, image_shape=(FRAME, FRAME))
    print(f"{len(images)} photographs, {FRAME}px frame, {probe.n_pos} positions")
    print("H16: if capacity binds, temporal - shuffled grows with filters\n",
          flush=True)

    rows = []
    for sd in SEEDS:
        seqs = build_sequences(images, sd)
        tr, te = split(y, sd)
        rec = {}
        for cells in CAPACITY:
            nf = cells // probe.n_pos
            for arm in ARMS:
                v1 = grow(arm, seqs, cells, sd)
                R = np.array([v1.drive(f) for f in frames], np.float32)
                ad = PopulationAdaptation(R.shape[1])
                V = np.array([_unit(ad(r)) for r in R], np.float32)
                rec[f"{cells}|{arm}"] = cluster_auc(
                    V[te], y[te], V[tr], y[tr], np.random.default_rng(sd))
            g = rec[f"{cells}|temporal"] - rec[f"{cells}|shuffled-time"]
            print(f"  seed {sd}  {nf:>3} filters: "
                  f"static {rec[f'{cells}|static']:.3f}  "
                  f"temporal {rec[f'{cells}|temporal']:.3f}  "
                  f"shuffled {rec[f'{cells}|shuffled-time']:.3f}  "
                  f"gap {g:+.4f}", flush=True)
        rows.append(rec)

    res = {"seeds": list(SEEDS), "per_seed": rows, "capacity": {}}
    print(f"\n{'filters':>8}{'static':>9}{'temporal':>10}{'shuffled':>10}"
          f"{'temporal-shuffled':>20}")
    gaps = []
    for cells in CAPACITY:
        nf = cells // probe.n_pos
        m = {a: float(np.mean([r[f"{cells}|{a}"] for r in rows]))
             for a in ARMS}
        g = np.array([r[f"{cells}|temporal"] - r[f"{cells}|shuffled-time"]
                      for r in rows])
        d = float(g.mean() / (g.std(ddof=1) + 1e-12)) if len(g) > 1 else float("nan")
        gaps.append(float(g.mean()))
        res["capacity"][nf] = {**{a: round(m[a], 4) for a in ARMS},
                               "gap": round(float(g.mean()), 4),
                               "cohens_d": round(d, 2),
                               "wins": int((g > 0).sum()), "n": len(g)}
        print(f"{nf:>8}{m['static']:>9.3f}{m['temporal']:>10.3f}"
              f"{m['shuffled-time']:>10.3f}"
              f"{g.mean():>+13.4f} d={d:>+5.2f} {int((g>0).sum())}/{len(g)}")

    print(f"\n--- H16: does the gap grow with capacity? ---")
    nfs = [c // probe.n_pos for c in CAPACITY]
    grew = bool(gaps[-1] > gaps[0] + 0.02)
    res["H16_gap_grows_with_capacity"] = grew
    print(f"  gap at {nfs[0]} filters {gaps[0]:+.4f}  ->  "
          f"at {nfs[-1]} filters {gaps[-1]:+.4f}")
    if grew:
        print("  Capacity IS what binds: the temporal rule separates from its "
              "own control once there are filters to separate with.\n"
              "  H14's falsification is about this architecture's width, not "
              "about the rule.")
    else:
        print("  Capacity is NOT what binds. The gap does not grow with "
              "filters, so nine was not the constraint and the temporal\n"
              "  rule buys nothing here at any width tested. H14's "
              "falsification stands on its own terms.")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
