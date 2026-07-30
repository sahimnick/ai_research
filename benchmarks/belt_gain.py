"""How much tonotopy should the belt keep? Swept on real audio AND synthetic.

`AuditoryBelt` was built fully shift-invariant -- its code says only *which
filter fired*, with frequency and time both pooled away. That was measured, and
on the designed 8-class bank it was worth about 50 points, because there the
classes differ by pattern while absolute pitch is randomised over an octave and
a half. Discarding pitch was discarding a nuisance.

Real environmental sound inverts the premise. Rain, wind, sea waves and crickets
are stationary textures whose *spectral profile is their identity*, so pooling
frequency away deletes the feature that separates them. On 600 ESC-50 clips the
pooled belt scored 0.198 on the 50-way task while a plain pooled cochleagram
scored 0.350 -- a 1024-cell spiking layer losing to the spectrum it is built on.

Adding a tonotopic channel alongside the invariant one fixes the real-audio side
(+0.04 to +0.15 on every ESC-50 category) and costs the designed bank 17 points
at equal weight. Neither extreme is right, so the question is not *which
channel* but *how much of each* -- which is a gain, and gains are measured.

The sweep re-weights without recomputing: `code_channels` returns both blocks
once per clip, and each gain is a different mix of the same two vectors. So the
cost is one coding pass, not one per gain.

Usage:  python3 benchmarks/belt_gain.py out_belt_gain.json [n_shards]
"""
import json
import sys

import numpy as np

from neurobrain.audition.audio import Cochleagram, sound_dataset
from neurobrain.audition.auditorycortex import linear_probe
from neurobrain.sensing.natural import esc50_category_of, load_esc50
from neurobrain.sensing.streams import AuditoryBelt, _unit
from neurobrain.vision.widev1 import _nearest_prototype

sys.path.insert(0, ".")
from real_audio import knn, stratified          # noqa: E402

GAINS = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
N_FREQ, SR, DUR = 36, 8000, 2.0
CATEGORIES = ("animals", "nature", "human", "interior", "urban")
# Splits, not front ends. The spiking layer is coded once and every seed is a
# different train/test partition of the SAME codes, so the seeds are paired --
# gain 0.5 and gain 0 are compared on identical clips in identical roles, and
# the spread is the split's, which is the only thing varying.
SPLIT_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)


def score(INV, TONO, y, tr, te, gain, k):
    """One gain's three read-outs. The mix is the only thing that changes."""
    X = (INV if gain <= 0 else
         np.array([_unit(np.concatenate([i, gain * t]))
                   for i, t in zip(INV, TONO)], np.float32))
    return (float(np.mean(_nearest_prototype(X[tr], y[tr], X[te], k) == y[te])),
            float(np.mean(knn(X[tr], y[tr], X[te]) == y[te])),
            float(linear_probe(X[tr], y[tr], X[te], y[te])))


def channels(belt, cochs):
    """Both blocks for every clip, computed once."""
    pairs = [belt.code_channels(c) for c in cochs]
    return (np.array([p[0] for p in pairs], np.float32),
            np.array([p[1] for p in pairs], np.float32))


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_belt_gain.json"
    n_shards = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    coch = Cochleagram(sr=SR, n_freq=N_FREQ)
    belt = AuditoryBelt(n_freq=N_FREQ, seed=0)

    # ---- real ------------------------------------------------------------
    waves, y_esc, _ = load_esc50(n_shards=n_shards, sr=SR, dur_s=DUR)
    cats = np.array([esc50_category_of(v) for v in y_esc])
    INV, TONO = channels(belt, [coch.forward(w)[0] for w in waves])
    print(f"ESC-50: {len(waves)} clips  invariant dim={INV.shape[1]}  "
          f"tonotopic dim={TONO.shape[1]}", flush=True)

    # ---- synthetic -------------------------------------------------------
    sigs, y_syn, names = sound_dataset(n_per_class=40, seed=0)
    coch_s = Cochleagram(n_freq=N_FREQ)
    SINV, STONO = channels(belt, [coch_s.forward(s)[0] for s in sigs])
    str_, ste = stratified(np.asarray(y_syn), seed=0)
    print(f"designed bank: {len(sigs)} clips, {len(names)} classes\n", flush=True)

    res = {"gains": list(GAINS), "split_seeds": list(SPLIT_SEEDS),
           "real": {}, "synthetic": {}, "summary": []}

    subsets = []
    for cat in CATEGORIES:
        idx = np.flatnonzero(cats == cat)
        uniq = np.unique(y_esc[idx])
        subsets.append((cat, idx, np.searchsorted(uniq, y_esc[idx]), len(uniq)))

    # per_seed[gain] = list over seeds of (real_mean, synthetic)
    per_seed = {g: [] for g in GAINS}
    for sd in SPLIT_SEEDS:
        for g in GAINS:
            row = []
            for cat, idx, lab, k in subsets:
                tr, te = stratified(lab, seed=sd)
                p, kn, lp = score(INV[idx], TONO[idx], lab, tr, te, g, k)
                res.setdefault("real", {}).setdefault(cat, {}).setdefault(
                    str(g), []).append(dict(seed=sd, proto=round(p, 4),
                                            knn5=round(kn, 4),
                                            probe=round(lp, 4)))
                row.append(max(p, kn))
            s_tr, s_te = stratified(np.asarray(y_syn), seed=sd)
            sp, skn, slp = score(SINV, STONO, np.asarray(y_syn), s_tr, s_te, g,
                                 len(names))
            res.setdefault("synthetic", {}).setdefault(str(g), []).append(
                dict(seed=sd, proto=round(sp, 4), knn5=round(skn, 4),
                     probe=round(slp, 4)))
            per_seed[g].append((float(np.mean(row)), sp))
        print(f"   seed {sd} done", flush=True)

    R = {g: np.array([v[0] for v in per_seed[g]]) for g in GAINS}
    S = {g: np.array([v[1] for v in per_seed[g]]) for g in GAINS}
    res["summary"] = [dict(gain=g, real=round(float(R[g].mean()), 4),
                           real_sd=round(float(R[g].std(ddof=1)), 4),
                           synthetic=round(float(S[g].mean()), 4),
                           synthetic_sd=round(float(S[g].std(ddof=1)), 4))
                      for g in GAINS]

    print(f"\n{'gain':>6}{'real':>9}{'+/-':>7}{'synthetic':>12}{'+/-':>7}"
          f"{'d(real)':>10}{'wins':>7}")
    for g in GAINS:
        d = R[g] - R[0.0]
        sd_ = float(d.std(ddof=1)) if g else 0.0
        cd = float(d.mean() / (sd_ + 1e-12)) if g else 0.0
        res["summary"][GAINS.index(g)].update(
            cohens_d=round(cd, 3), wins=int((d > 0).sum()), n=len(d))
        print(f"{g:>6.3f}{R[g].mean():>9.3f}{R[g].std(ddof=1):>7.3f}"
              f"{S[g].mean():>12.3f}{S[g].std(ddof=1):>7.3f}"
              f"{cd:>10.2f}{int((d > 0).sum()):>4}/{len(d)}")

    # -------------------------------------------------------- the choice ---
    base_r, base_s = float(R[0.0].mean()), float(S[0.0].mean())
    print(f"\ngain 0 (the belt as it was): real {base_r:.3f}  "
          f"synthetic {base_s:.3f}   over {len(SPLIT_SEEDS)} paired splits")
    print("\n=== which gain? the largest real gain that does not cost the "
          "designed bank more than 2 points ===")
    cand = [r for r in res["summary"][1:]
            if r["synthetic"] >= base_s - 0.02 and r["cohens_d"] >= 0.8
            and r["wins"] >= 0.75 * r["n"]]
    best = max(cand, key=lambda r: r["real"]) if cand else None
    for r in res["summary"][1:]:
        why = []
        if r["synthetic"] < base_s - 0.02:
            why.append("synthetic loses")
        if r["cohens_d"] < 0.8:
            why.append("real gain not reliable")
        star = "  <-- chosen" if best is not None and r is best else ""
        print(f"  gain {r['gain']:<6.3f} real {r['real']:.3f} "
              f"({r['real']-base_r:+.3f}, d={r['cohens_d']:+.2f}, "
              f"{r['wins']}/{r['n']})  synthetic {r['synthetic']:.3f} "
              f"({r['synthetic']-base_s:+.3f})"
              f"{'   (' + ', '.join(why) + ')' if why else ''}{star}")
    if best is None:
        print("  NO gain both helps real audio reliably and keeps the designed "
              "bank -- the two worlds want different belts, and the default "
              "must stay at 0.")
    else:
        print(f"\n  -> tono_gain = {best['gain']}: real {base_r:.3f} -> "
              f"{best['real']:.3f} ({best['real']-base_r:+.3f}, "
              f"d={best['cohens_d']:+.2f}), synthetic {base_s:.3f} -> "
              f"{best['synthetic']:.3f} ({best['synthetic']-base_s:+.3f})")
    res["chosen"] = best

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
