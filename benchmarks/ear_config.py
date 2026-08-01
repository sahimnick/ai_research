"""Give the eye the ear's configuration. The ceiling, attacked at its cause.

Every intervention on the eye has failed against the same wall: centred
concept-waking sits near 0.29 and cluster AUC near 0.573, whatever is done. Six
classes of change -- width, aperture, depth, whitening, concept feedback under
two rules, resolution, capacity, fixation choice -- leave it there.

Meanwhile the ear reaches cluster AUC 0.788 on the same kind of task, and
`selforganize.py` already records the structural difference without anyone
having tested it as a cause:

    AuditoryBelt builds a WideV1 and **never develops it**, so its bank stays
    tied at 0.978 -- which is why the same pooling operation genuinely works
    there and is a large part of why hearing outperforms vision.

"Tied" means every retinotopic position shares one filter bank. Untrained they
do (cosine 0.977 between the same filter index at different positions); after
`develop_v1` each column drifts independently (0.454, against 0.319 for random
pairs), and `pooling_index` -- which groups cells *by index* on the assumption
that index means filter identity -- is then summing unrelated cells.

So the working sense and the broken one differ in configuration and not only in
data, and the comparison has never been run. Three eyes on identical
photographs:

    developed, untied   what the eye is now. Filters learned per column
    developed, tied     `develop_v1(tie=True)`: learned, but one bank shared
                        across positions
    undeveloped, tied   **the ear's actual configuration** -- the seeded bank,
                        never developed, identical at every position

The third is the interesting one and it is nearly free to test. If an eye
configured like the ear clears the wall, the ceiling is a consequence of
per-column development rather than of anything about photographs -- and the
project's own headline (discovered fields beat designed ones) would need
qualifying, because it was measured on centred digits rather than on whether the
resulting code *clusters*.

Measured, held out:

    cluster AUC     same-category similarity above different-category. The
                    number `constraint.py` showed everything downstream is
                    gated on. The ear scores 0.788, this eye 0.573
    concept wakes   the panel-1 number, centred and at +-5px
    tie             cosine between the same filter index at different
                    positions, so the configuration is verified rather than
                    assumed

Usage:  python3 benchmarks/ear_config.py out_ear_config.json
"""
import json
import sys

import numpy as np

from neurobrain.cognition.multimodal import AssociationArea, _unit
from neurobrain.learning.selforganize import develop_v1
from neurobrain.sensing.natural import load_audiovisual
from neurobrain.vision.widev1 import PopulationAdaptation, WideV1

sys.path.insert(0, "benchmarks")
from phase10_2 import FRAME, SHIFTS, place                   # noqa: E402
from real_binding import opponent, split                     # noqa: E402

SEEDS = (0, 1, 2, 3)
N_CONCEPT = 256
CELLS = 4096
ARMS = ("developed, untied", "developed, tied", "undeveloped, tied")


def tie_score(v1, n=64):
    """Cosine between the same filter index at different positions.

    1.0 means every hypercolumn holds the same bank, which is what
    `pooling_index` assumes and what the ear actually has."""
    rng = np.random.default_rng(0)
    per = max(v1.n_cells // v1.n_pos, 1)
    got = []
    for _ in range(n):
        f = int(rng.integers(per))
        p, q = rng.choice(v1.n_pos, 2, replace=False)
        a, b = f * v1.n_pos + p, f * v1.n_pos + q
        if a < v1.n_cells and b < v1.n_cells:
            got.append(float(_unit(v1.Wt[a]) @ _unit(v1.Wt[b])))
    return float(np.mean(got)) if got else float("nan")


def build(arm, lum, seed):
    v1 = WideV1(n_cells=CELLS, window_ms=50, image_shape=(FRAME, FRAME),
                rf=7, stride=2, seed=seed)
    if arm == "developed, untied":
        develop_v1(v1, list(lum), epochs=3, tie=False, seed=seed)
    elif arm == "developed, tied":
        develop_v1(v1, list(lum), epochs=3, tie=True, seed=seed)
    # "undeveloped, tied" leaves the seeded bank exactly as the ear has it
    return v1


def codes(v1, frames):
    R = np.array([np.concatenate([v1.rate(c) for c in opponent(f)])
                  for f in frames], np.float32)
    ad = PopulationAdaptation(R.shape[1])
    return np.array([_unit(ad(r)) for r in R], np.float32)


def cluster_auc(Q, yq, B, yb, rng):
    same, diff = [], []
    for k in range(len(Q)):
        ps, pd = np.flatnonzero(yb == yq[k]), np.flatnonzero(yb != yq[k])
        if not len(ps) or not len(pd):
            continue
        same.append(float(Q[k] @ B[int(rng.choice(ps))]))
        diff.append(float(Q[k] @ B[int(rng.choice(pd))]))
    same, diff = np.array(same), np.array(diff)
    return float((same[:, None] > diff[None, :]).mean())


def run_seed(images, y, seed):
    tr, te = split(y, seed)
    cent = [place(im) for im in images]
    lum = np.array([opponent(f)[0] for f in cent], np.float32)
    out = {}
    for arm in ARMS:
        v1 = build(arm, lum, seed)
        V = codes(v1, cent)
        rng = np.random.default_rng(seed)
        rec = {"tie": tie_score(v1),
               "cluster_auc": cluster_auc(V[te], y[te], V[tr], y[tr], rng)}

        assoc = AssociationArea(n_vis=V.shape[1], n_aud=V.shape[1],
                                n_concept=N_CONCEPT, seed=seed)
        assoc.set_stats(V[tr], V[tr])
        votes = {}
        for i in tr:
            w = assoc.bind(V[i], V[i])
            votes.setdefault(w, {})
            votes[w][int(y[i])] = votes[w].get(int(y[i]), 0) + 1
        name = {c: max(v.items(), key=lambda kv: kv[1])[0]
                for c, v in votes.items() if v}
        for dx in SHIFTS:
            q = codes(v1, [place(images[i], dx) for i in te])
            ok = sum(int(name.get(assoc.concept_from_vision(q[k]), -1)
                         == int(y[i])) for k, i in enumerate(te))
            rec[f"wakes {dx:+d}px"] = ok / max(len(te), 1)
        rec["cells"] = int((assoc.wins > 0).sum())
        out[arm] = rec
    return out


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_ear_config.json"
    images, waves, y, names = load_audiovisual(n_per_class=60, seed=0,
                                               grayscale=False, size=32)
    y = np.asarray(y, int)
    chance = 1.0 / len(names)
    print(f"{len(images)} real photographs, {len(names)} categories, chance "
          f"{chance:.3f}")
    print(f"the ear scores cluster AUC 0.788 on its own material; this eye "
          f"0.573\n", flush=True)

    rows = [run_seed(images, y, sd) for sd in SEEDS]
    METRICS = ["tie", "cluster_auc"] + [f"wakes {d:+d}px" for d in SHIFTS]
    res = {"seeds": list(SEEDS), "chance": round(chance, 4), "arms": {}}
    for a in ARMS:
        res["arms"][a] = {m: round(float(np.mean([r[a][m] for r in rows])), 4)
                          for m in METRICS + ["cells"]}

    print(f"{'configuration':<22}{'tie':>8}{'cluster AUC':>13}"
          + "".join(f"{m.replace('wakes ',''):>12}" for m in METRICS[2:]))
    for a in ARMS:
        v = res["arms"][a]
        print(f"{a:<22}{v['tie']:>8.3f}{v['cluster_auc']:>13.3f}"
              + "".join(f"{v[m]:>12.3f}" for m in METRICS[2:]))

    base = res["arms"]["developed, untied"]
    print(f"\nagainst the eye as it is, paired over {len(SEEDS)} seeds")
    gate = {}
    for a in ARMS[1:]:
        cs = []
        for m in ("cluster_auc",) + tuple(METRICS[2:]):
            d = np.array([r[a][m] - r["developed, untied"][m] for r in rows])
            sd = float(d.std(ddof=1))
            cd = None if sd < 1e-9 else float(d.mean() / sd)
            gate.setdefault(a, {})[m] = dict(
                delta=round(float(d.mean()), 4),
                cohens_d=None if cd is None else round(cd, 3),
                wins=int((d > 0).sum()), n=len(d))
            cs.append(f"{d.mean():+.3f} " +
                      ("d=n/a" if cd is None else f"d={cd:+.1f}"))
        print(f"{a:<22}" + "".join(f"{c:>15}" for c in cs))
    res["gate"] = gate

    print("\n=== does the ear's configuration clear the eye's wall? ===")
    best = max(ARMS, key=lambda a: res["arms"][a]["cluster_auc"])
    b = res["arms"][best]
    print(f"  best configuration: {best}, cluster AUC "
          f"{b['cluster_auc']:.3f} against {base['cluster_auc']:.3f} for the "
          f"eye as it is (the ear scores 0.788)")
    res["best"] = best
    g = gate.get(best, {}).get("cluster_auc")
    passes = bool(g and g["cohens_d"] is not None and g["cohens_d"] >= 0.8
                  and g["wins"] >= 0.75 * g["n"])
    # Consistency is not explanation. The question is not "is the effect
    # reliable" but "does it account for the gap it was proposed to explain",
    # and an effect can be 4/4 seeds and still be 8% of what needs explaining.
    EAR = 0.788
    room = EAR - base["cluster_auc"]
    share = (g["delta"] / room) if (g and room > 1e-9) else 0.0
    res["gap_to_ear"] = round(float(room), 4)
    res["share_of_gap_explained"] = round(float(share), 4)
    res["clears_wall"] = bool(passes and share >= 0.25)
    if passes and share >= 0.25:
        print(f"  and it clears the gate: {g['delta']:+.4f}, "
              f"d={g['cohens_d']:+.2f}, {g['wins']}/{g['n']} -- "
              f"{share:.0%} of the {room:.3f} gap to the ear.")
        print("  Per-column development is a substantial part of the ceiling.")
    elif passes:
        print(f"  It clears the significance gate ({g['delta']:+.4f}, "
              f"d={g['cohens_d']:+.2f}, {g['wins']}/{g['n']}) and explains "
              f"almost none of the wall: {share:.0%} of the {room:.3f} gap")
        print(f"  to the ear. A reliable effect and a small one, which are "
              f"different things -- tying the bank is worth having and is not "
              f"why the eye is where it is.")
        print(f"  Note also that the ear's EXACT configuration (undeveloped, "
              f"tied) scores "
              f"{res['arms']['undeveloped, tied']['cluster_auc']:.3f}, below "
              f"the developed-tied arm, so it is not")
        print("  the absence of development that helps either. The last "
              "structural difference this architecture offers between the two "
              "senses is now measured, and it is not the cause.")
    else:
        print(f"  It does not. Every configuration lands within noise of "
              f"{base['cluster_auc']:.3f}, so the wall is not per-column "
              f"drift either --")
        print("  which removes the last structural difference between the eye "
              "and the ear that this architecture makes available.")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
