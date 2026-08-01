"""Translation invariance that keeps structure: the ear's trick, done properly.

Nine classes of intervention have left the eye's cluster AUC at 0.578 against
the ear's 0.788, and `ear_config.py` closed the last configuration difference
between them (tying the bank is worth +0.016, 7.8% of the gap). What is left is
a difference of *kind*, and it is visible once stated:

A cochleagram is a **time × frequency** map. The same sound occurring later
fills different columns of the same rows, so when the belt pools over time it
discards a genuine nuisance dimension and keeps the thing that identifies the
sound. The retinotopic grid has no such split. Pooling over position discards
*everything* spatial and returns a bag of features -- `translation.py` measured
fully pooled codes at **0.233** against 0.773 for position-specific ones.
Invariance bought by throwing away what made the code worth having.

`WideV1.relational_code` is the operation the ear's actually corresponds to:

    C[k, f, g] = sum_p  A[f, p] * A[g, p + offset_k]

Summing over p is what makes it translation-invariant -- shift the image and
every p shifts, leaving the sum alone. The offset index keeps **how the parts
are arranged relative to each other**, which is what a bag of features throws
away. Second-order statistics over relative position; nothing learned, nothing
differentiated, products and sums of drive the layer already produces.

Smoke-tested before running: an image and its 4-px shift sit at cosine **0.994**
under this code and **0.529** under the standard rate code.

The arms, and the one that decides it
-------------------------------------
    rate                the eye as it is. 0.578, shift-fragile
    pooled              `pooling_index("both")` -- invariance by discarding
                        position. The 0.233 arm, carried over because a new
                        invariant code has to beat the invariant code that
                        already exists, not just the position-specific one
    relational          the proposal

An invariant code that scores like `pooled` has bought nothing. The claim is
specifically that relative arrangement is *recoverable* structure that absolute
position is not.

Measured: cluster AUC (what everything downstream is gated on), and the panel-1
concept-waking number centred and at ±5 px.

Usage:  python3 benchmarks/relational.py out_relational.json
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
N_FEAT = 12
ARMS = ("rate", "pooled", "relational")


def make_coder(v1, arm):
    if arm == "rate":
        return lambda f: _unit(v1.rate(f))
    if arm == "pooled":
        idx, n = v1.pooling_index("both")
        return lambda f: v1.pooled_code(v1.drive(f), idx, n)
    return lambda f: v1.relational_code(f, n_feat=N_FEAT)


def codes(coder, frames):
    R = np.array([np.concatenate([coder(c) for c in opponent(f)])
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
    v1 = WideV1(n_cells=CELLS, window_ms=50, image_shape=(FRAME, FRAME),
                rf=7, stride=2, seed=seed)
    # tied, because both `pooling_index` and `relational_code` need filter
    # index to mean the same filter at every location
    develop_v1(v1, list(lum), epochs=3, tie=True, seed=seed)

    out = {}
    for arm in ARMS:
        coder = make_coder(v1, arm)
        V = codes(coder, cent)
        rng = np.random.default_rng(seed)
        rec = {"dim": int(V.shape[1]),
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
            q = codes(coder, [place(images[i], dx) for i in te])
            ok = sum(int(name.get(assoc.concept_from_vision(q[k]), -1)
                         == int(y[i])) for k, i in enumerate(te))
            rec[f"wakes {dx:+d}px"] = ok / max(len(te), 1)
        out[arm] = rec
    return out


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_relational.json"
    images, waves, y, names = load_audiovisual(n_per_class=60, seed=0,
                                               grayscale=False, size=32)
    y = np.asarray(y, int)
    chance = 1.0 / len(names)
    print(f"{len(images)} real photographs, {len(names)} categories, chance "
          f"{chance:.3f}; the ear scores cluster AUC 0.788\n", flush=True)

    rows = [run_seed(images, y, sd) for sd in SEEDS]
    KEYS = ["cluster_auc"] + [f"wakes {d:+d}px" for d in SHIFTS]
    res = {"seeds": list(SEEDS), "chance": round(chance, 4), "arms": {}}
    for a in ARMS:
        res["arms"][a] = {k: round(float(np.mean([r[a][k] for r in rows])), 4)
                          for k in KEYS}
        res["arms"][a]["dim"] = int(rows[0][a]["dim"])
        sh = [res["arms"][a][f"wakes {d:+d}px"] for d in SHIFTS if d]
        res["arms"][a]["shifted"] = round(float(np.mean(sh)), 4)

    print(f"{'code':<14}{'dim':>8}{'cluster AUC':>13}{'centred':>10}"
          f"{'shifted':>10}{'gap':>8}")
    for a in ARMS:
        v = res["arms"][a]
        gap = v["wakes +0px"] - v["shifted"]
        v["gap"] = round(float(gap), 4)
        print(f"{a:<14}{v['dim']:>8}{v['cluster_auc']:>13.3f}"
              f"{v['wakes +0px']:>10.3f}{v['shifted']:>10.3f}{gap:>8.3f}")

    base, pooled, rel = (res["arms"]["rate"], res["arms"]["pooled"],
                         res["arms"]["relational"])
    print(f"\nagainst the eye as it is, paired over {len(SEEDS)} seeds")
    gate = {}
    for a in ARMS[1:]:
        cs = []
        for k in KEYS:
            d = np.array([r[a][k] - r["rate"][k] for r in rows])
            sd = float(d.std(ddof=1))
            cd = None if sd < 1e-9 else float(d.mean() / sd)
            gate.setdefault(a, {})[k] = dict(
                delta=round(float(d.mean()), 4),
                cohens_d=None if cd is None else round(cd, 3),
                wins=int((d > 0).sum()), n=len(d))
            cs.append(f"{d.mean():+.3f} " +
                      ("d=n/a" if cd is None else f"d={cd:+.1f}"))
        print(f"{a:<14}" + "".join(f"{c:>15}" for c in cs))
    res["gate"] = gate

    EAR = 0.788
    room = EAR - base["cluster_auc"]
    g = gate["relational"]["cluster_auc"]
    share = g["delta"] / room if room > 1e-9 else 0.0
    res["gap_to_ear"] = round(float(room), 4)
    res["share_of_gap"] = round(float(share), 4)

    print("\n=== does relative arrangement buy invariance AND keep structure? ===")
    print(f"  invariance  centred-minus-shifted: rate {base['gap']:.3f}, "
          f"pooled {pooled['gap']:.3f}, relational {rel['gap']:.3f}")
    print(f"  structure   cluster AUC:           rate "
          f"{base['cluster_auc']:.3f}, pooled {pooled['cluster_auc']:.3f}, "
          f"relational {rel['cluster_auc']:.3f}")
    beats_pooled = rel["cluster_auc"] > pooled["cluster_auc"] + 0.02
    passes = (g["cohens_d"] is not None and g["cohens_d"] >= 0.8
              and g["wins"] >= 0.75 * g["n"])
    res["beats_pooled"] = bool(beats_pooled)
    res["clears_gate"] = bool(passes)
    if passes and beats_pooled:
        print(f"\n  Both. Relational clears the gate on clustering "
              f"({g['delta']:+.4f}, d={g['cohens_d']:+.2f}, "
              f"{g['wins']}/{g['n']}) -- {share:.0%} of the {room:.3f} gap to "
              f"the ear -- and beats the")
        print(f"  invariant code that already existed "
              f"({rel['cluster_auc']:.3f} against {pooled['cluster_auc']:.3f}),"
              f" which is the comparison that matters: invariance is easy and "
              f"invariance that keeps structure is not.")
    elif beats_pooled:
        print(f"\n  It beats the existing invariant code "
              f"({rel['cluster_auc']:.3f} vs {pooled['cluster_auc']:.3f}) but "
              f"does not clear the gate against the position-specific one "
              f"({g['delta']:+.4f}, d={g['cohens_d']})."
              )
    else:
        print(f"\n  No. Relational scores {rel['cluster_auc']:.3f} against "
              f"{pooled['cluster_auc']:.3f} for plain pooling, so relative "
              f"arrangement is not recoverable structure here --")
        print("  the invariance is real and it costs the same as the "
              "invariance that was already available.")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
