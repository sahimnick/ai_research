"""Phase 10.2, retried with Phase 10.3's perception instead of one fixed look.

Two results from this roadmap sit next to each other and were never combined.

`phase10_2.py`: a **5-pixel shift annihilates the code**. The concept a held-out
photograph wakes is right 0.290 of the time centred and **0.021** shifted --
below the 0.167 chance rate, so the shifted code wakes systematically wrong
categories. No teaching rule repaired it, and a deliberately misleading teacher
did as well as the correct one.

`uncertain_look.py`: choosing glances that **challenge** the current belief,
scored over the concept cells, is worth +0.0163 at k=5 (d=+0.89, 7/8) where
choosing glances that confirm it is worth −0.0200 -- and the gain *grows* with
the number of looks.

The second is a translation mechanism and the first is a translation failure.
`phase10_2.py` gave the eye exactly one look, always at the frame centre, so a
shifted object was simply somewhere else. This asks the question that pairing
them makes available: **if the eye takes several disagreement-chosen glances and
pools them, does the 5-pixel collapse survive?**

Nothing new is invented here. `glance` is `uncertain_look.py`'s, the
disagreement criterion is Phase 10.3's over the concept cells, and the measured
quantity is Phase 10.2's -- the panel-1 number, at 0 and ±5 px.

The arms
--------
    one centred look        `phase10_2.py`'s perception. The 0.021 baseline
    k random glances        pooled. Separates "more looks" from "better-chosen
                            looks", which is the whole question
    k disagreeing glances   Phase 10.3's rule

The pooled code is used for **both** forming concepts and querying them, because
a mind that gathers evidence one way and is tested another is being asked a
different question than the one it was built for.

Usage:  python3 benchmarks/phase10_2b.py out_phase10_2b.json
"""
import json
import sys

import numpy as np

from neurobrain.cognition.multimodal import AssociationArea, _unit
from neurobrain.learning.selforganize import develop_v1
from neurobrain.sensing.natural import load_audiovisual
from neurobrain.sensing.streams import StreamingBrain
from neurobrain.vision.widev1 import PopulationAdaptation

sys.path.insert(0, "benchmarks")
from phase10_2 import FRAME, SHIFTS, place                   # noqa: E402
from real_binding import opponent, split                     # noqa: E402

SEEDS = (0, 1, 2, 3)
N_CONCEPT = 256
KS = (1, 3, 5)
N_CAND = 6
JITTER = 6


def look(v1, frame_img, dy, dx):
    """One fixation: the eye's input re-centred on (dy, dx) of the frame."""
    sh = np.roll(np.roll(np.asarray(frame_img), -int(dy), axis=-2),
                 -int(dx), axis=-1)
    return np.concatenate([v1.rate(c) for c in opponent(sh)])


def disagreement(kept, g, basis):
    """How much this candidate would change the concept-cell profile."""
    cur = basis @ _unit(np.mean(kept, 0))
    return float(np.linalg.norm(basis @ _unit(g) - cur))


def gather(v1, frame_img, k, rng, basis=None):
    """k glances, pooled. ``basis`` present -> choose them by disagreement."""
    kept = [look(v1, frame_img, 0, 0)]
    for _ in range(k - 1):
        cands = rng.integers(-JITTER, JITTER + 1, size=(N_CAND, 2))
        if basis is None:
            a, b = cands[int(rng.integers(N_CAND))]
            kept.append(look(v1, frame_img, a, b))
            continue
        best, best_d = None, -np.inf
        for a, b in cands:
            g = look(v1, frame_img, a, b)
            d = disagreement(kept, g, basis)
            if d > best_d:
                best, best_d = g, d
        kept.append(best)
    return np.mean(kept, 0)


def codes(R):
    ad = PopulationAdaptation(R.shape[1])
    return np.array([_unit(ad(r)) for r in R], np.float32)


def concept_wakes(v1, images, y, tr, te, seed, k, choose):
    """The panel-1 number, with k pooled glances as the percept."""
    rng = np.random.default_rng(seed + 11)
    basis = None
    if choose:
        # the concept cells the criterion is scored over, grown on single
        # centred looks so the basis itself is not circular with the pooling
        b0 = codes(np.array([look(v1, place(images[i]), 0, 0) for i in tr],
                            np.float32))
        ca = AssociationArea(n_vis=b0.shape[1], n_aud=b0.shape[1],
                             n_concept=N_CONCEPT, seed=seed)
        ca.set_stats(b0, b0)
        for r in b0:
            ca.bind(r, r)
        basis = ca.Wv[np.flatnonzero(ca.wins > 0)]

    base = codes(np.array([gather(v1, place(images[i]), k, rng, basis)
                           for i in tr], np.float32))
    assoc = AssociationArea(n_vis=base.shape[1], n_aud=base.shape[1],
                            n_concept=N_CONCEPT, seed=seed)
    assoc.set_stats(base, base)
    votes = {}
    for j, i in enumerate(tr):
        w = assoc.bind(base[j], base[j])
        votes.setdefault(w, {})
        votes[w][int(y[i])] = votes[w].get(int(y[i]), 0) + 1
    name = {c: max(v.items(), key=lambda kv: kv[1])[0]
            for c, v in votes.items() if v}
    out = {}
    for dx in SHIFTS:
        q = codes(np.array([gather(v1, place(images[i], dx), k, rng, basis)
                            for i in te], np.float32))
        ok = sum(int(name.get(assoc.concept_from_vision(q[j]), -1) == int(y[i]))
                 for j, i in enumerate(te))
        out[f"shift {dx:+d}px"] = ok / max(len(te), 1)
    return out


def run_seed(images, y, seed):
    tr, te = split(y, seed)
    lum = [opponent(place(im))[0] for im in images]
    v1 = StreamingBrain(seed=seed, image_shape=(FRAME, FRAME), v1_cells=4096,
                        rf=7, stride=2).v1
    develop_v1(v1, lum, epochs=3, seed=seed)
    out = {}
    for k in KS:
        if k == 1:
            out["one centred look"] = concept_wakes(v1, images, y, tr, te,
                                                    seed, 1, False)
            continue
        out[f"{k} random glances"] = concept_wakes(v1, images, y, tr, te,
                                                   seed, k, False)
        out[f"{k} disagreeing glances"] = concept_wakes(v1, images, y, tr, te,
                                                        seed, k, True)
    return out


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_phase10_2b.json"
    images, waves, y, names = load_audiovisual(n_per_class=60, seed=0,
                                               grayscale=False, size=32)
    y = np.asarray(y, int)
    chance = 1.0 / len(names)
    print(f"{len(images)} real pairs, {len(names)} categories, chance "
          f"{chance:.3f}")
    print(f"32px object in a {FRAME}px frame; phase 10.2's measurement with "
          f"phase 10.3's perception\n", flush=True)

    rows = []
    for sd in SEEDS:
        rows.append(run_seed(images, y, sd))
        print(f"  seed {sd} done", flush=True)
    arms = list(rows[0])
    keys = [f"shift {d:+d}px" for d in SHIFTS]
    res = {"seeds": list(SEEDS), "chance": round(chance, 4), "arms": {}}
    for a in arms:
        res["arms"][a] = {k: round(float(np.mean([r[a][k] for r in rows])), 4)
                          for k in keys}
        res["arms"][a]["shifted mean"] = round(float(np.mean(
            [res["arms"][a][k] for k in keys[1:]])), 4)

    print(f"\n{'arm':<26}" + "".join(f"{k:>13}" for k in keys)
          + f"{'shifted mean':>15}")
    for a in arms:
        v = res["arms"][a]
        print(f"{a:<26}" + "".join(f"{v[k]:>13.3f}" for k in keys)
              + f"{v['shifted mean']:>15.3f}")

    base = res["arms"]["one centred look"]
    print(f"\n=== does pooling chosen glances survive the 5px shift? ===")
    print(f"  one centred look, shifted: {base['shifted mean']:.3f} "
          f"(chance {chance:.3f})")
    gate = {}
    for a in arms[1:]:
        d = np.array([np.mean([r[a][k] for k in keys[1:]])
                      - np.mean([r["one centred look"][k] for k in keys[1:]])
                      for r in rows])
        sd = float(d.std(ddof=1))
        cd = None if sd < 1e-9 else float(d.mean() / sd)
        gate[a] = dict(delta=round(float(d.mean()), 4),
                       cohens_d=None if cd is None else round(cd, 3),
                       wins=int((d > 0).sum()), n=len(d))
        dt = "d=n/a" if cd is None else f"d={cd:+.2f}"
        print(f"  {a:<26}{res['arms'][a]['shifted mean']:>8.3f}   "
              f"{d.mean():+.4f} {dt} {int((d > 0).sum())}/{len(d)}")
    res["gate"] = gate

    best = max(arms[1:], key=lambda a: res["arms"][a]["shifted mean"])
    b = res["arms"][best]["shifted mean"]
    res["best_arm"], res["best_shifted"] = best, b
    res["meets_prediction"] = bool(b >= 0.55)
    ch = [a for a in arms if "disagree" in a]
    rd = [a for a in arms if "random" in a]
    if ch and rd:
        bc = max(res["arms"][a]["shifted mean"] for a in ch)
        br = max(res["arms"][a]["shifted mean"] for a in rd)
        res["choosing_over_random"] = round(float(bc - br), 4)
        print(f"\n  best chosen {bc:.3f} against best random {br:.3f} "
              f"({bc - br:+.4f}) -- so {'CHOOSING' if bc > br + 0.01 else 'the extra looks, not the choosing,'} "
              f"is what carries it")
    if res["meets_prediction"]:
        print(f"\n  The Phase 10.2 prediction is MET: {b:.3f} >= 0.55, from "
              f"{base['shifted mean']:.3f} with a single fixed look.")
    else:
        print(f"\n  Still short of the 0.55-0.60 predicted: {b:.3f}. But "
              f"against the single-look baseline of {base['shifted mean']:.3f} "
              f"this is {b / max(base['shifted mean'], 1e-9):.1f}x,")
        print(f"  and {'above' if b > chance else 'still below'} the "
              f"{chance:.3f} chance rate that a single look falls under when "
              f"the object moves.")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
