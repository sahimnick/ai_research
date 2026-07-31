"""Censoring the meaningless recombinations -- where that is possible at all.

`imagine_factored` will build any crossing of any two concepts, and nothing
judges them: a bus with a frog's colour and a bus with a bird's colour are
equally available and equally unexamined. The proposal is a prior -- "if form =
bus, yellow is likely" -- that censors the implausible ones and pulls the output
toward where real data sits.

`constraint.py` already measured the thing that decides whether such a prior can
work, and the answer is uncomfortable: over (form, colour) on CIFAR the **entire
available signal** is AUC 0.545, and a Hebbian compatibility already captures
92% of it. Form barely constrains colour in a world of cars and buses of every
colour, so a prior over those two factors has almost nothing to censor with.
That is a property of the factorisation, and it is why this benchmark does not
only run there.

Two worlds, and the contrast is the experiment
----------------------------------------------
    form x colour     the eye's own factors on CIFAR-10. Predicted to do nearly
                      nothing, and included so the prediction is on the record
                      rather than assumed
    sight x sound     the same prior over the same code, on the factorisation
                      the world genuinely constrains -- a bark goes with a dog.
                      `constraint.py` measured AUC 0.879 here once the visual
                      side clusters

If the prior works in the second world and not the first, the lesson is about
which factors to build on, not about whether censoring is a good idea.

What censoring is measured against
----------------------------------
Generating N candidate crossings and keeping the most plausible has to be scored
against **keeping one at random from the same N**, or the result is confounded
with having drawn more candidates at all. And it has to be scored against
keeping the *least* plausible, which inverts only the selection and holds
everything else -- if `best` and `worst` do not differ, the prior is not ranking
anything.

    novelty / coherence   where the kept candidate lands, against real data
    censor rate           how often the prior would reject its own first draw

Usage:  python3 benchmarks/prior.py out_prior.json
"""
import json
import sys

import numpy as np

from neurobrain.cognition.multimodal import (AssociationArea,
                                             FactorCompatibility, _unit)
from neurobrain.sensing.natural import load_audiovisual
from neurobrain.sensing.streams import StreamingBrain

sys.path.insert(0, "benchmarks")
from real_binding import encode, split                       # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
N_CONCEPT = 256
N_DRAW = 8            # candidate crossings per imagining
RANK = 256
CHANNELS = 3


def run_seed(V, A, y, n_cls, seed):
    tr, te = split(y, seed)
    d = V.shape[1]
    blk = d // CHANNELS
    bounds = [(0, blk), (blk, d)]

    assoc = AssociationArea(n_vis=d, n_aud=A.shape[1], n_concept=N_CONCEPT,
                            seed=seed)
    assoc.set_stats(V[tr], A[tr])
    votes = {}
    for i in tr:
        w = assoc.bind(V[i], A[i])
        votes.setdefault(w, {})
        votes[w][int(y[i])] = votes[w].get(int(y[i]), 0) + 1
    cell_cat = {c: max(v.items(), key=lambda kv: kv[1])[0]
                for c, v in votes.items() if v}

    Btr = np.stack([assoc.prep_v(V[i]) for i in tr])
    Bte = np.stack([assoc.prep_v(V[i]) for i in te])
    protos = np.stack([_unit(Btr[y[tr] == c].mean(0)) for c in range(n_cls)])
    rng = np.random.default_rng(seed + 77)
    cells = [assoc.concept_from_sound(A[i]) for i in te]
    want = [int(y[i]) for i in te]
    all_cells = sorted(cell_cat)

    fc = FactorCompatibility(bounds, rank=RANK, seed=seed)
    for i in tr:
        fc.observe(assoc.prep_v(V[i]))

    def place(M):
        s = (M @ Btr.T).max(1)
        coh = np.mean([int(np.argmax(protos @ M[k])) == want[k]
                       for k in range(len(M))])
        return dict(novelty=float(1.0 - s.mean()), coherence=float(coh))

    out = {"a real unseen photograph": place(Bte)}

    # ---- generate N crossings per item, then keep by three rules ----------
    picks = {"unjudged (first draw)": [], "most plausible": [],
             "least plausible": [], "random of the same draws": []}
    censored = []
    for k, c in enumerate(cells):
        cands = []
        for _ in range(N_DRAW):
            other = [dd for dd in all_cells
                     if cell_cat.get(dd) != cell_cat.get(c)] or all_cells
            j = int(rng.choice(other))
            cands.append(assoc.imagine_factored([c, j], bounds,
                                                temperature=0.0))
        s = np.array([fc.score(v) for v in cands])
        picks["unjudged (first draw)"].append(cands[0])
        picks["most plausible"].append(cands[int(np.argmax(s))])
        picks["least plausible"].append(cands[int(np.argmin(s))])
        picks["random of the same draws"].append(
            cands[int(rng.integers(N_DRAW))])
        censored.append(float(s[0] < np.median(s)))
    for tag, M in picks.items():
        out[tag] = place(np.stack(M))
    out["censor_rate"] = float(np.mean(censored))

    # ---- the same prior, over sight x sound ------------------------------
    nv, na = d, A.shape[1]
    b2 = [(0, nv), (nv, nv + na)]
    fc2 = FactorCompatibility(b2, rank=RANK, seed=seed)
    for i in tr:
        fc2.observe(np.concatenate([assoc.prep_v(V[i]), assoc.prep_a(A[i])]))
    real = [fc2.score(np.concatenate([assoc.prep_v(V[k]), assoc.prep_a(A[k])]))
            for k in te]
    bad = []
    for k in te:
        j = int(rng.choice([j for j in tr if int(y[j]) != int(y[k])]))
        bad.append(fc2.score(np.concatenate([assoc.prep_v(V[k]),
                                             assoc.prep_a(A[j])])))
    real, bad = np.array(real), np.array(bad)
    out["sight_sound_auc"] = float((real[:, None] > bad[None, :]).mean())

    # and the form x colour prior's own discrimination, for comparison
    rf = [fc.score(assoc.prep_v(V[k])) for k in te]
    rb = []
    for k in te:
        v = assoc.prep_v(V[k]).copy()
        v[blk:] = assoc.prep_v(V[int(rng.choice(tr))])[blk:]
        rb.append(fc.score(v))
    rf, rb = np.array(rf), np.array(rb)
    out["form_colour_auc"] = float((rf[:, None] > rb[None, :]).mean())
    return out


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_prior.json"
    images, waves, y, names = load_audiovisual(n_per_class=60, seed=0,
                                               grayscale=False, size=32)
    n_cls = len(names)
    brain = StreamingBrain(seed=0, image_shape=(32, 32), v1_cells=4096,
                           rf=7, stride=2)
    V, A = encode(brain, images, waves, colour=True, adapt=True, develop=True)
    print(f"{len(images)} real pairs, {n_cls} categories, "
          f"{N_DRAW} candidate crossings per imagining\n", flush=True)

    rows = [run_seed(V, A, y, n_cls, sd) for sd in SEEDS]
    arms = [k for k in rows[0] if isinstance(rows[0][k], dict)]
    res = {"seeds": list(SEEDS), "arms": {}}
    for k in arms:
        res["arms"][k] = {q: round(float(np.mean([r[k][q] for r in rows])), 4)
                          for q in rows[0][k]}
    for k in ("censor_rate", "form_colour_auc", "sight_sound_auc"):
        res[k] = round(float(np.mean([r[k] for r in rows])), 4)

    t = res["arms"]["a real unseen photograph"]
    print(f"{'selection rule':<28}{'novelty':>9}{'coherence':>11}"
          f"{'gap to real':>13}")
    for k in arms:
        a = res["arms"][k]
        a["gap"] = round(float(np.hypot(a["novelty"] - t["novelty"],
                                        a["coherence"] - t["coherence"])), 4)
        mark = "   <- the target" if k.startswith("a real") else ""
        print(f"{k:<28}{a['novelty']:>9.3f}{a['coherence']:>11.3f}"
              f"{a['gap']:>13.3f}{mark}")

    best = res["arms"]["most plausible"]
    worst = res["arms"]["least plausible"]
    rand = res["arms"]["random of the same draws"]
    d_sel = np.array([r["most plausible"]["coherence"]
                      - r["least plausible"]["coherence"] for r in rows])
    sd = float(d_sel.std(ddof=1))
    res["selection_gap"] = dict(delta=round(float(d_sel.mean()), 4),
                                sd=round(sd, 4),
                                cohens_d=round(float(d_sel.mean()
                                                     / (sd + 1e-12)), 3),
                                wins=int((d_sel > 0).sum()), n=len(d_sel))

    print(f"\n=== does the prior rank anything? ===")
    print(f"  best minus worst, on coherence: {d_sel.mean():+.4f} "
          f"(d={res['selection_gap']['cohens_d']:+.2f}, "
          f"{res['selection_gap']['wins']}/{len(d_sel)})")
    print(f"  and against keeping one of the same draws at random: "
          f"{best['coherence'] - rand['coherence']:+.4f}")
    print(f"\n  the prior's own discrimination:")
    print(f"    form x colour   AUC {res['form_colour_auc']:.3f}")
    print(f"    sight x sound   AUC {res['sight_sound_auc']:.3f}"
          f"   <- the same rule, a factorisation the world constrains")

    works = (res["selection_gap"]["cohens_d"] >= 0.8
             and res["selection_gap"]["wins"] >= 0.75 * len(d_sel))
    res["prior_ranks"] = bool(works)
    unj = res["arms"]["unjudged (first draw)"]
    # gap is a distance: SMALLER is closer to real data. Reporting the raw
    # difference reads as a gain when it is a loss, so state the direction.
    dgap = best["gap"] - unj["gap"]
    res["gap_change"] = round(float(dgap), 4)
    if works and dgap < 0:
        print(f"\n  The prior ranks crossings (coherence "
              f"{d_sel.mean():+.4f}, d={res['selection_gap']['cohens_d']:+.2f})"
              f" AND censoring moves the output closer to real data: gap "
              f"{unj['gap']:.3f} -> {best['gap']:.3f}.")
    elif works:
        print(f"\n  The prior DOES rank crossings -- coherence "
              f"{unj['coherence']:.3f} -> {best['coherence']:.3f} "
              f"({d_sel.mean():+.4f} best over worst, "
              f"d={res['selection_gap']['cohens_d']:+.2f}, "
              f"{res['selection_gap']['wins']}/{len(d_sel)}), and against "
              f"keeping one of the same draws at random it is "
              f"{best['coherence'] - rand['coherence']:+.4f}. The ranking is "
              f"real.")
        print(f"\n  But it does NOT move the output toward real data: the gap "
              f"gets *worse*, {unj['gap']:.3f} -> {best['gap']:.3f} "
              f"({dgap:+.4f}; a gap is a distance, so up is away).")
        print(f"  It buys coherence by spending novelty "
              f"({unj['novelty']:.3f} -> {best['novelty']:.3f}), and the "
              f"target needs both -- real data sits at novelty "
              f"{t['novelty']:.3f}, coherence {t['coherence']:.3f}.")
        print(f"  Censoring makes the imagining safer, not more real. That is "
              f"what a prior built on factors this weakly coupled can do: "
              f"AUC {res['form_colour_auc']:.3f} over form x colour,")
        print(f"  {res['sight_sound_auc']:.3f} over sight x sound. Both are "
              f"near the 0.500 of ranking nothing, and the ceiling "
              f"`constraint.py` measured says that is the world, not the rule.")
    else:
        print(f"\n  It does not. Over (form, colour) the prior scores AUC "
              f"{res['form_colour_auc']:.3f} -- barely above the 0.500 of "
              f"ranking nothing -- so 'most plausible' and")
        print("  'least plausible' select nearly the same population and land "
              "in the same place. This is the predicted result and it is "
              "about the")
        print(f"  factors, not the rule: the same Hebbian prior reaches AUC "
              f"{res['sight_sound_auc']:.3f} over sight x sound. A censor "
              f"needs factors that constrain each other,")
        print("  and form x colour are close to independent -- which is "
              "exactly why crossing them produced a coherent object in the "
              "first place.")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
