"""Is the memoriser a memoriser because of the architecture, or the data?

`heard_world.py` ends on a claim it cannot test. Six findings across this
section reduce to one fact -- the concept layer stores exemplars instead of
forming categories -- and that fact has two candidate causes:

    pressure   vigilance recruits an uncommitted cell whenever the match falls
               short, and nothing ever forces a merge
    scale      with 57 training clips over 6 classes there is nothing to
               generalise *from*, and an exemplar store is genuinely the better
               recogniser

Only the second can be settled by adding data, and the auditory world cannot:
ESC-50 has 40 clips per class, full stop. CIFAR-10 has **5,000**. So the same
layer, the same rules, the same measurements, on 100 -> 3000 photographs.

The two factors are the ones the eye already emits -- `[luminance | chroma]`,
form and colour -- so the association area binds one modality to the other
exactly as it binds sight to sound, and `bind_contrastive` has a completion to
make and an error to learn from.

What is measured, and why each number is needed
-----------------------------------------------
    pairs per cell    below ~1 the layer is a memory. This is the fact under
                      everything else, and the question is whether data moves it
    prediction error  1 - cos between what the layer completes from colour alone
                      and the form that actually arrived. **A memoriser is never
                      surprised**: measured at 0.054 on the auditory world, which
                      is why the error-driven rule there was inert
    plain vs contrastive
                      instar against the two-phase rule. If scale is the cause,
                      the gap should open as data grows; if pressure is, it
                      should not, however much is added

A held-out probe throughout, and the two rules see identical data in identical
order -- they differ only in whether the negative phase runs.

Usage:  python3 benchmarks/scale.py out_scale.json
"""
import json
import sys

import numpy as np

from neurobrain.cognition.multimodal import AssociationArea, _unit
from neurobrain.sensing.natural import load_cifar10
from neurobrain.sensing.streams import StreamingBrain
from neurobrain.vision.widev1 import PopulationAdaptation

sys.path.insert(0, "benchmarks")
from real_binding import opponent                            # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
SIZES = (100, 300, 1000, 3000)
N_TEST = 500
N_CONCEPT = 4096          # never the binding constraint at any size below
V1_CELLS = 2048


def encode(brain, images):
    """Form and colour, kept apart -- the eye's own factorisation."""
    F, C = [], []
    for im in images:
        lum, rg, by = opponent(im)
        F.append(brain.v1.rate(lum))
        C.append(np.concatenate([brain.v1.rate(rg), brain.v1.rate(by)]))
    F, C = np.array(F, np.float32), np.array(C, np.float32)
    for M in (F, C):
        ad = PopulationAdaptation(M.shape[1])
        M[:] = np.array([_unit(ad(r)) for r in M], np.float32)
    return F, C


def wake(F, C, y, tr, seed, rule, passes=2):
    a = AssociationArea(n_vis=F.shape[1], n_aud=C.shape[1],
                        n_concept=N_CONCEPT, seed=seed)
    a.set_stats(F[tr], C[tr])
    votes, errs = {}, []
    rng = np.random.default_rng(seed + 3)
    for _ in range(passes):
        for i in tr:
            if rule == "contrastive":
                w, _, e = a.bind_contrastive(F[i], C[i], rng=rng)
                errs.append(e)
            else:
                w = a.bind(F[i], C[i])
            votes.setdefault(w, {})
            votes[w][int(y[i])] = votes[w].get(int(y[i]), 0) + 1
    return a, votes, (float(np.mean(errs)) if errs else 0.0)


def probe(a, votes, F, C, y, te):
    name = {c: max(v.items(), key=lambda kv: kv[1])[0] for c, v in votes.items()}
    ok = 0
    for i in te:
        m = a.match(a.prep_v(F[i]), a.prep_a(C[i]))
        ok += int(name.get(int(np.argmax(m)), -1) == int(y[i]))
    return ok / max(len(te), 1)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_scale.json"
    n_max = max(SIZES)
    X, y, Xt, yt = load_cifar10(n_train=n_max, n_test=N_TEST,
                                grayscale=False, size=32)
    brain = StreamingBrain(seed=0, image_shape=(32, 32), v1_cells=V1_CELLS,
                           rf=7, stride=3)
    print(f"encoding {n_max + N_TEST} photographs...", flush=True)
    F, C = encode(brain, list(X) + list(Xt))
    Fte, Cte = F[n_max:], C[n_max:]
    F, C = F[:n_max], C[:n_max]
    n_cls = len(np.unique(y))
    print(f"form {F.shape[1]}d | colour {C.shape[1]}d, {n_cls} classes, "
          f"chance {1/n_cls:.3f}\n", flush=True)

    Fa = np.concatenate([F, Fte])
    Ca = np.concatenate([C, Cte])
    ya = np.concatenate([y, yt])
    te = np.arange(len(F), len(Fa))

    res = {"seeds": list(SEEDS), "sizes": list(SIZES), "n_class": n_cls,
           "rows": {}}
    print(f"{'n_train':>9}{'cells':>8}{'pairs/cell':>12}{'pred error':>12}"
          f"{'plain':>9}{'contrastive':>13}{'delta':>9}")
    for n in SIZES:
        pl, ct, er, pc, cl = [], [], [], [], []
        for sd in SEEDS:
            rng = np.random.default_rng(sd)
            tr = rng.permutation(len(F))[:n]
            ab, vb, _ = wake(Fa, Ca, ya, tr, sd, "bind")
            ac, vc, e = wake(Fa, Ca, ya, tr, sd, "contrastive")
            pl.append(probe(ab, vb, Fa, Ca, ya, te))
            ct.append(probe(ac, vc, Fa, Ca, ya, te))
            live = int((ab.wins > 0).sum())
            er.append(e); cl.append(live); pc.append(n / max(live, 1))
        d = np.array(ct) - np.array(pl)
        sd_ = float(d.std(ddof=1)) if len(d) > 1 else 0.0
        row = dict(cells=round(float(np.mean(cl)), 1),
                   pairs_per_cell=round(float(np.mean(pc)), 2),
                   error=round(float(np.mean(er)), 4),
                   plain=round(float(np.mean(pl)), 4),
                   contrastive=round(float(np.mean(ct)), 4),
                   delta=round(float(d.mean()), 4),
                   cohens_d=round(float(d.mean() / (sd_ + 1e-12)), 3),
                   wins=int((d > 0).sum()), n=len(d))
        res["rows"][str(n)] = row
        print(f"{n:>9}{row['cells']:>8.1f}{row['pairs_per_cell']:>12.2f}"
              f"{row['error']:>12.3f}{row['plain']:>9.3f}"
              f"{row['contrastive']:>13.3f}{row['delta']:>+9.4f}", flush=True)

    lo, hi = res["rows"][str(SIZES[0])], res["rows"][str(SIZES[-1])]
    print("\n=== does data turn the memoriser into a generaliser? ===")
    print(f"  pairs per cell   {lo['pairs_per_cell']:.2f} at {SIZES[0]} "
          f"-> {hi['pairs_per_cell']:.2f} at {SIZES[-1]} "
          f"({SIZES[-1]//SIZES[0]}x the data)")
    print(f"  prediction error {lo['error']:.3f} -> {hi['error']:.3f}")
    merged = hi["pairs_per_cell"] > 2.0 * lo["pairs_per_cell"]
    res["merges_with_data"] = bool(merged)
    if not merged:
        print("\n  -> it does NOT. Thirty times the data leaves the layer at "
              "roughly the same pairs per cell, so the exemplar store is a")
        print("     property of the RULE -- vigilance recruits whenever the "
              "match falls short and nothing ever forces a merge -- and not")
        print("     of how much was seen. That is a fixable thing in a way "
              "that 'needs more data' is not.")
    else:
        print(f"\n  -> it does: the layer merges as data arrives "
              f"({lo['pairs_per_cell']:.2f} -> {hi['pairs_per_cell']:.2f}).")
    g = hi
    if g["cohens_d"] >= 0.8 and g["wins"] >= 0.75 * g["n"]:
        print(f"  and at {SIZES[-1]} the error-driven rule finally pays: "
              f"{g['delta']:+.4f}, d={g['cohens_d']:+.2f}, "
              f"{g['wins']}/{g['n']}")
        res["contrastive_pays_at_scale"] = True
    else:
        print(f"  and the error-driven rule still does not pay at "
              f"{SIZES[-1]} ({g['delta']:+.4f}, d={g['cohens_d']:+.2f}, "
              f"{g['wins']}/{g['n']}).")
        res["contrastive_pays_at_scale"] = False

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
