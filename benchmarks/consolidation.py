"""Two mechanisms built to answer two failures. Do they earn their place?

The failures, both proven rather than suspected:

* **Consolidation was frequency-blind.** `AssociativeCortex.remember` allocates
  a new cell per call, so replaying one episode 200 times left 200 identical
  cells and moved recall by 0.0000. That is why prioritised replay measured the
  same as uniform replay across 6 seeds (d=0.21) however informative the
  priority signal became -- frequency had nowhere to land.
* **Reconstruction was verbatim.** `reconstruct` returns a stored row, so every
  "imagined" percept had cosine 1.000000 to a cell already in the store. The
  mind could only replay what it had seen.

The fixes: `consolidate()` folds a replay into an existing trace by the instar
rule (local, Hebbian, no gradient) and counts its strength; `blend()` reads out
a Dirichlet-weighted population of traces instead of one winner.

Three questions:

1. Does consolidating beat appending on recall?
2. **Does prioritisation matter now?** This has been open for two rounds and is
   the whole reason the consolidator was built.
3. Novelty vs coherence for the blend: novelty is worthless if the mind cannot
   recognise its own imagery, so both are measured together across
   `concentration`. That pair is the falsifiable form of "imaginative".

Usage:  python3 benchmarks/consolidation.py out_consolidation.json
"""
import json, sys
import numpy as np

import neurobrain as nb
from neurobrain.memory.development import SleepConsolidator

SEEDS = (0, 1, 2, 3, 4, 5)
N_PALLIUM, N_DAY, NOISE = 300, 400, 0.55


def zipf_day(trY, n, rng, alpha=1.4):
    f = 1.0 / np.arange(1, 11) ** alpha
    f = f / f.sum()
    order = rng.permutation(10)
    return np.array([int(rng.choice(np.where(trY == int(order[d]))[0]))
                     for d in rng.choice(10, size=n, p=f)])


def probe(mind, tex, teY, seed):
    rng = np.random.default_rng(seed)
    ok = 0
    for im, y in zip(tex, teY):
        v = im.reshape(-1).astype(np.float32) / 255.0
        v = v + rng.normal(0, NOISE, v.shape).astype(np.float32)
        ok += int(mind.space.recognise("image", v) == str(int(y)))
    return ok / len(teY)


def night(mind, consolidating, prioritised, cycles=3, per_cycle=400):
    cons = SleepConsolidator(mind.episodes, seed=0)
    if consolidating:
        learner = lambda p, l, lr: mind.space.consolidate(str(l), lr=lr, image=p)
    else:
        learner = lambda p, l, lr: mind.space.remember(str(l), image=p)
    return cons.sleep(learner, cycles=cycles, replays_per_cycle=per_cycle,
                      prioritised=prioritised)


def main():
    trx, trY, tex, teY = nb.load_mnist(n_train=4000, n_test=400)
    tex, teY = tex[:400], teY[:400]
    out = {}

    # ------------------------------------- 1 & 2. consolidating, prioritised
    arms = {}
    for sd in SEEDS:
        picks = zipf_day(trY, N_DAY, np.random.default_rng(sd + 100))
        for consolidating in (False, True):
            for prio in (True, False):
                key = ("consolidate" if consolidating else "append") + \
                      ("/prioritised" if prio else "/uniform")
                mind = nb.build_unified_mind(n_pallium=N_PALLIUM)
                for i in picks:
                    mind.perceive(trx[i])
                before = probe(mind, tex, teY, sd)
                night(mind, consolidating, prio)
                after = probe(mind, tex, teY, sd)
                arms.setdefault(key, []).append(
                    dict(seed=sd, before=before, after=after,
                         delta=after - before,
                         cells=mind.space.cortex.n_memories))
        print(f"seed {sd} done", flush=True)

    agg = {}
    for k, rs in arms.items():
        a = np.array([r["after"] for r in rs])
        d = np.array([r["delta"] for r in rs])
        agg[k] = dict(after=round(float(a.mean()), 4),
                      after_sd=round(float(a.std(ddof=1)), 4),
                      delta=round(float(d.mean()), 4),
                      cells=round(float(np.mean([r["cells"] for r in rs])), 1))

    def paired(a, b):
        x = np.array([r["after"] for r in arms[a]])
        y = np.array([r["after"] for r in arms[b]])
        dd = x - y
        return dict(mean=round(float(dd.mean()), 4),
                    sd=round(float(dd.std(ddof=1)), 4),
                    wins=int((dd > 0).sum()), n=len(dd),
                    cohens_d=round(float(dd.mean() / (dd.std(ddof=1) + 1e-12)), 3))

    out["arms"] = agg
    out["consolidate_vs_append"] = paired("consolidate/prioritised",
                                          "append/prioritised")
    out["prioritised_vs_uniform_APPEND"] = paired("append/prioritised",
                                                  "append/uniform")
    out["prioritised_vs_uniform_CONSOLIDATE"] = paired("consolidate/prioritised",
                                                       "consolidate/uniform")

    print(f"\n{'arm':<28}{'recall after':>14}{'cells':>9}")
    for k in sorted(agg):
        print(f"{k:<28}{agg[k]['after']:>8.3f} ±{agg[k]['after_sd']:.3f}"
              f"{agg[k]['cells']:>9.0f}")
    print()
    for name, k in (("consolidate vs append", "consolidate_vs_append"),
                    ("prioritised vs uniform  [append]",
                     "prioritised_vs_uniform_APPEND"),
                    ("prioritised vs uniform  [consolidate]",
                     "prioritised_vs_uniform_CONSOLIDATE")):
        p = out[k]
        print(f"  {name:38} {p['mean']:+.4f} ±{p['sd']:.4f}  "
              f"wins {p['wins']}/{p['n']}  d={p['cohens_d']}")

    # ------------------------------------- 3. novelty vs coherence, blended
    mind = nb.build_unified_mind(n_pallium=N_PALLIUM)
    picks = zipf_day(trY, N_DAY, np.random.default_rng(7))
    for i in picks:
        mind.perceive(trx[i])
    night(mind, True, True)
    M = mind.space.cortex.modalities["image"]
    rows = []
    for blend, conc in [(0, 0.0), (3, 0.6), (6, 0.6), (6, 0.9), (6, 2.0), (12, 0.9)]:
        nov, ok, n = [], 0, 0
        for sd in range(6):
            names, pcs = mind.space.imagine("0", steps=12, temperature=0.6,
                                            rng_seed=sd, blend=blend,
                                            concentration=max(conc, 1e-3))
            for nm, p in zip(names, pcs):
                p = np.asarray(p, np.float32)
                if np.linalg.norm(p) < 1e-9:
                    continue
                nov.append(1.0 - float((M @ (p / np.linalg.norm(p))).max()))
                back = mind.perceive(
                    p.reshape(28, 28) if p.size == 784 else p, remember=False)
                n += 1
                ok += int(str(back) == str(nm))
        rows.append(dict(blend=blend, concentration=conc,
                         novelty=round(float(np.mean(nov)), 5),
                         round_trip=round(ok / max(n, 1), 4), n=n))
        print(f"  blend={blend:<3} conc={conc:<4} novelty={rows[-1]['novelty']:.5f}"
              f"  round_trip={rows[-1]['round_trip']:.3f}", flush=True)
    out["novelty_coherence"] = rows

    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1
                        else "out_consolidation.json", "w"), indent=1)


if __name__ == "__main__":
    main()
