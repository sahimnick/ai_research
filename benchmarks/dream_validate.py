"""Re-evaluation of the dream() fix. What the first pass could not support.

The first pass ran ONE seed per arm. It reported +2.0 for replay and a
rare-class regression of -3.5, and neither number had an error bar, so neither
was safe to believe. It also never tested the thing the surprise signal exists
for.

Three questions:

1. **Does +2.0 survive repetition?** Paired across seeds, with a spread.
2. **Does prioritisation earn its place?** `EpisodicBuffer.sample_replay` takes
   `prioritised=`. If prioritised and uniform replay score the same, the
   surprise signal is decoration -- and we already know it takes only two
   distinct values.
3. **Is the rare-class regression real?** Same paired treatment, rare and
   common scored separately.

Usage:  python3 benchmarks/dream_validate.py out_dream_validate.json
"""
import json, sys
import numpy as np

import neurobrain as nb
from neurobrain.memory.development import SleepConsolidator

SEEDS = (0, 1, 2, 3, 4, 5)
N_PALLIUM, N_DAY, NOISE = 300, 400, 0.55


def zipf_day(trY, n, rng, alpha=1.4):
    freq = 1.0 / np.arange(1, 11) ** alpha
    freq = freq / freq.sum()
    order = rng.permutation(10)
    picks = []
    for d in rng.choice(10, size=n, p=freq):
        picks.append(int(rng.choice(np.where(trY == int(order[d]))[0])))
    return np.array(picks)


def probe(mind, X, Y, seed, noise=NOISE):
    rng = np.random.default_rng(seed)
    ok = 0
    for im, y in zip(X, Y):
        v = im.reshape(-1).astype(np.float32) / 255.0
        v = v + rng.normal(0, noise, v.shape).astype(np.float32)
        ok += int(mind.space.recognise("image", v) == str(int(y)))
    return ok / max(len(Y), 1)


def one_seed(seed, trx, trY, tex, teY):
    """Build a mind, give it the SAME day, then vary only what happens at night."""
    picks = zipf_day(trY, N_DAY, np.random.default_rng(seed + 100))
    counts = np.array([sum(1 for i in picks if int(trY[i]) == d) for d in range(10)])
    rare = set(np.argsort(counts)[:5].tolist())
    m_rare = np.array([int(y) in rare for y in teY])

    res = {}
    for arm in ("no_sleep", "prioritised", "uniform", "shuffled"):
        mind = nb.build_unified_mind(n_pallium=N_PALLIUM)
        rng = np.random.default_rng(seed + 7)
        labels = [str(int(trY[i])) for i in picks]
        if arm == "shuffled":
            labels = list(rng.permutation(labels))
            for i, lab in zip(picks, labels):
                mind._record_episode(trx[i], lab)
        else:
            for i in picks:
                mind.perceive(trx[i])

        before = probe(mind, tex, teY, seed)
        before_rare = probe(mind, tex[m_rare], teY[m_rare], seed)
        if arm != "no_sleep":
            # drive the consolidator directly so `prioritised` can be varied
            cons = SleepConsolidator(mind.episodes, seed=0)
            cons.sleep(lambda p, l, lr: mind.space.remember(str(l), image=p),
                       cycles=3, replays_per_cycle=400,
                       prioritised=(arm != "uniform"))
        after = probe(mind, tex, teY, seed)
        after_rare = probe(mind, tex[m_rare], teY[m_rare], seed)
        res[arm] = dict(before=before, after=after, delta=after - before,
                        before_rare=before_rare, after_rare=after_rare,
                        delta_rare=after_rare - before_rare)
    return res


def main():
    trx, trY, tex, teY = nb.load_mnist(n_train=4000, n_test=400)
    tex, teY = tex[:400], teY[:400]
    per_seed = []
    for sd in SEEDS:
        r = one_seed(sd, trx, trY, tex, teY)
        per_seed.append(r)
        print(f"seed {sd}: " + "  ".join(
            f"{k}={v['after']:.3f}" for k, v in r.items()), flush=True)

    arms = ("no_sleep", "prioritised", "uniform", "shuffled")
    agg = {}
    for a in arms:
        after = np.array([s[a]["after"] for s in per_seed])
        d = np.array([s[a]["delta"] for s in per_seed])
        dr = np.array([s[a]["delta_rare"] for s in per_seed])
        agg[a] = dict(after_mean=round(float(after.mean()), 4),
                      after_sd=round(float(after.std(ddof=1)), 4),
                      delta_mean=round(float(d.mean()), 4),
                      delta_sd=round(float(d.std(ddof=1)), 4),
                      delta_rare_mean=round(float(dr.mean()), 4),
                      delta_rare_sd=round(float(dr.std(ddof=1)), 4))

    def paired(a, b, key="after"):
        x = np.array([s[a][key] for s in per_seed])
        y = np.array([s[b][key] for s in per_seed])
        d = x - y
        return dict(mean=round(float(d.mean()), 4),
                    sd=round(float(d.std(ddof=1)), 4),
                    wins=int((d > 0).sum()), losses=int((d < 0).sum()),
                    n=len(d),
                    cohens_d=round(float(d.mean() / (d.std(ddof=1) + 1e-12)), 3))

    out = dict(n_seeds=len(SEEDS), by_arm=agg,
               replay_vs_none=paired("prioritised", "no_sleep"),
               prioritised_vs_uniform=paired("prioritised", "uniform"),
               real_vs_shuffled=paired("prioritised", "shuffled"),
               rare_replay_vs_none=paired("prioritised", "no_sleep", "delta_rare"),
               per_seed=per_seed)

    print(f"\n=== {len(SEEDS)} seeds, paired ===")
    print(f"{'arm':>12} {'recall after':>14} {'delta':>16} {'delta rare':>16}")
    for a in arms:
        g = agg[a]
        print(f"{a:>12} {g['after_mean']:>8.3f} +/-{g['after_sd']:.3f}"
              f" {g['delta_mean']:>+10.3f} +/-{g['delta_sd']:.3f}"
              f" {g['delta_rare_mean']:>+10.3f} +/-{g['delta_rare_sd']:.3f}")
    print()
    for name, k in (("replay vs no sleep", "replay_vs_none"),
                    ("prioritised vs uniform", "prioritised_vs_uniform"),
                    ("real vs shuffled day", "real_vs_shuffled"),
                    ("RARE: replay vs no sleep", "rare_replay_vs_none")):
        p = out[k]
        print(f"  {name:26} {p['mean']:+.4f} +/-{p['sd']:.4f}  "
              f"wins {p['wins']}/{p['n']}  d={p['cohens_d']}")

    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1
                        else "out_dream_validate.json", "w"), indent=1)


if __name__ == "__main__":
    main()
