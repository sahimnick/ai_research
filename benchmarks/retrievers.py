"""Is the consolidator's recall loss information, or is it the retriever?

Consolidation compressed 1510 cells to 315 and cost 1.2 points of recall
(0.821 -> 0.809, losing 6/6 seeds). That was scored with ONE retriever --
`AssociativeCortex.recall`, a plain argmax over every cell. Nearest-neighbour
structurally rewards having more exemplars, so the comparison confounded *what
the memory knows* with *how it is read*.

Here the same two memories are read by five retrievers. If the append
advantage survives every retriever, consolidation genuinely destroyed
information. If it collapses under a retriever that does not count exemplars,
the loss was the read-out's and the summary is intact.

Retrievers (all local, all read-only -- none of them learn):
  nn            argmax over all cells                     (the shipped one)
  knn5          majority label of the 5 nearest cells
  proto         nearest per-label mean (a summary read-out)
  softmax       temperature-weighted vote over all cells
  strength_nn   nn, with each cell scaled by its strength counter

Usage:  python3 benchmarks/retrievers.py out_retrievers.json
"""
import json, sys
from collections import Counter
import numpy as np

import neurobrain as nb
from neurobrain.memory.development import SleepConsolidator

SEEDS = (0, 1, 2, 3, 4)
N_PALLIUM, N_DAY, NOISE = 300, 400, 0.55


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def zipf_day(trY, n, rng, alpha=1.4):
    f = 1.0 / np.arange(1, 11) ** alpha
    f = f / f.sum()
    order = rng.permutation(10)
    return np.array([int(rng.choice(np.where(trY == int(order[d]))[0]))
                     for d in rng.choice(10, size=n, p=f)])


class Retrievers:
    """Read-only views over one AssociativeCortex."""

    def __init__(self, cortex, modality="image"):
        self.M = cortex.modalities[modality]
        self.labels = np.array([str(l) for l in cortex.labels])
        self.strength = getattr(cortex, "strength", None)
        if self.strength is None or len(self.strength) != len(self.labels):
            self.strength = np.ones(len(self.labels), np.float32)
        self.uniq = sorted(set(self.labels.tolist()))
        self.proto = np.array(
            [_unit(self.M[self.labels == u].mean(0)) for u in self.uniq],
            np.float32)

    def nn(self, x):
        return self.labels[int(np.argmax(self.M @ x))]

    def knn5(self, x, k=5):
        s = self.M @ x
        top = np.argpartition(-s, min(k, len(s) - 1))[:k]
        return Counter(self.labels[top].tolist()).most_common(1)[0][0]

    def proto_nn(self, x):
        return self.uniq[int(np.argmax(self.proto @ x))]

    def softmax(self, x, temp=0.05):
        s = self.M @ x
        w = np.exp((s - s.max()) / temp)
        tot = {}
        for l, wi in zip(self.labels, w):
            tot[l] = tot.get(l, 0.0) + float(wi)
        return max(tot, key=tot.get)

    def strength_nn(self, x):
        return self.labels[int(np.argmax((self.M @ x) * np.log1p(self.strength)))]


NAMES = ("nn", "knn5", "proto", "softmax", "strength_nn")


def score_all(cortex, tex, teY, seed):
    R = Retrievers(cortex)
    fns = dict(nn=R.nn, knn5=R.knn5, proto=R.proto_nn,
               softmax=R.softmax, strength_nn=R.strength_nn)
    rng = np.random.default_rng(seed)
    cues, truth = [], []
    for im, y in zip(tex, teY):
        v = im.reshape(-1).astype(np.float32) / 255.0
        cues.append(_unit(v + rng.normal(0, NOISE, v.shape).astype(np.float32)))
        truth.append(str(int(y)))
    return {n: float(np.mean([fns[n](c) == t for c, t in zip(cues, truth)]))
            for n in NAMES}


def main():
    trx, trY, tex, teY = nb.load_mnist(n_train=4000, n_test=400)
    tex, teY = tex[:400], teY[:400]
    rows = {"append": [], "consolidate": []}
    cells = {"append": [], "consolidate": []}

    for sd in SEEDS:
        picks = zipf_day(trY, N_DAY, np.random.default_rng(sd + 100))
        for arm in ("append", "consolidate"):
            mind = nb.build_unified_mind(n_pallium=N_PALLIUM)
            for i in picks:
                mind.perceive(trx[i])
            cons = SleepConsolidator(mind.episodes, seed=0)
            if arm == "consolidate":
                learner = lambda p, l, lr: mind.space.consolidate(
                    str(l), lr=lr, image=p)
            else:
                learner = lambda p, l, lr: mind.space.remember(str(l), image=p)
            cons.sleep(learner, cycles=3, replays_per_cycle=400)
            rows[arm].append(score_all(mind.space.cortex, tex, teY, sd))
            cells[arm].append(mind.space.cortex.n_memories)
        print(f"seed {sd} done", flush=True)

    agg = {a: {n: dict(mean=round(float(np.mean([r[n] for r in rs])), 4),
                       sd=round(float(np.std([r[n] for r in rs], ddof=1)), 4))
               for n in NAMES}
           for a, rs in rows.items()}
    paired = {}
    for n in NAMES:
        d = np.array([a[n] for a in rows["append"]]) - \
            np.array([c[n] for c in rows["consolidate"]])
        paired[n] = dict(append_minus_consolidate=round(float(d.mean()), 4),
                         sd=round(float(d.std(ddof=1)), 4),
                         append_wins=int((d > 0).sum()), n=len(d),
                         cohens_d=round(float(d.mean() / (d.std(ddof=1) + 1e-12)), 3))

    out = dict(by_arm=agg, paired=paired,
               cells=dict(append=float(np.mean(cells["append"])),
                          consolidate=float(np.mean(cells["consolidate"]))))

    print(f"\ncells: append {out['cells']['append']:.0f}  "
          f"consolidate {out['cells']['consolidate']:.0f}")
    print(f"\n{'retriever':<14}{'append':>16}{'consolidate':>16}{'A-C':>10}{'wins':>7}{'d':>8}")
    for n in NAMES:
        a, c, p = agg["append"][n], agg["consolidate"][n], paired[n]
        print(f"{n:<14}{a['mean']:>10.3f} ±{a['sd']:.3f}"
              f"{c['mean']:>10.3f} ±{c['sd']:.3f}"
              f"{p['append_minus_consolidate']:>+10.4f}"
              f"{p['append_wins']:>4}/{p['n']}{p['cohens_d']:>8.2f}")

    best_a = max(NAMES, key=lambda n: agg["append"][n]["mean"])
    best_c = max(NAMES, key=lambda n: agg["consolidate"][n]["mean"])
    print(f"\nbest retriever for append      : {best_a} "
          f"({agg['append'][best_a]['mean']:.3f})")
    print(f"best retriever for consolidate : {best_c} "
          f"({agg['consolidate'][best_c]['mean']:.3f})")
    print(f"best-vs-best gap: "
          f"{agg['append'][best_a]['mean'] - agg['consolidate'][best_c]['mean']:+.4f}")
    out["best"] = dict(append=best_a, consolidate=best_c,
                       gap=round(agg["append"][best_a]["mean"]
                                 - agg["consolidate"][best_c]["mean"], 4))

    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1
                        else "out_retrievers.json", "w"), indent=1)


if __name__ == "__main__":
    main()
