"""Does the world model still recite the number line?

`build_unified_mind` used to teach its world model exactly one thing -- the
counting order `0->1->...->9`, twenty times, from a literal loop -- and nothing
else ever wrote to the transition matrix. It was peaked 400:1, temperature was
inert, and every "train of thought" was the number line.

`UnifiedMind._record_episode` now writes each lived transition, so the world
model learns whatever order the mind actually perceives things in.

Measured here, taught-only vs lived:

1. **Does the chain wander?** unique concepts, self-loops, entropy, and whether
   temperature does anything at all.
2. **Does it stay coherent?** Imagine a percept, feed it back into perception.
   Novelty is worthless if the mind cannot recognise its own imagery -- that
   pair (entropy up, round-trip high) is what makes "constantly imaginative"
   falsifiable rather than a vibe.
3. **Does the surprise signal come alive?** The dream re-evaluation found
   prioritised replay indistinguishable from uniform (d=0.21, 3/6 seeds),
   because surprise took only two distinct values off a world model that knew
   nothing but counting. If lived transitions are the cause, prioritisation
   should start to matter here.

Usage:  python3 benchmarks/imagination.py out_imagination.json
"""
import json, sys
from collections import Counter
import numpy as np

import neurobrain as nb

SEEDS = (0, 1, 2)
N_DAY = 400
TEMPS = (0.3, 0.6, 1.0)


def chain_stats(space, steps=40, n_seeds=8, temps=TEMPS):
    concepts = list(space.concepts)
    rows = []
    for temp in temps:
        for i, s in enumerate(concepts[:n_seeds]):
            names, _ = space.imagine(s, steps=steps, temperature=temp, rng_seed=i)
            c = np.array(list(Counter(names).values()), float)
            p = c / c.sum()
            rows.append(dict(
                temperature=temp, unique=len(set(names)),
                self_loops=sum(1 for a, b in zip(names, names[1:]) if a == b)
                / max(len(names) - 1, 1),
                entropy=float(-(p * np.log2(p)).sum())))
    out = {}
    for temp in temps:
        rs = [r for r in rows if r["temperature"] == temp]
        out[str(temp)] = dict(
            unique=round(float(np.mean([r["unique"] for r in rs])), 2),
            self_loops=round(float(np.mean([r["self_loops"] for r in rs])), 4),
            entropy=round(float(np.mean([r["entropy"] for r in rs])), 3))
    # temperature sensitivity: if this is ~0 the knob is inert
    ent = [out[str(t)]["entropy"] for t in temps]
    out["temperature_span"] = round(float(max(ent) - min(ent)), 4)
    out["max_entropy_bits"] = round(float(np.log2(len(concepts))), 3)
    return out


def round_trip(mind, steps=25, n_seeds=8):
    """Imagine, then re-perceive. remember=False so probing is not experience."""
    space = mind.space
    concepts = list(space.concepts)
    ok = n = 0
    for i, s in enumerate(concepts[:n_seeds]):
        names, percepts = space.imagine(s, steps=steps, temperature=0.6, rng_seed=i)
        for nm, pc in zip(names, percepts):
            if pc is None:
                continue
            img = np.asarray(pc, np.float32)
            try:
                back = mind.perceive(
                    img.reshape(28, 28) if img.size == 784 else img,
                    remember=False)
            except Exception:
                continue
            n += 1
            ok += int(str(back) == str(nm))
    return dict(n=n, correct=ok, round_trip=round(ok / max(n, 1), 4),
                chance=round(1 / max(len(concepts), 1), 4))


def transition_shape(space):
    T = np.asarray(space._T, float)
    row = T / np.maximum(T.sum(1, keepdims=True), 1e-12)
    ent = -(np.where(row > 0, row * np.log2(np.maximum(row, 1e-12)), 0)).sum(1)
    return dict(nonzero_frac=round(float((T > 0.06).mean()), 4),
                peak_ratio=round(float(T.max() / max(T[T > 0].min(), 1e-12)), 1),
                row_entropy_mean=round(float(ent.mean()), 3),
                uniform_row_entropy=round(float(np.log2(T.shape[1])), 3))


def surprise_shape(mind):
    s = np.array([e.surprise for e in mind.episodes.episodes])
    if not len(s):
        return dict(n=0)
    return dict(n=len(s), distinct=int(len(np.unique(np.round(s, 6)))),
                mean=round(float(s.mean()), 4), sd=round(float(s.std()), 4),
                min=round(float(s.min()), 4), max=round(float(s.max()), 4))


def run(arm, seed, trx, trY):
    """arm: 'taught_only' reproduces the old behaviour; 'lived' is the fix."""
    mind = nb.build_unified_mind(
        n_pallium=300, learn_transitions=(arm == "lived"))
    rng = np.random.default_rng(seed + 100)
    for i in rng.choice(len(trx), N_DAY, replace=False):
        mind.perceive(trx[i])
    return dict(arm=arm, seed=seed,
                transitions=transition_shape(mind.space),
                surprise=surprise_shape(mind),
                chains=chain_stats(mind.space),
                round_trip=round_trip(mind)), mind


def main():
    trx, trY, tex, teY = nb.load_mnist(n_train=4000, n_test=400)
    rows = []
    for arm in ("taught_only", "lived"):
        for sd in SEEDS:
            r, mind = run(arm, sd, trx, trY)
            rows.append(r)
            print(f"{arm:12} seed{sd}: entropy@0.6="
                  f"{r['chains']['0.6']['entropy']:.2f} "
                  f"unique={r['chains']['0.6']['unique']:.1f} "
                  f"temp_span={r['chains']['temperature_span']:.3f} "
                  f"round_trip={r['round_trip']['round_trip']:.3f} "
                  f"surprise_distinct={r['surprise']['distinct']}", flush=True)
            if sd == SEEDS[0]:
                names, _ = mind.space.imagine(
                    list(mind.space.concepts)[0], steps=15,
                    temperature=0.6, rng_seed=0)
                print(f"             chain: {' -> '.join(names)}", flush=True)

    agg = {}
    for arm in ("taught_only", "lived"):
        rs = [r for r in rows if r["arm"] == arm]
        m = lambda f: round(float(np.mean([f(r) for r in rs])), 4)
        agg[arm] = dict(
            entropy_at_0_6=m(lambda r: r["chains"]["0.6"]["entropy"]),
            unique_at_0_6=m(lambda r: r["chains"]["0.6"]["unique"]),
            self_loops=m(lambda r: r["chains"]["0.6"]["self_loops"]),
            temperature_span=m(lambda r: r["chains"]["temperature_span"]),
            round_trip=m(lambda r: r["round_trip"]["round_trip"]),
            transition_nonzero=m(lambda r: r["transitions"]["nonzero_frac"]),
            row_entropy=m(lambda r: r["transitions"]["row_entropy_mean"]),
            surprise_distinct=m(lambda r: r["surprise"]["distinct"]),
            surprise_sd=m(lambda r: r["surprise"]["sd"]))
    out = dict(rows=rows, by_arm=agg,
               max_entropy_bits=rows[0]["chains"]["max_entropy_bits"])

    print(f"\n=== {len(SEEDS)} seeds ===")
    print(f"{'metric':<26}{'taught only':>14}{'lived':>12}{'delta':>10}")
    for k in ("entropy_at_0_6", "unique_at_0_6", "self_loops",
              "temperature_span", "round_trip", "transition_nonzero",
              "row_entropy", "surprise_distinct", "surprise_sd"):
        a, b = agg["taught_only"][k], agg["lived"][k]
        print(f"{k:<26}{a:>14.3f}{b:>12.3f}{b - a:>+10.3f}")
    print(f"\n(max possible chain entropy over 10 concepts: "
          f"{out['max_entropy_bits']} bits)")

    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1
                        else "out_imagination.json", "w"), indent=1)


if __name__ == "__main__":
    main()
