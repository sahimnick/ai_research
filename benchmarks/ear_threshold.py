"""Choosing ContinuousEar.threshold on evidence rather than on one soundscape.

F1 on a single scape is not enough to move a shipped default. Three things are
swept here -- seed, event density, and background noise -- and the deciding
metric is not F1 but **named yield**: of all the events that really happened,
how many did the ear both find *and* name correctly. That is what a downstream
faculty actually receives.

The cost side is **false alarms per minute**: a detection with no real event
behind it still gets classified, so it injects a confident wrong percept into
whatever is listening. A threshold that wins on recall and floods the workspace
with junk is not a better threshold.

Usage:  python3 benchmarks/ear_threshold.py out_ear.json
"""
import json, sys
import numpy as np

import neurobrain as nb
from neurobrain.sensing.streams import (build_soundscape, ContinuousEar,
                                        StreamingBrain, onset_scores)
from neurobrain.vision.widev1 import _nearest_prototype
from neurobrain.audition.audio import sound_dataset

THRESHOLDS = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)
SEEDS = (0, 1, 2, 3, 4)
DENSITIES = {                      # gap between events, seconds
    "sparse": (0.30, 0.90),
    "normal": (0.15, 0.60),        # the build_soundscape default
    "dense":  (0.05, 0.20),
}
NOISES = (0.02, 0.05, 0.15, 0.30)


def _bank(brain, seed):
    sigs, labels, names = sound_dataset(n_per_class=8, seed=seed + 5)
    X = np.array([brain.belt.code(brain.ear.coch.forward(s)[0]) for s in sigs],
                 np.float32)
    perm = np.random.default_rng(seed + 11).permutation(len(X))
    X, y = X[perm], np.asarray(labels)[perm]
    half = len(X) // 2
    return X[:half], y[:half], len(names)


def evaluate(threshold, seed, gap, noise):
    """One condition, end to end."""
    brain = StreamingBrain(seed=seed)
    bank_X, bank_y, ncls = _bank(brain, seed)
    scape = build_soundscape(n_events=24, gap_s=gap, noise=noise, seed=seed)
    ear = ContinuousEar(threshold=threshold)
    found = ear.detect_onsets(scape.wave)
    s = onset_scores(found, scape.events, sr=scape.sr)

    tol = 0.12 * scape.sr
    named_ok, fa = 0, 0
    used = set()
    for f in found:
        cand = [(j, e) for j, e in enumerate(scape.events)
                if abs(e.start - f) <= tol and j not in used]
        code = brain.belt.code(ear.listen(scape.wave, f))
        pred = int(_nearest_prototype(bank_X, bank_y, code[None], ncls)[0])
        if cand:
            j, ev = cand[0]
            used.add(j)
            named_ok += int(pred == ev.label)
        else:
            fa += 1
    dur_min = len(scape.wave) / scape.sr / 60.0
    n_true = len(scape.events)
    return dict(
        threshold=threshold, seed=seed, noise=noise,
        precision=s["precision"], recall=s["recall"], f1=s["f1"],
        # of everything that really happened, how much arrived correctly named
        named_yield=named_ok / max(n_true, 1),
        false_alarms_per_min=fa / max(dur_min, 1e-9),
        n_found=int(s["n_found"]), n_true=n_true, named_ok=named_ok,
    )


def sweep(name, conditions):
    rows = []
    for th in THRESHOLDS:
        for cond in conditions:
            rows.append(evaluate(th, **cond))
    agg = {}
    for th in THRESHOLDS:
        rs = [r for r in rows if r["threshold"] == th]
        m = lambda k: round(float(np.mean([r[k] for r in rs])), 4)
        agg[str(th)] = dict(precision=m("precision"), recall=m("recall"),
                            f1=m("f1"), named_yield=m("named_yield"),
                            false_alarms_per_min=m("false_alarms_per_min"),
                            n=len(rs))
    best_f1 = max(agg, key=lambda t: agg[t]["f1"])
    best_yield = max(agg, key=lambda t: agg[t]["named_yield"])
    print(f"\n=== {name} ({len(rows)} runs) ===", flush=True)
    print(f"{'thr':>5} {'prec':>6} {'recall':>7} {'F1':>6} {'named_yield':>12} {'FA/min':>8}")
    for th in THRESHOLDS:
        a = agg[str(th)]
        mark = ""
        if str(th) == best_yield:
            mark += "  <- best yield"
        if str(th) == best_f1:
            mark += "  <- best F1"
        print(f"{th:>5} {a['precision']:>6.3f} {a['recall']:>7.3f} {a['f1']:>6.3f}"
              f" {a['named_yield']:>12.3f} {a['false_alarms_per_min']:>8.1f}{mark}",
              flush=True)
    return dict(rows=rows, by_threshold=agg,
                best_f1=best_f1, best_named_yield=best_yield)


out = {}
out["by_seed"] = sweep("across 5 seeds, default density and noise",
                       [dict(seed=s, gap=DENSITIES["normal"], noise=0.05)
                        for s in SEEDS])
out["by_density"] = sweep("across 3 event densities (seeds 0-2)",
                          [dict(seed=s, gap=g, noise=0.05)
                           for g in DENSITIES.values() for s in SEEDS[:3]])
out["by_noise"] = sweep("across 4 background-noise levels (seeds 0-2)",
                        [dict(seed=s, gap=DENSITIES["normal"], noise=nz)
                         for nz in NOISES for s in SEEDS[:3]])

# combined recommendation: the threshold that maximises mean named_yield over
# every condition tested, with its false-alarm cost stated
allrows = out["by_seed"]["rows"] + out["by_density"]["rows"] + out["by_noise"]["rows"]
comb = {}
for th in THRESHOLDS:
    rs = [r for r in allrows if r["threshold"] == th]
    comb[str(th)] = dict(
        named_yield=round(float(np.mean([r["named_yield"] for r in rs])), 4),
        f1=round(float(np.mean([r["f1"] for r in rs])), 4),
        false_alarms_per_min=round(float(np.mean([r["false_alarms_per_min"] for r in rs])), 3),
        n=len(rs))
best = max(comb, key=lambda t: comb[t]["named_yield"])
out["combined"] = comb
out["recommended"] = float(best)
out["shipped_default"] = 2.5
print(f"\n=== combined over {len(allrows)} runs ===", flush=True)
print(f"{'thr':>5} {'F1':>7} {'named_yield':>12} {'FA/min':>8}")
for th in THRESHOLDS:
    c = comb[str(th)]
    print(f"{th:>5} {c['f1']:>7.3f} {c['named_yield']:>12.3f}"
          f" {c['false_alarms_per_min']:>8.2f}"
          f"{'   <- RECOMMENDED' if str(th) == best else ''}", flush=True)
d, r = comb["2.5"], comb[best]
print(f"\nshipped 2.5 -> named_yield {d['named_yield']:.1%}, FA/min {d['false_alarms_per_min']:.2f}")
print(f"best {best:>7} -> named_yield {r['named_yield']:.1%}, FA/min {r['false_alarms_per_min']:.2f}")
print(f"gain: {r['named_yield'] - d['named_yield']:+.1%} of all real events, "
      f"at {r['false_alarms_per_min'] - d['false_alarms_per_min']:+.2f} false alarms/min")

json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "out_ear.json", "w"),
          indent=1)
