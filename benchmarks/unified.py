"""The unified mind: does it perceive, and does its imagination actually go anywhere?"""
import json, time, sys, traceback
from collections import Counter
import numpy as np
import neurobrain as nb

out = {}


def build():
    t0 = time.time()
    mind = nb.build_unified_mind()
    t = time.time() - t0
    return mind, dict(
        sec_build=round(t, 2),
        perceive_accuracy=round(float(mind.perceive_accuracy), 4),
        recall_accuracy=round(float(mind.recall_accuracy), 4),
        comprehend_accuracy=round(float(mind.comprehend_accuracy), 4),
        analogy_accuracy=round(float(mind.analogy_accuracy), 4),
        attention_benefit=round(float(mind.attention_benefit), 4),
        causal_accuracy=round(float(mind.causal_accuracy), 4),
        self_accuracy=round(float(mind.self_accuracy), 4),
        n_concepts=int(mind.ws.n_concepts) if mind.ws is not None else None,
        code_separation=round(float(mind.ws.separation()), 4) if mind.ws is not None else None,
    )


def imagination(mind, steps=40, n_seeds=8, temps=(0.3, 0.6, 1.0)):
    """Is the train of thought a wanderer or a stuck record?

    Measured: how many distinct concepts a chain visits, how often it repeats
    the previous one, and how long before it revisits (cycle length)."""
    space = mind.space
    concepts = list(space.concepts)
    rows = []
    for temp in temps:
        for i, seed in enumerate(concepts[:n_seeds]):
            names, percepts = space.imagine(seed, steps=steps, temperature=temp,
                                            rng_seed=i)
            uniq = len(set(names))
            self_loops = sum(1 for a, b in zip(names, names[1:]) if a == b)
            # first revisit distance = how quickly the chain closes a loop
            seen, cycle = {}, None
            for j, n in enumerate(names):
                if n in seen and cycle is None:
                    cycle = j - seen[n]
                seen.setdefault(n, j)
            counts = np.array(list(Counter(names).values()), float)
            p = counts / counts.sum()
            ent = float(-(p * np.log2(p)).sum())
            rows.append(dict(temperature=temp, seed=seed, steps=steps,
                             unique=uniq, unique_frac=round(uniq / len(names), 4),
                             self_loop_rate=round(self_loops / max(len(names) - 1, 1), 4),
                             first_cycle=cycle, entropy_bits=round(ent, 3),
                             max_entropy_bits=round(float(np.log2(len(concepts))), 3)))
    agg = {}
    for temp in temps:
        rs = [r for r in rows if r["temperature"] == temp]
        agg[str(temp)] = dict(
            mean_unique=round(float(np.mean([r["unique"] for r in rs])), 2),
            mean_unique_frac=round(float(np.mean([r["unique_frac"] for r in rs])), 4),
            mean_self_loop=round(float(np.mean([r["self_loop_rate"] for r in rs])), 4),
            mean_entropy=round(float(np.mean([r["entropy_bits"] for r in rs])), 3),
            mean_first_cycle=round(float(np.mean([r["first_cycle"] for r in rs
                                                  if r["first_cycle"]])), 2),
        )
    return dict(n_concepts_total=len(concepts), runs=rows, by_temperature=agg,
                max_entropy_bits=round(float(np.log2(len(concepts))), 3))


def closed_imagination_loop(mind, steps=25, n_seeds=8):
    """The real test: imagine a percept, then feed it BACK into perception.

    If the mind cannot recognise what it just imagined, imagination is
    decoration -- it is not producing anything the rest of the brain can use."""
    space = mind.space
    concepts = list(space.concepts)
    ok = tot = 0
    per_seed = []
    for i, seed in enumerate(concepts[:n_seeds]):
        names, percepts = space.imagine(seed, steps=steps, temperature=0.6, rng_seed=i)
        hit = n = 0
        for nm, pc in zip(names, percepts):
            if pc is None:
                continue
            img = np.asarray(pc, np.float32)
            try:
                back = mind.perceive(img.reshape(28, 28) if img.size == 784 else img)
            except Exception:
                continue
            n += 1
            hit += int(str(back) == str(nm))
        ok += hit
        tot += n
        per_seed.append(dict(seed=seed, n=n, recognised=hit,
                             rate=round(hit / max(n, 1), 4)))
    return dict(total=tot, recognised=ok,
                round_trip_accuracy=round(ok / max(tot, 1), 4),
                chance=round(1 / max(len(concepts), 1), 4), per_seed=per_seed)


def dreaming(mind):
    """Does a night's sleep help, and what does consolidation actually merge?"""
    before_concepts = int(mind.ws.n_concepts)
    before_sep = float(mind.ws.separation())
    t0 = time.time()
    d = mind.dream(cycles=3, replays_per_cycle=300)
    t = time.time() - t0
    return dict(sec_dream=round(t, 2), replays=int(d["replays"]),
                concepts_merged=int(d["concepts_merged"]),
                concepts_before=before_concepts,
                concepts_after=int(mind.ws.n_concepts),
                separation_before=round(before_sep, 4),
                separation_after=round(float(mind.ws.separation()), 4))


try:
    mind, out["build"] = build()
    print("build", json.dumps(out["build"]), flush=True)
    for name, fn in [("imagination", lambda: imagination(mind)),
                     ("closed_loop", lambda: closed_imagination_loop(mind)),
                     ("dream", lambda: dreaming(mind))]:
        try:
            out[name] = fn()
            print(name, "ok", flush=True)
        except Exception as e:
            out[name] = dict(ERROR=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-800:])
            print("FAIL", name, out[name]["ERROR"], flush=True)
        json.dump(out, open(sys.argv[1], "w"), indent=1)
    # imagination again AFTER the dream -- did consolidation change the wander?
    try:
        out["imagination_after_dream"] = imagination(mind)
    except Exception as e:
        out["imagination_after_dream"] = dict(ERROR=str(e))
except Exception as e:
    out["FATAL"] = dict(ERROR=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-1200:])
    print("FATAL", out["FATAL"]["ERROR"], flush=True)

json.dump(out, open(sys.argv[1], "w"), indent=1)
print("DONE", flush=True)
