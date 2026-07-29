"""Scene -> watch() -> dream() -> performance. Does anything actually improve?

Every replay result before this was measured on pre-cut centred digits handed
straight to `perceive`. `UnifiedMind.watch` now lets the saccadic eye free-view
a real scene and lay down what it finds, so for the first time the question can
be asked of *lived* experience:

    after a night's replay of what the eye actually saw, is the mind better at
    anything?

Four faculties are probed before and after the same night, on the same mind:

  recall        a noisy held-out digit -> concept, through the pallium
                (scored with nn AND knn5, since the retriever study showed nn
                is the wrong read-out for a consolidated store)
  detection     free-view a NEW scene and name what the eye lands on --
                named yield, the closed-loop number
  rare          recall restricted to the classes the day barely contained
  world_model   does the transition model predict the next fixation better

Three arms, paired per seed:
  real        watch a scene, then dream it
  shuffled    same images, mislabelled day, then dream  (content control)
  no_sleep    watch, no dream                            (drift control)

A before/after difference on its own proves nothing -- probing, ordering and
the mind's own state could move it. The controls are what make the number mean
something.

Usage:  python3 benchmarks/scene_to_dream.py out_scene_dream.json
"""
import json, sys
import numpy as np

import neurobrain as nb
from neurobrain.sensing.streams import build_scene, SaccadicEye
from neurobrain.memory.development import SleepConsolidator

SEEDS = (0, 1, 2, 3, 4)
N_PALLIUM = 300
N_SCENES, N_SAC = 3, 60          # the day
NOISE = 0.55


def _unit(v):
    v = np.asarray(v, np.float32).ravel()
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


# ------------------------------------------------------------- the probes ---
def probe_recall(mind, tex, teY, seed, k=5):
    """Noisy image -> concept, via the pallium. nn and knn5."""
    cx = mind.space.cortex
    M, labels = cx.modalities["image"], np.array([str(l) for l in cx.labels])
    rng = np.random.default_rng(seed)
    nn_ok = knn_ok = 0
    for im, y in zip(tex, teY):
        v = im.reshape(-1).astype(np.float32) / 255.0
        c = _unit(v + rng.normal(0, NOISE, v.shape).astype(np.float32))
        s = M @ c
        nn_ok += int(labels[int(np.argmax(s))] == str(int(y)))
        top = np.argpartition(-s, min(k, len(s) - 1))[:k]
        vals, cnt = np.unique(labels[top], return_counts=True)
        knn_ok += int(vals[int(np.argmax(cnt))] == str(int(y)))
    n = max(len(teY), 1)
    return nn_ok / n, knn_ok / n


def probe_detection(mind, trx, trY, seed):
    """Free-view an UNSEEN scene; how much of what is really there is named?
    remember=False so the probe is not itself experience."""
    scene = build_scene(trx, trY, size=256, n_objects=12, seed=seed + 500)
    fix = SaccadicEye(scene, seed=seed + 500).free_view(n_saccades=45,
                                                        correct=True)
    on = [f for f in fix if f.true_label >= 0]
    if not on:
        return 0.0, 0.0
    ok = 0
    for f in on:
        img = np.mean(f.frames, axis=0)
        ok += int(mind.perceive(np.asarray(img, np.float32),
                                remember=False) == str(int(f.true_label)))
    return ok / len(on), len(on) / max(len(fix), 1)


def probe_rare(mind, tex, teY, rare, seed):
    m = np.array([int(y) in rare for y in teY])
    if not m.any():
        return 0.0
    return probe_recall(mind, tex[m], teY[m], seed)[1]      # knn5


def probe_world(mind, seq):
    """Does the world model predict the next concept of a lived sequence?"""
    if len(seq) < 2:
        return 0.0
    ok = 0
    for a, b in zip(seq[:-1], seq[1:]):
        try:
            ok += int(mind.space.predict_next(a) == b)
        except Exception:
            pass
    return ok / max(len(seq) - 1, 1)


def probe_all(mind, trx, trY, tex, teY, rare, seq, seed):
    nn, knn = probe_recall(mind, tex, teY, seed)
    det, onrate = probe_detection(mind, trx, trY, seed)
    return dict(recall_nn=nn, recall_knn5=knn, detection=det,
                on_object_rate=onrate, rare=probe_rare(mind, tex, teY, rare, seed),
                world_model=probe_world(mind, seq))


# --------------------------------------------------------------- one seed ---
def run_seed(sd, trx, trY, tex, teY):
    out = {}
    for arm in ("real", "shuffled", "no_sleep"):
        mind = nb.build_unified_mind(n_pallium=N_PALLIUM)
        seq = []
        for s in range(N_SCENES):
            scene = build_scene(trx, trY, size=256, n_objects=12,
                                seed=sd * 10 + s)
            if arm == "shuffled":
                # same looking, but the day is mislabelled as it is laid down
                eye = SaccadicEye(scene, seed=sd * 10 + s)
                fix = eye.free_view(n_saccades=N_SAC, correct=True)
                rng = np.random.default_rng(sd + 3)
                for f in fix:
                    img = np.asarray(np.mean(f.frames, axis=0), np.float32)
                    mind._record_episode(img, str(int(rng.integers(10))))
                    seq.append(str(int(f.true_label)) if f.true_label >= 0 else "-1")
            else:
                fix = mind.watch(scene, n_saccades=N_SAC, seed=sd * 10 + s)
                seq += [str(int(f.true_label)) for f in fix if f.true_label >= 0]

        counts = np.array([seq.count(str(d)) for d in range(10)])
        rare = set(np.argsort(counts)[:5].tolist())

        before = probe_all(mind, trx, trY, tex, teY, rare, seq, sd)
        replays = 0
        if arm != "no_sleep":
            replays = mind.dream(cycles=3, replays_per_cycle=300)["replays"]
        after = probe_all(mind, trx, trY, tex, teY, rare, seq, sd)
        out[arm] = dict(before=before, after=after, replays=int(replays),
                        episodes=len(mind.episodes),
                        cells=mind.space.cortex.n_memories)
    return out


def main():
    trx, trY, tex, teY = nb.load_mnist(n_train=4000, n_test=400)
    tex, teY = tex[:300], teY[:300]
    per_seed = []
    for sd in SEEDS:
        per_seed.append(run_seed(sd, trx, trY, tex, teY))
        r = per_seed[-1]["real"]
        print(f"seed {sd}: episodes={r['episodes']} replays={r['replays']} "
              f"knn5 {r['before']['recall_knn5']:.3f}->{r['after']['recall_knn5']:.3f} "
              f"det {r['before']['detection']:.3f}->{r['after']['detection']:.3f}",
              flush=True)

    METRICS = ("recall_nn", "recall_knn5", "detection", "rare", "world_model")
    out = {"per_seed": per_seed, "deltas": {}, "paired_vs_nosleep": {}}
    print(f"\n{'metric':<14}{'arm':<10}{'before':>9}{'after':>9}{'delta':>10}"
          f"{'wins':>7}{'d':>8}")
    for m in METRICS:
        out["deltas"][m] = {}
        for arm in ("real", "shuffled", "no_sleep"):
            b = np.array([s[arm]["before"][m] for s in per_seed])
            a = np.array([s[arm]["after"][m] for s in per_seed])
            d = a - b
            out["deltas"][m][arm] = dict(
                before=round(float(b.mean()), 4), after=round(float(a.mean()), 4),
                delta=round(float(d.mean()), 4), sd=round(float(d.std(ddof=1)), 4),
                wins=int((d > 0).sum()), n=len(d),
                cohens_d=round(float(d.mean() / (d.std(ddof=1) + 1e-12)), 3))
            r = out["deltas"][m][arm]
            print(f"{m if arm=='real' else '':<14}{arm:<10}{r['before']:>9.3f}"
                  f"{r['after']:>9.3f}{r['delta']:>+10.4f}"
                  f"{r['wins']:>4}/{r['n']}{r['cohens_d']:>8.2f}")
        # the number that matters: real dream vs no dream, on the same measure
        x = np.array([s["real"]["after"][m] for s in per_seed])
        y = np.array([s["no_sleep"]["after"][m] for s in per_seed])
        d = x - y
        out["paired_vs_nosleep"][m] = dict(
            mean=round(float(d.mean()), 4), sd=round(float(d.std(ddof=1)), 4),
            wins=int((d > 0).sum()), n=len(d),
            cohens_d=round(float(d.mean() / (d.std(ddof=1) + 1e-12)), 3))

    print(f"\n=== real dream vs NO dream, after ({len(SEEDS)} seeds, paired) ===")
    for m in METRICS:
        p = out["paired_vs_nosleep"][m]
        verdict = ("IMPROVES" if p["cohens_d"] > 0.8 and p["wins"] >= 4
                   else "no effect" if abs(p["cohens_d"]) < 0.8 else "HURTS")
        print(f"  {m:<14}{p['mean']:>+9.4f} ±{p['sd']:.4f}  "
              f"wins {p['wins']}/{p['n']}  d={p['cohens_d']:>6.2f}   {verdict}")

    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1
                        else "out_scene_dream.json", "w"), indent=1)


if __name__ == "__main__":
    main()
