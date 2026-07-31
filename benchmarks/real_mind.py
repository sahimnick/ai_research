"""The assembled mind, on photographs. The gap that had never been crossed.

Ten benchmarks in this project use real CIFAR-10 photographs and ESC-50
recordings. Eleven use the assembled `UnifiedMind` -- episodes, `watch`,
`dream`, the transition model `_T`, the acceptance gate. The overlap was
**zero**. Every conclusion about replay, consolidation, the world model and
imagination was drawn on hand-written digits, and every conclusion about real
sensation was drawn without any of that machinery attached.

It was not an oversight so much as a wall: `build_unified_mind` loaded MNIST
inside itself, seeded a counting curriculum 0->1->...->9, hard-coded ten digit
cues, and `perceive` reshaped its input to `(1, 28, 28)`. Those are facts about
digits, not about minds. `build_mind_on` takes the data as an argument and drops
the two that are digit-specific -- the counting prior, because "after 3 comes 4"
is meaningless between a cat and an airplane, and the fixed ten cues.

This runs the **same acceptance gate** as `acceptance.py`, arm for arm and probe
for probe, on scenes of photographs instead of scenes of digits, so the two
tables can be read against each other:

    no_sleep      watch the day, then stop
    memory_only   dream into the memory store alone -- the night as originally
                  built, which measured exactly +0.0000 on detection and world
                  model because those read stores it never touched
    all_stores    dream into perception and the transition model as well

    detection     free-view an unseen scene and name what the eye lands on
    world_model   predict the next percept from the previous one, over the
                  order the eye actually chose
    recall_nn     nearest exemplar in the pallium
    recall_knn5   and by five, since a store holding more exemplars is
                  structurally favoured by 1-NN

The question is not whether the numbers are as high as on digits -- they cannot
be, the eye names photographs at 0.134 against 0.559 for digits. It is whether
the **mechanisms behave the same way**: does the day still populate the world
model, does replay still move it, does anything that worked on MNIST turn out to
have depended on MNIST.

Usage:  python3 benchmarks/real_mind.py out_real_mind.json
"""
import json
import sys

import numpy as np

import neurobrain as nb
from neurobrain.minds.unified import build_mind_on
from neurobrain.sensing.natural import load_cifar10
from neurobrain.sensing.streams import SaccadicEye, build_scene

SEEDS = (0, 1, 2, 3)
N_PALLIUM, N_SCENES, N_SAC = 800, 3, 60
N_TRAIN, N_TEST = 4000, 500
ARMS = ("no_sleep", "memory_only", "all_stores")
D_THRESHOLD, WIN_THRESHOLD = 0.8, 3


def luminance(X):
    """Scenes are one 2-D canvas, so colour is dropped -- stated, not hidden.

    Colour is worth about +0.04 to this eye on whole photographs, so every
    number below carries that handicap on top of everything else."""
    return X.mean(1).astype(np.uint8) if X.ndim == 4 else X


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def probe_recall(mind, tex, teY, names, seed, k=5, noise=0.6):
    """Recall a noised photograph from the pallium -- the same probe as
    `acceptance.py`, reading `cortex.modalities` exactly as it does, so the
    digit and photograph tables are measuring the same thing."""
    cx = mind.space.cortex
    M = cx.modalities["image"]
    labels = np.array([str(l) for l in cx.labels])
    rng = np.random.default_rng(seed)
    nn_ok = knn_ok = 0
    for im, y in zip(tex, teY):
        v = im.reshape(-1).astype(np.float32) / 255.0
        c = _unit(v + rng.normal(0, noise, v.shape).astype(np.float32))
        s = M @ c
        nn_ok += int(labels[int(np.argmax(s))] == names[int(y)])
        top = np.argpartition(-s, min(k, len(s) - 1))[:k]
        vals, cnt = np.unique(labels[top], return_counts=True)
        knn_ok += int(vals[int(np.argmax(cnt))] == names[int(y)])
    n = max(len(teY), 1)
    return nn_ok / n, knn_ok / n


def probe_detection(mind, trx, trY, names, seed):
    scene = build_scene(trx, trY, size=256, n_objects=12, seed=seed + 500)
    fix = SaccadicEye(scene, seed=seed + 500).free_view(n_saccades=45,
                                                        correct=True)
    on = [f for f in fix if f.true_label >= 0]
    if not on:
        return 0.0
    ok = 0
    for f in on:
        img = np.asarray(np.mean(f.frames, axis=0), np.float32)
        ok += int(mind.perceive(img, remember=False) == names[int(f.true_label)])
    return ok / len(on)


def _sharpness(T):
    """How far the transition model is from its uniform prior.

    Not `count_nonzero`: `_grow_T` initialises every entry to 0.05, so the
    nonzero count is n^2 before any experience and cannot show learning. The
    first version of this benchmark reported "100 of 100 entries" before and
    after a day and read it as saturation. Mean row max, normalised, actually
    moves when a row acquires structure."""
    if T is None or not T.size:
        return 0.0
    R = T / np.maximum(T.sum(1, keepdims=True), 1e-9)
    return float(R.max(1).mean())


def probe_world(mind, seq):
    if len(seq) < 2:
        return 0.0
    ok = sum(int(mind.space.predict_next(a) == b)
             for a, b in zip(seq[:-1], seq[1:]))
    return ok / (len(seq) - 1)


def probe(mind, trx, trY, tex, teY, names, seq, seed):
    nn, knn = probe_recall(mind, tex, teY, names, seed)
    return dict(detection=probe_detection(mind, trx, trY, names, seed),
                world_model=probe_world(mind, seq),
                recall_nn=nn, recall_knn5=knn)


def run_seed(sd, trx, trY, tex, teY, names):
    out = {}
    for arm in ARMS:
        mind = build_mind_on(trx, trY, tex, teY, names=names,
                             n_pallium=N_PALLIUM, seed=sd)
        seq = []
        for s in range(N_SCENES):
            scene = build_scene(trx, trY, size=256, n_objects=12,
                                seed=sd * 10 + s)
            fix = mind.watch(scene, n_saccades=N_SAC, seed=sd * 10 + s)
            seq += [names[int(f.true_label)] for f in fix if f.true_label >= 0]
        before = probe(mind, trx, trY, tex, teY, names, seq, sd)
        if arm == "memory_only":
            mind.dream(cycles=3, replays_per_cycle=300,
                       to_perception=False, to_world=False)
        elif arm == "all_stores":
            mind.dream(cycles=3, replays_per_cycle=300,
                       to_perception=True, to_world=True)
        out[arm] = dict(before=before,
                        after=probe(mind, trx, trY, tex, teY, names, seq, sd),
                        episodes=len(mind.episodes),
                        surprise_values=int(len({round(float(
                            getattr(e, "surprise", 0.0)), 4)
                            for e in mind.episodes.episodes})),
                        world_sharpness=_sharpness(mind.space._T))
    return out


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_real_mind.json"
    cx, cy, cxt, cyt = load_cifar10(n_train=N_TRAIN, n_test=N_TEST,
                                    grayscale=False, size=28)
    trx, tex = luminance(cx), luminance(cxt)
    names = list(nb.sensing.natural.CIFAR_CLASSES) if hasattr(
        nb, "sensing") else None
    from neurobrain.sensing.natural import CIFAR_CLASSES
    names = list(CIFAR_CLASSES)
    print(f"the assembled mind on {len(trx)} CIFAR-10 photographs, "
          f"{len(names)} classes, chance {1/len(names):.3f}\n", flush=True)

    per_seed = []
    for sd in SEEDS:
        r = run_seed(sd, trx, cy, tex, cyt, names)
        per_seed.append(r)
        n = r["no_sleep"]
        print(f"seed {sd}: episodes {n['episodes']}  world sharpness "
              f"{n['world_sharpness']:.3f}  surprise values {n['surprise_values']}  "
              f"| detection {n['before']['detection']:.3f}  world "
              f"{n['before']['world_model']:.3f}  recall "
              f"{n['before']['recall_nn']:.3f}", flush=True)

    res = {"seeds": list(SEEDS), "n_class": len(names), "per_seed": per_seed}
    METRICS = ("detection", "world_model", "recall_nn", "recall_knn5")

    print(f"\n{'arm':<14}" + "".join(f"{m:>14}" for m in METRICS))
    for arm in ARMS:
        row = [float(np.mean([s[arm]["after"][m] for s in per_seed]))
               for m in METRICS]
        res.setdefault("after", {})[arm] = {m: round(v, 4)
                                            for m, v in zip(METRICS, row)}
        print(f"{arm:<14}" + "".join(f"{v:>14.3f}" for v in row))

    print(f"\nchange from the night (all_stores minus no_sleep)")
    print(f"{'metric':<14}{'delta':>10}{'sd':>8}{'d':>8}{'wins':>7}")
    for m in METRICS:
        a = np.array([s["all_stores"]["after"][m] for s in per_seed])
        b = np.array([s["no_sleep"]["after"][m] for s in per_seed])
        d = a - b
        sd = float(d.std(ddof=1))
        cd = float(d.mean() / (sd + 1e-12))
        res.setdefault("gate", {})[m] = dict(
            delta=round(float(d.mean()), 4), sd=round(sd, 4),
            cohens_d=round(cd, 3), wins=int((d > 0).sum()),
            losses=int((d < 0).sum()), n=len(d))
        print(f"{m:<14}{d.mean():>+10.4f}{sd:>8.4f}{cd:>8.2f}"
              f"{int((d > 0).sum()):>4}/{len(d)}")

    print("\n=== does the machinery behave the same way on photographs? ===")
    ep = float(np.mean([s["no_sleep"]["episodes"] for s in per_seed]))
    we = float(np.mean([s["no_sleep"]["world_sharpness"] for s in per_seed]))
    sv = float(np.mean([s["no_sleep"]["surprise_values"] for s in per_seed]))
    print(f"  the day lands: {ep:.0f} episodes, world-model sharpness "
          f"{we:.3f} (uniform would be {1/len(names):.3f}), "
          f"{sv:.0f} distinct surprise values")
    if ep < 1:
        print("  -> the episodic buffer is EMPTY; nothing below means anything")
    g = res["gate"]
    for m in METRICS:
        p = g[m]
        v = ("gains" if p["cohens_d"] >= D_THRESHOLD
             and p["wins"] >= WIN_THRESHOLD
             else "loses" if p["cohens_d"] <= -D_THRESHOLD
             and p["losses"] >= WIN_THRESHOLD else "no effect")
        note = ""
        if abs(p["delta"]) < 1e-9:
            note = "   (EXACTLY zero -- a disconnection, not a weak effect)"
        print(f"  {m:<14}{p['delta']:>+9.4f}  {v}{note}")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
