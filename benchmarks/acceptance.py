"""The acceptance gate: does replay change BEHAVIOUR, not just memory?

Three criteria, all three required:

  1. replay produces a meaningful change in DETECTION
  2. replay produces a meaningful change in the WORLD MODEL
  3. it happens WITHOUT a drop in RECALL

Only if all three pass may the claim be made that replay no longer changes
memory alone but changes what the system does.

"Meaningful" is fixed in advance so the gate cannot be argued into passing:
a criterion passes when the paired effect against the no-dream arm has
|Cohen's d| >= 0.8 in the right direction AND wins on at least 4 of 5 seeds.
Criterion 3 passes when recall does NOT drop by that standard -- a real gain
is not required, only the absence of a real loss.

Two arms are compared, each dreaming the same lived day:
  memory_only   dream(to_perception=False, to_world=False)   -- the old night
  all_stores    dream(to_perception=True,  to_world=True)    -- the new night
against `no_sleep`, which watches and does not dream.

Usage:  python3 benchmarks/acceptance.py out_acceptance.json
"""
import json, sys
import numpy as np

import neurobrain as nb
from neurobrain.sensing.streams import build_scene, SaccadicEye

SEEDS = (0, 1, 2, 3, 4)
N_PALLIUM = 300
N_SCENES, N_SAC = 3, 60
NOISE = 0.55
D_THRESHOLD, WIN_THRESHOLD = 0.8, 4


def _unit(v):
    v = np.asarray(v, np.float32).ravel()
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def probe_recall(mind, tex, teY, seed, k=5):
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
    """Free-view an UNSEEN scene and name what the eye lands on."""
    scene = build_scene(trx, trY, size=256, n_objects=12, seed=seed + 500)
    fix = SaccadicEye(scene, seed=seed + 500).free_view(n_saccades=45, correct=True)
    on = [f for f in fix if f.true_label >= 0]
    if not on:
        return 0.0
    ok = 0
    for f in on:
        img = np.asarray(np.mean(f.frames, axis=0), np.float32)
        ok += int(mind.perceive(img, remember=False) == str(int(f.true_label)))
    return ok / len(on)


def probe_world(mind, seq):
    if len(seq) < 2:
        return 0.0
    ok = sum(int(mind.space.predict_next(a) == b)
             for a, b in zip(seq[:-1], seq[1:]))
    return ok / (len(seq) - 1)


def probe(mind, trx, trY, tex, teY, seq, seed):
    nn, knn = probe_recall(mind, tex, teY, seed)
    return dict(detection=probe_detection(mind, trx, trY, seed),
                world_model=probe_world(mind, seq),
                recall_nn=nn, recall_knn5=knn)


def run_seed(sd, trx, trY, tex, teY):
    out = {}
    for arm in ("no_sleep", "memory_only", "all_stores"):
        mind = nb.build_unified_mind(n_pallium=N_PALLIUM)
        seq = []
        for s in range(N_SCENES):
            scene = build_scene(trx, trY, size=256, n_objects=12, seed=sd * 10 + s)
            fix = mind.watch(scene, n_saccades=N_SAC, seed=sd * 10 + s)
            seq += [str(int(f.true_label)) for f in fix if f.true_label >= 0]
        before = probe(mind, trx, trY, tex, teY, seq, sd)
        if arm == "memory_only":
            mind.dream(cycles=3, replays_per_cycle=300,
                       to_perception=False, to_world=False)
        elif arm == "all_stores":
            mind.dream(cycles=3, replays_per_cycle=300,
                       to_perception=True, to_world=True)
        out[arm] = dict(before=before,
                        after=probe(mind, trx, trY, tex, teY, seq, sd))
    return out


def paired(per_seed, arm, metric):
    x = np.array([s[arm]["after"][metric] for s in per_seed])
    y = np.array([s["no_sleep"]["after"][metric] for s in per_seed])
    d = x - y
    sd = float(d.std(ddof=1))
    return dict(mean=round(float(d.mean()), 4), sd=round(sd, 4),
                wins=int((d > 0).sum()), losses=int((d < 0).sum()), n=len(d),
                cohens_d=round(float(d.mean() / (sd + 1e-12)), 3))


def main():
    trx, trY, tex, teY = nb.load_mnist(n_train=4000, n_test=400)
    tex, teY = tex[:300], teY[:300]
    per_seed = []
    for sd in SEEDS:
        per_seed.append(run_seed(sd, trx, trY, tex, teY))
        a = per_seed[-1]["all_stores"]["after"]
        n = per_seed[-1]["no_sleep"]["after"]
        print(f"seed {sd}: det {n['detection']:.3f}->{a['detection']:.3f}  "
              f"world {n['world_model']:.3f}->{a['world_model']:.3f}  "
              f"knn5 {n['recall_knn5']:.3f}->{a['recall_knn5']:.3f}", flush=True)

    METRICS = ("detection", "world_model", "recall_nn", "recall_knn5")
    out = {"per_seed": per_seed, "vs_no_sleep": {}}
    for arm in ("memory_only", "all_stores"):
        out["vs_no_sleep"][arm] = {m: paired(per_seed, arm, m) for m in METRICS}

    print(f"\n{'':<14}{'memory-only night':>26}{'all-stores night':>26}")
    print(f"{'metric':<14}{'delta   wins    d':>26}{'delta   wins    d':>26}")
    for m in METRICS:
        a, b = out["vs_no_sleep"]["memory_only"][m], out["vs_no_sleep"]["all_stores"][m]
        print(f"{m:<14}{a['mean']:>+10.4f}{a['wins']:>4}/{a['n']}{a['cohens_d']:>9.2f}"
              f"{b['mean']:>+10.4f}{b['wins']:>4}/{b['n']}{b['cohens_d']:>9.2f}")

    r = out["vs_no_sleep"]["all_stores"]
    def gained(m):
        p = r[m]
        return bool(p["cohens_d"] >= D_THRESHOLD and p["wins"] >= WIN_THRESHOLD)

    def not_lost(m):
        p = r[m]
        return not (p["cohens_d"] <= -D_THRESHOLD and p["losses"] >= WIN_THRESHOLD)

    c1 = gained("detection")
    c2 = gained("world_model")
    c3 = not_lost("recall_nn") and not_lost("recall_knn5")
    out["criteria"] = {
        "1_detection_changes": c1,
        "2_world_model_changes": c2,
        "3_no_recall_drop": c3,
        "ALL_PASS": bool(c1 and c2 and c3),
        "rule": f"|d| >= {D_THRESHOLD} and >= {WIN_THRESHOLD}/{len(SEEDS)} seeds",
    }
    print(f"\n=== ACCEPTANCE (all-stores night vs no dream) ===")
    print(f"  rule: {out['criteria']['rule']}")
    for k, label in (("1_detection_changes", "1. detection changes    "),
                     ("2_world_model_changes", "2. world model changes  "),
                     ("3_no_recall_drop", "3. recall does not drop ")):
        print(f"  {label} {'PASS' if out['criteria'][k] else 'FAIL'}")
    print(f"\n  ALL THREE: {'PASS' if out['criteria']['ALL_PASS'] else 'FAIL'}")
    if out["criteria"]["ALL_PASS"]:
        print("  -> replay no longer changes memory alone; it changes behaviour.")

    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1
                        else "out_acceptance.json", "w"), indent=1)


if __name__ == "__main__":
    main()
