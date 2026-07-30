"""Under what conditions does replay improve PERCEPTUAL representations?

Replay improves latent sequential knowledge -- the world model gained +14.6
points, 5/5 seeds. It does not improve perception on a familiar day, and a
ground-truth teacher only brings that back to parity. The standing hypothesis
for why: `vision.cortex` was already grown on 15,000 clean MNIST digits with
this same ART rule, so a 180-fixation day of the *same* material has nothing to
add and a perfect teacher has nothing to teach.

That predicts the condition under which replay *should* help: a day containing
something the original training did not.

The day is varied along exactly that axis, holding everything else fixed. The
mind is always MNIST-trained; the scenes it lives are built from:

  familiar   MNIST            -- what it was trained on
  mixed      half and half    -- a partial shift
  novel      Fashion-MNIST    -- classes it has never seen; its category cells
                                 are named for digits, so its beliefs about a
                                 shoe are close to meaningless

Detection is probed on held-out scenes from the SAME source as the day, so the
question is always "did the night help it see the world it is actually in".

Three arms per condition: no_dream, self-labelled replay, teacher-labelled
replay. If the hypothesis is right, the familiar column stays flat and the
novel column separates -- with teacher above self, because on unfamiliar
material the mind's own beliefs carry almost no information to reinforce.

Usage:  python3 benchmarks/novel_day.py out_novel_day.json
"""
import json, sys
import numpy as np

import neurobrain as nb
from neurobrain.sensing.streams import build_scene, SaccadicEye

SEEDS = (0, 1, 2, 3, 4, 5)
N_PALLIUM, N_SCENES, N_SAC = 300, 3, 60
PROBE_SCENES = 5
PERCEPTION_LR = 5.0            # a rate the sweep showed is actually active
ARMS = ("no_dream", "self", "teacher")
CONDITIONS = ("familiar", "mixed", "novel")


def sources():
    mx, my, _, _ = nb.load_mnist(n_train=4000, n_test=300)
    fx, fy, _, _ = nb.load_fashion_mnist(n_train=4000, n_test=300)
    rng = np.random.default_rng(0)
    half = rng.permutation(len(mx))[:len(mx) // 2]
    half2 = rng.permutation(len(fx))[:len(fx) // 2]
    mix_x = np.concatenate([mx[half], fx[half2]])
    mix_y = np.concatenate([my[half], fy[half2]])
    p = rng.permutation(len(mix_x))
    return {"familiar": (mx, my), "mixed": (mix_x[p], mix_y[p]),
            "novel": (fx, fy)}


def detection(mind, X, Y, seed):
    """Free-view held-out scenes from the day's own world and name what is hit."""
    ok = n = 0
    for k in range(PROBE_SCENES):
        sc = build_scene(X, Y, size=256, n_objects=12, seed=seed + 900 + k * 41)
        fix = SaccadicEye(sc, seed=seed + 900 + k * 41).free_view(
            n_saccades=45, correct=True)
        for f in fix:
            if f.true_label < 0:
                continue
            n += 1
            ok += int(mind.perceive(np.asarray(np.mean(f.frames, 0), np.float32),
                                    remember=False) == str(int(f.true_label)))
    return ok / max(n, 1), n


def run(cond, arm, sd, X, Y):
    mind = nb.build_unified_mind(n_pallium=N_PALLIUM)
    for s in range(N_SCENES):
        n0 = len(mind.episodes)
        fix = mind.watch(build_scene(X, Y, size=256, n_objects=12,
                                     seed=sd * 10 + s),
                         n_saccades=N_SAC, seed=sd * 10 + s)
        for ep, f in zip(mind.episodes.episodes[n0:], fix):
            ep.truth = str(int(f.true_label)) if f.true_label >= 0 else None
    if arm == "self":
        mind.dream(cycles=3, replays_per_cycle=300,
                   perception_lr=PERCEPTION_LR, relabel=lambda ep: ep.label)
    elif arm == "teacher":
        mind.dream(cycles=3, replays_per_cycle=300,
                   perception_lr=PERCEPTION_LR,
                   relabel=lambda ep: getattr(ep, "truth", None))
    acc, n = detection(mind, X, Y, sd)
    return acc, n


def main():
    src = sources()
    res = {c: {a: [] for a in ARMS} for c in CONDITIONS}
    probe_n = None
    for sd in SEEDS:
        for cond in CONDITIONS:
            X, Y = src[cond]
            for arm in ARMS:
                acc, probe_n = run(cond, arm, sd, X, Y)
                res[cond][arm].append(acc)
        print(f"seed {sd}: " + "  ".join(
            f"{c}[{res[c]['no_dream'][-1]:.3f}/{res[c]['self'][-1]:.3f}/"
            f"{res[c]['teacher'][-1]:.3f}]" for c in CONDITIONS), flush=True)

    out = {"probe_fixations": probe_n, "by_condition": {}, "raw": res}
    print(f"\nprobe = {probe_n} fixations (resolution {1/max(probe_n,1):.4f})")
    print(f"\n{'condition':<11}{'no_dream':>10}{'self':>10}{'teacher':>10}"
          f"{'self-nd':>10}{'teach-nd':>11}{'wins':>7}{'d':>8}")
    for cond in CONDITIONS:
        base = np.array(res[cond]["no_dream"])
        row = {"no_dream": round(float(base.mean()), 4)}
        for arm in ("self", "teacher"):
            a = np.array(res[cond][arm])
            d = a - base
            s = float(d.std(ddof=1))
            row[arm] = dict(after=round(float(a.mean()), 4),
                            delta=round(float(d.mean()), 4), sd=round(s, 4),
                            wins=int((d > 0).sum()), n=len(d),
                            cohens_d=round(float(d.mean() / (s + 1e-12)), 3))
        out["by_condition"][cond] = row
        t = row["teacher"]
        print(f"{cond:<11}{row['no_dream']:>10.4f}{row['self']['after']:>10.4f}"
              f"{t['after']:>10.4f}{row['self']['delta']:>+10.4f}"
              f"{t['delta']:>+11.4f}{t['wins']:>4}/{t['n']}{t['cohens_d']:>8.2f}")

    print("\n=== under what conditions does replay improve perception? ===")
    for cond in CONDITIONS:
        t = out["by_condition"][cond]["teacher"]
        s = out["by_condition"][cond]["self"]
        v = ("IMPROVES" if t["cohens_d"] >= 0.8 and t["wins"] >= 4
             else "HURTS" if t["cohens_d"] <= -0.8 else "no effect")
        print(f"  {cond:<9} teacher {t['delta']:+.4f} (d={t['cohens_d']:+.2f}) "
              f"{v:<10} | self {s['delta']:+.4f} (d={s['cohens_d']:+.2f})")

    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1
                        else "out_novel_day.json", "w"), indent=1)


if __name__ == "__main__":
    main()
