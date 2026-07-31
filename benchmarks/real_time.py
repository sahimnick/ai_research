"""A world model over real sound, where perception is not the bottleneck.

`real_mind.py` found that section 7's world-model result -- +14.6 points from
replaying the day's order -- **does not transfer to photographs**, and located
the reason upstream rather than in the replay machinery: the world model is
built out of percepts, and the eye names photographs at 0.129 against 0.559 for
digits. Replaying a sequence of mostly-wrong percepts cannot sharpen anything.

That explanation makes a sharp prediction, and this is the test of it. Hearing
in this project **works**: 1-NN 0.914 over six real ESC-50 categories, 17.5x
chance on the 50-way task. So if perception quality is really what gates the
world model, the same machinery on the same day-structure should recover when
the percepts come from the ear instead of the eye.

If it does not, the "perception is upstream" story is wrong and the fault is in
the transition model itself.

The world, and what is real about it
------------------------------------
The **sounds are real** -- ESC-50 field recordings, unmodified. The **order is
designed**, and it has to be said plainly: ESC-50 clips are independent
recordings with no natural succession between them, so a world model needs a
world, and there is none in the corpus. A street is imposed:

    car_horn  -> engine, siren        a horn, then traffic moving or a response
    engine    -> engine, train        traffic continues, or a train passes
    siren     -> engine, siren        the emergency goes by
    train     -> engine, train        rolling stock
    ...

The claim being tested is not "the mind discovers real-world statistics" -- it
is "given percepts of a certain quality, does replay sharpen a transition model".
The order being designed is what makes the ground truth knowable; the sounds
being real is what makes the percepts as hard as they are in life.

Measured, on the mind's own percept sequence and against the truth:

    world_self    predict the next percept from the previous one, over what the
                  mind believed it heard -- exactly `real_mind.py`'s probe.
                  **Degenerate if perception collapses**: a mind that names
                  everything "engine" predicts its own sequence at 1.000 and has
                  learned nothing, so this number is only readable beside the
                  count of distinct percepts
    world_true    predict the next TRUE class. Harder and more meaningful: a
                  mind can be perfectly self-consistent about a wrong sequence
    perceive      how often the ear names the clip correctly, which is the
                  quantity the whole prediction is about

Usage:  python3 benchmarks/real_time.py out_real_time.json
"""
import json
import sys

import numpy as np

from neurobrain.audition.audio import Cochleagram
from neurobrain.minds.unified import build_mind_on
from neurobrain.sensing.natural import ESC50_CLASSES, load_esc50
from neurobrain.sensing.streams import AuditoryBelt

SEEDS = (0, 1, 2, 3)
SR, DUR, N_FREQ = 8000, 2.0, 36
N_PALLIUM = 600
DAY_LEN, N_DAYS = 60, 3
ARMS = ("no_sleep", "memory_only", "all_stores")

#: A street. Sounds real, succession invented -- see the module docstring.
STREET = {
    "car_horn": ("engine", "siren"),
    "engine": ("engine", "train"),
    "siren": ("engine", "siren"),
    "train": ("engine", "train"),
    "helicopter": ("helicopter", "engine"),
    "church_bells": ("church_bells", "engine"),
}


def build_world(seed=0):
    """Real urban recordings, indexed by class, plus the transition structure."""
    waves, y, _ = load_esc50(n_shards=6, sr=SR, dur_s=DUR, seed=seed)
    names = [n for n in STREET]
    want = {ESC50_CLASSES.index(n): k for k, n in enumerate(names)}
    keep = [i for i, v in enumerate(y) if int(v) in want]
    if not keep:
        raise RuntimeError("no urban clips cached for this world")
    lab = np.array([want[int(y[i])] for i in keep])
    return [waves[i] for i in keep], lab, names


def encode(waves, seed=0, adapt=True):
    """The belt code is the 'image' this mind perceives.

    `build_mind_on` only flattens and contrast-normalises, so any vector works.
    Passing sound codes where a digit mind takes pixels is the whole point: the
    machinery was never about pixels.

    ``adapt`` is not optional in practice. Belt codes are non-negative and share
    a large component across every clip, and `contrast_normalise` removes each
    sample's *own* mean, which does nothing to a component that is common to all
    of them. Measured: without it, same-class cosine is 0.986 and different-class
    0.976 -- a separation of 0.010 -- and `GrowingCategoryMap` grows **one**
    category at every vigilance from 0.3 to 0.9, so the mind names every sound
    "engine" and every downstream number is measured on a mind that perceives a
    single thing.

    `PopulationAdaptation` is the fix already built and measured for vision,
    where it took 1-NN on photographs from 0.185 to 0.285: each feature stops
    transmitting its own running baseline. The same defect, in the other sense.
    """
    from neurobrain.vision.widev1 import PopulationAdaptation

    coch = Cochleagram(sr=SR, n_freq=N_FREQ)
    belt = AuditoryBelt(n_freq=N_FREQ, seed=seed)
    X = np.array([belt.code(coch.forward(w)[0]) for w in waves], np.float32)
    if adapt:
        ad = PopulationAdaptation(X.shape[1])
        X = np.array([ad(r) for r in X], np.float32)
    X = X - X.min()
    return (X / max(X.max(), 1e-6) * 255.0).astype(np.uint8)


def a_day(lab, names, rng, length=DAY_LEN):
    """Walk the street; return the clip indices heard, in order."""
    by_class = {c: np.flatnonzero(lab == c) for c in range(len(names))}
    cur = int(rng.integers(len(names)))
    out = []
    for _ in range(length):
        pool = by_class[cur]
        if len(pool):
            out.append(int(rng.choice(pool)))
        nxt = STREET[names[cur]]
        cur = names.index(str(rng.choice(nxt)))
    return out


def probe_world(mind, seq_self, seq_true):
    """Two questions: self-consistency, and agreement with the world."""
    def score(seq):
        if len(seq) < 2:
            return 0.0
        return sum(int(mind.space.predict_next(a) == b)
                   for a, b in zip(seq[:-1], seq[1:])) / (len(seq) - 1)
    return score(seq_self), score(seq_true)


def run_seed(sd, X, lab, names):
    # ONE day, shared by every arm. Generating it inside the arm loop advanced
    # the generator, so the three arms lived three different days and the
    # comparison was across days rather than across nights.
    rng = np.random.default_rng(sd)
    day = [i for _ in range(N_DAYS) for i in a_day(lab, names, rng)]
    out = {}
    for arm in ARMS:
        mind = build_mind_on(X, lab, X, lab, names=names,
                             n_pallium=N_PALLIUM, seed=sd)
        seq_self, seq_true = [], []
        for i in day:
            seq_self.append(mind.perceive(X[i]))
            seq_true.append(names[int(lab[i])])
        pa = float(np.mean([a == b for a, b in zip(seq_self, seq_true)]))
        b_self, b_true = probe_world(mind, seq_self, seq_true)
        if arm == "memory_only":
            mind.dream(cycles=3, replays_per_cycle=300,
                       to_perception=False, to_world=False)
        elif arm == "all_stores":
            mind.dream(cycles=3, replays_per_cycle=300,
                       to_perception=True, to_world=True)
        a_self, a_true = probe_world(mind, seq_self, seq_true)
        out[arm] = dict(perceive=pa, distinct_percepts=len(set(seq_self)),
                        before_self=b_self, before_true=b_true,
                        after_self=a_self, after_true=a_true,
                        episodes=len(mind.episodes))
    return out


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_real_time.json"
    waves, lab, names = build_world()
    X = encode(waves)
    print(f"{len(X)} real ESC-50 clips over {len(names)} street classes "
          f"({', '.join(names)}), code {X.shape[1]}d")
    print(f"chance for the next class: {1/len(names):.3f}; a day is "
          f"{DAY_LEN} sounds x {N_DAYS}\n", flush=True)

    per_seed = [run_seed(sd, X, lab, names) for sd in SEEDS]
    for sd, r in zip(SEEDS, per_seed):
        n = r["no_sleep"]
        print(f"seed {sd}: the ear names {n['perceive']:.3f} over "
              f"{n['distinct_percepts']} distinct percepts | world self "
              f"{n['before_self']:.3f} true {n['before_true']:.3f} | "
              f"{n['episodes']} episodes", flush=True)

    res = {"seeds": list(SEEDS), "n_class": len(names),
           "chance": round(1 / len(names), 4), "per_seed": per_seed}
    pa = float(np.mean([s["no_sleep"]["perceive"] for s in per_seed]))
    dp = float(np.mean([s["no_sleep"]["distinct_percepts"] for s in per_seed]))
    print(f"\nthe ear names its own clips at {pa:.3f} over {dp:.1f} distinct "
          f"percepts of {len(names)} (the eye names photographs at 0.129)")
    if dp < 2:
        print("  -- perception has COLLAPSED to one class; nothing below is "
              "readable")

    print(f"\n{'arm':<14}{'world self':>12}{'world true':>12}")
    for arm in ARMS:
        a = float(np.mean([s[arm]["after_self"] for s in per_seed]))
        b = float(np.mean([s[arm]["after_true"] for s in per_seed]))
        res.setdefault("after", {})[arm] = dict(self=round(a, 4),
                                                true=round(b, 4))
        print(f"{arm:<14}{a:>12.3f}{b:>12.3f}")

    print(f"\nchange from the night (all_stores minus no_sleep)")
    print(f"{'metric':<14}{'delta':>10}{'sd':>8}{'d':>8}{'wins':>7}")
    gate = {}
    for key, lbl in (("after_self", "world self"), ("after_true", "world true")):
        a = np.array([s["all_stores"][key] for s in per_seed])
        b = np.array([s["no_sleep"][key] for s in per_seed])
        d = a - b
        sd = float(d.std(ddof=1))
        cd = float(d.mean() / (sd + 1e-12))
        gate[lbl] = dict(delta=round(float(d.mean()), 4), sd=round(sd, 4),
                         cohens_d=round(cd, 3), wins=int((d > 0).sum()),
                         n=len(d))
        print(f"{lbl:<14}{d.mean():>+10.4f}{sd:>8.4f}{cd:>8.2f}"
              f"{int((d > 0).sum()):>4}/{len(d)}")
    res["gate"] = gate

    print("\n=== is perception what gates the world model? ===")
    base_true = float(np.mean([s["no_sleep"]["after_true"] for s in per_seed]))
    print(f"  hearing names {pa:.3f}; the world model predicts the true next "
          f"class at {base_true:.3f} (chance {1/len(names):.3f})")
    g = gate["world true"]
    if g["cohens_d"] >= 0.8 and g["wins"] >= 0.75 * g["n"]:
        print(f"  replay IMPROVES it by {g['delta']:+.4f} (d={g['cohens_d']:+.2f},"
              f" {g['wins']}/{g['n']}) -- so the machinery works when the")
        print("  percepts are right, and real_mind.py's upstream explanation "
              "holds.")
    elif abs(g["delta"]) < 1e-9:
        print("  replay changes it by EXACTLY zero -- a disconnection, and the "
              "upstream story is not what is wrong.")
    else:
        print(f"  replay does not reliably improve it ({g['delta']:+.4f}, "
              f"d={g['cohens_d']:+.2f}, {g['wins']}/{g['n']}) even with "
              f"perception at {pa:.3f}.")
        print("  Good percepts are not sufficient; the transition replay itself "
              "is the limit.")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
