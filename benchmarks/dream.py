"""Is dream() a mechanism now, or still a no-op -- and does replay actually help?

Before this, `UnifiedMind.episodes` was constructed empty and nothing ever
wrote to it, so `sleep()` returned 0 for the object's whole life and the
SleepConsolidator's "+13% with no new data" was unreachable from the assembled
mind. `perceive()` now lays down an episode.

Measuring it correctly matters more than it looks. A first pass probed
`vision.cortex` and found a flat zero for every arm *including the
shuffled-label control* -- which was the tell that the readout was blind, not
that replay was useless. Replay calls ``space.remember(label, image=pattern)``,
so it refines the **pallium's image-modality exemplars**; `vision.cortex` is a
different structure it never touches. The probe here is
``space.recognise("image", noisy)``, which reads what replay writes.

The day is also deliberately **imbalanced**. `development.sleep_benefit_experiment`
says why in its own docstring: online learning leaves the rare categories badly
estimated, and prioritised replay spends the night fixing them. A balanced day
gives replay nothing to fix.

Usage:  python3 benchmarks/dream.py out_dream.json
"""
import json, sys
import numpy as np

import neurobrain as nb

N_PALLIUM = 300      # small enough that one day of experience can matter
N_DAY = 400
NOISE = 0.55


def zipf_day(trx, trY, n, rng, alpha=1.4):
    """A Zipf-like day: a couple of common digits, a long rare tail."""
    freq = 1.0 / np.arange(1, 11) ** alpha
    freq = freq / freq.sum()
    order = rng.permutation(10)                 # which digits are the common ones
    picks = []
    for d in rng.choice(10, size=n, p=freq):
        digit = int(order[d])
        idx = np.where(trY == digit)[0]
        picks.append(int(rng.choice(idx)))
    return np.array(picks), order, freq


def probe(mind, X, Y, rng, noise=NOISE):
    """Noisy image -> concept, THROUGH THE PALLIUM (what replay refines)."""
    ok = 0
    for im, y in zip(X, Y):
        v = (im.reshape(-1).astype(np.float32) / 255.0)
        v = v + rng.normal(0, noise, v.shape).astype(np.float32)
        ok += int(mind.space.recognise("image", v) == str(int(y)))
    return ok / max(len(Y), 1)


def main():
    trx, trY, tex, teY = nb.load_mnist(n_train=4000, n_test=400)
    out = {}

    # ------------------------------------------------ 1. does the buffer fill
    mind = nb.build_unified_mind(n_pallium=N_PALLIUM)
    n0 = len(mind.episodes)
    for im in trx[:150]:
        mind.perceive(im)
    n1 = len(mind.episodes)
    for im in trx[150:200]:
        mind.perceive(im, remember=False)
    n2 = len(mind.episodes)
    out["buffer"] = dict(after_build=n0, after_150_perceive=n1,
                         after_50_probes=n2,
                         records_on_perceive=bool(n1 > n0),
                         probe_is_silent=bool(n2 == n1))
    print("1. buffer:", json.dumps(out["buffer"]), flush=True)

    # -------------------------------------------- 3. is replay prioritised?
    sur = np.array([e.surprise for e in mind.episodes.episodes])
    pr = mind.episodes.priorities()
    out["priority"] = dict(
        n=len(sur), surprise_mean=round(float(sur.mean()), 4),
        surprise_distinct=int(len(np.unique(np.round(sur, 6)))),
        priority_ratio=round(float(pr.max() / max(pr.min(), 1e-12)), 2),
        is_flat=bool(len(np.unique(np.round(sur, 6))) <= 1))
    print("3. priority:", json.dumps(out["priority"]), flush=True)

    # ------------------------------------------- 2 & 4. does replay help?
    rows = {}
    for arm in ("real", "shuffled", "no_sleep"):
        rng = np.random.default_rng(0)
        m = nb.build_unified_mind(n_pallium=N_PALLIUM)
        picks, order, freq = zipf_day(trx, trY, N_DAY, np.random.default_rng(1))
        labels = [str(int(trY[i])) for i in picks]
        if arm == "shuffled":
            labels = list(np.random.default_rng(2).permutation(labels))
        for i, lab in zip(picks, labels):
            if arm == "shuffled":
                m._record_episode(trx[i], lab)     # same images, wrong day
            else:
                m.perceive(trx[i])
        before = probe(m, tex[:400], teY[:400], np.random.default_rng(3))
        replays = 0 if arm == "no_sleep" else m.dream(
            cycles=3, replays_per_cycle=400)["replays"]
        after = probe(m, tex[:400], teY[:400], np.random.default_rng(3))

        # rare vs common, which is where replay is supposed to earn its keep
        seen = np.array([int(trY[i]) for i in picks])
        counts = np.array([(seen == d).sum() for d in range(10)])
        rare = set(np.argsort(counts)[:5].tolist())
        mask = np.array([int(y) in rare for y in teY[:400]])
        rng2 = np.random.default_rng(3)
        rare_after = probe(m, tex[:400][mask], teY[:400][mask], rng2)
        rows[arm] = dict(episodes=len(m.episodes), replays=int(replays),
                         before=round(before, 4), after=round(after, 4),
                         delta=round(after - before, 4),
                         rare_after=round(rare_after, 4),
                         day_counts=counts.tolist())
        print(f"2/4. {arm:9}", json.dumps({k: v for k, v in rows[arm].items()
                                           if k != "day_counts"}), flush=True)
    out["sleep"] = rows

    real, shuf, none = rows["real"], rows["shuffled"], rows["no_sleep"]
    out["verdict"] = dict(
        dream_is_live=bool(real["replays"] > 0),
        replay_helps=bool(real["delta"] > 0),
        gain_over_no_sleep=round(real["after"] - none["after"], 4),
        gain_over_shuffled_day=round(real["after"] - shuf["after"], 4))
    print("verdict:", json.dumps(out["verdict"]), flush=True)
    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "out_dream.json",
                        "w"), indent=1)


if __name__ == "__main__":
    main()
