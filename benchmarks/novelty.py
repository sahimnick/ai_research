"""Is the blend's "novelty" real, or a small wobble in the embedding?

Cosine against stored cells said 0.045. That number alone cannot distinguish
"a genuinely new percept" from "the same percept nudged", and it is measured in
raw pixel space, which is not where this system does its seeing.

Four things are added:

* **Pixel L2**, so the scale is interpretable next to cosine.
* **Perceptual distance through V1** -- the imagined image is pushed through the
  same `WideV1` the mind actually sees with, and compared in V1-rate space.
  That is high-level similarity as measured by this system's own front end,
  not by an arbitrary embedding.
* **A real-image baseline.** This is the one that settles it. A *held-out real
  digit* is scored by the identical metrics. If an imagined percept is roughly
  as far from the store as a genuine unseen digit is, the novelty is on the
  same scale as real novelty. If it is a fraction of that, the blend is
  wobbling, not imagining.
* **An off-manifold control.** Gaussian noise and a shuffled-pixel image are
  scored too, to put a ceiling on the scale: anything scoring like *those* is
  novel in the useless sense.

Usage:  python3 benchmarks/novelty.py out_novelty.json
"""
import json, sys
import numpy as np

import neurobrain as nb
from neurobrain.memory.development import SleepConsolidator
from neurobrain.vision.widev1 import WideV1

N_PALLIUM, N_DAY = 300, 400
SEEDS = (0, 1, 2)


def _unit(v):
    v = np.asarray(v, np.float32).ravel()
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def zipf_day(trY, n, rng, alpha=1.4):
    f = 1.0 / np.arange(1, 11) ** alpha
    f = f / f.sum()
    order = rng.permutation(10)
    return np.array([int(rng.choice(np.where(trY == int(order[d]))[0]))
                     for d in rng.choice(10, size=n, p=f)])


class Metrics:
    """Distance of a candidate percept from everything the mind has stored."""

    def __init__(self, M, v1):
        self.M = M                        # (cells, 784) unit rows
        self.v1 = v1
        self.V = np.array([_unit(v1.rate(m.reshape(28, 28))) for m in M],
                          np.float32)     # the store, as V1 sees it

    def __call__(self, p):
        p = np.asarray(p, np.float32).ravel()
        if np.linalg.norm(p) < 1e-9:
            return None
        pu = _unit(p)
        cos = 1.0 - float((self.M @ pu).max())
        l2 = float(np.min(np.linalg.norm(self.M - pu[None], axis=1)))
        pv = _unit(self.v1.rate(pu.reshape(28, 28)))
        v1cos = 1.0 - float((self.V @ pv).max())
        return dict(cosine=cos, pixel_l2=l2, v1_perceptual=v1cos)


def main():
    trx, trY, tex, teY = nb.load_mnist(n_train=4000, n_test=400)
    out = {}

    mind = nb.build_unified_mind(n_pallium=N_PALLIUM)
    picks = zipf_day(trY, N_DAY, np.random.default_rng(7))
    for i in picks:
        mind.perceive(trx[i])
    SleepConsolidator(mind.episodes, seed=0).sleep(
        lambda p, l, lr: mind.space.consolidate(str(l), lr=lr, image=p),
        cycles=3, replays_per_cycle=400)

    M = mind.space.cortex.modalities["image"]
    v1 = WideV1(n_cells=256, seed=0)
    metric = Metrics(M, v1)
    rng = np.random.default_rng(0)

    def summarise(name, samples, round_trip=None):
        vals = [metric(p) for p in samples]
        vals = [v for v in vals if v]
        row = {k: round(float(np.mean([v[k] for v in vals])), 5)
               for k in ("cosine", "pixel_l2", "v1_perceptual")}
        row["n"] = len(vals)
        if round_trip is not None:
            row["round_trip"] = round(round_trip, 4)
        print(f"  {name:<26} cos={row['cosine']:.5f}  L2={row['pixel_l2']:.4f}"
              f"  V1={row['v1_perceptual']:.5f}"
              + (f"  round_trip={row['round_trip']:.3f}" if round_trip is not None else ""),
              flush=True)
        return row

    print("REFERENCES — what the scale means")
    # a real held-out digit: the honest yardstick for "new but valid"
    out["real_heldout"] = summarise("real held-out digit",
                                    [tex[i].reshape(-1) / 255.0 for i in range(120)])
    # a training image the mind DID see: the floor
    out["seen_training"] = summarise("training image it saw",
                                     [trx[i].reshape(-1) / 255.0 for i in picks[:120]])
    # off-manifold ceilings
    out["gaussian_noise"] = summarise(
        "gaussian noise", [rng.normal(0.5, 0.3, 784).clip(0, 1) for _ in range(120)])
    out["shuffled_pixels"] = summarise(
        "shuffled-pixel digit",
        [rng.permutation(tex[i].reshape(-1) / 255.0) for i in range(120)])

    print("\nIMAGINED")
    rows = {}
    for blend, conc in [(0, 0.9), (3, 0.6), (6, 0.9), (12, 0.9), (24, 0.9)]:
        samples, ok, n = [], 0, 0
        for sd in SEEDS:
            names, pcs = mind.space.imagine("0", steps=14, temperature=0.6,
                                            rng_seed=sd, blend=blend,
                                            concentration=conc)
            for nm, p in zip(names, pcs):
                p = np.asarray(p, np.float32)
                if np.linalg.norm(p) < 1e-9:
                    continue
                samples.append(p)
                back = mind.perceive(_unit(p).reshape(28, 28), remember=False)
                n += 1
                ok += int(str(back) == str(nm))
        rows[f"blend{blend}"] = summarise(f"blend={blend} conc={conc}", samples,
                                          round_trip=ok / max(n, 1))
    out["imagined"] = rows

    # the verdict: imagined novelty as a FRACTION of real-held-out novelty
    print("\nVERDICT — imagined novelty as a fraction of a real unseen digit's")
    ref = out["real_heldout"]
    out["fraction_of_real"] = {}
    for k, r in rows.items():
        frac = {m: round(r[m] / max(ref[m], 1e-9), 3)
                for m in ("cosine", "pixel_l2", "v1_perceptual")}
        out["fraction_of_real"][k] = frac
        print(f"  {k:<10} cos={frac['cosine']:>6.2f}x  L2={frac['pixel_l2']:>6.2f}x"
              f"  V1={frac['v1_perceptual']:>6.2f}x   (1.00x = as novel as a real "
              f"unseen digit)")

    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1
                        else "out_novelty.json", "w"), indent=1)


if __name__ == "__main__":
    main()
