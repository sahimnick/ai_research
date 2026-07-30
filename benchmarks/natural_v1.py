"""What an eye needs to see photographs: colour, adaptation, and its own fields.

Section 7.8 of EVALUATION.md left vision as the bottleneck. `WideV1` scores
1-NN **0.818 on MNIST and 0.185 on CIFAR-10**, where raw opponent pixels score
0.308 -- a 1024-cell spiking layer doing worse than its own input. Everything
downstream is blocked behind it: two photographs of one category sit at cosine
~0, so concept cells never merge and a night has nothing to consolidate.

Three changes are measured here, cumulatively, because each was cheap and
biological and none of them is a new architecture:

    colour      opponent channels (L, R-G, B-Y). Grayscale was a modelling
                choice, not a property of the world, and these categories lean
                on colour -- sky is blue, frogs are green. Primate retina builds
                exactly these three
    adaptation  `PopulationAdaptation`: each cell stops transmitting its own
                running baseline. The V1 code on photographs has a large common
                component -- 63.5% of cells silent, the rest responding alike --
                so cosine similarity is dominated by what every code shares.
                This is the operation the belt and the workspace already had
                and vision did not
    discovered  receptive fields grown from natural images by `develop_v1`
                instead of the hand-written Gabor-like defaults. The project's
                own headline claim is that discovered fields beat designed ones;
                this asks whether that survives real photographs

Two probes, because they answer different questions. `proto` is the class-mean
read-out and tolerates a code whose individual samples are scattered. `1-NN` is
the one that matters for everything downstream, because the association area
stores exemplars: if two photographs of a cat are not neighbours, no concept can
form from them.

MNIST is carried through every row as a **regression check**, and the fields
developed on CIFAR are applied to it *without retraining* -- so the transfer
row is asking whether natural-image statistics make a better general-purpose V1
than hand-written Gabors do (Olshausen & Field 1996).

Usage:  python3 benchmarks/natural_v1.py out_natural_v1.json
"""
import json
import sys

import numpy as np

import neurobrain as nb
from neurobrain.learning.selforganize import develop_v1
from neurobrain.sensing.natural import load_cifar10
from neurobrain.vision.widev1 import (PopulationAdaptation, WideV1,
                                      _nearest_prototype, _unit)

SEEDS = (0, 1, 2)
N_TRAIN, N_TEST, N_DEV = 2500, 600, 1500
N_CELLS = 1024


def opponent(im):
    """Retinal colour opponency: luminance, red-green, blue-yellow."""
    r, g, b = [im[i].astype(np.float32) for i in range(3)]

    def n(c):
        c = c - c.min()
        m = c.max()
        return (c / m * 255.0) if m > 1e-6 else c
    return n((r + g + b) / 3.0), n(r - g), n(b - (r + g) / 2.0)


def knn(Xtr, ytr, Xte, k=1):
    S = Xte @ Xtr.T
    nn = np.argpartition(-S, k - 1, axis=1)[:, :k]
    return np.array([np.bincount(ytr[r]).argmax() for r in nn])


def codes(v1, images, colour, adapt, frozen=None):
    def rate(im):
        if colour and im.ndim == 3:
            return np.concatenate([v1.rate(c) for c in opponent(im)])
        return v1.rate(im[0] if im.ndim == 3 else im)

    R = np.array([rate(im) for im in images], np.float32)
    if not adapt:
        return np.array([_unit(r) for r in R], np.float32), None
    if frozen is not None:
        return np.array([_unit(frozen.apply(r)) for r in R], np.float32), frozen
    ad = PopulationAdaptation(R.shape[1])
    return np.array([_unit(ad(r)) for r in R], np.float32), ad


def score(v1, tr_x, tr_y, te_x, te_y, colour, adapt, n_cls=10):
    Tr, ad = codes(v1, tr_x, colour, adapt)
    Te, _ = codes(v1, te_x, colour, adapt, frozen=ad)
    return dict(proto=float(np.mean(_nearest_prototype(Tr, tr_y, Te, n_cls) == te_y)),
                knn1=float(np.mean(knn(Tr, tr_y, Te, 1) == te_y)),
                knn5=float(np.mean(knn(Tr, tr_y, Te, 5) == te_y)))


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_natural_v1.json"

    Xc, yc, Xtc, ytc = load_cifar10(n_train=N_TRAIN, n_test=N_TEST,
                                    grayscale=False, size=28)
    mx, my, mxt, myt = nb.load_mnist(n_train=N_TRAIN, n_test=N_TEST)
    print(f"CIFAR-10 {Xc.shape} colour, MNIST {mx.shape}, chance 0.100, "
          f"{len(SEEDS)} seeds\n", flush=True)

    rows = ("grayscale, designed fields",
            "colour, designed fields",
            "colour + adaptation, designed fields",
            "colour + adaptation, DISCOVERED fields")
    res = {"seeds": list(SEEDS), "cifar": {}, "mnist": {}}

    for sd in SEEDS:
        base = WideV1(n_cells=N_CELLS, window_ms=50, seed=sd)
        dev = develop_v1(WideV1(n_cells=N_CELLS, window_ms=50, seed=sd),
                         [opponent(im)[0] for im in Xc[:N_DEV]],
                         epochs=3, seed=sd)
        cfg = ((base, False, False), (base, True, False),
               (base, True, True), (dev, True, True))
        for name, (v1, col, ad) in zip(rows, cfg):
            res["cifar"].setdefault(name, []).append(
                score(v1, Xc, yc, Xtc, ytc, col, ad))
            # MNIST is grayscale; only the fields and the adaptation vary
            res["mnist"].setdefault(name, []).append(
                score(v1, mx, my, mxt, myt, False, ad))
        print(f"   seed {sd} done", flush=True)

    for tag in ("cifar", "mnist"):
        print(f"\n--- {tag.upper()} (chance 0.100) ---")
        print(f"{'':<42}{'proto':>8}{'1-NN':>8}{'5-NN':>8}")
        first = None
        for name in rows:
            m = {k: float(np.mean([r[k] for r in res[tag][name]]))
                 for k in ("proto", "knn1", "knn5")}
            s = float(np.std([r["knn1"] for r in res[tag][name]], ddof=1))
            res[tag][name] = dict(mean=m, knn1_sd=round(s, 4),
                                  runs=res[tag][name])
            first = first or m
            print(f"{name:<42}{m['proto']:>8.3f}{m['knn1']:>8.3f}"
                  f"{m['knn5']:>8.3f}")
        last = res[tag][rows[-1]]["mean"]
        print(f"{'':<42}{'':>8}{'':>8}")
        print(f"  cumulative 1-NN: {first['knn1']:.3f} -> {last['knn1']:.3f} "
              f"({last['knn1'] - first['knn1']:+.3f})")

    print("\n=== is the eye fixed? ===")
    c = res["cifar"]
    a = c["grayscale, designed fields"]["mean"]["knn1"]
    b = c["colour + adaptation, DISCOVERED fields"]["mean"]["knn1"]
    m0 = res["mnist"]["grayscale, designed fields"]["mean"]["knn1"]
    m1 = res["mnist"]["colour + adaptation, DISCOVERED fields"]["mean"]["knn1"]
    print(f"  photographs  1-NN {a:.3f} -> {b:.3f}  ({b/max(a,1e-9):.2f}x, "
          f"{b/0.1:.1f}x chance)")
    print(f"  digits       1-NN {m0:.3f} -> {m1:.3f}  "
          f"({'no regression' if m1 >= m0 - 0.01 else 'REGRESSION'})")
    print(f"  the gap to hearing (1-NN 0.924 on 6 real ESC-50 categories) is "
          f"{'closed' if b > 0.8 else 'still open'}")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
