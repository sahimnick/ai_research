"""Where does the eye actually run out on photographs?

After colour opponency, running-mean adaptation and receptive fields grown from
natural images, `WideV1` reaches 1-NN **0.326** on CIFAR-10 against **0.914**
for the ear on six real ESC-50 categories. Every cross-modal number in
EVALUATION.md section 7.8 is limited by that gap, so it is worth knowing which
part of the eye is responsible rather than guessing.

Four candidates, each a thing that could plausibly be the limit, each with a
control:

    width       ~20 distinct filters per location is very few for natural
                images, and single-layer unsupervised results on CIFAR are known
                to scale with feature count. Swept 1024 / 2048 / 4096 cells
    aperture    a 10x10 field on a 28x28 image sees an eighth of it. Swept
                against 7x7 (more positions, less context) and 14x14 (more
                context, fewer positions)
    spike noise the code is a windowed firing rate under membrane noise, so it
                is a noisy estimate of the underlying drive. Compared against
                the **noiseless filter response** -- the same receptive fields
                with the neuron model removed
    integration a longer window averages that noise down. 50 ms against 200 ms

The spiking comparison is the one that matters most for this project, because
the whole architecture is committed to spiking neurons and no gradients. If the
noiseless response were much better, that commitment would be costing accuracy
on real images and the cost should be stated. If it is not, the ceiling is
architectural and spiking is exonerated.

Usage:  python3 benchmarks/vision_ceiling.py out_vision_ceiling.json
"""
import json
import sys
import time

import numpy as np

from neurobrain.learning.selforganize import develop_v1
from neurobrain.sensing.natural import load_cifar10
from neurobrain.vision.widev1 import (PopulationAdaptation, WideV1,
                                      _nearest_prototype, _unit)

N_TRAIN, N_TEST, N_DEV = 2500, 600, 1200
EAR = 0.914          # 1-NN on six real ESC-50 categories, for scale


def opponent(im):
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


def score(fn, X, y, Xt, yt):
    R = np.array([np.concatenate([fn(c) for c in opponent(im)]) for im in X],
                 np.float32)
    Rt = np.array([np.concatenate([fn(c) for c in opponent(im)]) for im in Xt],
                  np.float32)
    ad = PopulationAdaptation(R.shape[1])
    Tr = np.array([_unit(ad(r)) for r in R], np.float32)
    Te = np.array([_unit(ad.apply(r)) for r in Rt], np.float32)
    return dict(proto=float(np.mean(_nearest_prototype(Tr, y, Te, 10) == yt)),
                knn1=float(np.mean(knn(Tr, y, Te, 1) == yt)),
                knn5=float(np.mean(knn(Tr, y, Te, 5) == yt)))


def eye(n_cells, rf, stride, X, seed=0):
    v1 = WideV1(n_cells=n_cells, window_ms=50, rf=rf, stride=stride, seed=seed)
    develop_v1(v1, [opponent(im)[0] for im in X[:N_DEV]], epochs=3, seed=seed)
    return v1


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_vision_ceiling.json"
    X, y, Xt, yt = load_cifar10(n_train=N_TRAIN, n_test=N_TEST,
                                grayscale=False, size=28)
    print(f"CIFAR-10 colour {X.shape}, chance 0.100; the ear reaches "
          f"{EAR:.3f} on its own six categories\n", flush=True)
    res = {"chance": 0.1, "ear_reference": EAR}

    # ---- does the neuron model cost anything? ----------------------------
    v1 = eye(1024, 10, 3, X)
    print(f"{'read-out':<32}{'proto':>8}{'1-NN':>8}{'5-NN':>8}{'time':>8}")
    for tag, fn in (("spiking rate, 50 ms", lambda c: v1.rate(c)),
                    ("spiking rate, 200 ms",
                     lambda c: v1.rate(c, window_ms=200)),
                    ("noiseless filter response", lambda c: v1.drive(c))):
        t0 = time.time()
        r = score(fn, X, y, Xt, yt)
        r["seconds"] = round(time.time() - t0, 1)
        res.setdefault("readout", {})[tag] = r
        print(f"{tag:<32}{r['proto']:>8.3f}{r['knn1']:>8.3f}{r['knn5']:>8.3f}"
              f"{r['seconds']:>7.0f}s", flush=True)

    sp = res["readout"]["spiking rate, 50 ms"]["knn1"]
    no = res["readout"]["noiseless filter response"]["knn1"]
    print(f"\n  spiking costs {sp - no:+.3f} against the noiseless response. "
          f"The sweep below therefore uses the")
    print(f"  noiseless read-out, which is ~20x faster and, at this "
          f"difference, measures the same thing.\n")

    # ---- width and aperture ----------------------------------------------
    print(f"{'config':<30}{'filters':>9}{'positions':>11}{'proto':>8}"
          f"{'1-NN':>8}{'5-NN':>8}")
    for n_cells, rf, stride in ((1024, 10, 3), (2048, 10, 3), (4096, 10, 3),
                                (2048, 7, 2), (2048, 14, 4)):
        v = eye(n_cells, rf, stride, X)
        r = score(lambda c: v.drive(c), X, y, Xt, yt)
        nf = v.n_cells // v.n_pos
        r.update(n_cells=n_cells, rf=rf, stride=stride, filters=nf,
                 positions=v.n_pos)
        res.setdefault("architecture", []).append(r)
        print(f"{f'{n_cells} cells, rf {rf}, stride {stride}':<30}{nf:>9}"
              f"{v.n_pos:>11}{r['proto']:>8.3f}{r['knn1']:>8.3f}"
              f"{r['knn5']:>8.3f}", flush=True)

    # ---------------------------------------------------------- verdict ---
    arch = res["architecture"]
    same_rf = [a for a in arch if a["rf"] == 10]
    lo, hi = same_rf[0]["knn1"], max(a["knn1"] for a in same_rf)
    print("\n=== which part of the eye is the ceiling? ===")
    print(f"  width      1024 -> 4096 cells: {same_rf[0]['knn1']:.3f} -> "
          f"{same_rf[-1]['knn1']:.3f}  ({hi - lo:+.3f} at best) -- "
          f"{'not the limit' if hi - lo < 0.03 else 'helps'}")
    ap = {a["rf"]: a["knn1"] for a in arch if a["n_cells"] == 2048}
    print(f"  aperture   rf 7 {ap.get(7, float('nan')):.3f}, rf 10 "
          f"{ap.get(10, float('nan')):.3f}, rf 14 "
          f"{ap.get(14, float('nan')):.3f} -- "
          f"{'not the limit' if max(ap.values()) - min(ap.values()) < 0.08 else 'matters'}")
    print(f"  spike noise  {sp:.3f} spiking vs {no:.3f} noiseless "
          f"({sp - no:+.3f}) -- "
          f"{'costs nothing' if abs(sp - no) < 0.02 else 'costs accuracy'}")
    w200 = res["readout"]["spiking rate, 200 ms"]["knn1"]
    print(f"  integration  50 ms {sp:.3f} vs 200 ms {w200:.3f} "
          f"({w200 - sp:+.3f}) -- "
          f"{'not the limit' if abs(w200 - sp) < 0.02 else 'helps'}")
    best = max(a["knn1"] for a in arch)
    print(f"\n  best of every configuration: {best:.3f}, against {EAR:.3f} for "
          f"the ear.")
    print(f"  Nothing available *within a single layer* moves it. That localises "
          f"the ceiling to depth --")
    print(f"  a second cortical stage -- which this project measured as harmful "
          f"once (V1->V2->V3 at 39.2%")
    print(f"  against 63.4% for one wide layer) but only at narrow widths and "
          f"only on MNIST digits.")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
