"""Does a second cortical stage help on photographs, where four levers did not?

This project measured depth as harmful: v0.21's V1->V2->V3 scored 39.2% against
63.4% for a single wide layer, and `integrated.py` records that "hierarchical
depth has failed four separate times". `widev1.py` exists because of that
result -- width instead of depth.

Every one of those failures was on **MNIST digits with narrow layers** (24 V1
cells feeding 160 feeding 300). A digit is one high-contrast figure on an empty
field, one wide layer already reaches ~0.82 on it, and a second stage built from
24 inputs has nothing to pass on. "Depth hurts" was close to guaranteed there.

`benchmarks/vision_ceiling.py` found the opposite situation on photographs: 1-NN
0.323 against 0.914 for the ear, with width flat from 1024 to 4096 cells, a
larger aperture worse, a longer window doing nothing, and spiking costing
nothing at all. Four levers, no movement -- which leaves depth as the only thing
untested under conditions where it could matter.

So this runs the comparison honestly and on both worlds:

    V1 alone          the current eye, at its best single-layer configuration
    V1 + V2           the same V1, with `SecondStage` on top, learned by
                      competition over static patches
    V1 + V2 (concat)  both codes together, since a downstream area is free to
                      read whichever it needs and discarding V1 is not required
    V1 + V2temporal   V2 learned by Foldiak's trace rule instead -- one winner
                      per sequence of views of the same object, updated on all
                      of them, so invariance comes from the world changing more
                      slowly than the retina does. This is the visual version of
                      what PredictiveA1 does for hearing, which is the front end
                      in this project that actually works
    V1 + V2temporal (concat)

on CIFAR-10 photographs **and** MNIST digits -- because the claim being tested
is not "depth is good" but "depth helps where a single layer is not already at
its ceiling", and that predicts a gain on one and not the other. If V2 helps
both, the old result was simply wrong. If it helps neither, depth has failed a
fifth time and on the case most favourable to it.

Usage:  python3 benchmarks/second_stage.py out_second_stage.json
"""
import json
import sys
import time

import numpy as np

import neurobrain as nb
from neurobrain.learning.selforganize import develop_v1
from neurobrain.sensing.natural import load_cifar10
from neurobrain.vision.secondstage import SecondStage
from neurobrain.vision.widev1 import (PopulationAdaptation, WideV1,
                                      _nearest_prototype, _unit)

SEEDS = (0, 1, 2)
N_TRAIN, N_TEST, N_DEV = 2000, 500, 800
# rf 7 / stride 2 tied for best at V1 and gives an 11x11 grid, which leaves
# room for a pooling stage; the 7x7 grid of rf 10 / stride 3 collapses to 3x3
# after pooling and yields four patches per image.
V1_CELLS, V1_RF, V1_STRIDE = 4096, 7, 2
V2_UNITS, V2_POOL, V2_SPAN = 512, 2, 2


def opponent(im):
    r, g, b = [im[i].astype(np.float32) for i in range(3)]

    def n(c):
        c = c - c.min()
        m = c.max()
        return (c / m * 255.0) if m > 1e-6 else c
    return n((r + g + b) / 3.0), n(r - g), n(b - (r + g) / 2.0)


def views(im, n=4, seed=0):
    """One object, several fixations. Small shifts stand in for fixational
    drift and micro-saccades -- what actually reaches a retina looking at a
    stationary thing."""
    rng = np.random.default_rng(seed)
    out = [im]
    for _ in range(n - 1):
        dy, dx = rng.integers(-2, 3, 2)
        out.append(np.roll(np.roll(im, int(dy), axis=-2), int(dx), axis=-1))
    return out


def channels(im):
    """Colour images become three opponent planes; digits stay one plane."""
    return opponent(im) if im.ndim == 3 else (im.astype(np.float32),)


def knn(Xtr, ytr, Xte, k=1):
    S = Xte @ Xtr.T
    nn = np.argpartition(-S, k - 1, axis=1)[:, :k]
    return np.array([np.bincount(ytr[r]).argmax() for r in nn])


def evaluate(Rtr, Rte, y, yt):
    ad = PopulationAdaptation(Rtr.shape[1])
    Tr = np.array([_unit(ad(r)) for r in Rtr], np.float32)
    Te = np.array([_unit(ad.apply(r)) for r in Rte], np.float32)
    return dict(proto=float(np.mean(_nearest_prototype(Tr, y, Te, 10) == yt)),
                knn1=float(np.mean(knn(Tr, y, Te, 1) == yt)),
                knn5=float(np.mean(knn(Tr, y, Te, 5) == yt)))


def run(X, y, Xt, yt, tag, seed):
    t0 = time.time()
    v1 = WideV1(n_cells=V1_CELLS, window_ms=50, rf=V1_RF, stride=V1_STRIDE,
                seed=seed)
    develop_v1(v1, [channels(im)[0] for im in X[:N_DEV]], epochs=3, seed=seed)
    v2 = SecondStage(v1, n_units=V2_UNITS, pool=V2_POOL, span=V2_SPAN,
                     seed=seed)
    curve = v2.learn([channels(im)[0] for im in X[:N_DEV]], epochs=2, seed=seed)

    v2t = SecondStage(v1, n_units=V2_UNITS, pool=V2_POOL, span=V2_SPAN,
                      seed=seed)
    curve_t = v2t.learn_temporal(
        [[channels(v)[0] for v in views(im, seed=seed + i)]
         for i, im in enumerate(X[:N_DEV])], epochs=2, seed=seed)

    def code1(im):
        return np.concatenate([v1.drive(c) for c in channels(im)])

    def code2(im):
        return np.concatenate([v2.code(c) for c in channels(im)])

    def code2t(im):
        return np.concatenate([v2t.code(c) for c in channels(im)])

    A1 = np.array([code1(im) for im in X], np.float32)
    A1t = np.array([code1(im) for im in Xt], np.float32)
    A2 = np.array([code2(im) for im in X], np.float32)
    A2t = np.array([code2(im) for im in Xt], np.float32)
    A3 = np.array([code2t(im) for im in X], np.float32)
    A3t = np.array([code2t(im) for im in Xt], np.float32)
    both = np.concatenate([_unit_rows(A1), _unit_rows(A2)], 1)
    botht = np.concatenate([_unit_rows(A1t), _unit_rows(A2t)], 1)
    bothT = np.concatenate([_unit_rows(A1), _unit_rows(A3)], 1)
    bothTt = np.concatenate([_unit_rows(A1t), _unit_rows(A3t)], 1)

    out = {"V1 alone": evaluate(A1, A1t, y, yt),
           "V1 + V2": evaluate(A2, A2t, y, yt),
           "V1 + V2 (concat)": evaluate(both, botht, y, yt),
           "V1 + V2temporal": evaluate(A3, A3t, y, yt),
           "V1 + V2temporal (concat)": evaluate(bothT, bothTt, y, yt),
           "v2_learn_curve": [round(c, 4) for c in curve],
           "v2t_learn_curve": [round(c, 4) for c in curve_t],
           "seconds": round(time.time() - t0, 1),
           "v1_filters": v1.n_cells // v1.n_pos, "v1_grid": [v1.n_rows, v1.n_cols],
           "v2_dim": v2.dim}
    print(f"  seed {seed}: V1 {out['V1 alone']['knn1']:.3f}  "
          f"V2 {out['V1 + V2']['knn1']:.3f}/{out['V1 + V2 (concat)']['knn1']:.3f}"
          f"  V2temporal {out['V1 + V2temporal']['knn1']:.3f}/"
          f"{out['V1 + V2temporal (concat)']['knn1']:.3f}   "
          f"({out['seconds']:.0f}s)", flush=True)
    return out


def _unit_rows(M):
    n = np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-6)
    return M / n


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_second_stage.json"
    cx, cy, cxt, cyt = load_cifar10(n_train=N_TRAIN, n_test=N_TEST,
                                    grayscale=False, size=28)
    mx, my, mxt, myt = nb.load_mnist(n_train=N_TRAIN, n_test=N_TEST)
    res = {"config": dict(v1_cells=V1_CELLS, rf=V1_RF, stride=V1_STRIDE,
                          v2_units=V2_UNITS, pool=V2_POOL, span=V2_SPAN),
           "seeds": list(SEEDS)}

    for tag, data in (("CIFAR-10 photographs", (cx, cy, cxt, cyt)),
                      ("MNIST digits", (mx, my, mxt, myt))):
        print(f"\n{tag} (chance 0.100)")
        runs = [run(*data, tag, sd) for sd in SEEDS]
        res[tag] = runs
        print(f"  {'code':<20}{'proto':>8}{'1-NN':>8}{'5-NN':>8}")
        for arm in ("V1 alone", "V1 + V2", "V1 + V2 (concat)",
                    "V1 + V2temporal", "V1 + V2temporal (concat)"):
            m = {k: float(np.mean([r[arm][k] for r in runs]))
                 for k in ("proto", "knn1", "knn5")}
            sd = float(np.std([r[arm]["knn1"] for r in runs], ddof=1))
            res.setdefault("summary", {}).setdefault(tag, {})[arm] = {
                **{k: round(v, 4) for k, v in m.items()}, "knn1_sd": round(sd, 4)}
            print(f"  {arm:<20}{m['proto']:>8.3f}{m['knn1']:>8.3f}"
                  f"{m['knn5']:>8.3f}")

    print("\n=== does depth help where width did not? ===")
    for tag in ("CIFAR-10 photographs", "MNIST digits"):
        s = res["summary"][tag]
        base = s["V1 alone"]["knn1"]
        best_arm = max(("V1 + V2", "V1 + V2 (concat)", "V1 + V2temporal",
                        "V1 + V2temporal (concat)"),
                       key=lambda a: s[a]["knn1"])
        d = np.array([r[best_arm]["knn1"] - r["V1 alone"]["knn1"] for r in res[tag]])
        sd = float(d.std(ddof=1))
        cd = float(d.mean() / (sd + 1e-12))
        verdict = ("HELPS" if cd >= 0.8 and (d > 0).all()
                   else "HURTS" if cd <= -0.8 else "no effect")
        print(f"  {tag:<22} {base:.3f} -> {s[best_arm]['knn1']:.3f} "
              f"({d.mean():+.3f}, d={cd:+.2f}, {int((d > 0).sum())}/{len(d)}) "
              f"via {best_arm:<17} {verdict}")
    c = res["summary"]["CIFAR-10 photographs"]
    m = res["summary"]["MNIST digits"]
    arms = ("V1 + V2", "V1 + V2 (concat)", "V1 + V2temporal",
            "V1 + V2temporal (concat)")
    dc = max(c[a]["knn1"] for a in arms) - c["V1 alone"]["knn1"]
    dm = max(m[a]["knn1"] for a in arms) - m["V1 alone"]["knn1"]
    print(f"  competition {c['V1 + V2 (concat)']['knn1']:.3f} vs trace rule "
          f"{c['V1 + V2temporal (concat)']['knn1']:.3f} on photographs "
          f"(V1 alone {c['V1 alone']['knn1']:.3f})")
    print()
    if dc > 0.02 and dm <= 0.02:
        print("  -> depth helps on photographs and not on digits, which is the")
        print("     prediction: a second stage adds where the first is not "
              "already at its ceiling.")
        print("     The four earlier failures were all measured on digits.")
    elif dc <= 0.02 and dm <= 0.02:
        print("  -> depth has now failed a FIFTH time, and on the case most "
              "favourable to it.")
        print("     Width instead of depth stands as this project's answer.")
    else:
        print("  -> the pattern does not match the prediction; read the table "
              "rather than this line.")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
