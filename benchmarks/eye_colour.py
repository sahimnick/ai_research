"""Does the four-stage eye gain from colour? And what does depth cost in it?

§9.17 established the four-stage stream is worth **+0.108 probe at matched
width** over the one-stage `WideV1` — the first architectural gain in this
project that survives its controls. It ran on luminance alone.

That was never a design decision. `SpikingConvLayer` has taken ``in_channels``
since it was written, and `benchmarks/real_binding.opponent` has built the
retina's three channels (luminance, red-green, blue-yellow) all along, where
they moved the one-stage read-out from 0.288 to 0.365. Nothing connected them.

`build_ventral_colour` puts a three-channel retina in front of V1 and changes
nothing else, so a difference between the two streams is the colour channels
and nothing else.

Hypotheses, before the run
--------------------------
**H18 (colour helps the four-stage eye).** The categories lean on colour — sky
is blue, frogs are green — and it was worth +0.077 to the one-stage eye. It
should be worth something here too.

**H18 is falsified if the colour stream ties or loses.** That would say the
hierarchy discards chroma somewhere between V1 and V4, which would be worth
knowing: V1 complex pools over *orientation*, and if the pooling is what throws
colour away then colour has to enter above it rather than below.

**H19 (the gain is not just width).** Three channels give V1 three times the
input, so any gain must survive a random projection back to the luminance
stream's width. §9.16 and §9.17 were both caught reading a width difference as
an effect; this one is checked before it is claimed.

Every stage is reported, because §9.17 found the hierarchy is not monotone and
the best stage is not the last one.

Usage:  python3 benchmarks/eye_colour.py out_eye_colour.json
"""
import json
import sys

import numpy as np
import torch
import torch.nn as nn

from neurobrain.cognition.multimodal import _unit
from neurobrain.sensing.natural import load_audiovisual, load_cifar10
from neurobrain.vision.ventral import (build_ventral_colour,
                                       build_ventral_stream_on,
                                       code_participation, upscale)
from neurobrain.vision.widev1 import PopulationAdaptation

sys.path.insert(0, "benchmarks")
from pathways import cluster_auc_full                          # noqa: E402
from real_binding import split                                 # noqa: E402

SEEDS = (0, 1, 2)
N_PER_CLASS = 40
N_DEVELOP = 600
CANVAS = 96
CID = {"airplane": 0, "automobile": 1, "bird": 2, "cat": 3, "dog": 5, "frog": 6}


def probe(V, y, tr, te, seed, epochs=300):
    torch.manual_seed(seed)
    lin = nn.Linear(V.shape[1], int(y.max()) + 1)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=1e-4)
    lf = nn.CrossEntropyLoss()
    X = torch.tensor(V[tr], dtype=torch.float32)
    t = torch.tensor(y[tr], dtype=torch.long)
    for _ in range(epochs):
        opt.zero_grad()
        lf(lin(X), t).backward()
        opt.step()
    with torch.no_grad():
        p = lin(torch.tensor(V[te], dtype=torch.float32)).argmax(1).numpy()
    return float((p == y[te]).mean())


def codes_per_stage(stream, images, colour):
    from neurobrain.vision.ventral import retinal_opponent
    h = stream.hierarchy
    names = [getattr(l, "name", type(l).__name__) for l in h.layers]
    per = [[] for _ in names]
    for im in images:
        x = (retinal_opponent(im) if colour
             else np.asarray(im, np.float32)[None])
        if not colour and x.max() > 1.5:
            x = x / 255.0
        h.forward(x)
        for k, l in enumerate(h.layers):
            per[k].append(np.asarray(l.log["output"], np.float32).ravel())
    return names, [np.stack(p) for p in per]


def score(M, y, tag=None):
    ad = PopulationAdaptation(M.shape[1])
    V = np.array([_unit(ad(r)) for r in M], np.float32)
    a, p = [], []
    for sd in SEEDS:
        tr, te = split(y, sd)
        a.append(cluster_auc_full(V[te], y[te], V[tr], y[tr]))
        p.append(probe(V, y, tr, te, sd))
    return {"dim": int(M.shape[1]),
            "cluster_auc": round(float(np.mean(a)), 4),
            "probe": round(float(np.mean(p)), 4),
            "participation": round(code_participation(V), 2)}


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_eye_colour.json"
    imgs, _w, y, names_cls = load_audiovisual(n_per_class=N_PER_CLASS, seed=0,
                                              grayscale=False, size=32)
    y = np.asarray(y, int)
    colour_eval = [upscale(np.asarray(im, np.float32), CANVAS) for im in imgs]
    grey_eval = [c.mean(0) if c.ndim == 3 else c for c in colour_eval]

    trX, trY, teX, teY = load_cifar10(n_train=50000, n_test=10000, seed=0,
                                      size=32, grayscale=False)
    X = np.concatenate([trX, teX])
    Y = np.concatenate([trY, teY])
    want = [CID[c] for c in names_cls]
    pool = np.concatenate([np.flatnonzero(Y == c)[-N_DEVELOP // 6:]
                           for c in want])
    dev_col = np.stack([upscale(np.asarray(X[i], np.float32), CANVAS)
                        for i in pool])
    dev_grey = np.stack([d.mean(0) for d in dev_col])
    print(f"{len(imgs)} evaluation photographs at {CANVAS}px, "
          f"{len(dev_col)} separate to develop on")
    print(f"colour input {dev_col.shape}, luminance input {dev_grey.shape}\n",
          flush=True)

    res = {"seeds": list(SEEDS), "luminance": {}, "colour": {}}

    print("A. luminance stream (what §9.17 measured)", flush=True)
    lum = build_ventral_stream_on(dev_grey, size=CANVAS, verbose=False)
    names, per_l = codes_per_stage(lum, grey_eval, colour=False)
    for nm, M in zip(names, per_l):
        res["luminance"][nm] = score(M, y)
        print(f"   {nm:<12} {res['luminance'][nm]}", flush=True)

    print("\nB. the same architecture with a THREE-CHANNEL retina", flush=True)
    col = build_ventral_colour(dev_col, size=CANVAS, verbose=True)
    _n2, per_c = codes_per_stage(col, colour_eval, colour=True)
    for nm, M in zip(names, per_c):
        res["colour"][nm] = score(M, y)
        print(f"   {nm:<12} {res['colour'][nm]}", flush=True)

    print(f"\n{'stage':<14}{'lum AUC':>10}{'col AUC':>10}"
          f"{'lum probe':>12}{'col probe':>12}{'delta':>9}")
    for nm in names:
        a, b = res["luminance"][nm], res["colour"][nm]
        print(f"{nm:<14}{a['cluster_auc']:>10.3f}{b['cluster_auc']:>10.3f}"
              f"{a['probe']:>12.3f}{b['probe']:>12.3f}"
              f"{b['probe'] - a['probe']:>+9.4f}")

    best_l = max(names, key=lambda n: res["luminance"][n]["probe"])
    best_c = max(names, key=lambda n: res["colour"][n]["probe"])
    gain = res["colour"][best_c]["probe"] - res["luminance"][best_l]["probe"]
    res["H18_colour_gain"] = round(float(gain), 4)

    print(f"\n--- H18: does colour help the four-stage eye? ---")
    print(f"  best luminance stage {best_l} {res['luminance'][best_l]['probe']:.3f}")
    print(f"  best colour stage    {best_c} {res['colour'][best_c]['probe']:.3f}")
    print(f"  gain {gain:+.4f}   (colour was worth +0.077 to the ONE-stage eye)")

    # ---- H19: is any gain just three times the input width? ---------------
    print(f"\n--- H19: is the gain width rather than chroma? ---")
    Mc = per_c[names.index(best_c)]
    Ml = per_l[names.index(best_l)]
    if Mc.shape[1] > Ml.shape[1]:
        rng = np.random.default_rng(0)
        P = rng.normal(0, 1, (Mc.shape[1], Ml.shape[1])).astype(np.float32)
        P /= np.linalg.norm(P, axis=0, keepdims=True)
        red = score(np.maximum(Mc @ P, 0.0), y)
        res["colour_matched_width"] = red
        print(f"  colour {best_c} projected to the luminance width "
              f"({Ml.shape[1]}): probe {red['probe']:.3f}")
        print(f"  against luminance {best_l}: "
              f"{res['luminance'][best_l]['probe']:.3f}  -> matched gain "
              f"{red['probe'] - res['luminance'][best_l]['probe']:+.4f}")
        res["H18_gain_matched"] = round(
            float(red["probe"] - res["luminance"][best_l]["probe"]), 4)
    else:
        print(f"  widths already comparable "
              f"({Mc.shape[1]} vs {Ml.shape[1]}); no projection needed")
        res["H18_gain_matched"] = res["H18_colour_gain"]

    m = res["H18_gain_matched"]
    res["H18_holds"] = bool(m > 0.02)
    print(f"\n  H18 holds (colour worth >0.02 at matched width): "
          f"{res['H18_holds']}")
    if not res["H18_holds"]:
        print("  Colour does not survive into this hierarchy. V1 complex pools "
              "over ORIENTATION, and pooling is where chroma\n  most plausibly "
              "goes -- which would mean colour has to enter above the complex "
              "stage, not at the retina.\n  Stated as the place to look, not "
              "as a fix.")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
