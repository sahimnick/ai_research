"""Grow all four areas on photographs instead of line drawings, and re-audit.

§9.15 measured the four-stage ventral stream on real photographs and found it
stops helping after V2: V4 clusters at 0.526 -- **below raw pixels' 0.532** --
with a participation ratio of 7.9 and an average of **12 spikes** per
photograph. Its own self-test is excellent (99% shapes, 92% complex objects,
87% IT purity), so the architecture is sound; each area was simply developed on
its own hand-made stimuli, `oriented_edges` / `corner_images` / `curve_images`.
Line drawings. The later areas were never shown the data the project is
evaluated on.

`build_ventral_stream_on` changes exactly that one thing. Same architecture,
same widths, same canvas, same competitive Hebbian rule, same patch sampler --
only the training stimuli become the photographs themselves. A difference
between the two streams can therefore only be the stimuli.

Hypothesis, before the run
--------------------------
**H17.** Developing the higher areas on photographs will revive them: V4's
spike count and participation ratio rise, and the hierarchy becomes monotone
(each stage at least as good as the one below) instead of 1 of 3.

**H17 is falsified if V4 stays silent.** Then the stimuli were not the problem
and something structural in the architecture cannot carry photographs -- which
would be a different and more serious finding than a training-set mismatch.

Reported per stage: cluster AUC, an independent linear probe, participation
ratio, live-unit fraction and spikes. A stage counts as revived only if spikes
and participation rise **and** the probe does not fall -- a stage can be made
loud by lowering a threshold without carrying anything more.

Usage:  python3 benchmarks/ventral_on_photos.py out_ventral_photos.json
"""
import json
import sys

import numpy as np
import torch
import torch.nn as nn

from neurobrain.cognition.multimodal import _unit
from neurobrain.sensing.natural import load_audiovisual, load_cifar10
from neurobrain.vision.ventral import (build_ventral_stream,
                                       build_ventral_stream_on, stage_extents,
                                       upscale)
from neurobrain.vision.widev1 import PopulationAdaptation

sys.path.insert(0, "benchmarks")
from pathways import cluster_auc_full                          # noqa: E402
from real_binding import split                                 # noqa: E402

SEEDS = (0, 1, 2)
N_PER_CLASS = 40
N_DEVELOP = 600
SZ = 96
CANVAS = 96
CID = {"airplane": 0, "automobile": 1, "bird": 2, "cat": 3, "dog": 5, "frog": 6}


def participation_ratio(X):
    X = np.asarray(X, np.float32)
    if len(X) < 2:
        return 0.0
    C = np.cov(X - X.mean(0), rowvar=False)
    ev = np.linalg.eigvalsh(C)
    ev = ev[ev > 0]
    return float(ev.sum() ** 2 / (ev ** 2).sum()) if len(ev) else 0.0


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


def audit(stream, canvas, y):
    """Every area's code, and whether it carries anything."""
    h = stream.hierarchy
    names = [getattr(l, "name", type(l).__name__) for l in h.layers]
    per = [[] for _ in names]
    for im in canvas:
        x = np.asarray(im, np.float32)
        if x.max() > 1.5:
            x = x / 255.0
        h.forward(x[None])
        for k, layer in enumerate(h.layers):
            per[k].append(np.asarray(layer.log["output"], np.float32).ravel())
    rows = []
    for k, nm in enumerate(names):
        raw = np.stack(per[k])
        ad = PopulationAdaptation(raw.shape[1])
        V = np.array([_unit(ad(r)) for r in raw], np.float32)
        a, p = [], []
        for sd in SEEDS:
            tr, te = split(y, sd)
            a.append(cluster_auc_full(V[te], y[te], V[tr], y[tr]))
            p.append(probe(V, y, tr, te, sd))
        rows.append({"stage": nm, "dim": int(raw.shape[1]),
                     "cluster_auc": round(float(np.mean(a)), 4),
                     "probe": round(float(np.mean(p)), 4),
                     "participation": round(participation_ratio(V), 2),
                     "live_frac": round(float((raw > 0).any(0).mean()), 3),
                     "spikes": round(float(raw.sum(1).mean()), 1)})
    return rows


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_ventral_photos.json"
    imgs, _w, y, names_cls = load_audiovisual(n_per_class=N_PER_CLASS, seed=0,
                                              grayscale=True, size=32)
    y = np.asarray(y, int)
    canvas = [upscale(im, CANVAS) for im in imgs]

    # a separate pool of photographs to DEVELOP on -- never the evaluation set
    trX, trY, teX, teY = load_cifar10(n_train=50000, n_test=10000, seed=0,
                                      size=32, grayscale=True)
    X = np.concatenate([trX, teX])
    Y = np.concatenate([trY, teY])
    want = [CID[c] for c in names_cls]
    pool = np.concatenate([np.flatnonzero(Y == c)[-N_DEVELOP // 6:]
                           for c in want])
    devel = np.stack([upscale(X[i], CANVAS)
                      for i in pool])
    print(f"{len(imgs)} evaluation photographs; {len(devel)} SEPARATE "
          f"photographs to develop on\n", flush=True)

    print("A. the stream as built -- areas grown on line drawings", flush=True)
    drawn = build_ventral_stream(verbose=False)
    rows_drawn = audit(drawn, canvas, y)

    print("B. the same architecture, areas grown on PHOTOGRAPHS", flush=True)
    photo = build_ventral_stream_on(devel, size=SZ, verbose=True)
    rows_photo = audit(photo, canvas, y)

    res = {"drawings": rows_drawn, "photographs": rows_photo,
           "seeds": list(SEEDS)}

    print(f"\n{'stage':<14}{'AUC (drawn)':>13}{'AUC (photo)':>13}"
          f"{'probe (drawn)':>15}{'probe (photo)':>15}")
    for a, b in zip(rows_drawn, rows_photo):
        print(f"{a['stage']:<14}{a['cluster_auc']:>13.3f}"
              f"{b['cluster_auc']:>13.3f}{a['probe']:>15.3f}"
              f"{b['probe']:>15.3f}")
    print(f"\n{'stage':<14}{'spikes (d)':>12}{'spikes (p)':>12}"
          f"{'partic (d)':>12}{'partic (p)':>12}{'live (p)':>10}")
    for a, b in zip(rows_drawn, rows_photo):
        print(f"{a['stage']:<14}{a['spikes']:>12.1f}{b['spikes']:>12.1f}"
              f"{a['participation']:>12.2f}{b['participation']:>12.2f}"
              f"{b['live_frac']:>10.2f}")

    d4 = next(r for r in rows_drawn if r["stage"] == "V4")
    p4 = next(r for r in rows_photo if r["stage"] == "V4")
    res["V4_spikes"] = [d4["spikes"], p4["spikes"]]
    res["V4_probe"] = [d4["probe"], p4["probe"]]
    revived = bool(p4["spikes"] > 2 * d4["spikes"]
                   and p4["participation"] > d4["participation"]
                   and p4["probe"] >= d4["probe"] - 0.01)
    res["H17_V4_revived"] = revived

    pl = [r["probe"] for r in rows_photo]
    rise = sum(1 for i in range(1, len(pl)) if pl[i] >= pl[i - 1] - 0.005)
    res["monotone_photo"] = f"{rise}/{len(pl)-1}"
    dl = [r["probe"] for r in rows_drawn]
    rise_d = sum(1 for i in range(1, len(dl)) if dl[i] >= dl[i - 1] - 0.005)
    res["monotone_drawn"] = f"{rise_d}/{len(dl)-1}"

    print(f"\n--- H17: does growing on photographs revive the later areas? ---")
    print(f"  V4 spikes {d4['spikes']:.1f} -> {p4['spikes']:.1f}, "
          f"participation {d4['participation']:.2f} -> "
          f"{p4['participation']:.2f}, probe {d4['probe']:.3f} -> "
          f"{p4['probe']:.3f}")
    print(f"  hierarchy non-decreasing: drawings {res['monotone_drawn']}, "
          f"photographs {res['monotone_photo']}")
    print(f"  H17 holds (V4 louder, wider, and no worse): {revived}")
    if not revived:
        print("  H17 is FALSIFIED. The training stimuli were not what silenced "
              "V4, so something structural in this\n  architecture cannot carry "
              "photographs -- a more serious finding than a training-set "
              "mismatch, and one\n  that no amount of re-developing will fix.")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
