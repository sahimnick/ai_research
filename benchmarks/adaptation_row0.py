"""How much does the zero-vector first row move a published number?

`PopulationAdaptation` starts at n_seen=0, so its first update uses
a = max(tau, 1/(n_seen+1)) = 1.0 and the running mean becomes the first code
exactly. `_unit(r - r)` is the zero vector. Every benchmark in this project
that builds `V = [_unit(ad(r)) for r in R]` therefore has a dead row 0.

On the live-camera test (23 frames) that destroyed the floor outright. On a
240-photograph benchmark it is 1 sample in 240. This measures whether it is
1/240 of nothing or something that moved a conclusion.

Three arms on the SAME codes:
    as-published   online adaptation, row 0 dead
    row0-repaired  same, but row 0 re-applied through the final baseline
    fit-frozen     baseline fitted on all codes, then applied to all of them
"""
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "benchmarks")
from neurobrain.cognition.multimodal import _unit
from neurobrain.learning.selforganize import develop_v1
from neurobrain.sensing.natural import load_audiovisual
from neurobrain.vision.widev1 import PopulationAdaptation, WideV1
from pathways import cluster_auc_full, place                     # noqa: E402
from real_binding import split                                   # noqa: E402

SEEDS = (0, 1, 2)


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


def main():
    imgs, _w, y, _c = load_audiovisual(n_per_class=40, seed=0,
                                       grayscale=True, size=32)
    y = np.asarray(y, int)
    frames = [place(im) for im in imgs]
    print(f"{len(imgs)} photographs\n", flush=True)

    got = {a: {"auc": [], "probe": []}
           for a in ("as-published", "row0-repaired", "fit-frozen")}
    for sd in SEEDS:
        v1 = WideV1(n_cells=4096, window_ms=50, rf=7, stride=2,
                    image_shape=frames[0].shape, seed=sd)
        develop_v1(v1, list(frames), epochs=3, tie=True, seed=sd)
        R = np.array([v1.drive(f) for f in frames], np.float32)

        ad = PopulationAdaptation(R.shape[1])
        Va = np.array([_unit(ad(r)) for r in R], np.float32)
        n_dead = int((np.linalg.norm(Va, axis=1) < 1e-6).sum())

        Vb = Va.copy()
        Vb[0] = _unit(ad.apply(R[0]))          # final baseline, not its own

        ad2 = PopulationAdaptation(R.shape[1])
        for r in R:
            ad2.observe(r)
        Vc = np.array([_unit(ad2(r, learn=False)) for r in R], np.float32)

        tr, te = split(y, sd)
        for nm, V in (("as-published", Va), ("row0-repaired", Vb),
                      ("fit-frozen", Vc)):
            got[nm]["auc"].append(cluster_auc_full(V[te], y[te], V[tr], y[tr]))
            got[nm]["probe"].append(probe(V, y, tr, te, sd))
        print(f"  seed {sd}: {n_dead} dead row(s) of {len(R)}", flush=True)

    print(f"\n{'arm':<16}{'cluster AUC':>13}{'probe':>10}")
    base = None
    for nm in ("as-published", "row0-repaired", "fit-frozen"):
        a = float(np.mean(got[nm]["auc"]))
        p = float(np.mean(got[nm]["probe"]))
        if base is None:
            base = (a, p)
            print(f"{nm:<16}{a:>13.4f}{p:>10.4f}")
        else:
            print(f"{nm:<16}{a:>13.4f}{p:>10.4f}   "
                  f"({a-base[0]:+.4f} / {p-base[1]:+.4f})")

    r0 = (float(np.mean(got['row0-repaired']['auc'])) - base[0],
          float(np.mean(got['row0-repaired']['probe'])) - base[1])
    print(f"\n--- does the dead row change a conclusion? ---")
    print(f"  repairing row 0 alone moves AUC {r0[0]:+.4f}, probe {r0[1]:+.4f}")
    if max(abs(r0[0]), abs(r0[1])) < 0.005:
        print("  Under 0.005 on both. The dead row is a real defect but it did "
              "not move any published visual number;\n  it mattered only where "
              "N was small (23 live cameras) or where the row WAS the "
              "measurement (the floor).")
    else:
        print("  Above 0.005. Published visual numbers are affected and the "
              "ones that moved must be restated.")


if __name__ == "__main__":
    main()
