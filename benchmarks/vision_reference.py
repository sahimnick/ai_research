"""How far is this eye from a gradient-trained one, on the SAME task?

The project's visual numbers have never had an external reference. cluster AUC
0.630 is meaningful against raw pixels (0.524) and against the ear (0.716), but
nothing said what a conventional vision model gets on the identical data, so
"the eye is weak" has been a comparison to the other sense rather than to the
state of the art.

This measures it, on the same 6 categories, the same 32px grayscale frames, and
the same `cluster_auc` computed the same way.

Three references, and each isolates something different:

    cnn-small-240     a convolutional net trained by BACKPROP on exactly the
                      240 images the eye develops on -- so the only difference
                      is the learning rule
    cnn-small-12k     the same net on 2000 images per class -- so the
                      difference is data, which the eye's rule cannot use
                      (`scale.py`: 30x the data moved pairs-per-cell not at all)
    probe-on-eye      a supervised linear read-out ON THE EYE'S OWN CODE. This
                      is the one that matters most: it separates "the
                      information is not in the code" from "the information is
                      there and the unsupervised read-out cannot reach it"

The comparison is deliberately unfair in the CNN's favour and is reported as
such: it uses gradients, a loss function and labels, all three of which this
project excludes by design. The point is the size of the gap, not a verdict on
which approach is better.

Usage:  python3 benchmarks/vision_reference.py out_vision_reference.json
"""
import json
import sys

import numpy as np
import torch
import torch.nn as nn

from neurobrain.cognition.multimodal import _unit
from neurobrain.learning.selforganize import develop_v1
from neurobrain.sensing.natural import load_audiovisual, load_cifar10
from neurobrain.vision.widev1 import PopulationAdaptation, WideV1

sys.path.insert(0, "benchmarks")
from pathways import FRAME, cluster_auc, place                 # noqa: E402
from real_binding import split                                 # noqa: E402

SEEDS = (0, 1, 2)
N_PER_CLASS = 40                      # what the eye gets: 240 images
BIG_PER_CLASS = 2000
EPOCHS, BATCH = 30, 64
AV_CLASSES = ("airplane", "automobile", "bird", "cat", "dog", "frog")
CIFAR_IDX = {"airplane": 0, "automobile": 1, "bird": 2, "cat": 3,
             "deer": 4, "dog": 5, "frog": 6}


class SmallCNN(nn.Module):
    """Three conv blocks. Small on purpose -- the question is what gradients
    buy on this data, not what a big model buys."""

    def __init__(self, n_cls=6):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(128, n_cls)

    def forward(self, x):
        return self.head(self.feat(x))


def train_cnn(X, y, Xte, yte, seed, epochs=EPOCHS):
    torch.manual_seed(seed)
    net = SmallCNN(int(y.max()) + 1)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    yt = torch.tensor(y, dtype=torch.long)
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            lossf(net(Xt[b]), yt[b]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        Ft = net.feat(torch.tensor(Xte, dtype=torch.float32).unsqueeze(1))
        acc = float((net(torch.tensor(Xte, dtype=torch.float32).unsqueeze(1))
                     .argmax(1).numpy() == yte).mean())
    return Ft.numpy(), acc


def linear_probe(V, y, tr, te, seed, epochs=300):
    """A supervised linear read-out. Adds no representation of its own -- it can
    only reach what the code already separates."""
    torch.manual_seed(seed)
    lin = nn.Linear(V.shape[1], int(y.max()) + 1)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    Xtr = torch.tensor(V[tr], dtype=torch.float32)
    ytr = torch.tensor(y[tr], dtype=torch.long)
    for _ in range(epochs):
        opt.zero_grad()
        lossf(lin(Xtr), ytr).backward()
        opt.step()
    with torch.no_grad():
        p = lin(torch.tensor(V[te], dtype=torch.float32)).argmax(1).numpy()
    return float((p == y[te]).mean())


def main():
    out_path = (sys.argv[1] if len(sys.argv) > 1
                else "out_vision_reference.json")
    imgs, _w, y, names = load_audiovisual(n_per_class=N_PER_CLASS, seed=0,
                                          grayscale=True, size=32)
    y = np.asarray(y, int)
    small = np.stack([np.asarray(im, np.float32) / 255.0 for im in imgs])
    frames = [place(im) for im in imgs]
    print(f"{len(imgs)} photographs, {len(names)} categories: {', '.join(names)}")
    print("every arm is measured with the SAME cluster_auc on the SAME split\n",
          flush=True)

    trX, trY, teX, teY = load_cifar10(n_train=50000, n_test=10000, seed=0,
                                      size=32, grayscale=True)
    X_all = np.concatenate([trX, teX]).astype(np.float32) / 255.0
    Y_all = np.concatenate([trY, teY])
    want = [CIFAR_IDX[c] for c in names if c in CIFAR_IDX]
    remap = {c: k for k, c in enumerate(want)}
    big_idx = np.concatenate([np.flatnonzero(Y_all == c)[:BIG_PER_CLASS]
                              for c in want])
    Xbig = X_all[big_idx]
    ybig = np.array([remap[int(Y_all[i])] for i in big_idx])
    print(f"big training set for the CNN reference: {len(Xbig)} images\n",
          flush=True)

    res = {"arms": {}}
    per = {}
    for sd in SEEDS:
        tr, te = split(y, sd)

        # ---- the eye, exactly as the project builds it --------------------
        v1 = WideV1(n_cells=4096, window_ms=50, rf=7, stride=2,
                    image_shape=(FRAME, FRAME), seed=sd)
        develop_v1(v1, list(frames), epochs=3, tie=True, seed=sd)
        R = np.array([v1.drive(f) for f in frames], np.float32)
        ad = PopulationAdaptation(R.shape[1])
        V = np.array([_unit(ad(r)) for r in R], np.float32)
        rng = np.random.default_rng(sd)
        per.setdefault("eye (this project)", []).append(
            cluster_auc(V[te], y[te], V[tr], y[tr], rng))
        per.setdefault("probe-on-eye (accuracy)", []).append(
            linear_probe(V, y, tr, te, sd))

        # ---- raw pixels ---------------------------------------------------
        P = np.stack([_unit(f.ravel()) for f in small])
        per.setdefault("raw pixels", []).append(
            cluster_auc(P[te], y[te], P[tr], y[tr], np.random.default_rng(sd)))
        per.setdefault("probe-on-pixels (accuracy)", []).append(
            linear_probe(P, y, tr, te, sd))

        # ---- CNN on the SAME 240 images -----------------------------------
        F, acc = train_cnn(small[tr], y[tr], small, y, sd)
        Fn = np.stack([_unit(f) for f in F])
        per.setdefault("cnn-small-240", []).append(
            cluster_auc(Fn[te], y[te], Fn[tr], y[tr], np.random.default_rng(sd)))
        per.setdefault("cnn-small-240 (accuracy)", []).append(
            float((np.asarray(acc),).__getitem__(0)))

        # ---- CNN on 12k images --------------------------------------------
        F2, acc2 = train_cnn(Xbig, ybig, small, y, sd, epochs=12)
        F2n = np.stack([_unit(f) for f in F2])
        per.setdefault("cnn-12k", []).append(
            cluster_auc(F2n[te], y[te], F2n[tr], y[tr],
                        np.random.default_rng(sd)))
        per.setdefault("cnn-12k (accuracy)", []).append(float(acc2))
        print(f"  seed {sd} done", flush=True)

    for k, v in per.items():
        res["arms"][k] = round(float(np.mean(v)), 4)

    print(f"\n{'arm':<30}{'cluster AUC':>13}")
    for k in ("raw pixels", "eye (this project)", "cnn-small-240", "cnn-12k"):
        print(f"{k:<30}{res['arms'][k]:>13.3f}")
    print(f"\n{'arm':<30}{'6-way accuracy':>16}   (chance 0.167)")
    for k in ("probe-on-pixels (accuracy)", "probe-on-eye (accuracy)",
              "cnn-small-240 (accuracy)", "cnn-12k (accuracy)"):
        print(f"{k:<30}{res['arms'][k]:>16.3f}")

    gap_same = res["arms"]["cnn-small-240"] - res["arms"]["eye (this project)"]
    gap_data = res["arms"]["cnn-12k"] - res["arms"]["eye (this project)"]
    res["gap_same_data"] = round(float(gap_same), 4)
    res["gap_with_data"] = round(float(gap_data), 4)
    print(f"\n  gradients on the SAME 240 images buy {gap_same:+.4f} cluster AUC")
    print(f"  gradients + 50x the data buy         {gap_data:+.4f}")
    print(f"\n  the read-out question: the eye's code carries "
          f"{res['arms']['probe-on-eye (accuracy)']:.3f} under a SUPERVISED "
          f"linear probe,")
    print(f"  against {res['arms']['probe-on-pixels (accuracy)']:.3f} for raw "
          f"pixels. If the probe is far above the unsupervised read-out, the")
    print("  information is in the code and the concept layer is what cannot "
          "reach it.")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
