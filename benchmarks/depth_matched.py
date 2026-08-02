"""V1 is the first cortical stage. What has this project been comparing it to?

The visual system is V1 -> V2 -> V4 -> IT. Object identity lives at the far end
of that, not at the near one: V1 codes oriented edges. Yet every visual number
in this project is `cluster_auc` on **V1's output**, asked to separate
*categories* -- a question V1 is not the stage that answers.

Worse, §9.11 compared that V1 code against a CNN's **final** layer, three
convolutional blocks deep. Those are not the same depth, so "the eye is 0.11
behind a CNN" mixed two claims: that the learning rule is worse, and that the
architecture is three stages shorter. This separates them.

The same trained CNN is read out at each depth:

    conv1   one 3x3 block  -- the honest analogue of V1
    conv2   two blocks     -- roughly V2
    conv3   three blocks   -- the layer §9.11 actually compared against

If `conv1` lands near this project's V1, then V1 is not the weak part and the
entire gap is missing depth. If `conv1` is already far ahead, the rule is
genuinely worse and depth is a second, separate problem.

This project HAS tried depth once: `second_stage.py` built a V2 and it lost a
third of the accuracy, with a concatenated arm -- which can only fail by V2
adding nothing -- landing at -0.014. That negative is in EVALUATION.md §7. But
it was built with the same competitive rule §9.11 later measured to be
insensitive to data, and it was tested at a single small data budget. Whether
depth failed, or that particular V2 failed, has never been separated.

Scored with `cluster_auc_full` (§9.13's corrected estimator) and an independent
linear probe.

Usage:  python3 benchmarks/depth_matched.py out_depth_matched.json
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
from pathways import FRAME, cluster_auc_full, place            # noqa: E402
from real_binding import split                                 # noqa: E402

SEEDS = (0, 1, 2)
N_PER_CLASS = 40
BIG_PER_CLASS = 2000
EPOCHS, BATCH = 12, 64
CID = {"airplane": 0, "automobile": 1, "bird": 2, "cat": 3, "dog": 5, "frog": 6}


class DepthCNN(nn.Module):
    """Three blocks, each tapped separately so a read-out can be taken at any
    depth from the SAME trained network -- not three networks of different
    sizes, which would confound depth with capacity and training."""

    def __init__(self, n_cls=6):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
                                nn.MaxPool2d(2))
        self.b2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                                nn.MaxPool2d(2))
        self.b3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(128, n_cls)

    def taps(self, x):
        h1 = self.b1(x)
        h2 = self.b2(h1)
        h3 = self.b3(h2)
        return [self.pool(h).flatten(1) for h in (h1, h2, h3)]

    def forward(self, x):
        return self.head(self.taps(x)[-1])


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
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_depth_matched.json"
    imgs, _w, y, names = load_audiovisual(n_per_class=N_PER_CLASS, seed=0,
                                          grayscale=True, size=32)
    y = np.asarray(y, int)
    small = np.stack([np.asarray(im, np.float32) / 255.0 for im in imgs])
    frames = [place(im) for im in imgs]

    trX, trY, teX, teY = load_cifar10(n_train=50000, n_test=10000, seed=0,
                                      size=32, grayscale=True)
    X = np.concatenate([trX, teX]).astype(np.float32) / 255.0
    Y = np.concatenate([trY, teY])
    want = [CID[c] for c in names]
    remap = {c: k for k, c in enumerate(want)}
    idx = np.concatenate([np.flatnonzero(Y == c)[:BIG_PER_CLASS] for c in want])
    Xb, yb = X[idx], np.array([remap[int(Y[i])] for i in idx])
    print(f"{len(imgs)} eval photographs, CNN trained on {len(Xb)} images")
    print("the same trained network read out at three depths\n", flush=True)

    res = {"arms": {}}
    per = {}
    for sd in SEEDS:
        tr, te = split(y, sd)

        # ---- this project's V1 -------------------------------------------
        v1 = WideV1(n_cells=4096, window_ms=50, rf=7, stride=2,
                    image_shape=(FRAME, FRAME), seed=sd)
        develop_v1(v1, list(frames), epochs=3, tie=True, seed=sd)
        R = np.array([v1.drive(f) for f in frames], np.float32)
        ad = PopulationAdaptation(R.shape[1])
        V = np.array([_unit(ad(r)) for r in R], np.float32)
        per.setdefault("this project's V1", []).append(
            (cluster_auc_full(V[te], y[te], V[tr], y[tr]),
             probe(V, y, tr, te, sd)))

        # ---- the CNN, tapped at each depth --------------------------------
        torch.manual_seed(sd)
        net = DepthCNN()
        opt = torch.optim.Adam(net.parameters(), lr=2e-3)
        lf = nn.CrossEntropyLoss()
        Xt = torch.tensor(Xb).unsqueeze(1)
        yt = torch.tensor(yb, dtype=torch.long)
        for _ in range(EPOCHS):
            pm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), BATCH):
                b = pm[i:i + BATCH]
                opt.zero_grad()
                lf(net(Xt[b]), yt[b]).backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            taps = net.taps(torch.tensor(small).unsqueeze(1))
        for k, t in enumerate(taps, start=1):
            F = np.stack([_unit(v) for v in t.numpy()])
            per.setdefault(f"CNN conv{k}", []).append(
                (cluster_auc_full(F[te], y[te], F[tr], y[tr]),
                 probe(F, y, tr, te, sd)))
        print(f"  seed {sd} done", flush=True)

    order = ["this project's V1", "CNN conv1", "CNN conv2", "CNN conv3"]
    for k in order:
        a = np.array([x[0] for x in per[k]])
        p = np.array([x[1] for x in per[k]])
        res["arms"][k] = {"cluster_auc": round(float(a.mean()), 4),
                          "probe": round(float(p.mean()), 4)}

    print(f"\n{'stage':<22}{'cluster AUC':>13}{'probe acc':>12}")
    for k in order:
        v = res["arms"][k]
        print(f"{k:<22}{v['cluster_auc']:>13.3f}{v['probe']:>12.3f}")

    v1a = res["arms"]["this project's V1"]["cluster_auc"]
    c1a = res["arms"]["CNN conv1"]["cluster_auc"]
    c3a = res["arms"]["CNN conv3"]["cluster_auc"]
    v1p = res["arms"]["this project's V1"]["probe"]
    c1p = res["arms"]["CNN conv1"]["probe"]
    c3p = res["arms"]["CNN conv3"]["probe"]
    res["depth_matched_gap"] = round(float(c1a - v1a), 4)
    res["depth_gain_in_cnn"] = round(float(c3a - c1a), 4)

    print(f"\n--- what §9.11's 0.11 gap was actually made of ---")
    print(f"  depth-MATCHED (V1 vs conv1):   {c1a - v1a:+.4f} AUC, "
          f"{c1p - v1p:+.4f} probe")
    print(f"  depth alone, inside the CNN:   {c3a - c1a:+.4f} AUC, "
          f"{c3p - c1p:+.4f} probe")
    if abs(c1a - v1a) < 0.02:
        print("\n  At matched depth the two are the same. V1 is NOT the weak "
              "part -- the whole gap is the three stages this\n  project does "
              "not have. Comparing V1 against a CNN's final layer was never a "
              "fair test of the rule.")
    elif c1a - v1a >= 0.02:
        print("\n  Even at matched depth the CNN's first stage is ahead, so the "
              "rule is genuinely weaker AND depth is missing.\n  Two separate "
              "problems, and this says so.")
    else:
        print("\n  This project's V1 is AHEAD of the CNN's first stage. The "
              "front end is not what is wrong; everything downstream is.")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
