"""The circuit: V1 -> higher area -> prediction back down -> settle -> repeat.

This is the loop drawn as the alternative to backpropagation -- a higher stage
sends its best guess down, the lower stage sends up only what the guess missed,
and the two run round until they agree. It is Rao & Ballard predictive coding,
and it is the family the literature reports can match backprop on small and
medium image tasks without a global backward pass.

`PredictiveStack` in `neurobrain/world/topdown.py` already implements it
exactly:

    prediction = W.T @ r
    error      = v1 - prediction
    r         <- relu(r + eta * (W @ error) - decay * r)     # inference
    W         <- W + lr * outer(r, error)                    # learning

Both updates are local products of quantities present in the tissue. No global
error signal, no weight transport, no loss function.

**It has never been measured.** It is exported from `neurobrain/__init__.py`,
appears in no benchmark, and appears nowhere in EVALUATION.md. This runs it on
the real audiovisual task, against the feedforward eye it is supposed to
improve.

What each arm isolates
----------------------
    v1-only            the current eye. The thing to beat.
    loop-1             the same stack with the settling loop cut to ONE
                       iteration -- feedback happens, but nothing iterates.
    loop-20            the full circuit, run to a fixed point.
    loop-20-noteach    settled the same way, but the generative weights are
                       never learned. Separates "the loop's inference helps"
                       from "the loop's learning helps".

`loop-1` against `loop-20` is the drawn idea's own claim -- "continue till best
result" -- isolated from everything else. If they tie, iteration is not what is
buying anything and the circuit is just another feedforward stage.

Measured with `cluster_auc_full`, the corrected estimator (§9.13), because the
sampled one is 3-6x too noisy and biased toward structured codes.

And the question that actually matters, from §9.11: **does the slope in data
change?** The eye is flat from 240 to 12 000 images while a CNN gains +0.123.
A circuit that beats the eye at 240 but is equally flat has not addressed the
defect.

Usage:  python3 benchmarks/closed_loop.py out_closed_loop.json
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
from neurobrain.world.topdown import PredictiveStack

sys.path.insert(0, "benchmarks")
from pathways import FRAME, cluster_auc_full, place            # noqa: E402
from real_binding import split                                 # noqa: E402

SEEDS = (0, 1, 2)
N_PER_CLASS = 40
N_HIGH = 256
DATA_STEPS = (40, 400, 2000)          # per class: 240, 2400, 12000 images
CID = {"airplane": 0, "automobile": 1, "bird": 2, "cat": 3, "dog": 5, "frog": 6}
ARMS = ("v1-only", "loop-1", "loop-20", "loop-20-noteach")


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


def build(arm, v1, devel_rates, seed):
    """Grow the higher stage on V1's own output, unsupervised."""
    if arm == "v1-only":
        return None
    iters = 1 if arm == "loop-1" else 20
    st = PredictiveStack(n_v1=devel_rates.shape[1], n_high=N_HIGH,
                         iters=iters, seed=seed)
    if arm != "loop-20-noteach":
        st.train(list(devel_rates), epochs=3)
    return st


def encode(arm, v1, stack, frames):
    R = np.array([v1.drive(f) for f in frames], np.float32)
    if stack is None:
        ad = PopulationAdaptation(R.shape[1])
        return np.array([_unit(ad(r)) for r in R], np.float32)
    H = np.array([stack.settle(r)[0] for r in R], np.float32)
    ad = PopulationAdaptation(H.shape[1])
    return np.array([_unit(ad(h)) for h in H], np.float32)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_closed_loop.json"
    imgs, _w, y, names = load_audiovisual(n_per_class=N_PER_CLASS, seed=0,
                                          grayscale=True, size=32)
    y = np.asarray(y, int)
    frames = [place(im) for im in imgs]
    trX, trY, teX, teY = load_cifar10(n_train=50000, n_test=10000, seed=0,
                                      size=32, grayscale=True)
    X = np.concatenate([trX, teX])
    Y = np.concatenate([trY, teY])
    want = [CID[c] for c in names]

    print(f"{len(imgs)} photographs, {len(names)} categories, {FRAME}px frame")
    print("metric: cluster_auc_full (the corrected estimator)\n", flush=True)

    res = {"seeds": list(SEEDS), "curve": {}}
    for n_per in DATA_STEPS:
        pool = np.concatenate([np.flatnonzero(Y == c)[:n_per] for c in want])
        devel = [place(X[i]) for i in pool]
        got = {a: {"auc": [], "probe": []} for a in ARMS}
        for sd in SEEDS:
            v1 = WideV1(n_cells=4096, window_ms=50, rf=7, stride=2,
                        image_shape=(FRAME, FRAME), seed=sd)
            develop_v1(v1, devel, epochs=3, tie=True, seed=sd)
            dev_rates = np.array([v1.drive(f) for f in devel], np.float32)
            tr, te = split(y, sd)
            for arm in ARMS:
                st = build(arm, v1, dev_rates, sd)
                V = encode(arm, v1, st, frames)
                got[arm]["auc"].append(
                    cluster_auc_full(V[te], y[te], V[tr], y[tr]))
                got[arm]["probe"].append(probe(V, y, tr, te, sd))
        res["curve"][len(devel)] = {
            a: {"auc": round(float(np.mean(got[a]["auc"])), 4),
                "probe": round(float(np.mean(got[a]["probe"])), 4)}
            for a in ARMS}
        row = res["curve"][len(devel)]
        print(f"  {len(devel):>5} images: " + "  ".join(
            f"{a} {row[a]['auc']:.3f}/{row[a]['probe']:.3f}" for a in ARMS),
            flush=True)

    ns = sorted(res["curve"])
    print(f"\n{'arm':<18}" + "".join(f"{n:>10}" for n in ns) + f"{'slope':>10}")
    for a in ARMS:
        v = [res["curve"][n][a]["auc"] for n in ns]
        print(f"{a:<18}" + "".join(f"{x:>10.4f}" for x in v)
              + f"{v[-1] - v[0]:>+10.4f}")
    print(f"\n{'arm (probe acc)':<18}" + "".join(f"{n:>10}" for n in ns)
          + f"{'slope':>10}")
    for a in ARMS:
        v = [res["curve"][n][a]["probe"] for n in ns]
        print(f"{a:<18}" + "".join(f"{x:>10.4f}" for x in v)
              + f"{v[-1] - v[0]:>+10.4f}")

    lo, hi = ns[0], ns[-1]
    it = (res["curve"][lo]["loop-20"]["auc"] - res["curve"][lo]["loop-1"]["auc"])
    beats = (res["curve"][lo]["loop-20"]["auc"]
             - res["curve"][lo]["v1-only"]["auc"])
    slope = (res["curve"][hi]["loop-20"]["auc"]
             - res["curve"][lo]["loop-20"]["auc"])
    res["iteration_worth"] = round(float(it), 4)
    res["beats_feedforward"] = round(float(beats), 4)
    res["loop_slope"] = round(float(slope), 4)

    print(f"\n--- the three questions the circuit has to answer ---")
    print(f"  1. does the loop beat the feedforward eye?      {beats:+.4f}")
    print(f"  2. does ITERATING buy anything (20 vs 1 step)?  {it:+.4f}")
    print(f"  3. does it fix the DATA slope?                  {slope:+.4f}"
          f"   (eye -0.013, CNN +0.123)")
    if slope > 0.02:
        print("\n  The circuit converts data into representation where the "
              "feedforward eye could not. That is the defect addressed.")
    elif beats > 0.02:
        print("\n  The circuit beats the feedforward eye but is equally flat in "
              "data, so it is a better stage and not a fix for\n  what §9.11 "
              "measured.")
    else:
        print("\n  The circuit neither beats the feedforward eye nor changes "
              "the data slope. Predictive coding as implemented here\n  does "
              "not transfer either, and the reason is worth stating rather than "
              "the next idea being tried.")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
