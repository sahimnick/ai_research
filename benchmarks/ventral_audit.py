"""The four-stage eye this project already had, measured for the first time.

The objection that prompted this: the visual system is V1 -> V2 -> V4 -> IT, and
this project put every visual claim on **V1 alone**. §9.14 measured what that
costs -- at matched depth this V1 is level with a CNN's first block (0.565
against 0.549) while the CNN's remaining blocks are worth **+0.157**.

Then a second thing turned up, which is worse. `neurobrain/vision/ventral.py`
already contains a complete spiking ventral stream --

    retina -> V1 (edges) -> V1 complex (phase-invariant) -> V2 (corners,
    junctions) -> pool -> V4 (curvature/shape)

-- every area convolutional and retinotopic, every filter learned by
competitive Hebbian plasticity, plus a `TraceInvarianceLayer` that learns
transformation invariance from objects moving in time. `VisionHierarchy` even
carries per-area input/output logging.

It appears in **no benchmark** and **zero times in EVALUATION.md**. Sixty
kilobytes of four-stage visual cortex, built and never measured, while thirteen
interventions were run on the one-stage `WideV1`.

What this measures
------------------
Each stage's output on the **same real photographs** every other visual number
in this project used, so the numbers can be read against them directly:

    cluster AUC     is category structure present?
    linear probe    is it reachable? (independent of the cosine metric)
    participation   has the code COLLAPSED? A stage can raise clustering by
                    discarding everything but one direction, and this catches it
    live fraction   what share of units ever fire across the whole set. A stage
                    whose units are dead carries nothing no matter what the
                    other numbers say
    spikes in/out   from `VisionHierarchy.logs()` -- the crude check that
                    anything is arriving at all

A stage earns its place only if AUC and probe rise **while** participation and
live fraction hold. That conjunction is the point: `second_stage.py` built a V2
that lost a third of the accuracy, and nothing in that benchmark could see
whether it had collapsed or merely reshuffled.

Usage:  python3 benchmarks/ventral_audit.py out_ventral_audit.json
"""
import json
import sys

import numpy as np
import torch
import torch.nn as nn

from neurobrain.cognition.multimodal import _unit
from neurobrain.learning.selforganize import develop_v1
from neurobrain.sensing.natural import load_audiovisual
from neurobrain.vision.ventral import build_ventral_stream, upscale, stage_extents
from neurobrain.vision.widev1 import PopulationAdaptation, WideV1

sys.path.insert(0, "benchmarks")
from pathways import FRAME, cluster_auc_full, place            # noqa: E402
from real_binding import split                                 # noqa: E402

SEEDS = (0, 1, 2)
N_PER_CLASS = 40
#: The canvas the hierarchy is actually run on. 32px is too small for four
#: stages: V4 comes out (44,1,1), one location, spike ceiling 12.
CANVAS = 96


from neurobrain.vision.ventral import code_participation as participation_ratio


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


def stage_codes(stream, images, size):
    """Every area's output for every image, plus the spike log."""
    h = stream.hierarchy
    names, per = None, None
    spikes = []
    for im in images:
        x = np.asarray(im, np.float32)
        if x.max() > 1.5:
            x = x / 255.0
        h.forward(x[None])
        logs = h.logs()
        if names is None:
            names = [d["layer"] for d in logs]
            per = [[] for _ in names]
        for k, layer in enumerate(h.layers):
            out = layer.log.get("output")
            per[k].append(np.asarray(out, np.float32).ravel())
        spikes.append([d["out_spikes"] for d in logs])
    return names, [np.stack(p) for p in per], np.array(spikes, np.float32)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_ventral_audit.json"
    imgs, _w, y, names_cls = load_audiovisual(n_per_class=N_PER_CLASS, seed=0,
                                              grayscale=True, size=32)
    y = np.asarray(y, int)
    print(f"{len(imgs)} photographs, {len(names_cls)} categories", flush=True)
    print("building the four-stage ventral stream (this is slow: every area is "
          "trained on its own stimuli)\n", flush=True)

    stream = build_ventral_stream(verbose=True)
    size = stream.size
    canvas = [np.asarray(upscale(im, CANVAS), np.float32)
              for im in imgs]
    frames = [place(im) for im in imgs]

    names, per_stage, spikes = stage_codes(stream, canvas, size)
    print(f"\nareas: {names}", flush=True)

    res = {"stages": [], "seeds": list(SEEDS)}
    rows = []

    # the one-stage eye every published number came from
    acc = {"auc": [], "probe": [], "pr": []}
    for sd in SEEDS:
        # SAME canvas as the ventral stream. Measuring WideV1 on a 48px
        # `place()` frame and the stream on a 96px one would compare across
        # input scales -- the error §9.14 and §9.16 were both caught making.
        v1 = WideV1(n_cells=4096, window_ms=50, rf=7, stride=2,
                    image_shape=(CANVAS, CANVAS), seed=sd)
        develop_v1(v1, list(canvas), epochs=3, tie=True, seed=sd)
        R = np.array([v1.drive(f) for f in canvas], np.float32)
        ad = PopulationAdaptation(R.shape[1])
        V = np.array([_unit(ad(r)) for r in R], np.float32)
        tr, te = split(y, sd)
        acc["auc"].append(cluster_auc_full(V[te], y[te], V[tr], y[tr]))
        acc["probe"].append(probe(V, y, tr, te, sd))
        acc["pr"].append(participation_ratio(V))
    rows.append({"stage": f"WideV1 @{CANVAS}px (one stage)",
                 "dim": int(V.shape[1]),
                 "cluster_auc": round(float(np.mean(acc["auc"])), 4),
                 "probe": round(float(np.mean(acc["probe"])), 4),
                 "participation": round(float(np.mean(acc["pr"])), 2),
                 "live_frac": round(float((V > 0).any(0).mean()), 3),
                 "spikes": None})

    for k, nm in enumerate(names):
        raw = per_stage[k]
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
                     "spikes": round(float(spikes[:, k].mean()), 1)})
    res["stages"] = rows

    print(f"\n{'stage':<34}{'dim':>7}{'AUC':>8}{'probe':>8}"
          f"{'partic':>9}{'live':>7}{'spikes':>9}")
    for r in rows:
        sp = "  -" if r["spikes"] is None else f"{r['spikes']:>9.1f}"
        print(f"{r['stage']:<34}{r['dim']:>7}{r['cluster_auc']:>8.3f}"
              f"{r['probe']:>8.3f}{r['participation']:>9.2f}"
              f"{r['live_frac']:>7.2f}{sp}")

    print(f"\n--- does each stage earn its place? ---")
    dead = [r["stage"] for r in rows[1:] if r["live_frac"] < 0.05]
    if dead:
        print(f"  DEAD AREAS (under 5% of units ever fire): {', '.join(dead)}")
        print("  Anything downstream of these is reading noise, and any loop "
              "built on them carries nothing.")
    best = max(rows[1:], key=lambda r: r["probe"]) if len(rows) > 1 else None
    if best:
        base = rows[0]
        print(f"  best ventral stage: {best['stage']} "
              f"probe {best['probe']:.3f} against WideV1's {base['probe']:.3f} "
              f"({best['probe'] - base['probe']:+.4f})")
        res["four_stage_beats_widev1"] = bool(
            best["probe"] - base["probe"] > 0.02)
        print(f"  four-stage beats the one-stage eye: "
              f"{res['four_stage_beats_widev1']}")
    ladder = [r["probe"] for r in rows[1:]]
    if len(ladder) > 1:
        rise = sum(1 for i in range(1, len(ladder)) if ladder[i] > ladder[i - 1])
        print(f"  stages that improve on the one below: {rise}/{len(ladder)-1}"
              f"   (a real hierarchy should mostly rise)")
        res["monotone_steps"] = f"{rise}/{len(ladder)-1}"

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
