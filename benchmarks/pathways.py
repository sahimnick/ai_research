"""Four visual pathways on the same photographs, and a bar fixed in advance.

The eye's defect has one description that survives every test: two photographs
of the same thing do not land near each other. Cluster AUC 0.578 against the
ear's 0.788, unmoved across ten interventions (0.540-0.605). All ten operated on
a representation that was already inadequate.

`LocalSpatialEye` is a second pathway rather than an eleventh repair: 8x8
overlapping patches, a feature bank per patch, and each local feature vector
placed by **where it sits in the object** rather than in the frame -- the patch
grid itself laid out from the object's centroid, so translation cancels in the
sampling and not merely in the labelling.

The bar, fixed before any measurement
-------------------------------------
From `localspatial.py`'s H2, and it is the whole point of requirement 11:

    cluster AUC must exceed the best existing arm (0.605) by >= 0.02,
    with d >= 0.8 and >= 75% of seeds.

**Invariance alone is not a result.** Plain pooling already achieves it (gap
0.029) and clusters at 0.540; `relational_code` achieves it better (0.005) and
clusters at 0.560. A third invariant code that fails to cluster is the same
negative a third time, and the honest response is to reject the architecture.

Cluster AUC is computed from **raw codes with no learning anywhere**, which is
what makes it a test of the representation rather than of the read-out (H5). A
gain that appears only in downstream accuracy means the concept layer adapted to
a different code, not that the code got better.

The arms
--------
    rate            the current pathway. One global WideV1, absolute position
    pooled          `pooling_index("both")` -- invariance by discarding position
    relational      second-order statistics over relative displacement
    local-spatial   the new pathway

and its ablations, one per hypothesis:

    no object frame     patches with frame-absolute coordinates (H3)
    no resampling       object-centred labels, frame-fixed patch grid -- the
                        half-implementation the first smoke test caught
    no position (bins=1)  a bag of local features (what geometry is worth)
    global bank         the same binning over one global WideV1 (H4)
    independent banks   a separate filter bank per patch

Colour is dropped for **every** arm. The existing pathway gains about 0.04 from
opponent channels, so this understates it, but running some arms in colour and
others not would confound the comparison with its input.

Usage:  python3 benchmarks/pathways.py out_pathways.json
"""
import json
import sys

import numpy as np

from neurobrain.learning.selforganize import develop_v1
from neurobrain.sensing.natural import load_audiovisual
from neurobrain.vision.localspatial import LocalSpatialEye
from neurobrain.vision.widev1 import PopulationAdaptation, WideV1, _unit

sys.path.insert(0, "benchmarks")
from real_binding import opponent, split                     # noqa: E402

SEEDS = (0, 1, 2, 3)
N_IMAGES = 240
FRAME = 48
SHIFTS = (0, 5, -5)
GLOBAL_CELLS = 4096
#: The pre-registered bar, from localspatial.py's H2 -- and a flaw in it that
#: has to be stated rather than quietly corrected. 0.605 was measured in
#: `relational.py` with **colour** opponent channels; this benchmark runs every
#: arm on luminance alone so the arms share their input, and the same `rate`
#: pathway scores ~0.562 here. Holding a luminance result to a colour bar is a
#: cross-condition comparison, which is the error this project has caught
#: repeatedly.
#:
#: So both are reported and neither is dropped. The absolute bar stays because
#: it was registered in advance and moving a bar after seeing the data is how
#: results get manufactured. The **within-experiment** bar -- beat the best arm
#: measured here, under identical conditions, by the same margin -- is the one
#: that actually tests the architecture, and it is stricter in spirit because it
#: is not comparable to any earlier number.
BEST_EXISTING = 0.605
MIN_GAIN, MIN_D, MIN_WINS = 0.02, 0.8, 0.75

ARMS = {
    "rate": {}, "pooled": {}, "relational": {},
    "local-spatial": {},
    "  ablate: no object frame": dict(centred=False),
    "  ablate: no resampling": dict(resample=False),
    "  ablate: no position": dict(bins=1),
    "  ablate: global bank": dict(local=False),
    "  ablate: independent banks": dict(shared_bank=False),
}
NEW = [a for a in ARMS if a.startswith("local-spatial") or a.startswith("  ")]


def place(im, dx=0, frame=FRAME):
    """A 32-px object in a 48-px frame, padded so the shift keeps it whole."""
    a = np.asarray(im, np.float32)
    h, w = a.shape
    out = np.zeros((frame, frame), np.float32)
    y0 = (frame - h) // 2
    x0 = int(np.clip((frame - w) // 2 + dx, 0, frame - w))
    out[y0:y0 + h, x0:x0 + w] = a
    return out


def build(arm, frames, seed):
    """One encoder per arm. Every one develops on the same frames."""
    if arm in ("rate", "pooled", "relational"):
        v1 = WideV1(n_cells=GLOBAL_CELLS, window_ms=50, rf=7, stride=2,
                    image_shape=(FRAME, FRAME), seed=seed)
        develop_v1(v1, list(frames), epochs=3, tie=True, seed=seed)
        if arm == "rate":
            return lambda f: _unit(v1.rate(f))
        if arm == "relational":
            return lambda f: v1.relational_code(f, n_feat=12)
        idx, n = v1.pooling_index("both")
        return lambda f: v1.pooled_code(v1.drive(f), idx, n)
    eye = LocalSpatialEye(image_shape=(FRAME, FRAME), seed=seed, **ARMS[arm])
    eye.develop(list(frames), epochs=3, seed=seed)
    return eye.code


def codes(coder, frames):
    R = np.array([coder(f) for f in frames], np.float32)
    ad = PopulationAdaptation(R.shape[1])
    return np.array([_unit(ad(r)) for r in R], np.float32)


def cluster_auc(Q, yq, B, yb, rng):
    """Same-category above different-category, from raw codes. No learning."""
    same, diff = [], []
    for k in range(len(Q)):
        ps, pd = np.flatnonzero(yb == yq[k]), np.flatnonzero(yb != yq[k])
        if not len(ps) or not len(pd):
            continue
        same.append(float(Q[k] @ B[int(rng.choice(ps))]))
        diff.append(float(Q[k] @ B[int(rng.choice(pd))]))
    same, diff = np.array(same), np.array(diff)
    return float((same[:, None] > diff[None, :]).mean())


def run_seed(images, y, seed, arms):
    tr, te = split(y, seed)
    cent = [place(im) for im in images]
    out = {}
    for arm in arms:
        coder = build(arm, cent, seed)
        V = codes(coder, cent)
        rng = np.random.default_rng(seed)
        rec = {"dim": int(V.shape[1]),
               "cluster_auc": cluster_auc(V[te], y[te], V[tr], y[tr], rng)}

        # same-object invariance: the SAME photograph, moved
        base = V[te]
        for dx in SHIFTS[1:]:
            S = codes(coder, [place(images[i], dx) for i in te])
            rec[f"self {dx:+d}px"] = float(np.mean(
                [base[k] @ S[k] for k in range(len(te))]))

        # downstream perception: a nearest-prototype read-out, which adds no
        # capacity of its own -- so it cannot rescue a code by learning around it
        protos = np.stack([_unit(V[tr][y[tr] == c].mean(0))
                           for c in np.unique(y[tr])])
        cls = np.unique(y[tr])
        rec["proto centred"] = float(np.mean(
            cls[(V[te] @ protos.T).argmax(1)] == y[te]))
        sh = []
        for dx in SHIFTS[1:]:
            S = codes(coder, [place(images[i], dx) for i in te])
            sh.append(float(np.mean(cls[(S @ protos.T).argmax(1)] == y[te])))
        rec["proto shifted"] = float(np.mean(sh))
        out[arm] = rec
        print(f"    {arm:<28} AUC {rec['cluster_auc']:.3f}  self+5 "
              f"{rec['self +5px']:.3f}  proto {rec['proto centred']:.3f}/"
              f"{rec['proto shifted']:.3f}", flush=True)
    return out


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_pathways.json"
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else list(ARMS)
    images, waves, y, names = load_audiovisual(n_per_class=N_IMAGES // 6,
                                               seed=0, grayscale=True, size=32)
    y = np.asarray(y, int)
    chance = 1.0 / len(names)
    print(f"{len(images)} photographs, {len(names)} categories, chance "
          f"{chance:.3f}; luminance only for every arm")
    print(f"pre-registered bar: cluster AUC > {BEST_EXISTING} + {MIN_GAIN}, "
          f"d >= {MIN_D}, >= {MIN_WINS:.0%} of seeds\n", flush=True)

    rows = []
    for sd in SEEDS:
        print(f"  seed {sd}", flush=True)
        rows.append(run_seed(images, y, sd, only))
    KEYS = ["cluster_auc", "self +5px", "self -5px", "proto centred",
            "proto shifted"]
    res = {"seeds": list(SEEDS), "chance": round(chance, 4),
           "bar": {"beat": BEST_EXISTING, "gain": MIN_GAIN, "d": MIN_D},
           "arms": {}}
    for a in only:
        res["arms"][a] = {k: round(float(np.mean([r[a][k] for r in rows])), 4)
                          for k in KEYS}
        res["arms"][a]["dim"] = int(rows[0][a]["dim"])
        v = res["arms"][a]
        v["self mean"] = round(float(np.mean(
            [v["self +5px"], v["self -5px"]])), 4)

    print(f"\n{'pathway':<28}{'dim':>7}{'cluster AUC':>13}{'self-sim':>10}"
          f"{'proto ctr':>11}{'proto shift':>13}")
    for a in only:
        v = res["arms"][a]
        print(f"{a:<28}{v['dim']:>7}{v['cluster_auc']:>13.3f}"
              f"{v['self mean']:>10.3f}{v['proto centred']:>11.3f}"
              f"{v['proto shifted']:>13.3f}")

    if "local-spatial" not in only or "rate" not in only:
        json.dump(res, open(out_path, "w"), indent=1)
        print(f"\nwrote {out_path}")
        return

    new, old = res["arms"]["local-spatial"], res["arms"]["rate"]
    d = np.array([r["local-spatial"]["cluster_auc"] - r["rate"]["cluster_auc"]
                  for r in rows])
    sd_ = float(d.std(ddof=1))
    cd = None if sd_ < 1e-9 else float(d.mean() / sd_)
    res["vs_rate"] = dict(delta=round(float(d.mean()), 4),
                          cohens_d=None if cd is None else round(cd, 3),
                          wins=int((d > 0).sum()), n=len(d))

    print("\n=== does the new pathway improve the REPRESENTATION? ===")
    print(f"  cluster AUC: rate {old['cluster_auc']:.3f} -> local-spatial "
          f"{new['cluster_auc']:.3f} ({d.mean():+.4f}, "
          f"d={'n/a' if cd is None else f'{cd:+.2f}'}, "
          f"{int((d > 0).sum())}/{len(d)})")
    print(f"  the bar was {BEST_EXISTING} + {MIN_GAIN} = "
          f"{BEST_EXISTING + MIN_GAIN:.3f}")
    # the within-experiment bar: the best arm measured HERE, same input, same
    # frames, same seeds. Registered late but stricter in kind -- it compares
    # only numbers produced under identical conditions.
    rivals = [a for a in ("rate", "pooled", "relational") if a in only]
    best_rival = max(rivals, key=lambda a: res["arms"][a]["cluster_auc"])
    br = res["arms"][best_rival]["cluster_auc"]
    dr = np.array([r["local-spatial"]["cluster_auc"] - r[best_rival]["cluster_auc"]
                   for r in rows])
    sdr = float(dr.std(ddof=1))
    cdr = None if sdr < 1e-9 else float(dr.mean() / sdr)
    res["vs_best_rival"] = dict(arm=best_rival, delta=round(float(dr.mean()), 4),
                                cohens_d=None if cdr is None else round(cdr, 3),
                                wins=int((dr > 0).sum()), n=len(dr))
    passes_abs = (new["cluster_auc"] >= BEST_EXISTING + MIN_GAIN
                  and cd is not None and cd >= MIN_D
                  and (d > 0).sum() >= MIN_WINS * len(d))
    passes_rel = (dr.mean() >= MIN_GAIN and cdr is not None and cdr >= MIN_D
                  and (dr > 0).sum() >= MIN_WINS * len(dr))
    passes = bool(passes_abs and passes_rel)
    res["H2_passes_absolute_bar"] = bool(passes_abs)
    res["H2_passes_within_experiment"] = bool(passes_rel)
    res["H2_passes"] = passes
    print(f"\n  the absolute bar ({BEST_EXISTING} + {MIN_GAIN}) was set from a "
          f"COLOUR run; every arm here is luminance-only and `rate` scores "
          f"{old['cluster_auc']:.3f}, so that bar")
    print(f"  compares across input conditions. Both are reported: the "
          f"registered one, and the best arm measured HERE.")
    print(f"  within this experiment, best rival is {best_rival} at {br:.3f}; "
          f"local-spatial is {dr.mean():+.4f} on it "
          f"(d={'n/a' if cdr is None else f'{cdr:+.2f}'}, "
          f"{int((dr > 0).sum())}/{len(dr)})")
    res["H1_passes"] = bool(new["self mean"] >= 0.95)
    print(f"\n  H1 invariance  self-similarity under +-5px "
          f"{new['self mean']:.3f} (bar 0.95): "
          f"{'MET' if res['H1_passes'] else 'not met'}")
    print(f"  H2 clustering  absolute bar "
          f"{'MET' if passes_abs else 'NOT MET'}"
          f"; within-experiment bar "
          f"{'MET' if passes_rel else 'NOT MET'}")
    if not passes:
        print("\n  REJECTED by the pre-registered criterion. The pathway is "
              "invariant and does not cluster better, which is the same "
              "negative pooling and relational_code")
        print("  already produced. Requirement 11 says an architecture that "
              "only changes downstream learning is to be rejected, and this "
              "one does not even do that.")
    else:
        print("\n  The representation itself is better, measured with no "
              "learning anywhere in the metric.")
    for a in NEW:
        if a in only and a != "local-spatial":
            v = res["arms"][a]
            print(f"    {a:<28} AUC {v['cluster_auc']:.3f} "
                  f"({v['cluster_auc'] - new['cluster_auc']:+.3f}), "
                  f"self-sim {v['self mean']:.3f}")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
