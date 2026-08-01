"""What actually limits the object-centred frame — measured, not argued.

`pathways.py` rejected `LocalSpatialEye` as specified (H1 0.931 against a 0.95
bar, H2 +0.015 against +0.02) and `pathway_downstream.py` confirmed the
rejection is not rescued by anything built on top of it (H9). The architecture
question is closed. What is left is the one thing the failure branch of a
scientific loop still owes: **the limiting factor**, stated as a measurement.

EVALUATION.md §9.6 currently answers it by argument:

> The centroid is estimated per image from contrast, and that estimate is
> stable when the *same* photograph moves and inconsistent *between*
> photographs — so it cancels translation and adds between-image jitter.

That is a mechanism, and no number in this project tests it.

Hypotheses, before the run
--------------------------
**H10 (the estimator is the limiting factor).** If between-image centroid
inconsistency is what costs clustering, then replacing the estimated centroid
with the object's **true** position -- which this benchmark knows, because it
placed the object itself -- should recover most of the 0.043 that removing the
object frame recovered, *while keeping* the invariance the object frame buys.
The oracle arm should land near `local-absolute` on cluster AUC and near
`local-spatial` on self-similarity: best of both.

**H10 is falsified if the oracle does not move clustering.** Then the estimator
is innocent, the object-centred frame costs clustering for some other reason,
and the limiting factor is the frame itself rather than how it is measured.

**H11 (the jitter is between images, not within one).** The mechanism claims two
things at once, and they are separately measurable: the centroid should track a
*shifted copy of one photograph* well (small within-image error) and scatter
*across photographs of one category* (large between-image spread). If both are
large the estimator is simply bad; if both are small the mechanism is not
present at all and H10 cannot be true for the stated reason.

A note on the first version of this file, which was vacuous
-----------------------------------------------------------
It placed every photograph at the frame centre. The object's true centre is
then the frame centre, so the "oracle" object-centred grid is laid out exactly
where the frame-absolute grid already is, and the two arms are the *same
computation*. It duly reported the oracle recovering **100.0%** of the gap,
with all four seeds identical to three decimals in both arms (0.661/0.661,
0.573/0.573, 0.651/0.651, 0.633/0.633). An oracle that never has to locate
anything cannot test a locator. Objects are now displaced per image on both
axes, and an assertion fails if the oracle origin does not vary.

The oracle is an **instrument, not a proposal**. Nothing in the mind can supply
the true object position; an arm that uses it is not an architecture and is not
reported as one. It exists to localise the fault.

Usage:  python3 benchmarks/limiting_factor.py out_limiting_factor.json
"""
import json
import sys

import numpy as np

from neurobrain.cognition.multimodal import _unit
from neurobrain.vision.localspatial import LocalSpatialEye
from neurobrain.vision.widev1 import PopulationAdaptation

sys.path.insert(0, "benchmarks")
from pathways import FRAME, cluster_auc, place                # noqa: E402
from real_binding import split                                # noqa: E402
from neurobrain.sensing.natural import load_audiovisual       # noqa: E402

SEEDS = (0, 1, 2, 3)
N_IMAGES = 240
SHIFT = 5
OBJ = 32                       # the photographs are 32 px in a 48 px frame
#: How far each photograph is displaced from centre, drawn per image. Without
#: this the whole benchmark is vacuous, and the first version of it was: with
#: every object at dx=0 the object's true centre IS the frame centre, so the
#: "oracle" object-centred grid and the frame-absolute grid are the same
#: computation. That run reported the oracle recovering 100% of the gap, and
#: the four seeds returned cluster AUC identical to three decimals in both
#: arms -- 0.661/0.661, 0.573/0.573, 0.651/0.651, 0.633/0.633 -- which is what
#: identity by construction looks like, not what a perfect estimator looks
#: like. An oracle that never has to locate anything tests nothing.
JITTER = 8                     # dx, dy each drawn from [-JITTER, +JITTER]


def place2(im, dx=0, dy=0, frame=FRAME):
    """`pathways.place`, but able to move the object on both axes.

    Position has to vary in two dimensions or the estimator is only ever asked
    half the question it is accused of failing.
    """
    a = np.asarray(im, np.float32)
    h, w = a.shape
    out = np.zeros((frame, frame), np.float32)
    y0 = int(np.clip((frame - h) // 2 + dy, 0, frame - h))
    x0 = int(np.clip((frame - w) // 2 + dx, 0, frame - w))
    out[y0:y0 + h, x0:x0 + w] = a
    return out


def true_centre(dx=0, dy=0, frame=FRAME, obj=OBJ):
    """Where `place2` actually put the object. The oracle, and it is exact."""
    y0 = int(np.clip((frame - obj) // 2 + dy, 0, frame - obj))
    x0 = int(np.clip((frame - obj) // 2 + dx, 0, frame - obj))
    return np.array([y0 + obj / 2.0, x0 + obj / 2.0], np.float32)


def build(seed, frames):
    """One developed bank, shared by every arm.

    The arms differ only in what origin the code is laid out from, so they must
    not differ in their filters as well -- `pathways._bank_key` makes the same
    argument for the same reason.
    """
    eye = LocalSpatialEye(image_shape=(FRAME, FRAME), seed=seed, spiking=False)
    eye.develop(list(frames), epochs=3, seed=seed)
    flat = LocalSpatialEye(image_shape=(FRAME, FRAME), seed=seed, spiking=False,
                           centred=False)
    flat.banks = eye.banks
    return eye, flat


def codes(raw):
    ad = PopulationAdaptation(raw.shape[1])
    return np.array([_unit(ad(r)) for r in raw], np.float32)


def run_seed(images, y, seed):
    tr, te = split(y, seed)
    # Each photograph gets its own displacement, so "where is the object" is a
    # real question with a different answer per image.
    jit = np.random.default_rng(seed + 7)
    dx = jit.integers(-JITTER, JITTER + 1, size=len(images))
    dy = jit.integers(-JITTER, JITTER + 1, size=len(images))
    frames = [place2(im, int(a), int(b)) for im, a, b in zip(images, dx, dy)]
    shifted = [place2(im, int(a) + SHIFT, int(b))
               for im, a, b in zip(images, dx, dy)]
    o0 = np.array([true_centre(int(a), int(b)) for a, b in zip(dx, dy)],
                  np.float32)
    oS = np.array([true_centre(int(a) + SHIFT, int(b))
                   for a, b in zip(dx, dy)], np.float32)
    # The guard the first version lacked: if the oracle origin never varies it
    # is the frame centre, the object-frame grid collapses onto the absolute
    # one, and the comparison is an identity rather than a measurement.
    assert o0.std(0).max() > 1.0, (
        "oracle origin does not vary across images -- the arms are the same "
        "computation and this benchmark tests nothing")
    eye, flat = build(seed, frames)

    arms = {
        # the specification: origin estimated from image contrast
        "local-spatial": (lambda i: eye.code(frames[i]),
                          lambda i: eye.code(shifted[i])),
        # the ablation that won in pathways.py: no object frame at all
        "local-absolute": (lambda i: flat.code(frames[i]),
                           lambda i: flat.code(shifted[i])),
        # the instrument: the object frame with a PERFECT origin
        "oracle-centroid": (lambda i: eye.code(frames[i], origin=o0[i]),
                            lambda i: eye.code(shifted[i], origin=oS[i])),
    }
    out = {}
    for name, (fc, sc) in arms.items():
        V = codes(np.array([fc(i) for i in range(len(frames))], np.float32))
        S = codes(np.array([sc(i) for i in range(len(frames))], np.float32))
        rng = np.random.default_rng(seed)
        out[name] = {
            "cluster_auc": cluster_auc(V[te], y[te], V[tr], y[tr], rng),
            "self_shift": float(np.mean([float(V[i] @ S[i])
                                         for i in range(len(V))])),
        }
    if (out["oracle-centroid"]["cluster_auc"]
            == out["local-absolute"]["cluster_auc"]):
        print("    WARNING: oracle and absolute arms scored identically -- "
              "check they are not the same computation", flush=True)

    # ---- H11: where does the estimator's error actually live? --------------
    est0 = np.array([eye.image_centroid(f) for f in frames], np.float32)
    estS = np.array([eye.image_centroid(s) for s in shifted], np.float32)
    # within-image: the same photograph moved SHIFT px to the right. A perfect
    # estimator moves by exactly (0, SHIFT).
    moved = estS - est0
    within = float(np.mean(np.linalg.norm(
        moved - np.array([0.0, float(SHIFT)], np.float32), axis=1)))
    # between-image: spread of the estimate across photographs of one category,
    # which is the quantity the mechanism blames
    # between-image: how far the estimate lands from the object's TRUE centre,
    # which is the error that matters now that the objects are in different
    # places. Measured per category so it is the spread the clustering sees.
    err = est0 - o0
    between = float(np.mean([
        np.linalg.norm(err[y == c] - err[y == c].mean(0), axis=1).mean()
        for c in np.unique(y)]))
    # and the absolute error against the position the object really occupies
    bias = float(np.mean(np.linalg.norm(err, axis=1)))
    out["_centroid"] = {"within_image_error_px": within,
                        "between_image_spread_px": between,
                        "bias_vs_truth_px": bias}
    return out


def main():
    out_path = (sys.argv[1] if len(sys.argv) > 1
                else "out_limiting_factor.json")
    images, _waves, y, names = load_audiovisual(n_per_class=N_IMAGES // 6,
                                                seed=0, grayscale=True, size=32)
    y = np.asarray(y, int)
    print(f"{len(images)} photographs, {len(names)} categories, "
          f"{FRAME}px frame, {SHIFT}px shift\n", flush=True)

    rows = []
    for sd in SEEDS:
        print(f"  seed {sd}", flush=True)
        r = run_seed(images, y, sd)
        rows.append(r)
        for k in ("local-spatial", "local-absolute", "oracle-centroid"):
            print(f"    {k:<18} AUC {r[k]['cluster_auc']:.3f}   "
                  f"self+{SHIFT} {r[k]['self_shift']:.3f}", flush=True)
        c = r["_centroid"]
        print(f"    centroid: within-image {c['within_image_error_px']:.2f}px  "
              f"between-image {c['between_image_spread_px']:.2f}px  "
              f"bias {c['bias_vs_truth_px']:.2f}px", flush=True)

    ARMS = ("local-spatial", "local-absolute", "oracle-centroid")
    res = {"seeds": list(SEEDS), "per_seed": rows, "arms": {}}
    for a in ARMS:
        res["arms"][a] = {
            m: round(float(np.mean([r[a][m] for r in rows])), 4)
            for m in ("cluster_auc", "self_shift")}
    res["centroid"] = {k: round(float(np.mean([r["_centroid"][k]
                                               for r in rows])), 3)
                       for k in rows[0]["_centroid"]}

    print(f"\n{'arm':<18}{'cluster AUC':>13}{'self+5':>10}")
    for a in ARMS:
        v = res["arms"][a]
        print(f"{a:<18}{v['cluster_auc']:>13.3f}{v['self_shift']:>10.3f}")

    print(f"\n--- H11: where is the estimator's error? ---")
    c = res["centroid"]
    print(f"  within-image  (same photo, shifted {SHIFT}px) {c['within_image_error_px']:>7.2f} px")
    print(f"  between-image (across one category)          {c['between_image_spread_px']:>7.2f} px")
    print(f"  bias against the true object centre          {c['bias_vs_truth_px']:>7.2f} px")
    res["H11_jitter_is_between_images"] = bool(
        c["between_image_spread_px"] > 2 * c["within_image_error_px"])
    print(f"  between-image spread > 2x within-image error: "
          f"{res['H11_jitter_is_between_images']}")

    print(f"\n--- H10: is the ESTIMATOR the limiting factor? ---")
    spec = np.array([r["local-spatial"]["cluster_auc"] for r in rows])
    absl = np.array([r["local-absolute"]["cluster_auc"] for r in rows])
    orac = np.array([r["oracle-centroid"]["cluster_auc"] for r in rows])
    gap = absl - spec                     # what removing the frame recovers
    got = orac - spec                     # what a perfect origin recovers
    frac = float(got.mean() / gap.mean()) if abs(gap.mean()) > 1e-9 else 0.0
    print(f"  removing the object frame recovers  {gap.mean():+.4f}")
    print(f"  a PERFECT origin recovers           {got.mean():+.4f} "
          f"(d={got.mean()/(got.std(ddof=1)+1e-12):+.2f}, "
          f"{int((got > 0).sum())}/{len(got)})")
    print(f"  fraction of the gap the estimator explains: {frac:.1%}")
    # The same bar `ear_config.py` uses: an effect has to explain a quarter of
    # the gap it was proposed to explain, not merely reach significance.
    res["H10_estimator_explains"] = round(frac, 4)
    res["H10_holds"] = bool(frac >= 0.25 and (got > 0).sum() >= 3)
    print(f"  H10 holds (>=25% of the gap, >=3/4 seeds): {res['H10_holds']}")

    inv = np.array([r["oracle-centroid"]["self_shift"] for r in rows])
    inv_s = np.array([r["local-spatial"]["self_shift"] for r in rows])
    print(f"\n  invariance kept by the oracle: {inv.mean():.3f} "
          f"against the estimator's {inv_s.mean():.3f}")

    if res["H10_holds"]:
        print("\n  The estimator IS the limiting factor: with the origin "
              "handed to it, the object-centred frame recovers most of what\n"
              "  removing it recovered. The architecture's idea is not what "
              "failed -- locating the object without a label is.")
    else:
        print("\n  H10 is FALSIFIED. A perfect origin does not buy the "
              "clustering back, so the centroid estimator is not what limits\n"
              "  this pathway. The cost is in the object-centred frame itself, "
              "not in how the centre is measured, and EVALUATION.md's\n"
              "  stated mechanism is wrong.")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
