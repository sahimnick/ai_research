"""Does invariance come from TIME rather than from geometry?

Eleven interventions moved this project's visual cluster AUC across 0.540-0.630
while the ear sat at 0.788, and `limiting_factor.py` localised why: the
object-centred frame's origin estimate carries an error correlated with image
content, and a contrast-weighted centroid cannot avoid that because contrast IS
image content. Every one of those interventions was geometric.

[Halvagal & Zenke, Nat Neuro 2023] report that Hebbian plasticity **alone fails
to produce invariant object representations** -- this project's own measured
result, arrived at independently -- and their fix is not a better estimator but
a predictive term over *time*. Everything this project has ever trained a filter
bank on was a static, independent image.

Hypotheses, fixed before the run
--------------------------------
**H14 (invariance comes from temporal continuity).** V1 filters grown with a
temporal-predictive term, on sequences of consecutive views of one moving
object, will produce a code whose cluster AUC exceeds the band eleven static
interventions could not.

**The bar, registered in advance** and matching `pathways.py` exactly
(luminance, `drive`, `cluster_auc` from raw codes with no learning, 4 seeds):

* beat the best existing arm -- `local-absolute` at **0.630** -- by **>= 0.02**
* Cohen's **d >= 0.8**, winning on **>= 75%** of seeds
* the code's participation ratio **>= 50%** of the static rule's, so a "win"
  cannot be bought by collapsing the representation

**H14 is falsified if `shuffled-time` matches it.** That arm is the whole
experiment: identical frames, identical count, identical rule, but the frames
are pooled and dealt out again so that "consecutive" no longer means "the same
object" (`sequences.shuffle_across`, which explains why shuffling *within* a
sequence would not do). If invariance survives that, the gain is data
augmentation and temporal continuity is not what produced it.

**H15 (both extra terms are load-bearing).** Already measured while building the
rule: without the decorrelation term the bank collapses to effective dimension
**0.00** with every filter pair identical. The ablations are here so that shows
up in the table rather than only in a commit message.

Usage:  python3 benchmarks/temporal_invariance.py out_temporal.json
"""
import json
import sys

import numpy as np

from neurobrain.cognition.multimodal import _unit
from neurobrain.learning.selforganize import develop_v1, develop_v1_temporal
from neurobrain.sensing.natural import load_audiovisual
from neurobrain.sensing.sequences import (object_sequences, sequence_stats,
                                          shuffle_across)
from neurobrain.sensing.streams import MovingScene
from neurobrain.tools.mapdebug import participation_ratio
from neurobrain.vision.widev1 import PopulationAdaptation, WideV1

sys.path.insert(0, "benchmarks")
from pathways import FRAME, cluster_auc, place                 # noqa: E402
from real_binding import split                                 # noqa: E402

SEEDS = (0, 1, 2, 3)
N_IMAGES = 240
N_SEQ, LEN_SEQ = 40, 5
SPIN, ZOOM, SPEED = 0.18, 0.03, 3.0
SHIFT = 5
CELLS = 4096
EPOCHS = 3
#: pre-registered, from the docstring above
BEST_EXISTING, MIN_GAIN, MIN_D, MIN_WINS, MIN_PR = 0.630, 0.02, 0.8, 0.75, 0.5

ARMS = ("static", "temporal", "shuffled-time",
        "temporal: no-predictive", "temporal: no-decorrelation")


def build_sequences(images, seed):
    """One moving world per seed, and the frames every arm shares."""
    scene = MovingScene(images, np.zeros(len(images), int), size=3 * FRAME,
                        n_objects=6, speed=SPEED, spin=SPIN, zoom=ZOOM,
                        seed=seed)
    return object_sequences(scene, n_seq=N_SEQ, len_seq=LEN_SEQ, crop=FRAME,
                            seed=seed)


def grow(arm, seqs, seed):
    v1 = WideV1(n_cells=CELLS, window_ms=50, rf=7, stride=2,
                image_shape=(FRAME, FRAME), seed=seed)
    if arm == "static":
        # the same frames, as independent images -- the current architecture
        frames = [f for s in seqs for f in s]
        develop_v1(v1, frames, epochs=EPOCHS, tie=True, seed=seed)
    elif arm == "shuffled-time":
        develop_v1_temporal(v1, shuffle_across(seqs, seed=seed),
                            epochs=EPOCHS, seed=seed)
    else:
        develop_v1_temporal(v1, seqs, epochs=EPOCHS, seed=seed,
                            predictive=(arm != "temporal: no-predictive"),
                            decorrelate=(arm != "temporal: no-decorrelation"))
    return v1


def run_seed(images, y, seed):
    tr, te = split(y, seed)
    seqs = build_sequences(images, seed)
    st = sequence_stats(seqs)
    # The sequences have to actually move, or the predictive term is being
    # credited with something that never happened -- the failure that flattered
    # `phase10_2.py` and `limiting_factor.py`.
    assert st["adjacent_cos"] < 0.999, (
        f"sequences barely change (adjacent cos {st['adjacent_cos']:.4f}); "
        f"the predictive term has nothing to pull together")
    frames = [place(im) for im in images]

    def enc(v1, ims):
        R = np.array([v1.drive(f) for f in ims], np.float32)
        ad = PopulationAdaptation(R.shape[1])
        return np.array([_unit(ad(r)) for r in R], np.float32)

    out = {"_seq": st}
    for arm in ARMS:
        v1 = grow(arm, seqs, seed)
        V = enc(v1, frames)
        # Same protocol as `pathways.py`: the SAME photographs moved, measured
        # on the test split only and averaged over both directions. The bar
        # this benchmark is judged against is that benchmark's number, so the
        # measurement has to be the same measurement.
        shifted = [enc(v1, [place(images[i], dx) for i in te])
                   for dx in (SHIFT, -SHIFT)]
        base = V[te]
        rng = np.random.default_rng(seed)
        out[arm] = {
            "cluster_auc": cluster_auc(V[te], y[te], V[tr], y[tr], rng),
            "self_shift": float(np.mean(
                [np.mean([base[k] @ S[k] for k in range(len(te))])
                 for S in shifted])),
            "code_pr": float(participation_ratio(V)),
            "bank_pr": float(participation_ratio(
                v1.Wt[:9 * v1.n_pos].reshape(-1, v1.n_pos,
                                             v1.Wt.shape[1])[:, 0, :])),
        }
    return out


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_temporal.json"
    images, _w, y, names = load_audiovisual(n_per_class=N_IMAGES // 6, seed=0,
                                            grayscale=True, size=32)
    y = np.asarray(y, int)
    print(f"{len(images)} photographs, {len(names)} categories, {FRAME}px frame")
    print(f"pre-registered bar: cluster AUC > {BEST_EXISTING} + {MIN_GAIN}, "
          f"d >= {MIN_D}, >= {MIN_WINS:.0%} of seeds, code PR >= "
          f"{MIN_PR:.0%} of static's\n", flush=True)

    rows = []
    for sd in SEEDS:
        print(f"  seed {sd}", flush=True)
        r = run_seed(images, y, sd)
        rows.append(r)
        s = r["_seq"]
        print(f"    sequences: adjacent cos {s['adjacent_cos']:.3f}, "
              f"endpoints {s['endpoints_cos']:.3f}, "
              f"across objects {s['across_objects_cos']:.3f}", flush=True)
        for a in ARMS:
            v = r[a]
            print(f"    {a:<28} AUC {v['cluster_auc']:.3f}  self+{SHIFT} "
                  f"{v['self_shift']:.3f}  code PR {v['code_pr']:.1f}",
                  flush=True)

    res = {"seeds": list(SEEDS), "per_seed": rows, "arms": {}}
    for a in ARMS:
        res["arms"][a] = {k: round(float(np.mean([r[a][k] for r in rows])), 4)
                          for k in rows[0][a]}

    print(f"\n{'arm':<28}{'cluster AUC':>12}{'self+5':>9}{'code PR':>10}"
          f"{'bank PR':>9}")
    for a in ARMS:
        v = res["arms"][a]
        print(f"{a:<28}{v['cluster_auc']:>12.3f}{v['self_shift']:>9.3f}"
              f"{v['code_pr']:>10.1f}{v['bank_pr']:>9.2f}")

    def col(a):
        return np.array([r[a]["cluster_auc"] for r in rows])

    def stat(g):
        d = (float(g.mean() / (g.std(ddof=1) + 1e-12)) if len(g) > 1
             else float("nan"))
        return float(g.mean()), d, int((g > 0).sum()), len(g)

    # identical scores mean two arms are the same computation -- that signature
    # caught three broken comparisons in this project already
    for i, a in enumerate(ARMS):
        for b in ARMS[i + 1:]:
            if res["arms"][a]["cluster_auc"] == res["arms"][b]["cluster_auc"]:
                print(f"\n  WARNING: {a!r} and {b!r} scored identically -- "
                      f"check they are not the same computation")

    print(f"\n--- H14: does temporal continuity beat the static rule? ---")
    tmp, sta, shf = col("temporal"), col("static"), col("shuffled-time")
    for nm, g in (("temporal - static", tmp - sta),
                  ("temporal - shuffled-time", tmp - shf)):
        m, d, w, n = stat(g)
        res.setdefault("H14_contrasts", {})[nm] = {
            "delta": round(m, 4), "cohens_d": round(d, 2), "wins": w, "n": n}
        print(f"  {nm:<26}{m:>+8.4f}  d={d:>+5.2f}  {w}/{n}")

    pr_ok = (res["arms"]["temporal"]["code_pr"]
             >= MIN_PR * res["arms"]["static"]["code_pr"])
    abs_gain = res["arms"]["temporal"]["cluster_auc"] - BEST_EXISTING
    m, d, w, n = stat(tmp - sta)
    passes = bool(abs_gain >= MIN_GAIN and d >= MIN_D
                  and w >= MIN_WINS * n and pr_ok)
    res["H14_passes_registered_bar"] = passes
    res["H14_code_pr_ok"] = bool(pr_ok)
    print(f"\n  against the PRE-REGISTERED bar ({BEST_EXISTING} + {MIN_GAIN}):")
    print(f"    temporal {res['arms']['temporal']['cluster_auc']:.3f}, "
          f"gain over best existing {abs_gain:+.4f}")
    print(f"    code participation ratio "
          f"{res['arms']['temporal']['code_pr']:.1f} against static's "
          f"{res['arms']['static']['code_pr']:.1f}  -> not collapsed: {pr_ok}")
    print(f"    H14 PASSES: {passes}")

    # The control decides the meaning of any gain, so it is judged separately.
    mc, dc, wc, nc = stat(tmp - shf)
    beats_control = bool(mc > 0 and dc >= MIN_D and wc >= MIN_WINS * nc)
    res["H14_beats_shuffled_control"] = beats_control
    print(f"\n  and against the control that decides what a gain MEANS:")
    print(f"    temporal - shuffled-time {mc:+.4f} (d={dc:+.2f}, {wc}/{nc}) "
          f"-> ordering matters: {beats_control}")
    # Two independent questions, and they are stated separately because a
    # single sentence covering both is how a verdict ends up asserting
    # something the numbers do not say.
    ms, _, _, _ = stat(tmp - sta)
    print(f"\n  the two questions, kept apart:")
    print(f"    1. is the temporal rule better than the static one?   "
          f"{ms:+.4f} over static")
    print(f"    2. does it clear the PRE-REGISTERED absolute bar?     "
          f"{passes} ({abs_gain:+.4f} against +{MIN_GAIN})")
    print(f"    3. is any gain due to ORDER rather than to the frames? "
          f"{beats_control}")
    if beats_control and passes:
        print("    -> H14 HOLDS: it clears the registered bar AND the control "
              "says the gain comes from temporal ordering.")
    elif beats_control and not passes:
        print("    -> Ordering demonstrably helps, but the result does not "
              "clear the bar registered in advance.\n       The mechanism is "
              "supported; the architecture is not yet better than the best "
              "existing arm.")
    elif passes and not beats_control:
        print("    -> The bar is cleared but the shuffled control matches it, "
              "so the gain is NOT from temporal continuity.\n       H14's "
              "mechanism is falsified even though its number passed -- this is "
              "data augmentation.")
    else:
        print("    -> H14 is FALSIFIED: it neither clears the registered bar "
              "nor beats its own shuffled control.")

    print(f"\n--- H15: are both extra terms load-bearing? ---")
    for a in ("temporal: no-predictive", "temporal: no-decorrelation"):
        v = res["arms"][a]
        print(f"  {a:<28} AUC {v['cluster_auc']:.3f}  bank PR "
              f"{v['bank_pr']:.2f}  (full rule {res['arms']['temporal']['bank_pr']:.2f})")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
