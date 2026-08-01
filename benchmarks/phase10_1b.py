"""Phase 10.1 step 5, measured directly: does alignment break V1's
single-image binding?

Every previous test of Phase 10.1 measured a **downstream** quantity -- cluster
AUC, prototype accuracy, concept-waking -- and all of them came back negative
with a misleading teacher scoring as well as a correct one. But the phase's own
step 5 is a claim about V1 itself: *this should break V1's single-image
binding*. That was never measured. A mechanism can do exactly what it was
designed to do and still not produce the benefit predicted from it, and those
are different failures worth telling apart.

What "single-image bound" means, as a number
--------------------------------------------
A filter is bound to one image if almost all of the drive it ever produces comes
from that image. Across the training set, take each cell's response vector and
ask what fraction of its total response its single best image accounts for:

    top1_share      max_i r[c, i] / sum_i r[c, i].  At 1/n the cell responds
                    equally to everything; near 1 it is a detector for one
                    photograph and nothing else
    participation   the participation ratio of the same distribution, which is
                    the effective number of images a cell responds to --
                    (sum r)^2 / sum r^2, a standard sparsity measure that does
                    not depend on picking a threshold
    lifetime kurt   excess kurtosis of each cell's response across images, the
                    classical measure of lifetime sparseness in visual cortex

All three say the same thing from different angles, which is the point: a claim
this load-bearing should not rest on one statistic's arbitrary choices.

The arms are Phase 10.1's, and the two controls are what make the answer
readable. If a **misleading** teacher breaks single-image binding as much as the
correct one, then the rule loosens filters by perturbing them and the concept
layer is not what did it.

Usage:  python3 benchmarks/phase10_1b.py out_phase10_1b.json
"""
import json
import sys

import numpy as np

from neurobrain.cognition.multimodal import AssociationArea, _unit
from neurobrain.learning.selforganize import (align_v1,
                                              align_v1_reconstructive,
                                              develop_v1)
from neurobrain.sensing.natural import load_audiovisual
from neurobrain.sensing.streams import StreamingBrain
from neurobrain.vision.widev1 import PopulationAdaptation

sys.path.insert(0, "benchmarks")
from real_binding import opponent, split                     # noqa: E402

SEEDS = (0, 1, 2, 3)
N_CONCEPT = 256
KEEP = 0.6
EPOCHS = 2
LR_DELTA, LR_REC = 0.02, 8.0
ARMS = ("no align", "align to concept", "reconstruct concept",
        "reconstruct own", "reconstruct wrong")


def responses(v1, images):
    """(n_cells, n_images) -- what every cell does to every photograph."""
    return np.array([v1.rate(im) for im in images], np.float32).T


def binding_stats(R):
    """How tied each cell is to a single image, three ways."""
    R = np.maximum(R, 0.0)
    tot = R.sum(1)
    live = tot > 1e-9
    R, tot = R[live], tot[live]
    n = R.shape[1]
    top1 = (R.max(1) / tot)
    pr = (tot ** 2) / np.maximum((R ** 2).sum(1), 1e-12)
    mu = R.mean(1, keepdims=True)
    sd = R.std(1, keepdims=True) + 1e-9
    kurt = (((R - mu) / sd) ** 4).mean(1) - 3.0
    return dict(top1_share=float(top1.mean()),
                participation=float(pr.mean()),
                lifetime_kurtosis=float(kurt.mean()),
                chance_top1=1.0 / n, n_images=n,
                live_cells=int(live.sum()))


def run_seed(images, waves, y, seed):
    tr, te = split(y, seed)
    lum = np.array([opponent(im)[0] for im in images], np.float32)
    base = StreamingBrain(seed=seed, image_shape=(32, 32), v1_cells=4096,
                          rf=7, stride=2)
    develop_v1(base.v1, list(lum), epochs=3, seed=seed)
    R = np.array([np.concatenate([base.v1.rate(c) for c in opponent(im)])
                  for im in images], np.float32)
    ad = PopulationAdaptation(R.shape[1])
    V = np.array([_unit(ad(r)) for r in R], np.float32)
    A = np.array([base.belt.code(base.ear.coch.forward(w)[0]) for w in waves],
                 np.float32)

    assoc = AssociationArea(n_vis=V.shape[1], n_aud=A.shape[1],
                            n_concept=N_CONCEPT, seed=seed)
    assoc.set_stats(V[tr], A[tr])
    owner = {}
    for i in tr:
        owner.setdefault(assoc.bind(V[i], A[i]), []).append(int(i))
    merged = assoc.consolidate_ranked(keep=KEEP)
    for old, new in merged.items():
        while new in merged:
            new = merged[new]
        owner.setdefault(new, []).extend(owner.pop(old, []))
    cmean = {c: R[m].mean(0) for c, m in owner.items() if m}
    cimg = {c: lum[m].mean(0) for c, m in owner.items() if m}
    cells = sorted(cmean)
    rng = np.random.default_rng(seed + 5)

    tgt, timg = {}, {}
    for i in tr:
        c = assoc.concept_from_vision(V[i])
        c = c if c in cmean else cells[0]
        other = int(rng.choice([d for d in cells if d != c] or cells))
        tgt.setdefault("align to concept", []).append(cmean[c])
        timg.setdefault("reconstruct concept", []).append(cimg[c])
        timg.setdefault("reconstruct own", []).append(lum[i])
        timg.setdefault("reconstruct wrong", []).append(cimg[other])

    lum_tr = [lum[i] for i in tr]
    out = {"no align": binding_stats(responses(base.v1, lum))}
    for arm in ARMS[1:]:
        b = StreamingBrain(seed=seed, image_shape=(32, 32), v1_cells=4096,
                           rf=7, stride=2)
        develop_v1(b.v1, list(lum), epochs=3, seed=seed)
        if arm in timg:
            align_v1_reconstructive(b.v1, lum_tr, timg[arm], epochs=EPOCHS,
                                    lr=LR_REC)
        else:
            align_v1(b.v1, lum_tr, [t[:b.v1.n_cells] for t in tgt[arm]],
                     epochs=EPOCHS, lr=LR_DELTA)
        out[arm] = binding_stats(responses(b.v1, lum))
    return out


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_phase10_1b.json"
    images, waves, y, names = load_audiovisual(n_per_class=60, seed=0,
                                               grayscale=False, size=32)
    y = np.asarray(y, int)
    print(f"{len(images)} real photographs; measuring how tied V1 filters are "
          f"to single images\n", flush=True)

    rows = [run_seed(images, waves, y, sd) for sd in SEEDS]
    METRICS = ("top1_share", "participation", "lifetime_kurtosis")
    res = {"seeds": list(SEEDS),
           "chance_top1": round(float(np.mean(
               [r["no align"]["chance_top1"] for r in rows])), 5),
           "n_images": int(rows[0]["no align"]["n_images"]), "arms": {}}
    for a in ARMS:
        res["arms"][a] = {m: round(float(np.mean([r[a][m] for r in rows])), 4)
                          for m in METRICS}

    print(f"{'arm':<24}{'top1 share':>12}{'participation':>15}"
          f"{'lifetime kurt':>15}")
    for a in ARMS:
        v = res["arms"][a]
        print(f"{a:<24}{v['top1_share']:>12.4f}{v['participation']:>15.2f}"
              f"{v['lifetime_kurtosis']:>15.2f}")
    print(f"\n  ({res['n_images']} photographs; a cell responding equally to "
          f"all of them would score top1 {res['chance_top1']:.4f}, "
          f"participation {res['n_images']}, kurtosis -1.2 or so)")

    base = res["arms"]["no align"]
    print(f"\nagainst 'no align', paired over {len(SEEDS)} seeds")
    gate = {}
    for a in ARMS[1:]:
        cs = []
        for m in METRICS:
            d = np.array([r[a][m] - r["no align"][m] for r in rows])
            sd = float(d.std(ddof=1))
            cd = None if sd < 1e-9 else float(d.mean() / sd)
            gate.setdefault(a, {})[m] = dict(
                delta=round(float(d.mean()), 4),
                cohens_d=None if cd is None else round(cd, 3),
                wins=int((d > 0).sum()), n=len(d))
            cs.append(f"{d.mean():+.4f} " +
                      ("d=n/a" if cd is None else f"d={cd:+.1f}"))
        print(f"{a:<24}" + "".join(f"{c:>16}" for c in cs))
    res["gate"] = gate

    print("\n=== phase 10.1 step 5: is single-image binding broken? ===")
    print(f"  before: a filter's best photograph accounts for "
          f"{base['top1_share']:.1%} of everything it ever does, and it "
          f"responds to {base['participation']:.1f} of "
          f"{res['n_images']} photographs")
    conc = res["arms"]["reconstruct concept"]
    wrong = res["arms"]["reconstruct wrong"]
    own = res["arms"]["reconstruct own"]
    print(f"  after (concept teacher): {conc['top1_share']:.1%}, "
          f"{conc['participation']:.1f} photographs")
    print(f"  after (MISLEADING)     : {wrong['top1_share']:.1%}, "
          f"{wrong['participation']:.1f} photographs")
    print(f"  after (uninformative)  : {own['top1_share']:.1%}, "
          f"{own['participation']:.1f} photographs")

    broke = conc["participation"] > base["participation"] * 1.1
    teacher = (conc["participation"] > wrong["participation"] * 1.05
               and conc["participation"] > own["participation"] * 1.05)
    res["binding_broken"] = bool(broke)
    res["teacher_did_it"] = bool(teacher)
    if broke and teacher:
        print("\n  Step 5 is met AND the concept layer is what met it: filters "
              "respond to more photographs after alignment, and a misleading "
              "teacher does not do the same.")
    elif broke:
        print(f"\n  Step 5 is met -- filters do become less tied to single "
              f"photographs ({base['participation']:.1f} -> "
              f"{conc['participation']:.1f}) -- but a MISLEADING teacher does "
              f"it too ({wrong['participation']:.1f}).")
        print("  So the rule loosens filters by perturbing them, not by "
              "carrying the concept layer's opinion. The mechanism does what "
              "it was designed to do; the design's premise is what fails.")
    else:
        print(f"\n  Step 5 is not met: filters are no less tied to single "
              f"photographs after alignment "
              f"({base['participation']:.1f} -> {conc['participation']:.1f}).")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
