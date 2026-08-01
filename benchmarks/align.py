"""Phase 10: can the concept layer teach the eye, with no labels?

The eye is this project's critical path. `constraint.py` proved it with an
oracle: the same Hebbian judge reaches AUC 0.879 given a visual code that
clusters and 0.533 given the code the eye actually produces. Eleven mechanisms
have been tried on the eye and none moved it, and every one of them worked
*inside* vision -- wider banks, bigger apertures, second stages, whitening,
pooling. Nothing above V1 has ever spoken back to it.

Something above it now has an opinion worth hearing. Once
`consolidate_ranked` makes the concept layer generalise (singletons 53% -> 0%),
each concept holds the **average of everything that turned out to be the same
thing** -- and that average is a teaching signal that costs no labels. It is the
mind's own view of what it is looking at.

`align_v1` moves each filter by the local delta rule

    dWt[c] = lr * (target[c] - rate[c]) * patch[position of c]

two factors both present at the synapse, no gradient propagated anywhere. A cell
that already predicts its concept's expectation does not move, which is exactly
what `develop_v1`'s purely Hebbian rule cannot do.

The controls, which are the whole experiment
--------------------------------------------
Alignment is extra plasticity, and extra plasticity changes things. Three arms
share the identical schedule, learning rate, and number of updates, and differ
only in **what the target is**:

    no align              the eye as `develop_v1` leaves it
    align to concept      target = the mean raw rate of the concept's members.
                          The proposal
    align to own rate     target = the image's own rate. Same rule, same
                          budget, teacher carries no information beyond the
                          image itself -- so a gain here is plasticity, not
                          teaching
    align to wrong concept  target = a randomly chosen *other* concept's mean.
                          Same rule, same budget, teacher actively misleading

If "align to concept" does not beat both, the concept layer is not teaching
anything and the result is about the extra updates.

Phase 10.2, the translation test
--------------------------------
Everything is measured twice: centred, and with the object shifted +-5 px. The
prediction on the table is that alignment buys position tolerance, because a
concept mean is an average over members that sat in different places. Measured
here rather than assumed, with the shift applied only at test time so the shift
cannot be learned.

Measured, all held out:

    cluster AUC   same-category similarity above different-category. THE
                  number -- it is what `constraint.py` showed everything
                  downstream is gated on
    prototype     class-mean read-out
    1-NN          nearest stored photograph
    shifted       the same three, +-5 px

Usage:  python3 benchmarks/align.py out_align.json
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
from neurobrain.vision.widev1 import PopulationAdaptation, _nearest_prototype

sys.path.insert(0, "benchmarks")
from real_binding import opponent, split                     # noqa: E402

SEEDS = (0, 1, 2, 3)
N_CONCEPT = 256
KEEP = 0.6
EPOCHS, LR = 2, 0.02
SHIFT = 5
ARMS = ("no align", "align to concept", "align to own rate",
        "align to wrong concept", "reconstruct concept",
        "reconstruct own", "reconstruct wrong")


def shift(im, dx):
    """Move the object, leaving the frame the same size. Test-time only."""
    return np.roll(np.asarray(im), int(dx), axis=-1)


def rates(v1, images):
    return np.array([np.concatenate([v1.rate(c) for c in opponent(im)])
                     for im in images], np.float32)


def codes(R):
    ad = PopulationAdaptation(R.shape[1])
    return np.array([_unit(ad(r)) for r in R], np.float32)


def cluster_auc(Q, yq, B, yb, rng):
    same, diff = [], []
    for k in range(len(Q)):
        ps = np.flatnonzero(yb == yq[k])
        pd = np.flatnonzero(yb != yq[k])
        if not len(ps) or not len(pd):
            continue
        same.append(float(Q[k] @ B[int(rng.choice(ps))]))
        diff.append(float(Q[k] @ B[int(rng.choice(pd))]))
    same, diff = np.array(same), np.array(diff)
    return float((same[:, None] > diff[None, :]).mean())


def measure(v1, images, y, tr, te, n_cls, seed):
    out = {}
    for tag, ims in (("", images),
                     ("shifted ", [shift(im, SHIFT if i % 2 else -SHIFT)
                                   for i, im in enumerate(images)])):
        V = codes(rates(v1, ims))
        B, Q = V[tr], V[te]
        rng = np.random.default_rng(seed)
        out[f"{tag}cluster_auc"] = cluster_auc(Q, y[te], B, y[tr], rng)
        out[f"{tag}prototype"] = float(np.mean(
            _nearest_prototype(B, y[tr], Q, n_cls) == y[te]))
        s = Q @ B.T
        out[f"{tag}1nn"] = float(np.mean(y[tr][s.argmax(1)] == y[te]))
    return out


def run_seed(images, waves, y, n_cls, seed):
    tr, te = split(y, seed)
    out = {}
    base = StreamingBrain(seed=seed, image_shape=(32, 32), v1_cells=4096,
                          rf=7, stride=2)
    develop_v1(base.v1, [opponent(im)[0] for im in images], epochs=3,
               seed=seed)
    R = rates(base.v1, images)
    V = codes(R)
    A = np.array([base.belt.code(base.ear.coch.forward(w)[0]) for w in waves],
                 np.float32)

    # the teacher: a concept layer that has been made to generalise
    assoc = AssociationArea(n_vis=V.shape[1], n_aud=A.shape[1],
                            n_concept=N_CONCEPT, seed=seed)
    assoc.set_stats(V[tr], A[tr])
    owner = {}
    for i in tr:
        w = assoc.bind(V[i], A[i])
        owner.setdefault(w, []).append(int(i))
    merged = assoc.consolidate_ranked(keep=KEEP)
    for old, new in merged.items():
        while new in merged:
            new = merged[new]
        owner.setdefault(new, []).extend(owner.pop(old, []))

    # targets in RAW RATE space -- the space align_v1 requires. Building them
    # in the association area's prep_v space is the recurring bug in this
    # project and would silently teach the eye nonsense.
    cmean = {c: R[m].mean(0) for c, m in owner.items() if m}
    # the reconstructive rule needs a target *image*, not a target code: the
    # mean luminance frame over everything the concept came to hold
    lum = np.array([opponent(im)[0] for im in images], np.float32)
    cimg = {c: lum[m].mean(0) for c, m in owner.items() if m}
    cells = sorted(cmean)
    rng = np.random.default_rng(seed + 5)
    tgt = {"align to concept": [], "align to own rate": [],
           "align to wrong concept": []}
    timg = {"reconstruct concept": [], "reconstruct own": [],
            "reconstruct wrong": []}
    for i in tr:
        c = assoc.concept_from_vision(V[i])
        c = c if c in cmean else cells[0]
        other = int(rng.choice([d for d in cells if d != c] or cells))
        tgt["align to concept"].append(cmean[c])
        tgt["align to own rate"].append(R[i])
        tgt["align to wrong concept"].append(cmean[other])
        timg["reconstruct concept"].append(cimg[c])
        timg["reconstruct own"].append(lum[i])
        timg["reconstruct wrong"].append(cimg[other])

    out["no align"] = measure(base.v1, images, y, tr, te, n_cls, seed)
    ims_tr = [images[i] for i in tr]
    for arm in ARMS[1:]:
        b = StreamingBrain(seed=seed, image_shape=(32, 32), v1_cells=4096,
                           rf=7, stride=2)
        develop_v1(b.v1, [opponent(im)[0] for im in images], epochs=3,
                   seed=seed)
        # align on the LUMINANCE channel, which is the one develop_v1 grew on
        if arm in timg:
            align_v1_reconstructive(b.v1, [opponent(im)[0] for im in ims_tr],
                                    timg[arm], epochs=EPOCHS, lr=0.05)
        else:
            align_v1(b.v1, [opponent(im)[0] for im in ims_tr],
                     [t[:b.v1.n_cells] for t in tgt[arm]],
                     epochs=EPOCHS, lr=LR)
        out[arm] = measure(b.v1, images, y, tr, te, n_cls, seed)
    out["_cells"] = len(cells)
    return out


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_align.json"
    images, waves, y, names = load_audiovisual(n_per_class=60, seed=0,
                                               grayscale=False, size=32)
    n_cls = len(names)
    y = np.asarray(y, int)
    print(f"{len(images)} real pairs, {n_cls} categories, chance "
          f"{1/n_cls:.3f}; shift +-{SHIFT}px at test time only\n", flush=True)

    rows = []
    for sd in SEEDS:
        rows.append(run_seed(images, waves, y, n_cls, sd))
        print(f"  seed {sd}: {rows[-1]['_cells']} concepts | AUC "
              f"{rows[-1]['no align']['cluster_auc']:.3f} -> "
              f"{rows[-1]['align to concept']['cluster_auc']:.3f}", flush=True)

    METRICS = ("cluster_auc", "prototype", "1nn",
               "shifted cluster_auc", "shifted prototype", "shifted 1nn")
    res = {"seeds": list(SEEDS), "arms": {}}
    for arm in ARMS:
        res["arms"][arm] = {m: round(float(np.mean([r[arm][m] for r in rows])),
                                     4) for m in METRICS}

    print(f"\n{'arm':<24}" + "".join(f"{m:>13}" for m in METRICS[:3]))
    for arm in ARMS:
        print(f"{arm:<24}" + "".join(f"{res['arms'][arm][m]:>13.3f}"
                                     for m in METRICS[:3]))
    print(f"\n{'arm':<24}" + "".join(f"{m:>13}" for m in
                                     ("shift AUC", "shift proto", "shift 1nn")))
    for arm in ARMS:
        print(f"{arm:<24}" + "".join(f"{res['arms'][arm][m]:>13.3f}"
                                     for m in METRICS[3:]))

    print(f"\nagainst 'no align', paired over {len(SEEDS)} seeds")
    gate = {}
    for arm in ARMS[1:]:
        cells = []
        for m in METRICS:
            d = np.array([r[arm][m] - r["no align"][m] for r in rows])
            sd = float(d.std(ddof=1))
            cd = float(d.mean() / (sd + 1e-12))
            gate.setdefault(arm, {})[m] = dict(
                delta=round(float(d.mean()), 4), cohens_d=round(cd, 3),
                wins=int((d > 0).sum()), n=len(d))
        cells = [f"{gate[arm][m]['delta']:+.3f} d={gate[arm][m]['cohens_d']:+.1f}"
                 for m in METRICS[:3]]
        print(f"{arm:<24}" + "".join(f"{c:>18}" for c in cells))
    res["gate"] = gate

    print("\n=== can the concept layer teach the eye? ===")
    key = "cluster_auc"
    g = gate["align to concept"][key]
    own = gate["align to own rate"][key]
    wrong = gate["align to wrong concept"][key]
    rg = gate["reconstruct concept"][key]
    ro = gate["reconstruct own"][key]
    rw = gate["reconstruct wrong"][key]
    print(f"  -- the delta rule (scalar error per cell) --")
    print(f"  align to concept      {g['delta']:+.4f} "
          f"(d={g['cohens_d']:+.2f}, {g['wins']}/{g['n']})")
    print(f"  align to own rate     {own['delta']:+.4f} "
          f"(d={own['cohens_d']:+.2f}, {own['wins']}/{own['n']})"
          f"   <- same rule, teacher says nothing new")
    print(f"  align to wrong concept{wrong['delta']:+.4f} "
          f"(d={wrong['cohens_d']:+.2f}, {wrong['wins']}/{wrong['n']})"
          f"   <- same rule, teacher misleading")

    print(f"\n  -- the reconstructive rule (error is a vector over the patch) --")
    print(f"  reconstruct concept   {rg['delta']:+.4f} "
          f"(d={rg['cohens_d']:+.2f}, {rg['wins']}/{rg['n']})")
    print(f"  reconstruct own       {ro['delta']:+.4f} "
          f"(d={ro['cohens_d']:+.2f}, {ro['wins']}/{ro['n']})"
          f"   <- teacher says nothing new")
    print(f"  reconstruct wrong     {rw['delta']:+.4f} "
          f"(d={rw['cohens_d']:+.2f}, {rw['wins']}/{rw['n']})"
          f"   <- teacher misleading")
    res["reconstructive_teaches"] = bool(
        rg["cohens_d"] >= 0.8 and rg["wins"] >= 0.75 * rg["n"]
        and rg["delta"] > ro["delta"] and rg["delta"] > rw["delta"])
    if res["reconstructive_teaches"]:
        print("\n  The reconstructive rule DOES carry the teacher: a correct "
              "target beats an uninformative and a misleading one.")
    else:
        print("\n  The reconstructive rule does not carry it either -- correct,"
              " uninformative and misleading targets still agree.")

    beats = (g["cohens_d"] >= 0.8 and g["wins"] >= 0.75 * g["n"]
             and g["delta"] > own["delta"] and g["delta"] > wrong["delta"])
    res["teaching_works"] = bool(beats)
    if beats:
        sh = gate["align to concept"]["shifted cluster_auc"]
        print(f"\n  YES -- and it is the TEACHER, not the plasticity: the same "
              f"rule with an uninformative target gives {own['delta']:+.4f} "
              f"and with a misleading one {wrong['delta']:+.4f}.")
        print(f"  Under a +-{SHIFT}px shift: {sh['delta']:+.4f} "
              f"(d={sh['cohens_d']:+.2f}).")
    else:
        print(f"\n  No. Concept feedback does not raise the clustering of the "
              f"visual code above what the same number of updates with an "
              f"uninformative target does.")
        print(f"  The eye's problem is not that nothing above it was speaking "
              f"to it.")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
