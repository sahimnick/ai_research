"""Phase 10.2 as specified: the panel-1 test, with the object moved.

Panel 1 of the imagination report measured one number and it is the one this
phase is about: **the concept a held-out photograph wakes is the right category
only 35% of the time**. That is the eye's failure made visible, and the
prediction on the table was that concept feedback would take it to 55-60% once
V1 stopped being tied to individual photographs, with the test made harder by
shifting the object 5 pixels left and right.

Earlier runs of Phase 10 measured cluster AUC, prototype accuracy and 1-NN.
Those are reasonable numbers and they are *not* the number that was asked for,
so this benchmark measures the specified one directly:

    concept_wakes   show a held-out photograph; the concept cell it wakes has a
                    majority category from training; is that the right one?

taken three ways -- centred, shifted +5px, shifted -5px -- and across the same
arms as `align.py`, so the answer is attributable:

    no align              the eye as develop_v1 leaves it. The 35% baseline
    align to concept      the delta rule with the concept mean as teacher
    reconstruct concept   the reconstructive rule, which was verified to carry
                          its teacher (two different targets leave the filters
                          0.341 apart rather than 0.030)
    reconstruct own       same rule, teacher says nothing new
    reconstruct wrong     same rule, teacher actively misleading

The last two are what make the first three readable. A rule that improves the
number with a *misleading* teacher improved it by adding plasticity, not by
learning anything from the concept layer.

**The shift is applied at test time only**, so it cannot be learned, and the
frame is padded rather than cropped so the object stays whole -- a 32-pixel
object shifted inside a 32-pixel frame is an occlusion test, which is the
mistake `translation.py` exists to correct.

Usage:  python3 benchmarks/phase10_2.py out_phase10_2.json
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
# 32-px object in a 48-px frame. 40 was the first choice and it was wrong: it
# leaves only (48-32)//2 = 4px of margin, so a requested +-5 CLIPS to +-4 and the
# specified shift is silently not delivered. Verified by measuring which columns
# the object occupies. 48 gives 8px of margin, so +-5 is exact.
FRAME = 48
N_CONCEPT = 256
KEEP = 0.6
EPOCHS = 2
LR_DELTA, LR_REC = 0.02, 8.0
SHIFTS = (0, 5, -5)
ARMS = ("no align", "align to concept", "reconstruct concept",
        "reconstruct own", "reconstruct wrong")


def place(im, dx=0, frame=FRAME):
    """Put the object in a larger frame, offset horizontally.

    Padding rather than rolling: a rolled image wraps the object round the edge,
    which is a different manipulation wearing translation's name."""
    a = np.asarray(im)
    c, h, w = (a.shape if a.ndim == 3 else (1,) + a.shape)
    a = a.reshape(c, h, w)
    out = np.zeros((c, frame, frame), a.dtype)
    y0 = (frame - h) // 2
    x0 = int(np.clip((frame - w) // 2 + dx, 0, frame - w))
    out[:, y0:y0 + h, x0:x0 + w] = a
    return out if a.ndim == 3 else out[0]


def rates(v1, frames):
    return np.array([np.concatenate([v1.rate(c) for c in opponent(f)])
                     for f in frames], np.float32)


def codes(R):
    ad = PopulationAdaptation(R.shape[1])
    return np.array([_unit(ad(r)) for r in R], np.float32)


def concept_wakes(v1, images, y, tr, te, seed):
    """The panel-1 number, at each shift.

    Concepts are formed on centred training photographs; the held-out
    photograph is presented centred and shifted, and the question each time is
    whether the concept it wakes is one whose training majority is its own
    category."""
    base = codes(rates(v1, [place(images[i]) for i in tr]))
    A = np.stack([_unit(base[i] + 0.0) for i in range(len(base))])
    assoc = AssociationArea(n_vis=base.shape[1], n_aud=base.shape[1],
                            n_concept=N_CONCEPT, seed=seed)
    assoc.set_stats(base, A)
    votes = {}
    for k, i in enumerate(tr):
        w = assoc.bind(base[k], A[k])
        votes.setdefault(w, {})
        votes[w][int(y[i])] = votes[w].get(int(y[i]), 0) + 1
    name = {c: max(v.items(), key=lambda kv: kv[1])[0]
            for c, v in votes.items() if v}
    out = {}
    for dx in SHIFTS:
        q = codes(rates(v1, [place(images[i], dx) for i in te]))
        ok = sum(int(name.get(assoc.concept_from_vision(q[k]), -1)
                     == int(y[i])) for k, i in enumerate(te))
        out[f"shift {dx:+d}px"] = ok / max(len(te), 1)
    return out


def run_seed(images, waves, y, n_cls, seed):
    tr, te = split(y, seed)
    cent = [place(im) for im in images]
    lum = np.array([opponent(f)[0] for f in cent], np.float32)

    base = StreamingBrain(seed=seed, image_shape=(FRAME, FRAME),
                          v1_cells=4096, rf=7, stride=2)
    develop_v1(base.v1, list(lum), epochs=3, seed=seed)
    R = rates(base.v1, cent)
    V = codes(R)
    A = np.array([base.belt.code(base.ear.coch.forward(w)[0]) for w in waves],
                 np.float32)

    # the teacher: a concept layer made to generalise
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

    out = {"no align": concept_wakes(base.v1, images, y, tr, te, seed)}
    lum_tr = [lum[i] for i in tr]
    for arm in ARMS[1:]:
        b = StreamingBrain(seed=seed, image_shape=(FRAME, FRAME),
                           v1_cells=4096, rf=7, stride=2)
        develop_v1(b.v1, list(lum), epochs=3, seed=seed)
        if arm in timg:
            align_v1_reconstructive(b.v1, lum_tr, timg[arm], epochs=EPOCHS,
                                    lr=LR_REC)
        else:
            align_v1(b.v1, lum_tr, [t[:b.v1.n_cells] for t in tgt[arm]],
                     epochs=EPOCHS, lr=LR_DELTA)
        out[arm] = concept_wakes(b.v1, images, y, tr, te, seed)
    return out


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_phase10_2.json"
    images, waves, y, names = load_audiovisual(n_per_class=60, seed=0,
                                               grayscale=False, size=32)
    y = np.asarray(y, int)
    n_cls = len(names)
    print(f"{len(images)} real pairs, {n_cls} categories, chance "
          f"{1/n_cls:.3f}")
    margin = (FRAME - 32) // 2
    assert margin >= max(abs(d) for d in SHIFTS), (
        f"a {FRAME}px frame gives {margin}px of margin, so the requested "
        f"shifts would clip -- the test would not be the one specified")
    print(f"32px object in a {FRAME}px frame ({margin}px margin, so "
          f"+-{max(SHIFTS)}px is exact); shift applied at TEST time only\n",
          flush=True)

    rows = [run_seed(images, waves, y, n_cls, sd) for sd in SEEDS]
    keys = [f"shift {d:+d}px" for d in SHIFTS]
    res = {"seeds": list(SEEDS), "chance": round(1 / n_cls, 4), "arms": {}}
    for arm in ARMS:
        res["arms"][arm] = {k: round(float(np.mean([r[arm][k] for r in rows])),
                                     4) for k in keys}

    print(f"{'arm':<24}" + "".join(f"{k:>14}" for k in keys))
    for arm in ARMS:
        print(f"{arm:<24}" + "".join(f"{res['arms'][arm][k]:>14.3f}"
                                     for k in keys))

    print(f"\nagainst 'no align', paired over {len(SEEDS)} seeds")
    gate = {}
    for arm in ARMS[1:]:
        cs = []
        for k in keys:
            d = np.array([r[arm][k] - r["no align"][k] for r in rows])
            sd = float(d.std(ddof=1))
            cd = None if sd < 1e-9 else float(d.mean() / sd)
            gate.setdefault(arm, {})[k] = dict(
                delta=round(float(d.mean()), 4),
                cohens_d=None if cd is None else round(cd, 3),
                wins=int((d > 0).sum()), n=len(d))
            cs.append(f"{d.mean():+.3f} " +
                      ("d=n/a" if cd is None else f"d={cd:+.1f}"))
        print(f"{arm:<24}" + "".join(f"{c:>14}" for c in cs))
    res["gate"] = gate

    base = res["arms"]["no align"]
    print("\n=== phase 10.2: the prediction was 35% -> 55-60% ===")
    print(f"  baseline, centred      {base['shift +0px']:.3f}")
    print(f"  baseline, shifted +-5px "
          f"{(base['shift +5px'] + base['shift -5px']) / 2:.3f}")
    best_arm = max(ARMS[1:], key=lambda a: np.mean(
        [res["arms"][a][k] for k in keys]))
    b = res["arms"][best_arm]
    got = float(np.mean([b[k] for k in keys]))
    res["best_arm"], res["best_mean"] = best_arm, round(got, 4)
    print(f"  best aligned arm ({best_arm}): "
          f"{got:.3f} across all three shifts")
    res["meets_prediction"] = bool(got >= 0.55)
    # and the control that decides whether any gain is teaching
    wrong = float(np.mean([res["arms"]["reconstruct wrong"][k] for k in keys]))
    conc = float(np.mean([res["arms"]["reconstruct concept"][k] for k in keys]))
    res["teacher_matters"] = bool(conc > wrong + 0.02)
    if res["meets_prediction"]:
        print(f"\n  The prediction is met ({got:.3f} >= 0.55).")
    else:
        print(f"\n  The prediction is NOT met: {got:.3f} against the 0.55-0.60 "
              f"predicted, from a baseline of "
              f"{np.mean([base[k] for k in keys]):.3f}.")
    if res["teacher_matters"]:
        print(f"  And the teacher is what did it: the correct concept gives "
              f"{conc:.3f} against {wrong:.3f} for a misleading one.")
    else:
        print(f"  And the teacher is not what matters: the correct concept "
              f"gives {conc:.3f} against {wrong:.3f} for a deliberately "
              f"MISLEADING one -- so whatever moved, the concept layer did "
              f"not move it.")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
