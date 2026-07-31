"""Does looking at a thing more than once help name it?

`natural_scene.py` found the sharpest split in the project: on 256x256 scenes of
scattered CIFAR-10 photographs the eye reaches an on-object rate of **0.883**,
better than the 0.830 it manages on digit scenes, and then names what it found at
**0.134** against a chance of 0.100. Looking transfers to the real world
completely; recognising does not transfer at all.

Seven mechanisms inside the recognition path have since been eliminated with
controls -- width, aperture, spike noise, integration window, and a second stage
under two learning rules -- and one cropping decision was worth more than all of
them. So this asks a different kind of question. Instead of a better code, use
**more of the faculty that already works**: the eye lands on the same object
several times during a free view, and every one of those glances is currently
classified alone and then thrown away.

A real visual system does not decide from one glance. Evidence accumulates
across fixations until it crosses a bound -- that is the standard account of
perceptual decision-making, and it is the one thing this project's eye is
already producing the raw material for and never using.

Measured, on the eye's own free-viewing behaviour with nothing staged:

    per-fixation    the current number: each glance named on its own
    per-object      the glances that landed on the SAME object, pooled, then
                    named once
    ...by k         the same as a function of how many glances were pooled, so
                    the shape of the accumulation is visible rather than a
                    single average

Two controls, because "more vectors averaged" is an alternative explanation for
any gain:

    scrambled   pool k glances drawn from *different* objects and score against
                the majority object's label. If averaging unrelated glances helps
                as much, the gain is variance reduction and not evidence
                accumulation
    matched-n   the per-fixation baseline restricted to objects that received at
                least k glances, so the two arms are scored on the same objects
                and an easy-object bias cannot masquerade as a gain

MNIST scenes are carried through as a reference, since the question is whether
this recovers ground specifically where recognition is failing.

Usage:  python3 benchmarks/multi_fixation.py out_multi_fixation.json
"""
import json
import sys
from collections import defaultdict

import numpy as np

import neurobrain as nb
from neurobrain.learning.selforganize import develop_v1
from neurobrain.sensing.natural import load_cifar10
from neurobrain.sensing.streams import SaccadicEye, build_scene
from neurobrain.vision.widev1 import (PopulationAdaptation, WideV1,
                                      _nearest_prototype, _unit)

SEEDS = (0, 1, 2, 3, 4)
N_SCENES, N_SACCADES, N_OBJECTS = 4, 80, 12
N_TRAIN, N_DEV = 2500, 800
CANVAS = 256
KS = (1, 2, 3, 4, 6)
# How far the eye is allowed to land from the corrected point, so that
# returning to an object scans a different part of it. 0 is the eye as built.
EXPLORE = (0, 4, 8, 12)


def luminance(X):
    return X.mean(1).astype(np.uint8) if X.ndim == 4 else X


def which_object(scene, f, tol=None):
    """Which object this glance landed on, by index, or -1 for background.

    `Fixation.true_label` already says *what* was there but not *which one*, and
    two dogs in the same scene must not have their glances pooled together."""
    tol = tol if tol is not None else scene.tile // 2
    for k, (pr, pc) in enumerate(scene.positions):
        if abs(pr - f.row) <= tol and abs(pc - f.col) <= tol:
            return k
    return -1


def collect(scene, v1, adapt, seed, explore=0):
    """Free-view once; return {object index: [codes]} and the object's label."""
    fix = SaccadicEye(scene, seed=seed).free_view(n_saccades=N_SACCADES,
                                                  correct=True,
                                                  explore=explore)
    by_obj = defaultdict(list)
    for f in fix:
        if f.true_label < 0:
            continue
        k = which_object(scene, f)
        if k < 0:
            continue
        by_obj[k].append(_unit(adapt.apply(v1.rate_over(f.frames))))
    return by_obj, len(fix)


def pool(codes):
    """Mean and max over glances, as the belt pools over time.

    The max half matters: a glance that caught the object badly should not be
    able to average away a glance that caught it well."""
    R = np.asarray(codes, np.float32)
    return _unit(np.concatenate([R.mean(0), R.max(0)]))


def run(images, labels, seed):
    v1 = WideV1(n_cells=1024, window_ms=50, seed=seed)
    develop_v1(v1, [im.astype(np.float32) for im in images[:N_DEV]],
               epochs=3, seed=seed)
    ad = PopulationAdaptation(v1.n_cells)
    n_cls = int(labels.max()) + 1

    # the bank is built in BOTH codes, so per-fixation and per-object are each
    # scored against a bank in their own space -- a pooled query against an
    # unpooled bank would be a different comparison wearing this one's name
    R = np.array([v1.rate(im.astype(np.float32)) for im in images[:1500]],
                 np.float32)
    bank1 = np.array([_unit(ad(r)) for r in R], np.float32)
    bank2 = np.array([_unit(np.concatenate([b, b])) for b in bank1], np.float32)
    ybank = labels[:1500]

    single, by_k, ctrl_k, matched_k = [], defaultdict(list), defaultdict(list), defaultdict(list)
    rng = np.random.default_rng(seed + 5)
    for s in range(N_SCENES):
        sc = build_scene(images, labels, size=CANVAS, n_objects=N_OBJECTS,
                         seed=seed * 100 + s)
        by_obj, n_fix = collect(sc, v1, ad, seed * 100 + s)
        if not by_obj:
            continue
        # ---- per fixation: every glance on its own -----------------------
        flat, flat_y = [], []
        for k, codes in by_obj.items():
            flat.extend(codes)
            flat_y.extend([sc.labels[k]] * len(codes))
        if flat:
            pred = _nearest_prototype(bank1, ybank, np.asarray(flat, np.float32),
                                      n_cls)
            single.append(float(np.mean(pred == np.asarray(flat_y))))
        # ---- per object: k glances pooled --------------------------------
        for k in KS:
            q, qy, cq = [], [], []
            for oi, codes in by_obj.items():
                if len(codes) < k:
                    continue
                q.append(pool(codes[:k]))
                qy.append(sc.labels[oi])
                # control: k glances from DIFFERENT objects, scored against the
                # label most of them came from
                pick = [by_obj[j][0] for j in
                        rng.choice(list(by_obj), size=k, replace=len(by_obj) < k)]
                cq.append(pool(pick))
            if not q:
                continue
            qy = np.asarray(qy)
            by_k[k].append(float(np.mean(
                _nearest_prototype(bank2, ybank, np.asarray(q, np.float32),
                                   n_cls) == qy)))
            ctrl_k[k].append(float(np.mean(
                _nearest_prototype(bank2, ybank, np.asarray(cq, np.float32),
                                   n_cls) == qy)))
            # matched-n baseline: first glance only, same objects
            first = [by_obj[oi][0] for oi in by_obj if len(by_obj[oi]) >= k]
            matched_k[k].append(float(np.mean(
                _nearest_prototype(bank1, ybank, np.asarray(first, np.float32),
                                   n_cls) == qy)))
    return (float(np.mean(single)) if single else 0.0,
            {k: float(np.mean(v)) for k, v in by_k.items() if v},
            {k: float(np.mean(v)) for k, v in ctrl_k.items() if v},
            {k: float(np.mean(v)) for k, v in matched_k.items() if v})


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_multi_fixation.json"
    cx, cy, _, _ = load_cifar10(n_train=N_TRAIN, n_test=200, grayscale=False,
                                size=28)
    mx, my, _, _ = nb.load_mnist(n_train=N_TRAIN, n_test=200)
    res = {"n_saccades": N_SACCADES, "n_objects": N_OBJECTS, "ks": list(KS),
           "seeds": list(SEEDS)}

    for tag, X, y in (("CIFAR-10 photographs", luminance(cx), cy),
                      ("MNIST digits (reference)", mx, my)):
        print(f"\n{tag} -- {N_OBJECTS} objects, {N_SACCADES} saccades, "
              f"{N_SCENES} scenes x {len(SEEDS)} seeds")
        rows = [run(X, y, sd) for sd in SEEDS]
        base = float(np.mean([r[0] for r in rows]))
        print(f"  {'k glances':<12}{'pooled':>9}{'matched-n':>11}"
              f"{'scrambled':>11}{'gain':>9}{'d':>7}{'wins':>7}")
        rec = {"per_fixation": round(base, 4), "by_k": {}}
        for k in KS:
            p = [r[1].get(k) for r in rows if r[1].get(k) is not None]
            mm = [r[3].get(k) for r in rows if r[3].get(k) is not None]
            cc = [r[2].get(k) for r in rows if r[2].get(k) is not None]
            if len(p) < 2:
                continue
            d = np.array(p) - np.array(mm)
            sd = float(d.std(ddof=1))
            cd = float(d.mean() / (sd + 1e-12))
            rec["by_k"][k] = dict(pooled=round(float(np.mean(p)), 4),
                                  matched=round(float(np.mean(mm)), 4),
                                  scrambled=round(float(np.mean(cc)), 4),
                                  gain=round(float(d.mean()), 4),
                                  cohens_d=round(cd, 3),
                                  wins=int((d > 0).sum()), n=len(d))
            r = rec["by_k"][k]
            print(f"  {k:<12}{r['pooled']:>9.3f}{r['matched']:>11.3f}"
                  f"{r['scrambled']:>11.3f}{r['gain']:>+9.3f}{cd:>7.2f}"
                  f"{r['wins']:>4}/{r['n']}")
        res[tag] = rec

    print("\n=== does accumulating glances recover recognition? ===")
    for tag in ("CIFAR-10 photographs", "MNIST digits (reference)"):
        r = res[tag]["by_k"]
        if not r:
            continue
        best = max(r, key=lambda k: r[k]["pooled"])
        b = r[best]
        ok = b["cohens_d"] >= 0.8 and b["wins"] >= 0.8 * b["n"]
        beats_ctrl = b["pooled"] > b["scrambled"] + 0.02
        print(f"  {tag:<26} k={best}: {b['matched']:.3f} -> {b['pooled']:.3f} "
              f"({b['gain']:+.3f}, d={b['cohens_d']:+.2f}, {b['wins']}/{b['n']})"
              f"  {'YES' if ok and beats_ctrl else 'no'}")
        if ok and not beats_ctrl:
            print(f"    -- but scrambled glances score {b['scrambled']:.3f} too,"
                  f" so this is variance reduction, not evidence accumulation")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
