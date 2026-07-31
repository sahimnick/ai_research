"""The one property eight failed mechanisms were all blocked by.

`multi_fixation.py` ended by naming it. Accumulating glances is impossible
because the corrective saccade lands on the same pixel every time (median offset
**0.0**, glance-to-glance cosine 0.879), and letting the eye scan an object's
parts instead removes the redundancy (0.856 -> 0.168) while destroying accuracy
(MNIST 0.733 -> 0.257). Diverse glances must be off-centre; off-centre glances
cannot be read. The wide V1 code is retinotopic and has **no translation
invariance at all**.

`WideV1.pooling_index` already builds an invariant code -- it is what makes the
auditory belt work -- and on vision it was measured as *harmful*: on centred
MNIST crops, position-specific scores 1-NN 0.818 against 0.350 for fully pooled.
But a centred crop is exactly the condition where invariance buys nothing and
costs everything, so that measurement answered a different question from the one
that matters.

This asks the one that matters: **how does each code degrade as the object moves
off centre?** A code that starts lower and stays flat beats a code that starts
higher and falls off a cliff, at every offset past the crossing point -- and the
free-viewing regime lives past that point.

    position-specific   the code as shipped: which filter fired *where*
    pooled cols         frequency/row kept, the other axis pooled away
    pooled rows         the reverse
    pooled both         only *which filter fired*, fully shift-invariant
    mixed g             position-specific concatenated with fully-pooled at
                        gain g -- the same two-channel trick that fixed the
                        auditory belt, where neither extreme was right

Two experiments, because one alone can be argued with:

1. **Controlled offset.** Objects are shifted by a known number of pixels inside
   the fovea. This isolates translation and nothing else.
2. **The real regime.** Free-viewing with `explore`, so the offsets are the ones
   the eye actually produces, and the question becomes whether an invariant code
   makes glance accumulation possible where the retinotopic one made it
   impossible.

Usage:  python3 benchmarks/translation.py out_translation.json
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

sys.path.insert(0, "benchmarks")
from multi_fixation import which_object                     # noqa: E402

SEEDS = (0, 1, 2)
N_TRAIN, N_DEV, N_TEST = 2500, 800, 500
OFFSETS = (0, 2, 4, 6, 8, 10)
# The frame must be BIGGER than the object, or 'translation' is occlusion.
# A 28x28 object shifted 10px inside a 28x28 frame loses a third of itself,
# and the first version of this benchmark measured exactly that: every code,
# including the fully shift-invariant one, fell to chance by 8px. That is not
# a property of the codes. FRAME leaves room to move.
FRAME = 48
MIX_GAINS = (0.5, 1.0, 2.0)
N_SCENES, N_SACCADES, N_OBJECTS = 3, 80, 12


def luminance(X):
    return X.mean(1).astype(np.uint8) if X.ndim == 4 else X


def place(im, dy=0, dx=0, frame=FRAME, clutter=0.08, rng=None):
    """Put the object in a larger frame, offset from centre.

    The frame is what makes this a translation test: the object stays whole and
    only its retinal position changes. Background clutter matches `build_scene`
    so the surround is not an information-free void that a pooled code could
    exploit."""
    rng = rng or np.random.default_rng(0)
    out = (clutter * 255.0 * rng.random((frame, frame))).astype(np.float32)
    h, w = im.shape
    y = (frame - h) // 2 + int(dy)
    x = (frame - w) // 2 + int(dx)
    y = int(np.clip(y, 0, frame - h))
    x = int(np.clip(x, 0, frame - w))
    out[y:y + h, x:x + w] = np.maximum(out[y:y + h, x:x + w],
                                       im.astype(np.float32))
    return out


class Codes:
    """Every read-out of one V1, so they are compared on identical spikes."""

    def __init__(self, v1):
        self.v1 = v1
        self.idx = {a: v1.pooling_index(a) for a in ("both", "rows", "cols")}

    def variants(self, drive):
        out = {"position-specific": _unit(drive)}
        for a in ("cols", "rows", "both"):
            i, n = self.idx[a]
            out[f"pooled {a}"] = self.v1.pooled_code(drive, i, n)
        for g in MIX_GAINS:
            out[f"mixed {g}"] = _unit(np.concatenate(
                [out["position-specific"], g * out["pooled both"]]))
        return out

    def names(self):
        return (["position-specific"] + [f"pooled {a}" for a in
                                         ("cols", "rows", "both")]
                + [f"mixed {g}" for g in MIX_GAINS])


def controlled(X, y, seed, tie=False):
    """Accuracy against a known offset, with the bank built from centred objects."""
    v1 = WideV1(n_cells=1024, window_ms=50, image_shape=(FRAME, FRAME),
                seed=seed)
    rng = np.random.default_rng(seed)
    develop_v1(v1, [place(im, rng=rng) for im in X[:N_DEV]], epochs=3,
               tie=tie, seed=seed)
    C = Codes(v1)

    # the bank is centred in the same frame, so the only thing that varies
    # between bank and query is WHERE the object is
    bank_raw = [v1.drive(place(im, rng=rng)) for im in X[:1500]]
    ybank = y[:1500]
    banks = defaultdict(list)
    for d in bank_raw:
        for k, v in C.variants(d).items():
            banks[k].append(v)
    banks = {k: np.asarray(v, np.float32) for k, v in banks.items()}

    te = X[1500:1500 + N_TEST]
    yte = y[1500:1500 + N_TEST]
    out = {}
    for off in OFFSETS:
        q = defaultdict(list)
        for im in te:
            dy, dx = (0, 0) if off == 0 else tuple(
                rng.integers(-off, off + 1, 2))
            d = v1.drive(place(im, int(dy), int(dx), rng=rng))
            for k, v in C.variants(d).items():
                q[k].append(v)
        for k in C.names():
            out.setdefault(k, {})[off] = float(np.mean(
                _nearest_prototype(banks[k], ybank,
                                   np.asarray(q[k], np.float32), 10) == yte))
    return out


def in_the_wild(X, y, seed, tie=False):
    """The real regime: free-viewing, and does accumulation work now?"""
    v1 = WideV1(n_cells=1024, window_ms=50, seed=seed)
    develop_v1(v1, [im.astype(np.float32) for im in X[:N_DEV]], epochs=3,
               tie=tie, seed=seed)
    C = Codes(v1)
    bank_raw = [v1.drive(im.astype(np.float32)) for im in X[:1500]]
    ybank = y[:1500]
    banks = defaultdict(list)
    for d in bank_raw:
        for k, v in C.variants(d).items():
            banks[k].append(v)
    banks = {k: np.asarray(v, np.float32) for k, v in banks.items()}

    res = defaultdict(lambda: defaultdict(list))
    for explore in (0, 8):
        for s in range(N_SCENES):
            sc = build_scene(X, y, size=256, n_objects=N_OBJECTS,
                             seed=seed * 100 + s)
            fix = SaccadicEye(sc, seed=seed * 100 + s).free_view(
                n_saccades=N_SACCADES, correct=True, explore=explore)
            by = defaultdict(lambda: defaultdict(list))
            for f in fix:
                if f.true_label < 0:
                    continue
                oi = which_object(sc, f)
                if oi < 0:
                    continue
                d = np.mean([v1.drive(fr) for fr in f.frames], 0)
                for k, v in C.variants(d).items():
                    by[oi][k].append(v)
            for k in C.names():
                for nglance in (1, 4):
                    q, qy = [], []
                    for oi, codes in by.items():
                        if len(codes[k]) < nglance:
                            continue
                        q.append(_unit(np.asarray(codes[k][:nglance],
                                                  np.float32).mean(0)))
                        qy.append(sc.labels[oi])
                    if q:
                        res[explore][(k, nglance)].append(float(np.mean(
                            _nearest_prototype(banks[k], ybank,
                                               np.asarray(q, np.float32),
                                               10) == np.asarray(qy))))
    return {e: {k: float(np.mean(v)) for k, v in d.items()}
            for e, d in res.items()}


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_translation.json"
    cx, cy, _, _ = load_cifar10(n_train=N_TRAIN + N_TEST, n_test=200,
                                grayscale=False, size=28)
    mx, my, _, _ = nb.load_mnist(n_train=N_TRAIN + N_TEST, n_test=200)
    res = {"offsets": list(OFFSETS), "seeds": list(SEEDS)}

    for tag, X, y in (("MNIST digits", mx, my),
                      ("CIFAR-10 photographs", luminance(cx), cy)):
      for tie in (False, True):
        rows = [controlled(X, y, sd, tie=tie) for sd in SEEDS]
        names = list(rows[0])
        tag = f"{tag.split(' [')[0]} [{'TIED' if tie else 'untied'} filters]"
        print(f"\n{tag} -- accuracy against a known offset (chance 0.100)")
        print("  " + f"{'code':<20}" + "".join(f"{o:>7}px" for o in OFFSETS)
              + f"{'fall':>8}")
        rec = {}
        for k in names:
            v = [float(np.mean([r[k][o] for r in rows])) for o in OFFSETS]
            rec[k] = {str(o): round(x, 4) for o, x in zip(OFFSETS, v)}
            rec[k]["fall"] = round(v[0] - v[-1], 4)
            print(f"  {k:<20}" + "".join(f"{x:>9.3f}" for x in v)
                  + f"{v[0] - v[-1]:>+8.3f}")
        res.setdefault("controlled", {})[tag] = rec
        best_far = max(names, key=lambda k: rec[k][str(OFFSETS[-1])])
        print(f"  -> at {OFFSETS[-1]}px the best code is **{best_far}** "
              f"({rec[best_far][str(OFFSETS[-1])]:.3f}); at 0px it was "
              f"{rec[best_far]['0']:.3f} against "
              f"{rec['position-specific']['0']:.3f} for position-specific")

    print("\n\nfree-viewing, the regime that actually matters")
    if "--controlled-only" in sys.argv:
        json.dump(res, open(out_path, "w"), indent=1); print("wrote", out_path); return
    for tag, X, y in (("MNIST digits", mx, my),
                      ("CIFAR-10 photographs", luminance(cx), cy)):
      for tie in (False, True):
        rows = [in_the_wild(X, y, sd, tie=tie) for sd in SEEDS]
        print(f"\n{tag} [{'TIED' if tie else 'untied'}]")
        print(f"  {'code':<20}{'e=0 k=1':>10}{'e=0 k=4':>10}"
              f"{'e=8 k=1':>10}{'e=8 k=4':>10}{'accum':>8}")
        rec = {}
        for k in list(rows[0][0]):
            pass
        names = sorted({kk[0] for r in rows for kk in r[0]})
        for k in names:
            def g(e, n):
                v = [r[e].get((k, n)) for r in rows if r[e].get((k, n)) is not None]
                return float(np.mean(v)) if v else float("nan")
            a, b, c, d = g(0, 1), g(0, 4), g(8, 1), g(8, 4)
            rec[k] = dict(e0_k1=round(a, 4), e0_k4=round(b, 4),
                          e8_k1=round(c, 4), e8_k4=round(d, 4),
                          accumulation=round(d - c, 4))
            print(f"  {k:<20}{a:>10.3f}{b:>10.3f}{c:>10.3f}{d:>10.3f}"
                  f"{d - c:>+8.3f}")
        res.setdefault("free_viewing", {})[f"{tag} tie={tie}"] = rec

    print("\n=== is translation invariance the missing property? ===")
    for tag in res["controlled"]:
        c = res["controlled"][tag]
        ps = c["position-specific"]
        best = max(c, key=lambda k: c[k][str(OFFSETS[-1])])
        print(f"  {tag}: position-specific falls {ps['fall']:+.3f} over "
              f"{OFFSETS[-1]}px; best-at-distance is {best} "
              f"({c[best]['fall']:+.3f})")
        if c[best][str(OFFSETS[-1])] > ps[str(OFFSETS[-1])] + 0.02:
            print(f"    -> yes for this dataset: {best} wins by "
                  f"{c[best][str(OFFSETS[-1])] - ps[str(OFFSETS[-1])]:+.3f} "
                  f"once the object is off centre")
        else:
            print("    -> no: position-specific still wins even off centre, so "
                  "invariance is not what is missing")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
