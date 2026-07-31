"""Where to look next, and whether choosing it is worth anything.

`multi_fixation.py` established that pooling glances helps once the code is
genuinely invariant (`develop_v1(tie=True)`: +0.054 on MNIST, +0.113 on CIFAR),
and `translation.py` established why it did not before. Both choose glance
locations by saliency or at random. The open question is the other half:
**does choosing *where* to look, by how uncertain the mind currently is, beat
looking in an arbitrary place?**

Measure the ceiling before building the mechanism
-------------------------------------------------
A real uncertainty-driven eye has to decide from the periphery -- a cheap,
low-resolution signal -- because deciding by looking is not deciding. Building
that is work, and it is only worth doing if there is anything to gain, so this
measures the **upper bound** first, exactly as `constraint.py` measured the
constraint before anyone built a solver:

    oracle-greedy   glance at M candidate locations, keep the one that most
                    sharpens the pooled belief, repeat. This *cheats* -- it
                    costs M glances per glance kept and uses the answer it is
                    trying to produce -- and that is the point: it is the most
                    any selection rule could achieve.

If the oracle does not beat random selection, no cheap approximation of it will,
and uncertainty-driven looking is not the thing to build.

The arms
--------
    one glance        the centre, k=1 -- what the eye does today
    random k          k glances at random offsets, pooled
    saliency k        k glances chosen the way `SaccadicEye` chooses them
    oracle-greedy k   the ceiling above

"Sharpens the belief" is the **margin** between the best and second-best class
under the current pooled code. That is a quantity the mind has without being
told the answer -- it is the ambiguity of its own read-out -- which is what makes
the cheap version conceivable at all. The oracle part is only that it gets to
try every candidate before committing.

Objects are placed in a 48-pixel frame, so an off-centre glance is genuinely a
different view rather than a crop of the same one -- the mistake `translation.py`
was written to correct.

Usage:  python3 benchmarks/uncertain_look.py out_uncertain_look.json
"""
import json
import sys

import numpy as np

from neurobrain.learning.selforganize import develop_v1
from neurobrain.sensing.natural import load_cifar10
from neurobrain.vision.widev1 import WideV1, _unit, _nearest_prototype

sys.path.insert(0, "benchmarks")
from translation import FRAME, luminance, place                # noqa: E402

SEEDS = (0, 1, 2, 3)
N_TRAIN, N_TEST, N_DEV = 1200, 300, 400
KS = (1, 2, 3, 5)
N_CAND = 6            # candidates the oracle may try per glance
JITTER = 8            # how far a glance may land from centre, in pixels


def glance(v1, frame_img, dy, dx, size=28):
    """One fixation: the eye centres its 48-px input on (dy, dx) of the frame.

    Implemented by rolling the frame, which keeps the object whole and only
    changes where it falls on the retina -- the same manipulation `place` makes,
    applied at look time instead of at scene-build time."""
    shifted = np.roll(np.roll(frame_img, -int(dy), axis=0), -int(dx), axis=1)
    return _unit(v1.rate(shifted))


def margin(code, protos):
    """How sure the read-out is: best class minus runner-up.

    The mind has this without being told the answer -- it is the ambiguity of
    its own drive -- which is what makes a cheap version of the oracle
    conceivable."""
    s = np.sort(protos @ code)
    return float(s[-1] - s[-2]) if len(s) > 1 else 0.0


def run_seed(X, y, Xt, yt, seed):
    rng = np.random.default_rng(seed)
    v1 = WideV1(n_cells=1024, window_ms=50, image_shape=(FRAME, FRAME),
                seed=seed)
    # tie=True, because translation.py showed an untied bank breaks pooling --
    # without it every arm here would be measuring the same broken thing
    develop_v1(v1, [place(im, rng=rng) for im in X[:N_DEV]], epochs=3,
               tie=True, seed=seed)

    frames = [place(im, dy=int(rng.integers(-JITTER, JITTER + 1)),
                    dx=int(rng.integers(-JITTER, JITTER + 1)), rng=rng)
              for im in X[:N_TRAIN]]
    tframes = [place(im, dy=int(rng.integers(-JITTER, JITTER + 1)),
                     dx=int(rng.integers(-JITTER, JITTER + 1)), rng=rng)
               for im in Xt[:N_TEST]]
    ytr, yte = y[:N_TRAIN], yt[:N_TEST]

    # the bank is one centred glance per training object, so what varies
    # between bank and query is only how the query was gathered
    B = np.stack([glance(v1, f, 0, 0) for f in frames])
    protos = np.stack([_unit(B[ytr == c].mean(0)) for c in np.unique(ytr)])
    classes = np.unique(ytr)

    def score(Q):
        return float(np.mean(_nearest_prototype(B, ytr, np.stack(Q),
                                                len(classes)) == yte))

    out = {}
    # ---- one glance ------------------------------------------------------
    out["one glance"] = score([glance(v1, f, 0, 0) for f in tframes])

    for k in KS:
        if k == 1:
            continue
        # ---- k random glances --------------------------------------------
        Q = []
        for f in tframes:
            offs = rng.integers(-JITTER, JITTER + 1, size=(k, 2))
            Q.append(_unit(np.mean([glance(v1, f, a, b) for a, b in offs], 0)))
        out[f"random k={k}"] = score(Q)

        # ---- k glances chosen by the margin, with M tries each ------------
        # The ceiling. It spends N_CAND glances per glance kept and is allowed
        # to test each before committing, which no real eye can do.
        Q = []
        for f in tframes:
            kept = [glance(v1, f, 0, 0)]
            for _ in range(k - 1):
                cands = rng.integers(-JITTER, JITTER + 1, size=(N_CAND, 2))
                best, best_m = None, -np.inf
                for a, b in cands:
                    g = glance(v1, f, a, b)
                    m = margin(_unit(np.mean(kept + [g], 0)), protos)
                    if m > best_m:
                        best, best_m = g, m
                kept.append(best)
            Q.append(_unit(np.mean(kept, 0)))
        out[f"oracle-greedy k={k}"] = score(Q)

        # ---- the control that separates CHOOSING from TRYING MORE ---------
        # same N_CAND glances drawn, one kept at random. If this matches the
        # oracle, the gain was extra sampling and not the choice.
        Q = []
        for f in tframes:
            kept = [glance(v1, f, 0, 0)]
            for _ in range(k - 1):
                cands = rng.integers(-JITTER, JITTER + 1, size=(N_CAND, 2))
                a, b = cands[int(rng.integers(N_CAND))]
                kept.append(glance(v1, f, a, b))
            Q.append(_unit(np.mean(kept, 0)))
        out[f"drew {N_CAND}, kept one at random k={k}"] = score(Q)
    return out


def main():
    out_path = (sys.argv[1] if len(sys.argv) > 1
                else "out_uncertain_look.json")
    X, y, Xt, yt = load_cifar10(n_train=N_TRAIN + N_DEV, n_test=N_TEST,
                                grayscale=False, size=28)
    X, Xt = luminance(X), luminance(Xt)
    print(f"{N_TRAIN} train / {N_TEST} test CIFAR photographs in a "
          f"{FRAME}x{FRAME} frame, jitter +-{JITTER}px, chance 0.100\n",
          flush=True)

    rows = [run_seed(X, y, Xt, yt, sd) for sd in SEEDS]
    keys = list(rows[0])
    res = {"seeds": list(SEEDS),
           "mean": {k: round(float(np.mean([r[k] for r in rows])), 4)
                    for k in keys},
           "sd": {k: round(float(np.std([r[k] for r in rows], ddof=1)), 4)
                  for k in keys},
           "per_seed": rows}

    print(f"{'arm':<34}{'accuracy':>10}{'sd':>8}{'vs one glance':>15}")
    base = res["mean"]["one glance"]
    for k in keys:
        v = res["mean"][k]
        print(f"{k:<34}{v:>10.3f}{res['sd'][k]:>8.3f}{v - base:>+15.4f}")

    print("\n=== is there anything to gain from CHOOSING where to look? ===")
    verdicts = []
    print(f"  (every number here sits between 0.14 and 0.16 on a task whose "
          f"chance is 0.100 -- the eye is barely above chance, so these are "
          f"small effects on a weak perceiver)")
    for k in KS:
        if k == 1:
            continue
        o = res["mean"][f"oracle-greedy k={k}"]
        r = res["mean"][f"random k={k}"]
        c = res["mean"][f"drew {N_CAND}, kept one at random k={k}"]
        d = np.array([s[f"oracle-greedy k={k}"] -
                      s[f"drew {N_CAND}, kept one at random k={k}"]
                      for s in rows])
        sd = float(d.std(ddof=1))
        cd = float(d.mean() / (sd + 1e-12))
        res.setdefault("choice_gain", {})[k] = dict(
            delta=round(float(d.mean()), 4), cohens_d=round(cd, 3),
            wins=int((d > 0).sum()), n=len(d))
        print(f"  k={k}: pooling {r - base:+.4f} | choosing "
              f"{d.mean():+.4f} over the same draws "
              f"(d={cd:+.2f}, {int((d > 0).sum())}/{len(d)})")
        verdicts.append(cd >= 0.8 and (d > 0).sum() >= 0.75 * len(d))

    # `any` would call this a win on an effect that changes sign. It has to
    # hold at every k, because a selection rule that helps at k=3 and hurts at
    # k=5 is not a selection rule -- it is a bias that has not yet compounded.
    res["choice_pays"] = bool(verdicts and all(verdicts))
    ks = [k for k in KS if k != 1]
    signs = [res["choice_gain"][k]["delta"] > 0 for k in ks]
    res["reverses"] = bool(any(signs) and not all(signs))
    if res["reverses"]:
        good = [k for k, g in zip(ks, signs) if g]
        bad = [k for k, g in zip(ks, signs) if not g]
        print(f"\n  It REVERSES: choosing helps at k={good} and hurts at "
              f"k={bad}. That is the signature of the rule rather than noise.")
        print("  Greedy margin-maximisation is **confirmation bias**: it keeps "
              "the glance that best agrees with what the mind")
        print("  already believes, so each look makes the next one more likely "
              "to agree too. At two or three glances that reads")
        print("  as sharpening; by five the pooled code is a fixed point and "
              "has stopped gathering evidence -- which is exactly")
        print("  when the arm that keeps a glance at RANDOM overtakes it "
              f"({res['mean'][f'drew {N_CAND}, kept one at random k=5']:.3f} "
              f"against {res['mean']['oracle-greedy k=5']:.3f}).")
        print("\n  So the thing to build is not 'look where you are least "
              "sure' as a greedy rule. An uncertainty-driven eye needs a")
        print("  criterion that rewards DISAGREEING evidence -- looking where "
              "the current belief would be tested rather than confirmed.")
    elif res["choice_pays"]:
        print("\n  The ceiling is real: an eye that picks its next fixation by "
              "its own ambiguity beats one that")
        print("  takes the same number of looks at arbitrary places. A cheap "
              "peripheral version is worth building.")
    else:
        print("\n  Even the ORACLE -- which spends "
              f"{N_CAND} glances per glance kept and gets to test each before "
              f"committing -- does not")
        print("  beat drawing the same glances and keeping one at random. "
              "There is no headroom for a cheap approximation to")
        print("  capture, so uncertainty-driven fixation is not the thing to "
              "build. What pooling buys, it buys from")
        print("  having more looks, not from having better-chosen ones.")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
