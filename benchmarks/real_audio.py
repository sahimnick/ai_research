"""Does the auditory front end work on REAL environmental sound?

Every auditory number in this project so far came from synthetic stimuli: pure
tones, linear chirps, harmonic stacks, amplitude modulations. Those were built
to differ along one axis at a time, which is what made them useful for isolating
mechanisms -- and also what makes them a poor test of whether hearing works.

ESC-50 is the other thing. 50 classes of real field recording, 5 s each,
grouped by ESC-50's own five categories. The two this benchmark cares about most
are the two the goal names:

    nature   rain, sea waves, crackling fire, crickets, chirping birds, water
             drops, wind, pouring water, toilet flush, thunderstorm
    urban    helicopter, chainsaw, siren, car horn, engine, train, church
             bells, airplane, fireworks, hand saw

A siren has a doppler shift and reverberation. Rain has no onset at all -- it is
stationary noise with a spectral signature and nothing else. An engine's
harmonics drift with load. Nothing here is a 400 ms tone.

Five front ends are compared on identical clips:

    raw_cochleagram   the mel/log spectrum, pooled -- the control
    belt_pooled       AuditoryBelt as it was: fully shift-invariant, keeping
                      only WHICH filter fired
    belt_tonotopic    AuditoryBelt with a second, frequency-resolved channel
                      alongside the invariant one
    gabors            spectrotemporal Gabor bank -- an inherited layout,
                      unlearned
    predictive_a1     PredictiveA1, learned by predicting its own next input
    multiscale_a1     MultiScaleA1, three windows at once (8/16/32 lags)

The two belts are both here because the choice between them cannot be made on
real audio alone. The fully-pooled code was set by measurement on the synthetic
8-class bank, where it was worth +50 points, so replacing it needs the synthetic
result re-measured too -- which `--synthetic` does in the same run.

and three read-outs are reported for each, because they answer different
questions:

    proto   nearest class-mean in the code's own cosine metric. A strong
            requirement: the classes must be arranged so raw similarity finds
            them.
    knn5    5-nearest-neighbour. Same metric, but a class may be several
            clusters -- "rain" and "wind" are not single points.
    probe   one linear layer, fitted in closed form. Says how much class
            information is present *at all*, regardless of whether cosine
            similarity happens to be the right way to read it.

A large probe-minus-proto gap is itself a finding: it means the information is
there but the geometry is wrong, which is a fixable problem and a different one
from the code being empty.

Methodology notes
-----------------
* The learned front ends are trained on the **training split only**. The earlier
  inline version of this measurement trained them on everything, which leaks.
* Labels are remapped to ``0..k-1`` per subset. They are not remapped because
  the read-outs need it -- they no longer do -- but because a subset of 10
  classes should report against a 10-class chance level, and a stray label range
  should never again be able to masquerade as a score.
* The split is stratified per class, so chance is exactly ``1/k``.

Usage:  python3 benchmarks/real_audio.py out_real_audio.json [n_shards]
"""
import json
import sys
import time

import numpy as np

from neurobrain.audition.audio import Cochleagram
from neurobrain.audition.auditorycortex import (MultiScaleA1, PredictiveA1,
                                                linear_probe,
                                                spectrotemporal_gabor_bank,
                                                spectrotemporal_context,
                                                on_off_channels)
from neurobrain.sensing.natural import (ESC50_CATEGORIES, esc50_category_of,
                                        load_esc50)
from neurobrain.sensing.streams import AuditoryBelt
from neurobrain.vision.widev1 import _nearest_prototype, _unit

SR, DUR = 8000, 2.0
N_FREQ = 36
TRAIN_FRAC = 0.6
SUBSETS = ("all", "animals", "nature", "human", "interior", "urban",
           "nature+urban")


# ------------------------------------------------------------------ split ---
def stratified(y, frac=TRAIN_FRAC, seed=0):
    """Per-class split, so chance is exactly 1/k and no class is train-only."""
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        cut = max(1, int(round(len(idx) * frac)))
        tr.extend(idx[:cut].tolist())
        te.extend(idx[cut:].tolist())
    return np.array(sorted(tr)), np.array(sorted(te))


def knn(Xtr, ytr, Xte, k=5):
    """Cosine k-NN with a plurality vote -- a class may be several clusters."""
    S = Xte @ Xtr.T
    k = min(k, Xtr.shape[0])
    nn = np.argpartition(-S, k - 1, axis=1)[:, :k]
    out = np.empty(len(Xte), ytr.dtype)
    for i in range(len(Xte)):
        lab, cnt = np.unique(ytr[nn[i]], return_counts=True)
        out[i] = lab[cnt.argmax()]
    return out


# ------------------------------------------------------------- front ends ---
def pooled_cochleagram(coch, w=16, step=8):
    """The control: mean+max over time of the raw spectrum, unit-normed.

    Pooling matters even here. A clip's spectrum at frame 3 and frame 300 are
    different vectors describing the same source, so a time-specific code pays
    for the alignment it never had."""
    C = np.asarray(coch, np.float32)
    sl = [C[:, i:i + w] for i in range(0, max(C.shape[1] - w + 1, 1), step)]
    R = np.array([s.mean(1) for s in sl], np.float32)
    return _unit(np.concatenate([R.mean(0), R.max(0)]))


class GaborFront:
    """An inherited spectrotemporal layout, never trained.

    Gabors in the frequency-by-lag plane: tilt is a sweep rate, wavelength a
    modulation rate. A cortex has this before it has experience, so it is the
    right baseline for asking what learning actually added."""

    def __init__(self, n_units=128, n_lags=16, seed=0):
        self.W = spectrotemporal_gabor_bank(n_units, N_FREQ, n_lags, seed=seed)
        self.n_lags = n_lags

    def code(self, coch, step=4):
        ctx, _ = spectrotemporal_context(on_off_channels(coch), self.n_lags, step)
        if not len(ctx):
            return np.zeros(2 * self.W.shape[0], np.float32)
        X = np.asarray(ctx, np.float32)
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-6)
        R = np.maximum(X @ self.W.T, 0.0)
        return _unit(np.concatenate([R.mean(0), R.max(0)]))


def build_frontends(train_cochs, seed=0, verbose=True):
    """Every front end, with the learned ones fitted on TRAIN cochleagrams only."""
    fe = {}
    fe["raw_cochleagram"] = lambda c: pooled_cochleagram(c)

    bp = AuditoryBelt(n_freq=N_FREQ, tonotopic=False, seed=seed)
    fe["belt_pooled"] = lambda c: bp.code(c)

    bt = AuditoryBelt(n_freq=N_FREQ, tonotopic=True, seed=seed)
    fe["belt_tonotopic"] = lambda c: bt.code(c)

    g = GaborFront(seed=seed)
    fe["gabors"] = lambda c: g.code(c)

    t0 = time.time()
    a1 = PredictiveA1(n_freq=N_FREQ, n_units=128, n_lags=16, seed=seed)
    curve = a1.train(train_cochs, epochs=2, lr=0.03, step=6, seed=seed)
    fe["predictive_a1"] = lambda c: a1.code(c, step=4)
    if verbose:
        print(f"   PredictiveA1 trained: err {curve[0]:.3f} -> {curve[-1]:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    t0 = time.time()
    ms = MultiScaleA1(n_freq=N_FREQ, lags=(8, 16, 32), n_units=64, seed=seed)
    last = ms.train(train_cochs, epochs=2, lr=0.05, step=6, seed=seed)
    fe["multiscale_a1"] = lambda c: ms.code(c, step=4)
    if verbose:
        print(f"   MultiScaleA1 trained: {[round(v,3) for v in last]} "
              f"({time.time()-t0:.0f}s)", flush=True)
    return fe


# ------------------------------------------------- the synthetic control ---
def synthetic_control(seed=0, n_per_class=40):
    """The 8-class designed bank, where the fully-pooled belt was chosen.

    This is the result the pooled code was set by: classes that differ by
    *pattern* with their absolute pitch randomised over an octave and a half, so
    a frequency-specific code cannot generalise and pooling frequency away is
    worth ~50 points. If a tonotopic channel is to be added by default, this
    number must not fall -- otherwise the fix on real audio is a regression
    everywhere else, and the honest answer is two belts rather than one.
    """
    from neurobrain.audition.audio import sound_dataset

    sigs, y, names = sound_dataset(n_per_class=n_per_class, seed=seed)
    coch = Cochleagram(n_freq=N_FREQ)
    C = [coch.forward(s)[0] for s in sigs]
    tr, te = stratified(np.asarray(y), seed=seed)
    out = {}
    print(f"\n--- synthetic control: {len(C)} clips, {len(names)} classes, "
          f"chance {1/len(names):.3f} ---")
    print(f"   {'front end':<17}{'proto':>8}{'knn5':>8}{'probe':>8}")
    for nm, belt in (("belt_pooled", AuditoryBelt(n_freq=N_FREQ, tonotopic=False,
                                                  seed=seed)),
                     ("belt_tonotopic", AuditoryBelt(n_freq=N_FREQ,
                                                     tonotopic=True, seed=seed))):
        X = np.array([belt.code(c) for c in C], np.float32)
        p = float(np.mean(_nearest_prototype(X[tr], y[tr], X[te], len(names)) == y[te]))
        kn = float(np.mean(knn(X[tr], y[tr], X[te]) == y[te]))
        lp = float(linear_probe(X[tr], y[tr], X[te], y[te]))
        out[nm] = dict(proto=round(p, 4), knn5=round(kn, 4), probe=round(lp, 4))
        print(f"   {nm:<17}{p:>8.3f}{kn:>8.3f}{lp:>8.3f}")
    d = out["belt_tonotopic"]["proto"] - out["belt_pooled"]["proto"]
    print(f"   tonotopic costs {d:+.3f} on the designed bank"
          f"{'  -- REGRESSION' if d < -0.05 else ''}")
    return out


# ------------------------------------------------------------------- main ---
def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_real_audio.json"
    n_shards = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    waves, y_raw, names = load_esc50(n_shards=n_shards, sr=SR, dur_s=DUR)
    coch = Cochleagram(sr=SR, n_freq=N_FREQ)
    C = [coch.forward(w)[0] for w in waves]
    cats = np.array([esc50_category_of(v) for v in y_raw])
    print(f"ESC-50: {len(C)} clips, {len(np.unique(y_raw))} classes present, "
          f"cochleagram {C[0].shape}", flush=True)
    for c in ESC50_CATEGORIES:
        print(f"   {c:<9} {int((cats == c).sum())} clips", flush=True)

    # the split is on the FULL set, once, so every front end and every subset
    # sees the same clips in the same roles
    tr_all, te_all = stratified(y_raw, seed=0)
    print(f"\ntraining front ends on {len(tr_all)} clips "
          f"(held out {len(te_all)})", flush=True)
    fes = build_frontends([C[i] for i in tr_all], seed=0)

    res = {"n_clips": len(C), "n_train": int(len(tr_all)),
           "n_test": int(len(te_all)), "subsets": {}, "by_frontend": {}}

    codes = {}
    for fname, fn in fes.items():
        t0 = time.time()
        codes[fname] = np.array([fn(c) for c in C], np.float32)
        print(f"   coded {fname:<16} dim={codes[fname].shape[1]:<5} "
              f"({time.time()-t0:.0f}s)", flush=True)

    for sub in SUBSETS:
        if sub == "all":
            keep = np.ones(len(C), bool)
        else:
            wanted = sub.split("+")
            keep = np.isin(cats, wanted)
        idx = np.flatnonzero(keep)
        if len(idx) < 20:
            continue
        lab_raw = y_raw[idx]
        uniq = np.unique(lab_raw)
        remap = {int(v): i for i, v in enumerate(uniq)}       # -> 0..k-1
        lab = np.array([remap[int(v)] for v in lab_raw])
        k = len(uniq)
        tr = np.flatnonzero(np.isin(idx, tr_all))
        te = np.flatnonzero(np.isin(idx, te_all))
        res["subsets"][sub] = {"n": int(len(idx)), "n_class": k,
                              "chance": round(1.0 / k, 4),
                              "n_train": int(len(tr)), "n_test": int(len(te))}
        print(f"\n{sub}  n={len(idx)}  classes={k}  chance={1/k:.3f}  "
              f"train={len(tr)} test={len(te)}", flush=True)
        print(f"   {'front end':<17}{'proto':>8}{'knn5':>8}{'probe':>8}"
              f"{'best/chance':>13}", flush=True)
        for fname in fes:
            X = codes[fname]
            Xtr, Xte = X[idx][tr], X[idx][te]
            ytr, yte = lab[tr], lab[te]
            p = float(np.mean(_nearest_prototype(Xtr, ytr, Xte, k) == yte))
            kn = float(np.mean(knn(Xtr, ytr, Xte) == yte))
            lp = float(linear_probe(Xtr, ytr, Xte, yte))
            best = max(p, kn, lp)
            res["by_frontend"].setdefault(fname, {})[sub] = dict(
                proto=round(p, 4), knn5=round(kn, 4), probe=round(lp, 4),
                over_chance=round(best * k, 2))
            print(f"   {fname:<17}{p:>8.3f}{kn:>8.3f}{lp:>8.3f}"
                  f"{best * k:>12.1f}x", flush=True)

    # -------------------------------------------------------- the verdict ---
    print("\n=== does the front end hear the real world? ===")
    for sub in ("nature", "urban", "animals", "all"):
        if sub not in res["subsets"]:
            continue
        ch = res["subsets"][sub]["chance"]
        rows = [(f, res["by_frontend"][f][sub]) for f in fes]
        bf, br = max(rows, key=lambda r: max(r[1]["proto"], r[1]["knn5"],
                                             r[1]["probe"]))
        best = max(br["proto"], br["knn5"], br["probe"])
        how = ("proto" if best == br["proto"] else
               "knn5" if best == br["knn5"] else "probe")
        verdict = ("yes" if best >= 3 * ch else
                   "weakly" if best >= 1.6 * ch else "no")
        print(f"  {sub:<9} best {best:.3f} ({bf}, {how}) vs chance {ch:.3f} "
              f"-> {best/ch:.1f}x  {verdict}")
        geom = br["probe"] - max(br["proto"], br["knn5"])
        if geom > 0.08:
            print(f"            probe beats similarity by {geom:+.3f}: the "
                  f"information is present but the GEOMETRY is wrong")

    # does the tonotopic channel earn its place, on real audio and on the
    # synthetic bank the pooled code was originally chosen by?
    print("\n=== the belt's two channels ===")
    for sub in ("all", "nature", "urban", "animals", "interior"):
        if sub not in res["subsets"]:
            continue
        a = res["by_frontend"]["belt_pooled"][sub]
        b = res["by_frontend"]["belt_tonotopic"][sub]
        ba, bb = max(a["proto"], a["knn5"]), max(b["proto"], b["knn5"])
        print(f"  {sub:<9} pooled {ba:.3f} -> tonotopic {bb:.3f}  ({bb-ba:+.3f})")
    res["synthetic_control"] = synthetic_control(seed=0)

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
