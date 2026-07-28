"""Re-evaluation of the threshold=1.5 decision. Three things the sweep did not prove.

1. **Held-out validation.** 1.5 was *chosen* on seeds 0-4. A value picked on a
   sweep and reported on the same sweep is fitted to it. Re-run on seeds 10-24,
   never used in the selection.

2. **Where does high noise actually break it?** The sweep showed recall staying
   at 100% while named yield collapsed, and the docstring now claims the fault
   is in the classifier rather than the onset detector. That was inferred, not
   tested. Here the classifier is fed *ground-truth-segmented* events -- the
   detector removed from the path entirely -- so if it still fails, the claim
   holds causally.

3. **Does the extra false-alarm rate poison anything downstream?** More
   detections means more junk entering binding. Measured, not assumed.

Usage:  python3 benchmarks/ear_threshold_validate.py out_validate.json
"""
import json, sys
import numpy as np

import neurobrain as nb
from neurobrain.sensing.streams import (build_soundscape, ContinuousEar,
                                        StreamingBrain, onset_scores, _unit)
from neurobrain.vision.widev1 import _nearest_prototype
from neurobrain.audition.audio import sound_dataset

HELD_OUT = tuple(range(10, 25))          # never used to choose the threshold
OLD, NEW = 2.5, 1.5
out = {}


def _bank(brain, seed):
    sigs, labels, names = sound_dataset(n_per_class=8, seed=seed + 5)
    X = np.array([brain.belt.code(brain.ear.coch.forward(s)[0]) for s in sigs],
                 np.float32)
    perm = np.random.default_rng(seed + 11).permutation(len(X))
    X, y = X[perm], np.asarray(labels)[perm]
    return X[:len(X) // 2], y[:len(X) // 2], len(names)


def named_yield(threshold, seed, noise=0.05, gap=(0.15, 0.60)):
    brain = StreamingBrain(seed=seed)
    bx, by, ncls = _bank(brain, seed)
    scape = build_soundscape(n_events=24, gap_s=gap, noise=noise, seed=seed)
    ear = ContinuousEar(threshold=threshold)
    found = ear.detect_onsets(scape.wave)
    s = onset_scores(found, scape.events, sr=scape.sr)
    tol, used, ok, fa = 0.12 * scape.sr, set(), 0, 0
    for f in found:
        cand = [(j, e) for j, e in enumerate(scape.events)
                if abs(e.start - f) <= tol and j not in used]
        pred = int(_nearest_prototype(
            bx, by, brain.belt.code(ear.listen(scape.wave, f))[None], ncls)[0])
        if cand:
            j, ev = cand[0]; used.add(j); ok += int(pred == ev.label)
        else:
            fa += 1
    mins = len(scape.wave) / scape.sr / 60.0
    return dict(named_yield=ok / len(scape.events), recall=s["recall"],
                precision=s["precision"], f1=s["f1"],
                fa_per_min=fa / max(mins, 1e-9))


# ---------------------------------------------------- 1. held-out validation
def validate():
    rows = {OLD: [], NEW: []}
    for sd in HELD_OUT:
        for th in (OLD, NEW):
            rows[th].append(named_yield(th, sd))
    res = {}
    for th, rs in rows.items():
        m = lambda k: float(np.mean([r[k] for r in rs]))
        sd_ = lambda k: float(np.std([r[k] for r in rs]))
        res[str(th)] = dict(named_yield=round(m("named_yield"), 4),
                            named_yield_sd=round(sd_("named_yield"), 4),
                            recall=round(m("recall"), 4),
                            precision=round(m("precision"), 4),
                            f1=round(m("f1"), 4),
                            fa_per_min=round(m("fa_per_min"), 3), n=len(rs))
    a = [r["named_yield"] for r in rows[NEW]]
    b = [r["named_yield"] for r in rows[OLD]]
    diff = np.array(a) - np.array(b)
    res["paired_delta_mean"] = round(float(diff.mean()), 4)
    res["paired_delta_sd"] = round(float(diff.std(ddof=1)), 4)
    res["seeds_where_new_wins"] = int((diff > 0).sum())
    res["seeds_where_new_loses"] = int((diff < 0).sum())
    res["n_seeds"] = len(diff)
    # simple effect size; with n=15 paired seeds this is enough to be decisive
    res["cohens_d"] = round(float(diff.mean() / (diff.std(ddof=1) + 1e-12)), 3)
    print("1. HELD-OUT SEEDS 10-24 (never used to pick the threshold)", flush=True)
    for th in (OLD, NEW):
        r = res[str(th)]
        print(f"   thr={th}: named_yield {r['named_yield']:.3f} (sd {r['named_yield_sd']:.3f})"
              f"  recall {r['recall']:.3f}  FA/min {r['fa_per_min']:.2f}", flush=True)
    print(f"   paired delta {res['paired_delta_mean']:+.3f}, "
          f"new wins on {res['seeds_where_new_wins']}/{res['n_seeds']} seeds, "
          f"d={res['cohens_d']}", flush=True)
    return res


# ------------------------------- 2. is the noise failure the classifier's fault?
def noise_locus(noises=(0.02, 0.05, 0.15, 0.30), seeds=(10, 11, 12)):
    """Remove the detector from the path: hand the classifier the TRUE onsets.

    If accuracy still collapses with perfect segmentation, the detector is
    exonerated and the belt code / prototype bank is where the loss lives."""
    rows = []
    for nz in noises:
        per_true, per_det = [], []
        for sd in seeds:
            brain = StreamingBrain(seed=sd)
            bx, by, ncls = _bank(brain, sd)
            scape = build_soundscape(n_events=24, noise=nz, seed=sd)
            ear = ContinuousEar(threshold=NEW)
            # (a) perfect segmentation -- classifier alone
            ok = 0
            for ev in scape.events:
                pred = int(_nearest_prototype(
                    bx, by, brain.belt.code(ear.listen(scape.wave, ev.start))[None],
                    ncls)[0])
                ok += int(pred == ev.label)
            per_true.append(ok / len(scape.events))
            # (b) the ear's own onsets
            per_det.append(named_yield(NEW, sd, noise=nz)["named_yield"])
        rows.append(dict(noise=nz,
                         classifier_on_true_onsets=round(float(np.mean(per_true)), 4),
                         full_pipeline=round(float(np.mean(per_det)), 4)))
        print(f"   noise {nz:<5} classifier given TRUE onsets "
              f"{rows[-1]['classifier_on_true_onsets']:.3f}   "
              f"full pipeline {rows[-1]['full_pipeline']:.3f}", flush=True)
    drop = rows[0]["classifier_on_true_onsets"] - rows[-1]["classifier_on_true_onsets"]
    print(f"   -> classifier alone loses {drop:.1%} from noise 0.02 to 0.30 "
          f"with segmentation handed to it", flush=True)
    return dict(rows=rows, classifier_loss_from_noise=round(float(drop), 4))


# --------------------------- 3. do the extra false alarms poison the binding?
def binding_cost(seeds=(10, 11, 12)):
    """Bind every detected event to a visual code and see whether the noisier
    detector degrades cross-modal recall. False alarms have no true label, so
    they are bound to whatever the classifier calls them -- exactly the junk
    path a real system would suffer."""
    trx, trY, _, _ = nb.load_mnist(n_train=1200, n_test=400)
    res = {}
    for th in (OLD, NEW):
        accs, junk = [], []
        for sd in seeds:
            brain = StreamingBrain(seed=sd)
            bx, by, ncls = _bank(brain, sd)
            scape = build_soundscape(n_events=24, seed=sd)
            ear = ContinuousEar(threshold=th)
            found = ear.detect_onsets(scape.wave)
            tol, used, n_junk = 0.12 * scape.sr, set(), 0
            pairs = []
            for f in found:
                a = brain.belt.code(ear.listen(scape.wave, f))
                pred = int(_nearest_prototype(bx, by, a[None], ncls)[0])
                cand = [(j, e) for j, e in enumerate(scape.events)
                        if abs(e.start - f) <= tol and j not in used]
                if cand:
                    j, ev = cand[0]; used.add(j); lab = ev.label
                else:
                    lab = pred; n_junk += 1          # junk enters with a guess
                vi = np.where(trY == lab % 10)[0][0]
                pairs.append((_unit(brain.v1.rate(trx[vi])), a, lab))
            if not pairs:
                continue
            brain.calibrate(np.array([p[0] for p in pairs], np.float32),
                            np.array([p[1] for p in pairs], np.float32))
            for v, a, lab in pairs:
                brain.bind(v, a, int(lab))
            # recall on the TRUE events only -- junk must not be scored as a win
            ok = n = 0
            for ev in scape.events:
                got = brain.recall_visual_from_sound(
                    brain.belt.code(ear.listen(scape.wave, ev.start)))
                if got is not None:
                    n += 1; ok += int(got == ev.label)
            accs.append(ok / max(n, 1)); junk.append(n_junk)
        res[str(th)] = dict(binding_recall=round(float(np.mean(accs)), 4),
                            junk_bindings=round(float(np.mean(junk)), 2))
        print(f"   thr={th}: binding recall on true events "
              f"{res[str(th)]['binding_recall']:.3f}  "
              f"(junk bindings admitted: {res[str(th)]['junk_bindings']:.1f})", flush=True)
    return res


out["heldout"] = validate()
print("\n2. WHERE DOES NOISE BREAK IT?", flush=True)
out["noise_locus"] = noise_locus()
print("\n3. DO FALSE ALARMS POISON BINDING?", flush=True)
out["binding_cost"] = binding_cost()

json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "out_validate.json", "w"),
          indent=1)
print("\nDONE", flush=True)
