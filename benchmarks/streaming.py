"""Audio, detection-in-clutter, and cross-modal binding -- each against a control."""
import json, time, sys, traceback
import numpy as np
import neurobrain as nb
from neurobrain.sensing.streams import (build_scene, build_soundscape, SaccadicEye,
                                        StreamingBrain, onset_scores, _unit)
from neurobrain.vision.widev1 import _nearest_prototype
from neurobrain.audition.audio import sound_dataset

out = {}
SEED = 0


# ---------------------------------------------------------------- AUDIO ----
def audio(scape_seeds=(0, 1, 2)):
    """Does the ear find its own event boundaries, and does the code survive it?"""
    rows = []
    for sd in scape_seeds:
        brain = StreamingBrain(seed=sd)
        scape = build_soundscape(n_events=24, seed=sd)
        t0 = time.time()
        found = brain.ear.detect_onsets(scape.wave)
        t_onset = time.time() - t0
        s = onset_scores(found, scape.events, sr=scape.sr)

        sigs, labels, names = sound_dataset(n_per_class=8, seed=sd + 5)
        Xa = np.array([brain.belt.code(brain.ear.coch.forward(sg)[0]) for sg in sigs], np.float32)
        Xraw = np.array([_unit(brain.ear.coch.forward(sg)[0].reshape(-1)) for sg in sigs], np.float32)
        perm = np.random.default_rng(sd + 11).permutation(len(Xa))
        Xa, Xraw, labels = Xa[perm], Xraw[perm], np.asarray(labels)[perm]
        half, ncls = len(Xa) // 2, len(names)
        bx, by = Xa[:half], labels[:half]

        preseg = float(np.mean(_nearest_prototype(bx, by, Xa[half:], ncls) == labels[half:]))
        raw = float(np.mean(_nearest_prototype(Xraw[:half], by, Xraw[half:], ncls) == labels[half:]))

        # streaming: classify what the ear itself decided was an event
        tol = 0.12 * scape.sr
        hits = ok = 0
        for f in found:
            truth = [e for e in scape.events if abs(e.start - f) <= tol]
            if not truth:
                continue
            hits += 1
            code = brain.belt.code(brain.ear.listen(scape.wave, f))
            ok += int(int(_nearest_prototype(bx, by, code[None], ncls)[0]) == truth[0].label)
        rows.append(dict(seed=sd, onset_precision=round(s["precision"], 4),
                         onset_recall=round(s["recall"], 4), onset_f1=round(s["f1"], 4),
                         n_found=int(s["n_found"]), n_true=int(s["n_true"]),
                         belt_presegmented=round(preseg, 4),
                         raw_cochleagram=round(raw, 4),
                         streaming=round(ok / max(hits, 1), 4),
                         classifiable_events=hits, chance=round(1 / ncls, 4),
                         sec_onset=round(t_onset, 3),
                         wave_seconds=round(len(scape.wave) / scape.sr, 1)))
        print("audio", json.dumps(rows[-1]), flush=True)
    agg = lambda k: round(float(np.mean([r[k] for r in rows])), 4)
    return dict(runs=rows, mean_onset_f1=agg("onset_f1"),
                mean_streaming=agg("streaming"), mean_presegmented=agg("belt_presegmented"),
                mean_raw=agg("raw_cochleagram"), chance=rows[0]["chance"])


# ------------------------------------------------------------ DETECTION ----
def detection(dataset, loader, seeds=(0, 1, 2)):
    """Does saliency-driven looking beat looking at random pixels?"""
    trx, trY, tex, teY = loader(n_train=1200, n_test=400)
    rows = []
    for sd in seeds:
        brain = StreamingBrain(seed=sd)
        scene = build_scene(trx, trY, size=256, n_objects=12, seed=sd)
        tile = scene.tile
        chance = (12 * tile * tile) / (256.0 * 256.0)

        t0 = time.time()
        fix = SaccadicEye(scene, seed=sd).free_view(n_saccades=60, correct=True)
        t_view = time.time() - t0
        raw = [f for f in SaccadicEye(scene, seed=sd).free_view(n_saccades=60, correct=False)
               if f.true_label >= 0]
        on = [f for f in fix if f.true_label >= 0]

        # read-out trained on centred crops, applied to what the eye actually got
        Xc = np.array([_unit(brain.v1.rate(im)) for im in trx[:600]], np.float32)
        yc = trY[:600]
        Xt = np.array([_unit(brain.v1.rate(im)) for im in tex[:300]], np.float32)
        preseg = float(np.mean(_nearest_prototype(Xc, yc, Xt, 10) == teY[:300]))
        stream = corr = None
        if on:
            Xs = np.array([_unit(brain.v1.rate_over(f.frames)) for f in on], np.float32)
            stream = float(np.mean(_nearest_prototype(Xc, yc, Xs, 10)
                                   == np.array([f.true_label for f in on])))
        if raw:
            Xr = np.array([_unit(brain.v1.rate_over(f.frames)) for f in raw], np.float32)
            corr = float(np.mean(_nearest_prototype(Xc, yc, Xr, 10)
                                 == np.array([f.true_label for f in raw])))
        rows.append(dict(seed=sd, on_object_rate=round(len(on) / max(len(fix), 1), 4),
                         chance_on_object=round(chance, 4),
                         lift_over_chance=round((len(on) / max(len(fix), 1)) / max(chance, 1e-9), 2),
                         n_saccades=len(fix), n_on_object=len(on),
                         presegmented_acc=round(preseg, 4),
                         streaming_acc=round(stream, 4) if stream is not None else None,
                         uncorrected_acc=round(corr, 4) if corr is not None else None,
                         sec_freeview=round(t_view, 2)))
        print(f"detect[{dataset}]", json.dumps(rows[-1]), flush=True)
    agg = lambda k: round(float(np.mean([r[k] for r in rows if r[k] is not None])), 4)
    return dict(dataset=dataset, runs=rows, mean_on_object=agg("on_object_rate"),
                mean_chance=agg("chance_on_object"), mean_lift=agg("lift_over_chance"),
                mean_presegmented=agg("presegmented_acc"),
                mean_streaming=agg("streaming_acc"), mean_uncorrected=agg("uncorrected_acc"))


# -------------------------------------------------------------- BINDING ----
def binding(seeds=(0, 1, 2)):
    """Bind a seen digit to a heard sound, then recall one from the other.

    Control: bind the SAME codes to shuffled labels. If the shuffled condition
    scores as well, nothing was learned about the pairing."""
    trx, trY, tex, teY = nb.load_mnist(n_train=1200, n_test=400)
    sigs, labels, names = sound_dataset(n_per_class=8, seed=3)
    rows = []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        for shuffled in (False, True):
            brain = StreamingBrain(seed=sd)
            # one visual code + one audio code per class, bound together
            n_cls = min(10, len(names))
            pairs = []
            for c in range(n_cls):
                vi = np.where(trY == c)[0][:6]
                ai = np.where(np.asarray(labels) == c % len(names))[0][:6]
                if not len(vi) or not len(ai):
                    continue
                for k in range(min(len(vi), len(ai))):
                    v = _unit(brain.v1.rate(trx[vi[k]]))
                    a = brain.belt.code(brain.ear.coch.forward(sigs[ai[k]])[0])
                    pairs.append((v, a, c))
            tgt = [p[2] for p in pairs]
            if shuffled:
                tgt = list(rng.permutation(tgt))
            for (v, a, _), lab in zip(pairs, tgt):
                brain.bind(v, a, int(lab))
            # recall the visual identity from sound alone
            ok = n = 0
            for (v, a, c), lab in zip(pairs, tgt):
                got = brain.recall_visual_from_sound(a)
                if got is not None:
                    n += 1
                    ok += int(got == lab)
            rows.append(dict(seed=sd, shuffled=shuffled, n_pairs=len(pairs),
                             answered=n, answer_rate=round(n / max(len(pairs), 1), 4),
                             recall_acc=round(ok / max(n, 1), 4),
                             chance=round(1 / max(len(set(tgt)), 1), 4)))
            print("bind", json.dumps(rows[-1]), flush=True)
    real = [r for r in rows if not r["shuffled"]]
    ctrl = [r for r in rows if r["shuffled"]]
    m = lambda rs, k: round(float(np.mean([r[k] for r in rs])), 4)
    return dict(runs=rows, real_recall=m(real, "recall_acc"), shuffled_recall=m(ctrl, "recall_acc"),
                real_answer_rate=m(real, "answer_rate"), shuffled_answer_rate=m(ctrl, "answer_rate"),
                chance=real[0]["chance"])


for name, fn in [("audio", lambda: audio()),
                 ("detect_mnist", lambda: detection("mnist", nb.load_mnist)),
                 ("detect_fashion", lambda: detection("fashion", nb.load_fashion_mnist)),
                 ("binding", lambda: binding())]:
    try:
        out[name] = fn()
    except Exception as e:
        out[name] = dict(ERROR=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-800:])
        print("FAIL", name, out[name]["ERROR"], flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
print("DONE", flush=True)
