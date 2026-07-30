"""Point the eye at a cluttered scene made of photographs, not digits.

This project built a real eye -- a fovea at full resolution, a heavily blurred
periphery, saliency-driven saccades, inhibition of return, and fixational drift
inside every 50 ms window -- and then measured it entirely on scenes made of
**MNIST and Fashion-MNIST**. Section 3 reports 56.7-100% on-object rate and
79-90% naming on digit scenes, 27.6-47.9% on Fashion.

Every audiovisual measurement in section 7.8, meanwhile, hands the mind a whole
photograph at once, centred and static. The eye is not used at all. So the one
faculty built for looking at the world has never been pointed at a real image,
and the two halves of "sees the world" have never been measured together.

This does that. Scenes are 256x256 canvases with twelve CIFAR-10 photographs
scattered on a noisy background, and the eye free-views them exactly as it does
digit scenes -- no positions given, no segmentation, saliency and inhibition of
return deciding where to look.

Three things are measured, each against the digit scenes as a reference so the
cost of the real world is visible rather than absorbed:

    on-object rate   does saliency find photographs at all, or only the
                     high-contrast strokes it was tuned on? Against the chance
                     of landing on an object by accident
    naming           of the fixations that did land on something, how many are
                     named correctly. Free-viewing, so this is detection and
                     recognition together
    corrected        the same with the foveation correction the streaming path
                     uses, which re-centres on the saliency peak before reading

A photograph has no empty background and no single bright stroke, so saliency
built on local contrast has a much harder job than it does on a digit. If the
on-object rate holds up and only naming falls, the eye works and the recognition
behind it is the limit -- which is what every other measurement in section 7.8
also concluded. If the on-object rate collapses, the eye itself was tuned to
digits and that is a separate defect.

Usage:  python3 benchmarks/natural_scene.py out_natural_scene.json
"""
import json
import sys

import numpy as np

import neurobrain as nb
from neurobrain.learning.selforganize import develop_v1
from neurobrain.sensing.natural import load_cifar10
from neurobrain.sensing.streams import SaccadicEye, build_scene
from neurobrain.vision.widev1 import (PopulationAdaptation, WideV1,
                                      _nearest_prototype, _unit)

SEEDS = (0, 1, 2)
N_SCENES, N_SACCADES, N_OBJECTS = 3, 60, 12
N_TRAIN, N_DEV = 2500, 800
CANVAS = 256


def luminance(X):
    """Scenes are a single 2-D canvas, so colour is dropped here.

    Worth stating rather than hiding: colour is worth +0.04 to this eye on whole
    photographs, so the scene numbers below carry that handicap on top of
    everything the scene itself costs."""
    return X.mean(1).astype(np.uint8) if X.ndim == 4 else X


def bank(images, labels, v1, adapt):
    """Codes for whole, centred objects -- the pre-segmented reference."""
    R = np.array([v1.rate(im.astype(np.float32)) for im in images], np.float32)
    return np.array([_unit(adapt(r)) for r in R], np.float32), labels


def free_view(scene, v1, adapt, Xc, yc, n_cls, seed, correct):
    eye = SaccadicEye(scene, seed=seed)
    fix = eye.free_view(n_saccades=N_SACCADES, correct=correct)
    on = [f for f in fix if f.true_label >= 0]
    if not on:
        return dict(n_fix=len(fix), n_on=0, on_rate=0.0, named=None)
    codes = np.array([_unit(adapt.apply(v1.rate_over(f.frames))) for f in on],
                     np.float32)
    pred = _nearest_prototype(Xc, yc, codes, n_cls)
    truth = np.array([f.true_label for f in on])
    return dict(n_fix=len(fix), n_on=len(on), on_rate=len(on) / max(len(fix), 1),
                named=float(np.mean(pred == truth)))


def run(images, labels, tag, seed):
    v1 = WideV1(n_cells=1024, window_ms=50, seed=seed)
    develop_v1(v1, [im.astype(np.float32) for im in images[:N_DEV]],
               epochs=3, seed=seed)
    ad = PopulationAdaptation(v1.n_cells)
    Xc, yc = bank(images[:1500], labels[:1500], v1, ad)
    n_cls = int(labels.max()) + 1
    rows = []
    for s in range(N_SCENES):
        sc = build_scene(images, labels, size=CANVAS, n_objects=N_OBJECTS,
                         seed=seed * 100 + s)
        # chance of landing on an object by accident: the fraction of canvas
        # they cover
        chance = (N_OBJECTS * images.shape[1] ** 2) / float(CANVAS ** 2)
        for correct in (False, True):
            r = free_view(sc, v1, ad, Xc, yc, n_cls, seed * 100 + s, correct)
            r.update(corrected=correct, chance_on_object=round(chance, 4))
            rows.append(r)
    out = {}
    for correct in (False, True):
        sub = [r for r in rows if r["corrected"] == correct]
        named = [r["named"] for r in sub if r["named"] is not None]
        out["corrected" if correct else "raw"] = dict(
            on_rate=round(float(np.mean([r["on_rate"] for r in sub])), 4),
            named=round(float(np.mean(named)), 4) if named else None,
            n_on=int(np.sum([r["n_on"] for r in sub])),
            chance_on_object=sub[0]["chance_on_object"])
    print(f"  seed {seed}: on-object {out['raw']['on_rate']:.3f}  "
          f"named {out['raw']['named']:.3f}  |  corrected "
          f"{out['corrected']['on_rate']:.3f} / "
          f"{out['corrected']['named']:.3f}", flush=True)
    return out


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_natural_scene.json"
    cx, cy, _, _ = load_cifar10(n_train=N_TRAIN, n_test=200, grayscale=False,
                                size=28)
    mx, my, _, _ = nb.load_mnist(n_train=N_TRAIN, n_test=200)
    res = {"n_saccades": N_SACCADES, "n_objects": N_OBJECTS, "canvas": CANVAS,
           "seeds": list(SEEDS)}

    for tag, X, y in (("CIFAR-10 photographs", luminance(cx), cy),
                      ("MNIST digits (reference)", mx, my)):
        print(f"\n{tag} -- {N_OBJECTS} objects on a {CANVAS}x{CANVAS} canvas, "
              f"{N_SACCADES} saccades")
        runs = [run(X, y, tag, sd) for sd in SEEDS]
        res[tag] = runs
        for arm in ("raw", "corrected"):
            m = {k: float(np.mean([r[arm][k] for r in runs]))
                 for k in ("on_rate", "named")}
            res.setdefault("summary", {}).setdefault(tag, {})[arm] = {
                k: round(v, 4) for k, v in m.items()}
        s = res["summary"][tag]
        ch = runs[0]["raw"]["chance_on_object"]
        res["summary"][tag]["chance_on_object"] = ch
        print(f"  on-object {s['raw']['on_rate']:.3f} raw / "
              f"{s['corrected']['on_rate']:.3f} corrected  "
              f"(chance {ch:.3f}, lift {s['raw']['on_rate']/max(ch,1e-9):.1f}x)")
        print(f"  named     {s['raw']['named']:.3f} raw / "
              f"{s['corrected']['named']:.3f} corrected  "
              f"(chance {1/10:.3f})")

    print("\n=== does the eye work on photographs? ===")
    c = res["summary"]["CIFAR-10 photographs"]
    m = res["summary"]["MNIST digits (reference)"]
    ch = c["chance_on_object"]
    look = c["corrected"]["on_rate"] / max(ch, 1e-9)
    print(f"  looking      on-object {c['corrected']['on_rate']:.3f} vs "
          f"{m['corrected']['on_rate']:.3f} on digits, {look:.1f}x chance "
          f"-> {'the eye finds them' if look > 2 else 'saliency fails'}")
    print(f"  recognising  named {c['corrected']['named']:.3f} vs "
          f"{m['corrected']['named']:.3f} on digits")
    if look > 2 and c["corrected"]["named"] < 0.5 * m["corrected"]["named"]:
        print("\n  -> the eye finds photographs about as well as it finds "
              "digits, and then cannot name them.")
        print("     Looking is not the bottleneck; recognition is -- the same "
              "conclusion every other")
        print("     measurement in section 7.8 reached, now including the one "
              "faculty they all skipped.")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
