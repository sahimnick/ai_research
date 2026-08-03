"""H24 -- is the canvas the limiting factor on top-down attention?

The plan (``docs/PLAN_BRAIN_IN_THE_LOOP.fa.md``) phase 2, and it is independent
of H23: nothing here asks a representation to transfer to an unseen scene. It
asks whether, *within* one scene, a bigger map lets a cue steer what the system
picks out.

The hypothesis, and where it comes from
---------------------------------------
§9.32 tried to improve ``attend()`` by swapping its averaged prototype for the
spatial template of §9.21, and the fix made it **worse** -- 0.275 cue-following
against 0.350. The reason it gave was not the tag but the map: on the 56 px
canvas V4 comes out 7x21 with **100% of cells above 55% of peak energy**.
Uniformly active, so a method that depends on spatial layout has no layout to
work with. §9.21's 0.943 was at 224 px, where V4 is 49x49.

> **H24.** Attention follows a cue better on a 224 px canvas than on a 56 px
> one, because the V4 map is no longer saturated.

Falsified if 224 px does not beat 56 px. Then "the canvas is the limiting
factor" is wrong and the limiting factor is elsewhere.

Why the baseline is re-measured rather than quoted
--------------------------------------------------
The archived figures are 0.275 / 0.350. **They are not reproducible from this
tree**: nothing in the repository calls ``attend_spatial``, so whatever script
produced that table is gone. Quoting 0.350 as this run's control would compare
a fresh number against an unreproducible one and call the difference a result.

So both canvases are measured **here, in one run, from one protocol**, and the
archived pair is reported alongside as context only. If the re-measured 56 px
arm lands far from 0.350 that is itself worth knowing, and it is visible rather
than hidden.

Protocol
--------
``two_object_image`` composites two of the 12 shape classes into one canvas --
the same helper §9.32's description implies. The image is **identical** across
the two trials of a pair; only the cue changes. That is the signature no
feed-forward pass can produce: whatever a stimulus-driven mechanism picks, it
picks the same thing both times, so its switch rate is 0 **by construction**.

    cue-following   is the returned class the cued one?   (chance 1/12 = 0.083)
    switch rate     does cueing A vs B change the answer on the SAME image?
    wrong-cue       a third class, absent from the image. If the answer is the
                    same as under the correct cue, no top-down signal arrives.

Arms: ``attend`` (averaged profile, as published) and ``attend_spatial``
(§9.21's spatial template), each at 56 px and at 224 px. Two methods x two
canvases isolates canvas size from tag type -- the confound §9.32 could not
separate, because it only ever had one canvas.

Also reported, because it is the *mechanism* the hypothesis names and not just
the outcome: the V4 map shape and the **fraction of cells above 55% of peak**
at each canvas. If 224 px wins while saturation is unchanged, the explanation
is wrong even if the prediction is right.

Usage:  python3 benchmarks/attend_canvas.py out_attend_canvas.json [n_pairs]
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from neurobrain.vision.ventral import (SHAPE_CLASSES,            # noqa: E402
                                       build_ventral_stream,
                                       two_object_image)

OUT = sys.argv[1] if len(sys.argv) > 1 else "out_attend_canvas.json"
N_PAIRS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
CANVASES = (56, 224)
ARCHIVED = {"attend": 0.350, "attend_spatial": 0.275, "canvas": 56,
            "source": "§9.32 -- not reproducible from this tree"}
SEED = 0


def saturation(stream, img, thresh=0.55):
    """Fraction of V4 cells above ``thresh`` of peak energy -- §9.32's number."""
    stream.hierarchy.forward(img)
    v4 = np.asarray(stream.V4.log["output"], np.float32)
    e = np.linalg.norm(v4, axis=0)
    peak = float(e.max())
    frac = float((e >= thresh * peak).mean()) if peak > 1e-9 else 1.0
    return v4.shape, frac


def gate_diagnostics(stream, size, cases):
    """Where the top-down gain actually lands, and what survives it.

    Cue-following alone cannot say *which* half of the mechanism failed. The
    gain map is built from the cue, and IT then reads the gated map, so a wrong
    answer is either "the gain pointed at the wrong object" or "the gain pointed
    correctly and IT could not read what was left". These separate them.

    ``two_object_image`` puts the cued object in one half of the canvas, so the
    fraction of gain mass falling on that half has a known chance level of 0.5.
    ``energy_kept`` is what reaches IT: ``attend`` multiplies V4 *by* the gain
    rather than by ``1 + gain``, so a peaked gain does not merely emphasise --
    it deletes.
    """
    on_cued, kept, part = [], [], []
    for a, b, _w, s in cases:
        img = two_object_image(a, b, size=size, seed=s)
        for cue, left in ((a, True), (b, False)):
            stream.hierarchy.forward(img)
            v4 = np.asarray(stream.V4.log["output"], np.float32)
            h = stream._attention_heatmap(v4, cue)          # (H4, W4), peak 1
            tot = float(h.sum())
            if tot <= 1e-9:
                continue
            half = h.shape[1] // 2
            mass = float(h[:, :half].sum() if left else h[:, half:].sum())
            on_cued.append(mass / tot)
            g = h ** 3.0
            e = np.linalg.norm(v4, axis=0)
            kept.append(float((e * g).sum() / (e.sum() + 1e-9)))
            part.append(float(g.sum() ** 2 / ((g ** 2).sum() + 1e-12) / g.size))
    return {"gain_mass_on_cued_half": round(float(np.mean(on_cued)), 4),
            "v4_energy_kept": round(float(np.mean(kept)), 4),
            "gain_map_participation": round(float(np.mean(part)), 4)}


def trials(n, rng):
    """``(class_a, class_b, wrong_class, seed)`` -- three distinct classes."""
    out = []
    for i in range(n):
        a, b, w = rng.choice(len(SHAPE_CLASSES), 3, replace=False)
        out.append((SHAPE_CLASSES[a], SHAPE_CLASSES[b], SHAPE_CLASSES[w], i))
    return out


def run(stream, size, cases, method):
    """cue-following, switch rate and wrong-cue agreement for one arm."""
    fn = getattr(stream, method)
    follow, switch, wrong_follow, wrong_same = [], [], [], []
    for a, b, w, s in cases:
        img = two_object_image(a, b, size=size, seed=s)
        ra, rb = fn(img, a), fn(img, b)
        follow += [ra == a, rb == b]
        switch.append(ra != rb)
        rw = fn(img, w)                 # a class that is not in the picture
        wrong_follow.append(rw == w)
        wrong_same.append(rw == ra)     # unchanged answer => no top-down signal
    return {"cue_following": round(float(np.mean(follow)), 4),
            "switch_rate": round(float(np.mean(switch)), 4),
            "wrong_cue_following": round(float(np.mean(wrong_follow)), 4),
            "wrong_cue_same_answer": round(float(np.mean(wrong_same)), 4),
            "n_trials": len(follow)}


def main():
    rng = np.random.default_rng(SEED)
    cases = trials(N_PAIRS, rng)
    chance = 1.0 / len(SHAPE_CLASSES)
    print(f"H24: {len(SHAPE_CLASSES)} classes, chance {chance:.3f}, "
          f"{N_PAIRS} image pairs", flush=True)

    arms, v4info = {}, {}
    for size in CANVASES:
        print(f"\nbuilding the stream at {size} px ...", flush=True)
        stream = build_ventral_stream(size=size)
        probe = two_object_image(SHAPE_CLASSES[0], SHAPE_CLASSES[1],
                                 size=size, seed=999)
        shape, frac = saturation(stream, probe)
        v4info[size] = {"v4_map": list(shape),
                        "frac_above_55pct_peak": round(frac, 4)}
        print(f"  V4 map {shape}, {frac:.1%} of cells above 55% of peak",
              flush=True)
        v4info[size].update(gate_diagnostics(stream, size, cases))
        print(f"  gain mass on cued half "
              f"{v4info[size]['gain_mass_on_cued_half']:.3f} (chance 0.500), "
              f"V4 energy reaching IT {v4info[size]['v4_energy_kept']:.3f}",
              flush=True)
        print(f"  shape readout {stream.test_accuracy:.0%}, "
              f"object readout {stream.hard_accuracy:.0%}", flush=True)
        for method in ("attend", "attend_spatial"):
            r = run(stream, size, cases, method)
            arms[f"{method}@{size}"] = r
            print(f"  {method:16s} cue-following {r['cue_following']:.3f}  "
                  f"switch {r['switch_rate']:.3f}  "
                  f"wrong-cue same answer {r['wrong_cue_same_answer']:.3f}",
                  flush=True)

    a56, a224 = arms["attend@56"], arms["attend@224"]
    s56, s224 = arms["attend_spatial@56"], arms["attend_spatial@224"]
    out = {
        "hypothesis": "H24: a 224 px canvas lets a cue steer selection better "
                      "than 56 px, because V4 is no longer saturated",
        "n_classes": len(SHAPE_CLASSES), "chance": round(chance, 4),
        "n_pairs": N_PAIRS, "canvases": list(CANVASES),
        "archived_reference": ARCHIVED,
        "v4": v4info,
        "arms": arms,
        "verdict": {
            "attend_224_beats_56": bool(a224["cue_following"]
                                        > a56["cue_following"]),
            "spatial_224_beats_56": bool(s224["cue_following"]
                                         > s56["cue_following"]),
            "best_224_beats_archived_0350": bool(
                max(a224["cue_following"], s224["cue_following"]) > 0.350),
            "saturation_falls_with_canvas": bool(
                v4info[224]["frac_above_55pct_peak"]
                < v4info[56]["frac_above_55pct_peak"]),
            "cue_carries_signal_224": bool(
                a224["cue_following"] > a224["wrong_cue_following"]),
        },
        # The two halves of the mechanism, separated. Reported as MARGINS and
        # not as booleans: a bare `mass > 0.5` reads as "the gain finds the
        # cued object" while being satisfied by +0.009, which is nothing. The
        # number has to carry its own size.
        "gain_margin_over_chance": {
            str(c): round(v4info[c]["gain_mass_on_cued_half"] - 0.5, 4)
            for c in CANVASES},
    }
    json.dump(out, open(OUT, "w"), indent=1)
    print("\n" + json.dumps(out["verdict"], indent=1))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
