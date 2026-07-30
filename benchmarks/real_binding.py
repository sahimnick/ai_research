"""Concepts from sensor fusion, on two real corpora that share nothing but meaning.

Every cross-modal number in this project so far came from MNIST digits paired
with synthetic tones -- a pairing invented for the test. This one is not
invented. CIFAR-10 and ESC-50 were built by different people for different
tasks, and they happen to contain the same real categories: a cat, a dog, a
frog, a bird, an airplane, a car. So the mind can be shown a *photograph* of a
cat and played a *field recording* of a cat, and the only thing those two
signals have in common is what they are of.

That is the strongest form this test can take. In a purpose-built multimodal
corpus the two channels are recorded together, so a binding can be learned from
incidental correlation -- same room, same session, same microphone, same
lighting. Here there is no such correlation to find. Whatever the concept cells
learn has to be semantic, because nothing else is shared.

Measured, all with the queries **held out**:

    sound -> label      hear a recording, name what would be seen
    sound -> vision     hear a recording, retrieve a visual code, and identify
                        *that code* against visual prototypes. Vision is on the
                        output path here, so this one cannot be passed without it
    vision -> sound     the reverse direction
    concept cells       how many the layer recruited, and how purely each maps
                        to one category

Three controls, and each answers a different objection:

    shuffled    labels permuted pair by pair. The right control for the two
                *label* read-outs -- and the wrong one for the cross-modal
                paths, which is worth stating because the first version of this
                benchmark used it for all four and got +0.000 on two of them.
                Binding is unsupervised: shuffling labels changes the vote map
                and leaves `Wv` and `Wa` bit-identical, so a read-out that never
                consults a label cannot possibly move.
    mismatched  the sight and the sound are re-paired at random *across*
                categories -- a cat photo with an engine recording. This is the
                control the cross-modal paths need, because it destroys the
                correspondence itself rather than its name.
    unimodal    the same recall attempted from the raw sound code by nearest
                prototype, with no concept layer at all -- so a gain over this
                is a gain from *fusion*, not from the audio front end

Usage:  python3 benchmarks/real_binding.py out_real_binding.json [n_per_class]
"""
import json
import sys

import numpy as np

from neurobrain.cognition.multimodal import AssociationArea
from neurobrain.sensing.natural import load_audiovisual
from neurobrain.sensing.streams import StreamingBrain, _unit
from neurobrain.vision.widev1 import _nearest_prototype

SEEDS = (0, 1, 2, 3, 4)
TRAIN_FRAC = 0.6
# Large enough that the number of concepts is set by the data. At 64 every seed
# reported exactly 64 cells, which is the pool reporting its own size.
N_CONCEPT = 256


def opponent(im):
    """Retinal colour opponency: luminance, red-green, blue-yellow.

    Photographs are in colour and these categories lean on it -- sky is blue,
    frogs are green. Grayscale was a modelling choice, not a property of the
    world, and it costs: the class-mean read-out goes 0.288 grayscale -> 0.365
    with opponent channels. Primate retina builds exactly these three, so
    keeping them is the less invented option, not the more."""
    r, g, b = [im[i].astype(np.float32) for i in range(3)]

    def n(c):
        c = c - c.min()
        m = c.max()
        return (c / m * 255.0) if m > 1e-6 else c
    return n((r + g + b) / 3.0), n(r - g), n(b - (r + g) / 2.0)


def encode(brain, images, waves, colour=True, adapt=True, develop=True):
    """Both senses, through the same front ends the streaming path uses.

    ``adapt`` subtracts each cell's running baseline before the code is read.
    Without it the V1 code on photographs scores 1-NN 0.185 -- below raw pixels
    at 0.308 -- because a large component is common to every image and the
    cosine between any two codes is dominated by it. With it, 0.285. MNIST is
    unaffected (0.830 -> 0.840), which is why it never surfaced before.

    ``develop`` grows the receptive fields from these photographs instead of
    using the hand-written Gabor-like defaults, which is the project's own
    standing claim applied to real images: 1-NN 0.288 -> 0.362 on CIFAR, and the
    fields transfer to MNIST and beat the designed ones there too."""
    from neurobrain.learning.selforganize import develop_v1
    from neurobrain.vision.widev1 import PopulationAdaptation

    if develop:
        develop_v1(brain.v1, [opponent(im)[0] if im.ndim == 3 else im
                              for im in images], epochs=3, seed=0)

    def rate(im):
        if colour and im.ndim == 3:
            return np.concatenate([brain.v1.rate(c) for c in opponent(im)])
        return brain.v1.rate(im)

    R = np.array([rate(im) for im in images], np.float32)
    if adapt:
        ad = PopulationAdaptation(R.shape[1])
        V = np.array([_unit(ad(r)) for r in R], np.float32)
    else:
        V = np.array([_unit(r) for r in R], np.float32)
    A = np.array([brain.belt.code(brain.ear.coch.forward(w)[0])
                  for w in waves], np.float32)
    return V, A


def split(y, seed):
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        cut = max(1, int(round(len(idx) * TRAIN_FRAC)))
        tr.extend(idx[:cut].tolist())
        te.extend(idx[cut:].tolist())
    return np.array(tr), np.array(te)


def fit(V, A, y, tr, seed, mode, n_cls):
    """Bind the training pairs; return the association area and its cell names.

    ``mode`` is ``"real"``, ``"shuffled"`` (labels permuted) or ``"mismatched"``
    (sight re-paired with the wrong sound)."""
    rng = np.random.default_rng(seed + 7)
    assoc = AssociationArea(n_vis=V.shape[1], n_aud=A.shape[1],
                            n_concept=N_CONCEPT, seed=seed)
    assoc.set_stats(V[tr], A[tr])
    labs = y[tr].copy()
    vis = tr.copy()
    if mode == "shuffled":
        labs = rng.permutation(labs)
    elif mode == "mismatched":
        vis = rng.permutation(tr)          # this sound, someone else's sight
    votes = {}
    for iv, ia, lab in zip(vis, tr, labs):
        win = assoc.bind(V[iv], A[ia])
        votes.setdefault(win, {})
        votes[win][int(lab)] = votes[win].get(int(lab), 0) + 1
    name = {c: max(v.items(), key=lambda kv: kv[1])[0] for c, v in votes.items()}
    return assoc, name, votes


def purity(votes):
    """Fraction of binds landing on a cell's own majority category.

    1.0 means every recruited cell answers to exactly one real category."""
    tot = sum(sum(v.values()) for v in votes.values())
    top = sum(max(v.values()) for v in votes.values())
    return top / max(tot, 1)


def run_seed(V, A, y, names, seed):
    n_cls = len(names)
    tr, te = split(y, seed)
    out = {"n_train": int(len(tr)), "n_test": int(len(te))}

    # ---- unimodal controls: the sound code alone, no concept layer ---------
    # Two of them, and the second is the one that matters. With vigilance high
    # the association area recruits about one cell per training pair, so
    # "hear a sound, find its concept cell, read its name" is structurally a
    # nearest-neighbour lookup over the audio codes. Comparing that to a
    # *prototype* baseline would credit fusion with a gain that is really
    # exemplar-versus-centroid. The 1-NN control removes that confound.
    out["unimodal_sound"] = float(np.mean(
        _nearest_prototype(A[tr], y[tr], A[te], n_cls) == y[te]))
    out["unimodal_vision"] = float(np.mean(
        _nearest_prototype(V[tr], y[tr], V[te], n_cls) == y[te]))
    S = A[te] @ A[tr].T
    out["unimodal_sound_1nn"] = float(np.mean(y[tr][S.argmax(1)] == y[te]))
    Sv = V[te] @ V[tr].T
    out["unimodal_vision_1nn"] = float(np.mean(y[tr][Sv.argmax(1)] == y[te]))

    # visual prototypes, for scoring a retrieved visual CODE
    protoV = np.stack([_unit(V[tr][y[tr] == c].mean(0)) for c in range(n_cls)])
    protoA = np.stack([_unit(A[tr][y[tr] == c].mean(0)) for c in range(n_cls)])

    # A reference point for `sound_to_vision`, and it is not optional -- that
    # read-out scores a *retrieved* visual code against these prototypes, so it
    # inherits whatever the visual front end's class-mean geometry happens to
    # be. Changing the eye changes the ruler: developing the receptive fields
    # moved sound_to_vision 0.882 -> 0.581 while making vision better by every
    # other measure, purely because discovered fields trade class-mean structure
    # for exemplar structure. So the *true* visual code for each held-out item
    # is scored under the identical probe.
    #
    # It is called a reference and not a ceiling because the retrieved code
    # beats it -- 174% and 233% in the two configurations. That is not an error:
    # a concept cell's `Wv` is an average over everything bound to it, so the
    # recalled sight is *denoised toward the category* while a single real
    # photograph is not. Recalling a cleaner cat than any cat you have seen is
    # what having a concept means. It also means this probe is easy, which is
    # why the 1-NN version below is reported beside it.
    out["vision_reference"] = float(np.mean(
        [int(np.argmax(protoV @ V[i])) == int(y[i]) for i in te]))
    out["sound_reference"] = float(np.mean(
        [int(np.argmax(protoA @ A[i])) == int(y[i]) for i in te]))

    for tag in ("real", "shuffled", "mismatched"):
        assoc, name, votes = fit(V, A, y, tr, seed, tag, n_cls)
        s2l = v2l = s2v = v2s = s2v1 = n = 0
        for i in te:
            n += 1
            cs = assoc.concept_from_sound(A[i])
            cv = assoc.concept_from_vision(V[i])
            s2l += int(name.get(cs, -1) == int(y[i]))
            v2l += int(name.get(cv, -1) == int(y[i]))
            # the cross-modal paths: the ANSWER is in the other modality
            s2v += int(int(np.argmax(protoV @ _unit(assoc.Wv[cs]))) == int(y[i]))
            v2s += int(int(np.argmax(protoA @ _unit(assoc.Wa[cv]))) == int(y[i]))
            # the harder version: does the imagined sight look like a SPECIFIC
            # real photograph of the right thing, rather than like the average
            # of that category? Nearest training image, not nearest class mean.
            s2v1 += int(int(y[tr][np.argmax(V[tr] @ _unit(assoc.Wv[cs]))])
                        == int(y[i]))
        out[tag] = dict(sound_to_label=round(s2l / n, 4),
                        vision_to_label=round(v2l / n, 4),
                        sound_to_vision=round(s2v / n, 4),
                        sound_to_vision_1nn=round(s2v1 / n, 4),
                        vision_to_sound=round(v2s / n, 4),
                        cells=int((assoc.wins > 0).sum()),
                        purity=round(purity(votes), 4))
    return out


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_real_binding.json"
    n_per = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    develop = "--no-develop" not in sys.argv

    images, waves, y, names = load_audiovisual(n_per_class=n_per, seed=0,
                                               grayscale=False)
    n_cls = len(names)
    print(f"{len(images)} audiovisual samples, {n_cls} shared categories: "
          f"{', '.join(names)}")
    print(f"vision: CIFAR-10 photographs {images.shape[1:]} in colour  "
          f"audio: ESC-50 field recordings\n", flush=True)

    brain = StreamingBrain(seed=0)
    V, A = encode(brain, images, waves, colour=True, adapt=True,
                  develop=develop)
    print(f"eye: colour + adaptation + {'DISCOVERED' if develop else 'designed'} fields", flush=True)
    print(f"visual code {V.shape[1]}d, sound code {A.shape[1]}d  "
          f"(chance {1/n_cls:.3f})\n", flush=True)

    per_seed = [run_seed(V, A, y, names, sd) for sd in SEEDS]
    for sd, r in zip(SEEDS, per_seed):
        print(f"seed {sd}: s->l {r['real']['sound_to_label']:.3f}  "
              f"s->v {r['real']['sound_to_vision']:.3f}  "
              f"v->s {r['real']['vision_to_sound']:.3f}  "
              f"cells {r['real']['cells']}  purity {r['real']['purity']:.2f}",
              flush=True)

    res = {"n_class": n_cls, "names": list(names), "chance": round(1 / n_cls, 4),
           "n_samples": int(len(images)), "per_seed": per_seed, "summary": {}}
    chance = 1.0 / n_cls

    # each read-out is judged against the control that can actually move it
    CONTROL = {"sound_to_label": "shuffled", "vision_to_label": "shuffled",
               "sound_to_vision": "mismatched",
               "sound_to_vision_1nn": "mismatched",
               "vision_to_sound": "mismatched"}
    print(f"\n{'path':<18}{'real':>9}{'+/-':>8}{'control':>10}{'(which)':>13}"
          f"{'spread':>9}{'d':>8}{'wins':>7}{'x chance':>10}")
    for key, ctrl in CONTROL.items():
        r = np.array([s["real"][key] for s in per_seed])
        c = np.array([s[ctrl][key] for s in per_seed])
        d = r - c
        sd_ = float(d.std(ddof=1))
        rec = dict(real=round(float(r.mean()), 4),
                   real_sd=round(float(r.std(ddof=1)), 4), control=ctrl,
                   control_score=round(float(c.mean()), 4),
                   shuffled=round(float(np.mean([s["shuffled"][key]
                                                 for s in per_seed])), 4),
                   mismatched=round(float(np.mean([s["mismatched"][key]
                                                   for s in per_seed])), 4),
                   spread=round(float(d.mean()), 4),
                   cohens_d=round(float(d.mean() / (sd_ + 1e-12)), 3),
                   wins=int((d > 0).sum()), n=len(d),
                   over_chance=round(float(r.mean()) / chance, 2))
        res["summary"][key] = rec
        print(f"{key:<18}{rec['real']:>9.3f}{rec['real_sd']:>8.3f}"
              f"{rec['control_score']:>10.3f}{ctrl:>13}"
              f"{rec['spread']:>+9.3f}"
              f"{rec['cohens_d']:>8.2f}{rec['wins']:>4}/{rec['n']}"
              f"{rec['over_chance']:>9.2f}x")

    us = float(np.mean([s["unimodal_sound"] for s in per_seed]))
    uv = float(np.mean([s["unimodal_vision"] for s in per_seed]))
    us1 = float(np.mean([s["unimodal_sound_1nn"] for s in per_seed]))
    uv1 = float(np.mean([s["unimodal_vision_1nn"] for s in per_seed]))
    cells = float(np.mean([s["real"]["cells"] for s in per_seed]))
    pur = float(np.mean([s["real"]["purity"] for s in per_seed]))
    n_tr = float(np.mean([s["n_train"] for s in per_seed]))
    res.update(unimodal_sound=round(us, 4), unimodal_vision=round(uv, 4),
               unimodal_sound_1nn=round(us1, 4),
               unimodal_vision_1nn=round(uv1, 4),
               cells=cells, purity=round(pur, 4), n_train=n_tr)
    vc = float(np.mean([s["vision_reference"] for s in per_seed]))
    ac = float(np.mean([s["sound_reference"] for s in per_seed]))
    sv_r = res["summary"]["sound_to_vision"]["real"]
    sv1 = res["summary"]["sound_to_vision_1nn"]["real"]
    res.update(vision_reference=round(vc, 4), sound_reference=round(ac, 4),
               sound_to_vision_vs_reference=round(sv_r / max(vc, 1e-9), 4))
    print(f"\nthe same probe applied to the TRUE visual code of each held-out "
          f"item:")
    print(f"   reference {vc:.3f}   |   recalled from sound {sv_r:.3f}  "
          f"= {sv_r/max(vc,1e-9):.0%}")
    if sv_r > vc:
        print(f"   the recalled sight beats the real one, because a concept "
              f"cell's Wv is an average and a photograph is not.")
    sl_r = res["summary"]["sound_to_label"]["real"]
    print(f"   the 1-NN version of the same probe reads {sv1:.3f}")
    if abs(sv1 - sl_r) < 1e-6:
        print(f"   -- and that is EXACTLY sound_to_label ({sl_r:.3f}), which "
              f"is not a coincidence. Recruitment sets Wv[cell] = the pair's")
        print(f"      own visual code, so with one cell per pair the nearest "
              f"training image to a recalled sight IS that pair's image, and")
        print(f"      every sound -> X probe collapses to 'find the nearest "
              f"stored sound, read off what was stored beside it'.")
        print(f"      Until cells < pairs, the prototype probe above is the "
              f"only one measuring something a lookup cannot do.")

    print(f"\nunimodal controls, no concept layer:")
    print(f"   prototype   sound {us:.3f}   vision {uv:.3f}")
    print(f"   1-NN        sound {us1:.3f}   vision {uv1:.3f}   <- the honest "
          f"floor for a layer holding one cell per example")
    print(f"   {cells:.1f} concept cells for {n_tr:.0f} training pairs "
          f"({cells/max(n_tr,1):.2f} per pair), purity {pur:.2f}")
    if cells > 0.8 * n_tr:
        print("   NOTE: that is an exemplar memory, not a set of concepts. "
              "Compare against 1-NN, not against the prototype.")

    print("\n=== does fusion build concepts on real data? ===")
    sl = res["summary"]["sound_to_label"]
    sv = res["summary"]["sound_to_vision"]
    for nm, rec, floor, what in (
            ("sound -> label", sl, max(us, us1), "beats hearing alone"),
            ("sound -> vision", sv, chance, "retrieves the right SIGHT")):
        ok = (rec["cohens_d"] >= 0.8 and rec["wins"] >= 4
              and rec["real"] > floor)
        print(f"  {nm:<17} {rec['real']:.3f} vs {floor:.3f} "
              f"({rec['real']-floor:+.3f}), {rec['control']} control "
              f"{rec['control_score']:.3f} -> {'YES' if ok else 'no'}, {what}")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
