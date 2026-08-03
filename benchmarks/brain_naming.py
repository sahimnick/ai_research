"""H23 -- can a real percept ground into a concept, above chance, in both senses?

The plan (``docs/PLAN_BRAIN_IN_THE_LOOP.fa.md``) phase 1. ``minds/mind.py`` has
been an integrated brain since it was written; it was fed drawn circles and
400 ms chirps while every number in EVALUATION §9.18-§9.36 came from real video.
This is the first time it is given real frames and real recordings.

The hypothesis
--------------
**H23.** A real ``UnifiedEye`` region code and a real ESC-50 sound code can be
bound by the association area into shared concept cells, and either sense alone
can then name the concept above chance.

Falsified if naming does not beat chance, or does not beat raw pixels. Nothing
downstream in the plan survives that -- H24-H28 all stand on this.

The task
--------
Five visual categories, each present in **two or three different VOT2019
sequences**, each paired with one real ESC-50 sound class:

    drone     drone1, drone_across, drone_flip     <-> helicopter
    vehicle   car1, motocross1, road               <-> engine
    fish      fish1, fish2, zebrafish1             <-> water_drops
    bird      birds1, flamingo1                    <-> chirping_birds
    sport     handball1, handball2                 <-> clapping

Two scenes per category is the whole point. With one scene per label,
"generalises to a new scene" is not a question that can be asked, and §9.27
(cross-scene R^2 = 0.00) says it is *the* question for this project.

The pairing is arbitrary and consistent, which is what a word is. It is not
claimed that a fish sounds like a water drop.

Arms
----
``eye``           pooled multi-area code from the 224 px ``UnifiedEye``, bound
                  to the matching sound class.
``eye_no_sound``  identical, except the clip bound to each crop is drawn from a
                  **random** concept. Sound is still present, still real, still
                  binding -- it just says nothing about what is being seen. The
                  gap ``eye - eye_no_sound`` is what hearing contributes to
                  seeing, which is the only way to answer that as a number
                  rather than an intuition.
``pixels``        the same crop, greyscale, downsampled to a comparable budget
                  -- the floor. If the eye does not beat this, the eye is not
                  what is doing the naming.
``shuffled``      identical pipeline, training labels permuted. Keeps code
                  statistics, class balance and cell count; destroys only the
                  correspondence. Run ``N_PERM`` times, because one permutation
                  is a single draw from the null, not the null. The verdict
                  compares against the **worst case** over those draws.

Both directions of the binding are read out: ``hear_then_see`` names the concept
from a **held-out ESC-50 recording alone**, through the same concept cells the
crops trained. ``see_then_hear`` is not reported separately -- with a fixed
concept-to-sound pairing it is a dictionary lookup on the naming result and
would be the same number twice.

Splits, and why they are reported separately
--------------------------------------------
``within``    per sequence, first half of frames trains, second half tests.
``cross``     leave-one-sequence-out: the held-out sequence is never trained on,
              but its category still is, via its sibling scenes.

§9.27 is why these are never averaged together. A ``within`` number that looks
strong is a statement about frames, not about the world.

Known caveat, stated up front
-----------------------------
The eye is developed **once**, unsupervised, on frames drawn from all thirteen
sequences, and that single eye is used for every fold. No label is involved in
development, and the association area's training is strictly split -- but the
*representation* has seen the held-out scene's pixels. So the ``cross`` number
here is an upper bound on what a genuinely unseen scene would give. Developing
per fold would cost thirteen developments; the honest move is to run it as-is
and say so rather than quietly report ``cross`` as fully held out.

Usage:  python3 benchmarks/brain_naming.py out_brain_naming.json [vot_dir]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from neurobrain.minds.eyebrain import EyeBrain                     # noqa: E402
from neurobrain.minds.mind import Mind                             # noqa: E402
from neurobrain.cognition.multimodal import AssociationArea        # noqa: E402
from neurobrain.audition.audio import build_auditory_stream        # noqa: E402
from neurobrain.vision.unified_eye import UnifiedEye               # noqa: E402
from neurobrain.sensing.natural import load_esc50, ESC50_CLASSES   # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "out_brain_naming.json"
VOT = sys.argv[2] if len(sys.argv) > 2 else "vot"
SIZE = 224                 # the canvas §9.35 measured at 0.902-0.910
PER_SEQ = 24               # crops sampled per sequence
N_CELL = 20                # concept cells: 4 per category, so cells can split
N_PERM = 5                 # label permutations -- one draw is not a null
SEED = 0

CATEGORIES = {
    "drone":   (["drone1", "drone_across", "drone_flip"], "helicopter"),
    "vehicle": (["car1", "motocross1", "road"],           "engine"),
    "fish":    (["fish1", "fish2", "zebrafish1"],         "water_drops"),
    "bird":    (["birds1", "flamingo1"],                  "chirping_birds"),
    "sport":   (["handball1", "handball2"],               "clapping"),
}
NAMES = sorted(CATEGORIES)
SOUND_OF = {k: v[1] for k, v in CATEGORIES.items()}
CHANCE = 1.0 / len(NAMES)


# ----------------------------------------------------------------- loading
def load_gt(path):
    out = []
    for line in open(path):
        v = [float(x) for x in line.replace(",", " ").split()]
        if len(v) == 8:
            xs, ys = v[0::2], v[1::2]
            out.append((min(ys), min(xs), max(ys) - min(ys), max(xs) - min(xs)))
        elif len(v) == 4:
            out.append((v[1], v[0], v[3], v[2]))
        else:
            out.append(None)
    return out


def resize(a, nh, nw):
    """Bilinear -- no scipy in this project, and nearest would alias."""
    h, w = a.shape
    yi, xi = np.linspace(0, h - 1, nh), np.linspace(0, w - 1, nw)
    y0, x0 = np.floor(yi).astype(int), np.floor(xi).astype(int)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    wy, wx = (yi - y0)[:, None], (xi - x0)[None, :]
    return (a[np.ix_(y0, x0)] * (1 - wy) * (1 - wx)
            + a[np.ix_(y1, x0)] * wy * (1 - wx)
            + a[np.ix_(y0, x1)] * (1 - wy) * wx
            + a[np.ix_(y1, x1)] * wy * wx).astype(np.float32)


def crops_of(seq_dir, n, rng):
    """``n`` ground-truth regions from a sequence, each as a 224 px view.

    The box is padded to 1.6x and made square before resizing, so the eye sees
    the object *in a little context* at a fixed scale -- otherwise "how big the
    annotation is" and "which category" would be the same variable.
    """
    from PIL import Image
    jpgs = sorted(f for f in os.listdir(seq_dir) if f.endswith(".jpg"))
    gt = load_gt(os.path.join(seq_dir, "groundtruth.txt"))
    ok = [i for i in range(min(len(jpgs), len(gt)))
          if gt[i] is not None and min(gt[i][2], gt[i][3]) >= 8]
    if len(ok) < 4:
        return []
    idx = np.linspace(0, len(ok) - 1, min(n, len(ok))).astype(int)
    out = []
    for i in idx:
        j = ok[i]
        a = np.asarray(Image.open(os.path.join(seq_dir, jpgs[j])).convert("L"),
                       np.float32) / 255.0
        y, x, bh, bw = gt[j]
        s = max(bh, bw) * 1.6
        cy, cx = y + bh / 2, x + bw / 2
        y0 = int(np.clip(cy - s / 2, 0, max(0, a.shape[0] - 1)))
        x0 = int(np.clip(cx - s / 2, 0, max(0, a.shape[1] - 1)))
        patch = a[y0:y0 + int(s), x0:x0 + int(s)]
        if patch.shape[0] < 4 or patch.shape[1] < 4:
            continue
        out.append(resize(patch, SIZE, SIZE))
    return out


# ------------------------------------------------------------------ scoring
def assign(assoc, V, y, n_name):
    """Majority label per concept cell, from **training** data only."""
    lab = -np.ones(assoc.n_concept, int)
    for c in range(assoc.n_concept):
        m = [y[i] for i in range(len(V))
             if assoc.concept_from_vision(V[i]) == c]
        if m:
            lab[c] = np.bincount(m, minlength=n_name).argmax()
    return lab


def train_assoc(V, A, y, n_name, seed):
    """Bind visual and sound codes into shared cells, then name the cells."""
    assoc = AssociationArea(V.shape[1], A.shape[1], N_CELL, seed=seed)
    assoc.set_stats(V, A)
    order = np.random.default_rng(seed).permutation(len(V))
    for _ in range(3):
        for i in order:
            assoc.bind(V[i], A[i])
    return assoc, assign(assoc, V, y, n_name)


def score(assoc, lab, X, y, by_sound=False):
    if len(X) == 0:
        return float("nan")
    f = assoc.concept_from_sound if by_sound else assoc.concept_from_vision
    pred = np.array([lab[f(x)] for x in X])
    return float((pred == np.asarray(y)).mean())


# --------------------------------------------------------------------- main
def main():
    rng = np.random.default_rng(SEED)
    seqs = [(s, n) for n in NAMES for s in CATEGORIES[n][0]
            if os.path.isdir(os.path.join(VOT, s))]
    missing = [s for n in NAMES for s in CATEGORIES[n][0]
               if not os.path.isdir(os.path.join(VOT, s))]
    if missing:
        print(f"missing sequences: {missing}", flush=True)
    if len(seqs) < 6:
        print(f"need the VOT sequences under {VOT}/ -- found {len(seqs)}")
        return 1

    print(f"H23: {len(seqs)} sequences, {len(NAMES)} concepts, "
          f"chance {CHANCE:.3f}", flush=True)

    # -- crops ---------------------------------------------------------
    views, vseq, vlab = [], [], []
    for s, n in seqs:
        c = crops_of(os.path.join(VOT, s), PER_SEQ, rng)
        views += c
        vseq += [s] * len(c)
        vlab += [NAMES.index(n)] * len(c)
        print(f"  {s:14s} -> {len(c):3d} crops ({n})", flush=True)
    views = np.stack(views)
    vseq, vlab = np.array(vseq), np.array(vlab)

    # -- the eye, developed on these frames (§9.27: it does not transfer)
    print("developing the eye on real frames ...", flush=True)
    eye = UnifiedEye(size=SIZE)
    d = rng.permutation(len(views))[:min(120, len(views))]
    eye.develop(views[d])

    # -- the ear, and real recordings ----------------------------------
    print("building the ear ...", flush=True)
    ear = build_auditory_stream()
    print("loading ESC-50 ...", flush=True)
    waves, wlab, _ = load_esc50(n_shards=12)
    want = {ESC50_CLASSES.index(SOUND_OF[n]): NAMES.index(n) for n in NAMES}
    snd = {i: [] for i in range(len(NAMES))}
    for w, l in zip(waves, wlab):
        if int(l) in want:
            snd[want[int(l)]].append(np.asarray(w, np.float32))
    for i, n in enumerate(NAMES):
        print(f"  {n:8s} <- {SOUND_OF[n]:15s} {len(snd[i]):3d} clips",
              flush=True)
    if min(len(v) for v in snd.values()) < 4:
        print("not enough ESC-50 clips per class")
        return 1

    # sound codes, split into a train and a held-out half per class
    print("coding sounds ...", flush=True)
    A_tr, A_te, ay_te = {}, [], []
    for i in range(len(NAMES)):
        codes = [np.asarray(ear.sound_code(w), np.float32) for w in snd[i]]
        h = max(2, len(codes) // 2)
        A_tr[i] = codes[:h]
        A_te += codes[h:]
        ay_te += [i] * (len(codes) - h)
    A_te = np.stack(A_te) if A_te else np.zeros((0, 1), np.float32)

    # -- visual codes --------------------------------------------------
    print(f"coding {len(views)} crops through the eye ...", flush=True)
    brain = EyeBrain(eye, ear, None, [], SOUND_OF)
    V = np.stack([brain.vision_code(v) for v in views])
    print(f"  visual code {V.shape[1]}-d, sound code {A_te.shape[1]}-d",
          flush=True)

    # raw-pixel floor at a comparable budget
    P = np.stack([resize(v, 16, 16).ravel() for v in views])

    def paired_sound(y, seed, matched=True):
        """One real recording per crop.

        ``matched`` draws it from that crop's own concept -- the pairing a word
        is. With ``matched=False`` the clip is drawn from a random concept
        instead: sound is still present, still real, still bound, but carries
        no information about what is being seen. The gap between the two arms
        is what hearing contributes to *seeing*.
        """
        r = np.random.default_rng(seed)
        out = []
        for l in y:
            c = int(l) if matched else int(r.integers(len(NAMES)))
            out.append(A_tr[c][int(r.integers(len(A_tr[c])))])
        return np.stack(out)

    # arm -> (visual code, is the sound informative, how many label permutations)
    ARMS = (("eye", V, True, 1),
            ("eye_no_sound", V, False, 1),
            ("pixels", P, True, 1),
            ("shuffled", V, True, N_PERM))

    results = {}
    for arm, X, matched, reps in ARMS:
        within, cross, s2v = [], [], []
        for rep in range(reps):                    # one permutation is not a
            sd = SEED + 1000 * rep                 # null distribution
            for held, _ in seqs:                   # leave-one-sequence-out
                m = vseq == held
                ytr = vlab[~m].copy()
                if arm == "shuffled":
                    ytr = np.random.default_rng(sd).permutation(ytr)
                a_tr = paired_sound(ytr, sd, matched)
                assoc, lab = train_assoc(X[~m], a_tr, ytr, len(NAMES), SEED)
                cross.append(score(assoc, lab, X[m], vlab[m]))
                if len(A_te):                      # sound alone -> concept
                    s2v.append(score(assoc, lab, A_te, ay_te, by_sound=True))
            for s, _ in seqs:                      # within-scene, per sequence
                m = np.where(vseq == s)[0]
                h = len(m) // 2
                tr, te = m[:h], m[h:]
                if len(tr) < 2 or len(te) < 1:
                    continue
                # a within-scene fold has ONE label, so the cell must be named
                # from the whole training pool; only the *frames* are held out
                keep = vseq != s
                ypool = np.concatenate([vlab[keep], vlab[tr]])
                Xpool = np.concatenate([X[keep], X[tr]])
                if arm == "shuffled":              # the WHOLE pool, or the
                    ypool = np.random.default_rng( # other 12 sequences would
                        sd).permutation(ypool)     # keep their correspondence
                assoc, l2 = train_assoc(Xpool, paired_sound(ypool, sd, matched),
                                        ypool, len(NAMES), SEED)
                within.append(score(assoc, l2, X[te], vlab[te]))
        results[arm] = {
            "hear_then_see": round(float(np.mean(s2v)), 4) if s2v else None,
            "within_scene": round(float(np.mean(within)), 4),
            "cross_scene": round(float(np.mean(cross)), 4),
        }
        if reps == 1:                       # per-sequence only when the folds
            results[arm]["cross_scene_per_seq"] = {   # line up one-to-one
                s: round(float(c), 4) for (s, _), c in zip(seqs, cross)}
            # within-scene per sequence too, or "sound helps" cannot be given a
            # paired test and would rest on two means with no spread
            results[arm]["within_scene_per_seq"] = {
                s: round(float(w), 4) for (s, _), w in zip(seqs, within)}
        else:                               # the null's spread, not just its
            results[arm]["permutations"] = reps       # mean
            results[arm]["cross_scene_max_perm"] = round(float(max(
                np.mean(cross[i * len(seqs):(i + 1) * len(seqs)])
                for i in range(reps))), 4)
        print(f"{arm:13s} within {results[arm]['within_scene']:.3f}  "
              f"cross {results[arm]['cross_scene']:.3f}  "
              f"hear->see {results[arm]['hear_then_see']}", flush=True)

    out = {
        "hypothesis": "H23: real percepts ground into concepts above chance",
        "chance": round(CHANCE, 4),
        "n_sequences": len(seqs), "n_concepts": len(NAMES),
        "crops_per_sequence": PER_SEQ, "concept_cells": N_CELL,
        "canvas_px": SIZE,
        "pairing": SOUND_OF,
        "eye_developed_on": "all sequences, unsupervised -- see module docstring",
        "arms": results,
    }
    e, sh = results["eye"], results["shuffled"]
    out["verdict"] = {
        "cross_beats_chance": bool(e["cross_scene"] > CHANCE),
        "cross_beats_pixels": bool(e["cross_scene"]
                                   > results["pixels"]["cross_scene"]),
        # against the WORST-case permutation, not the mean of the null
        "cross_beats_shuffled": bool(e["cross_scene"]
                                     > sh["cross_scene_max_perm"]),
        "within_beats_chance": bool(e["within_scene"] > CHANCE),
        "sound_helps_vision": bool(
            e["cross_scene"] > results["eye_no_sound"]["cross_scene"]),
        "sound_alone_names_above_chance": bool(
            (e["hear_then_see"] or 0.0) > CHANCE),
    }
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out["verdict"], indent=1))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
