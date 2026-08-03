"""The imagination system, measured on the real brain -- five tests and a report.

Everything this project knows about its own imagination was measured on drawn
shapes, CIFAR photographs and ESC-50: §5 (the imagine->perceive round trip),
§9.19 (*"it does not imagine, it recalls an average"*), and composition's
finding that 36.8% of what it "imagined" was a stored photograph byte for byte.

None of it was ever measured on **this eye, on real video**. H23 built the
adapter that makes that possible (`minds/eyebrain.py`), so it is now a question
that can be asked rather than an assumption.

The five tests, and why each one is here
----------------------------------------
1. **Mental imagery** -- ``imagine_vision`` on the novelty x coherence plane of
   §9.19, kept identical so the two are comparable. The target is not "maximum
   novelty"; it is **where real held-out data sits**. Novelty alone is gameable
   from the low side (a coherence collapse buys it), so an arm must be at least
   as coherent as real data before its novelty counts.

2. **Sequential imagination** -- ``Mind.imagine`` walks the transition matrix,
   and *nothing has ever measured that walk*. Its two failure modes are
   opposite: **collapse** (every imagined sequence is one state repeated) and
   **uniform** (the walk is the prior, nothing was learned). Both look like
   "it produced a sequence". Scored by bigram KL to real experience against a
   shuffled-time control, so the reference is what the world actually does.

3. **Forward imagination** -- rollout at horizons 1..8 against **persistence**,
   the baseline §9.22 lost to once already. This is the plan's H27 question and
   it is run here because it is imagination, not because the gate opened.

4. **Surprise** -- ``PredictiveWorldModel.surprise`` must rise on a transition
   the world did not produce. Within-sequence transitions against ones spliced
   from a different video; if the two distributions coincide, surprise is not a
   prediction error, it is a constant.

5. **Cross-modal imagery** -- hear a **held-out** recording, imagine the sight.
   The query is entirely auditory and the answer entirely visual, which is what
   makes it imagery rather than a lookup.

State space, and an honest limitation
-------------------------------------
Tests 2-4 need a *trajectory*, and the five category labels do not give one: one
VOT sequence contains one object, so a category sequence is a constant and every
transition is a self-loop. The state is therefore the **concept cell** the mind
itself recruited -- its own vocabulary, not the experimenter's -- which does
move as an object's appearance changes.

And the limitation that bounds all of it: H23 measured cross-scene concept
naming at **chance**. Imagination here runs on concepts that are reliable
*within* a scene and not across one, so every number below inherits that. It is
reported per test rather than mentioned once.

Usage:  python3 benchmarks/imagination_real.py out_imagination_real.json [vot]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "benchmarks")

import matplotlib                                             # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                               # noqa: E402

from neurobrain.cognition.multimodal import AssociationArea   # noqa: E402
from neurobrain.minds.eyebrain import EyeBrain                # noqa: E402
from neurobrain.minds.mind import Mind                        # noqa: E402
from neurobrain.audition.audio import build_auditory_stream   # noqa: E402
from neurobrain.vision.unified_eye import UnifiedEye          # noqa: E402
from neurobrain.sensing.natural import (load_esc50,           # noqa: E402
                                        ESC50_CLASSES)
from neurobrain.sensing.streams import _unit                  # noqa: E402

import brain_naming as bn                                     # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "out_imagination_real.json"
VOT = sys.argv[2] if len(sys.argv) > 2 else "vot"
FIGDIR = "reports/imagination_real"
PER_SEQ = 32                # frames per sequence -- trajectories need length
N_CELL = 24
TEMPS = (0.0, 1.0, 2.0, 4.0, 8.0)
N_COMPOSE = 3
VERBATIM = 0.999            # §9.19's threshold, kept identical
HORIZONS = (1, 2, 4, 8)
SEED = 0
SEEDS = (0, 1, 2, 3, 4)     # §9.19 used five; margins here are ~0.02 wide


# ------------------------------------------------------------------ metrics
def place(M, Btr, protos, want):
    """§9.19's plane, unchanged: novelty, coherence, verbatim.

    Everything must already be in ``prep_v`` space -- mixing it with raw code
    space is the error §9.19 was written to correct, and it is worth 0.1 of
    cosine.
    """
    s = (M @ Btr.T).max(1)
    coh = np.mean([int(np.argmax(protos @ M[k])) == int(want[k])
                   for k in range(len(M))])
    return {"novelty": round(float(1.0 - s.mean()), 4),
            "coherence": round(float(coh), 4),
            "verbatim": round(float((s > VERBATIM).mean()), 4)}


def bigram(seqs, n):
    """Normalised transition counts over ``n`` states."""
    T = np.zeros((n, n), np.float64)
    for s in seqs:
        for a, b in zip(s[:-1], s[1:]):
            T[a, b] += 1.0
    return T


def kl(P, Q, eps=1e-9):
    """KL(P||Q) over the joint transition distribution."""
    p = P / max(P.sum(), eps)
    q = (Q + eps) / (Q + eps).sum()
    m = p > 0
    return float((p[m] * np.log(p[m] / q[m])).sum())


# --------------------------------------------------------------------- data
def build(rng):
    """The real brain: eye on real frames, ear on real recordings, bound."""
    seqs = [(s, n) for n in bn.NAMES for s in bn.CATEGORIES[n][0]
            if os.path.isdir(os.path.join(VOT, s))]
    if len(seqs) < 6:
        raise SystemExit(f"need VOT sequences under {VOT}/ -- found {len(seqs)}")

    views, vseq, vlab = [], [], []
    for s, n in seqs:
        c = bn.crops_of(os.path.join(VOT, s), PER_SEQ, rng)
        views += c
        vseq += [s] * len(c)
        vlab += [bn.NAMES.index(n)] * len(c)
    views = np.stack(views)
    vseq, vlab = np.array(vseq), np.array(vlab)
    print(f"  {len(views)} crops from {len(seqs)} sequences", flush=True)

    eye = UnifiedEye(size=bn.SIZE)
    d = rng.permutation(len(views))[:min(120, len(views))]
    eye.develop(views[d])
    ear = build_auditory_stream()

    waves, wlab, _ = load_esc50(n_shards=12)
    want = {ESC50_CLASSES.index(bn.SOUND_OF[n]): bn.NAMES.index(n)
            for n in bn.NAMES}
    snd = {i: [] for i in range(len(bn.NAMES))}
    for w, l in zip(waves, wlab):
        if int(l) in want:
            snd[want[int(l)]].append(np.asarray(w, np.float32))

    A_tr, A_te, ay_te = {}, [], []
    for i in range(len(bn.NAMES)):
        codes = [np.asarray(ear.sound_code(w), np.float32) for w in snd[i]]
        h = max(2, len(codes) // 2)
        A_tr[i], = [codes[:h]]
        A_te += codes[h:]
        ay_te += [i] * (len(codes) - h)

    brain = EyeBrain(eye, ear, None, [], bn.SOUND_OF)
    V = np.stack([brain.vision_code(v) for v in views])
    print(f"  visual {V.shape[1]}-d, sound {len(A_te[0])}-d", flush=True)
    return dict(views=views, V=V, vseq=vseq, vlab=vlab, seqs=seqs,
                A_tr=A_tr, A_te=np.stack(A_te), ay_te=np.array(ay_te),
                eye=eye, ear=ear, brain=brain)


def bind_all(D, rng, seed=SEED):
    """Train the association area; return it plus each crop's winning cell."""
    y = D["vlab"]
    A = np.stack([D["A_tr"][int(l)][rng.integers(len(D["A_tr"][int(l)]))]
                  for l in y])
    assoc = AssociationArea(D["V"].shape[1], A.shape[1], N_CELL, seed=seed)
    assoc.set_stats(D["V"], A)
    order = rng.permutation(len(D["V"]))
    for _ in range(3):
        for i in order:
            assoc.bind(D["V"][i], A[i])
    cells = np.array([assoc.concept_from_vision(v) for v in D["V"]])
    return assoc, cells


# -------------------------------------------------------------------- tests
def test_imagery(D, assoc, rng):
    """Test 1 -- §9.19's plane, on real video instead of CIFAR."""
    V, y = D["V"], D["vlab"]
    tr = np.arange(len(V)) % 2 == 0            # alternate frames
    te = ~tr
    Btr = np.stack([assoc.prep_v(v) for v in V[tr]])
    Bte = np.stack([assoc.prep_v(v) for v in V[te]])
    protos = np.stack([_unit(Btr[y[tr] == c].mean(0))
                       for c in range(len(bn.NAMES))])
    cells = [assoc.concept_from_sound(a) for a in D["A_te"]]
    want = list(D["ay_te"])
    n = min(len(cells), len(want))
    cells, want = cells[:n], want[:n]

    votes = {}
    for i in np.flatnonzero(tr):
        votes.setdefault(int(assoc.concept_from_vision(V[i])), []).append(int(y[i]))
    cell_cat = {c: int(np.bincount(v).argmax()) for c, v in votes.items()}
    by_cat = {}
    for c, cat in cell_cat.items():
        by_cat.setdefault(cat, []).append(c)

    out = {}
    same = [int(rng.choice(np.flatnonzero(y[tr] == w))) for w in want]
    out["a stored crop"] = place(Btr[same], Btr, protos, want)
    out["gaussian noise"] = place(
        np.stack([_unit(rng.standard_normal(V.shape[1]).astype(np.float32))
                  for _ in want]), Btr, protos, want)
    ridx = rng.permutation(len(Bte))[:len(want)]
    out["a real unseen crop"] = place(Bte[ridx], Btr, protos,
                                      list(y[te][ridx]))

    # the control that decides whether the concept layer buys anything
    by_cat_ex = {}
    for k, i in enumerate(np.flatnonzero(tr)):
        by_cat_ex.setdefault(int(y[i]), []).append(k)
    mix = []
    for c in cells:
        pool = by_cat_ex.get(cell_cat.get(c), list(range(len(Btr))))
        mix.append(_unit(Btr[[int(rng.choice(pool))
                              for _ in range(N_COMPOSE)]].mean(0)))
    out[f"{N_COMPOSE} stored crops, averaged"] = place(np.stack(mix), Btr,
                                                       protos, want)
    for T in TEMPS:
        out[f"sampled T={T}"] = place(
            np.stack([assoc.imagine_vision(c, temperature=T, rng=rng)
                      for c in cells]), Btr, protos, want)
    for T in (1.0, 4.0):
        M = []
        for c in cells:
            pool = by_cat.get(cell_cat.get(c), [c])
            sib = [c] + [int(rng.choice(pool)) for _ in range(N_COMPOSE - 1)]
            M.append(assoc.imagine_composite(sib, temperature=T, rng=rng))
        out[f"composed T={T}"] = place(np.stack(M), Btr, protos, want)

    used = np.flatnonzero(assoc.wins > 0)
    out["_diag"] = {"cells_used": int(len(used)),
                    "singletons": int((assoc.wins[used] == 1).sum()),
                    "no_subspace": int((assoc.mode_var[used].max(1) <= 0).sum()),
                    "woke_a_singleton": round(float(np.mean(
                        [assoc.mode_var[c].max() <= 0 for c in cells])), 4)}
    return out


def test_sequences(D, mind, cells, rng):
    """Test 2 -- does the generative walk reproduce the world's statistics?"""
    real = [list(cells[D["vseq"] == s]) for s, _ in D["seqs"]]
    n = N_CELL
    T_real = bigram(real, n)

    # the mind experiences the real trajectories
    names = mind.concepts
    for s in real:
        mind.experience([names[c] for c in s])

    lens = [len(s) for s in real]
    imag = []
    for k, s in enumerate(real):
        seed_state = names[s[0]]
        walk = mind.imagine(seed_state, steps=lens[k] - 1, temperature=0.6,
                            rng_seed=SEED + k)
        imag.append([names.index(w) for w in walk])
    T_imag = bigram(imag, n)

    # control: a mind that experienced the SAME states in shuffled order, so
    # the marginal is identical and only the temporal structure is destroyed
    ctl = Mind(mind.brain, list(names), dict(mind.sound_of))
    for s in real:
        ctl.experience([names[c] for c in rng.permutation(s)])
    imag_c = []
    for k, s in enumerate(real):
        walk = ctl.imagine(names[s[0]], steps=lens[k] - 1, temperature=0.6,
                           rng_seed=SEED + k)
        imag_c.append([names.index(w) for w in walk])
    T_ctl = bigram(imag_c, n)

    def collapse(seqs):
        return float(np.mean([len(set(s)) == 1 for s in seqs]))

    def selfloop(seqs):
        num = den = 0
        for s in seqs:
            num += sum(a == b for a, b in zip(s[:-1], s[1:]))
            den += max(len(s) - 1, 1)
        return float(num / max(den, 1))

    def ent(seqs):
        f = np.bincount(np.concatenate(seqs), minlength=n).astype(float)
        f = f / max(f.sum(), 1e-9)
        m = f > 0
        return float(-(f[m] * np.log(f[m])).sum())

    # The full-matrix KL is dominated by the diagonal -- "it stayed" is most of
    # what this world does and the easiest thing to learn. Restricting to
    # off-diagonal transitions asks whether the mind learned anything about
    # what follows what, which is the only part a world model is for.
    off = ~np.eye(n, dtype=bool)
    return {"kl_imagined_to_real": round(kl(T_imag, T_real), 4),
            "kl_shuffled_control_to_real": round(kl(T_ctl, T_real), 4),
            "kl_offdiag_imagined": round(kl(T_imag * off, T_real * off), 4),
            "kl_offdiag_control": round(kl(T_ctl * off, T_real * off), 4),
            "diag_mass_real": round(float(np.trace(T_real) / T_real.sum()), 4),
            "diag_mass_imagined": round(float(np.trace(T_imag)
                                              / T_imag.sum()), 4),
            "self_loop_real": round(selfloop(real), 4),
            "self_loop_imagined": round(selfloop(imag), 4),
            "self_loop_control": round(selfloop(imag_c), 4),
            "collapsed_real": round(collapse(real), 4),
            "collapsed_imagined": round(collapse(imag), 4),
            "entropy_real": round(ent(real), 4),
            "entropy_imagined": round(ent(imag), 4),
            "entropy_control": round(ent(imag_c), 4),
            "states_visited_real": int(len(set(np.concatenate(real)))),
            "states_visited_imagined": int(len(set(np.concatenate(imag)))),
            "_T": {"real": T_real.tolist(), "imagined": T_imag.tolist(),
                   "control": T_ctl.tolist()}}


def test_rollout(D, mind, cells):
    """Test 3 -- imagining forward, against persistence.

    Skill is kept **per sequence**, not pooled. Frames inside one video are
    near-copies, so pooling 400 of them and quoting the total treats a
    correlated sample as 400 independent ones -- the mistake §9.35 made and
    corrected. Thirteen sequences are the independent units, and the paired
    bootstrap in :func:`main` runs over those.
    """
    names = mind.concepts
    res = {}
    for h in HORIZONS:
        per_seq_m, per_seq_p, tot = [], [], 0
        for s, _ in D["seqs"]:
            traj = list(cells[D["vseq"] == s])
            hit = per = n = 0
            for t in range(len(traj) - h):
                cur = names[traj[t]]
                pred = cur
                for _ in range(h):                # roll the model forward
                    nxt = mind.predict_next(pred)
                    pred = nxt if nxt is not None else pred
                truth = names[traj[t + h]]
                hit += pred == truth
                per += cur == truth               # persistence: nothing moved
                n += 1
            if n:
                per_seq_m.append(hit / n)
                per_seq_p.append(per / n)
                tot += n
        res[f"h={h}"] = {"model": round(float(np.mean(per_seq_m)), 4),
                         "persistence": round(float(np.mean(per_seq_p)), 4),
                         "skill": round(float(np.mean(per_seq_m)
                                              - np.mean(per_seq_p)), 4),
                         "per_seq_model": per_seq_m,
                         "per_seq_persistence": per_seq_p,
                         "n_frames": tot, "n_sequences": len(per_seq_m)}
    return res


def test_surprise(D, mind, cells, rng):
    """Test 4 -- does surprise rise on a transition the world did not make?"""
    names = mind.concepts
    within, spliced = [], []
    traj = {s: list(cells[D["vseq"] == s]) for s, _ in D["seqs"]}
    keys = list(traj)
    for s in keys:
        t = traj[s]
        for a, b in zip(t[:-1], t[1:]):
            within.append(mind.surprise(names[a], names[b]))
        other = traj[keys[int(rng.integers(len(keys)))]]
        for a in t[:-1]:
            b = other[int(rng.integers(len(other)))]
            spliced.append(mind.surprise(names[a], names[b]))
    w, sp = np.array(within), np.array(spliced)
    pooled = np.sqrt((w.var() + sp.var()) / 2) + 1e-9
    return {"within_sequence": round(float(w.mean()), 4),
            "spliced_across": round(float(sp.mean()), 4),
            "difference": round(float(sp.mean() - w.mean()), 4),
            "cohens_d": round(float((sp.mean() - w.mean()) / pooled), 4),
            "n_within": int(len(w)), "n_spliced": int(len(sp)),
            "_within": w.tolist(), "_spliced": sp.tolist()}


def test_cross_modal(D, assoc):
    """Test 5 -- hear a held-out recording, imagine the sight."""
    V, y = D["V"], D["vlab"]
    protos = np.stack([_unit(np.stack([assoc.prep_v(v) for v in V[y == c]]
                                      ).mean(0))
                       for c in range(len(bn.NAMES))])
    hit, margin, S = 0, [], []
    for a, lab in zip(D["A_te"], D["ay_te"]):
        img = assoc.vision_from_sound(a)          # entirely auditory query
        sc = protos @ img
        S.append(sc)
        hit += int(np.argmax(sc)) == int(lab)
        srt = np.sort(sc)[::-1]
        margin.append(float(srt[0] - srt[1]))
    return {"imagined_sight_names_its_class": round(hit / max(len(S), 1), 4),
            "chance": round(1.0 / len(bn.NAMES), 4),
            "mean_margin": round(float(np.mean(margin)), 4),
            "n": len(S), "_scores": np.array(S).tolist(),
            "_labels": D["ay_te"].tolist()}


# ------------------------------------------------------------------ figures
def figures(R, D):
    os.makedirs(FIGDIR, exist_ok=True)
    paths = []

    # 1 -- the novelty x coherence plane.  The interesting arms pile up on the
    # target, so labels get leader lines and staggered offsets: the one region
    # that decides the test is the one a default layout makes unreadable.
    im = {k: v for k, v in R["imagery"].items() if not k.startswith("_")}
    t = im["a real unseen crop"]

    def style(k):
        if k == "a real unseen crop":
            return "#c0392b", "*", 330
        if k.endswith("stored crops, averaged"):
            return "#8e44ad", "D", 120
        return "#2c3e50", "o", 105

    fig, (ax, zx) = plt.subplots(1, 2, figsize=(13.4, 6.0),
                                 gridspec_kw={"width_ratios": [1.25, 1]})
    for a in (ax, zx):
        a.axhspan(t["coherence"], 1.02, color="#27ae60", alpha=.07)
        a.axhline(t["coherence"], ls=":", c="#c0392b", lw=1.1)
        a.axvline(t["novelty"], ls=":", c="#c0392b", lw=1.1)
        a.grid(alpha=.22)

    order = sorted(im, key=lambda k: (-im[k]["coherence"], im[k]["novelty"]))
    off = [(11, 10), (11, -18), (11, 12), (11, -16), (11, 8),
           (11, -14), (11, 14), (11, -20), (11, 6), (11, -12), (11, 16)]
    for i, k in enumerate(order):
        v = im[k]
        c, mk, sz = style(k)
        ax.scatter(v["novelty"], v["coherence"], s=sz, marker=mk, c=c,
                   zorder=4, edgecolors="w", linewidths=.7)
        ax.annotate(k, (v["novelty"], v["coherence"]),
                    textcoords="offset points", xytext=off[i % len(off)],
                    fontsize=8, color=c, zorder=5,
                    arrowprops=dict(arrowstyle="-", lw=.6, color=c,
                                    shrinkA=0, shrinkB=3, alpha=.6))
    ax.text(.86, t["coherence"] + .008,
            "only above this line does novelty count",
            fontsize=8, c="#1e8449", ha="right")
    ax.set_xlim(-.06, .90)
    ax.set_ylim(.13, .66)
    ax.set_xlabel("novelty   (1 − cos to the nearest stored crop)")
    ax.set_ylabel("coherence   (does it still read as its own category)")
    ax.set_title("the whole plane", fontsize=10)

    # The arms that decide the test sit on top of one another, so the region
    # carrying the conclusion gets a panel of its own rather than a tangle.
    near = sorted((k for k in im
                   if abs(im[k]["novelty"] - t["novelty"]) < .08
                   and abs(im[k]["coherence"] - t["coherence"]) < .05),
                  key=lambda k: im[k]["novelty"])
    for j, k in enumerate(near):
        v = im[k]
        c, mk, sz = style(k)
        zx.scatter(v["novelty"], v["coherence"], s=sz * 1.5, marker=mk, c=c,
                   zorder=4, edgecolors="w", linewidths=.8)
        zx.annotate(f"{k}\n({v['novelty']:.3f}, {v['coherence']:.3f})",
                    (v["novelty"], v["coherence"]),
                    textcoords="offset points",
                    xytext=(0, 20 if j % 2 == 0 else -34),
                    ha="center", fontsize=8.4, color=c, zorder=5)
    zx.set_xlim(t["novelty"] - .055, t["novelty"] + .055)
    zx.set_ylim(t["coherence"] - .022, t["coherence"] + .022)
    zx.set_xlabel("novelty")
    zx.set_title("the region that decides it", fontsize=10)
    fig.suptitle("Test 1 — mental imagery on real video.  The target is the "
                 "red star, not the top-right corner.\nNo generative arm is as "
                 "coherent as real data; averaging three stored crops with no "
                 "concept layer sits exactly on it.", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, .93))
    p = f"{FIGDIR}/01_imagery_plane.png"
    fig.savefig(p, dpi=125)
    plt.close(fig)
    paths.append(p)

    # 2 -- verbatim vs temperature
    ts = [t for t in TEMPS]
    vb = [im[f"sampled T={t}"]["verbatim"] for t in ts]
    nv = [im[f"sampled T={t}"]["novelty"] for t in ts]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(ts, vb, "o-", c="#c0392b", label="verbatim (cos > 0.999)")
    ax.plot(ts, nv, "s-", c="#2980b9", label="novelty")
    ax.set_xlabel("temperature")
    ax.set_ylabel("fraction / novelty")
    ax.set_title("Test 1b — does turning up the noise move away from memory?",
                 fontsize=11)
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    p = f"{FIGDIR}/02_verbatim_temperature.png"
    fig.savefig(p, dpi=125)
    plt.close(fig)
    paths.append(p)

    # 3 -- transition matrices
    S = R["sequences"]["_T"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    for ax, key, ttl in zip(axes, ("real", "imagined", "control"),
                            ("what the world did",
                             "what the mind imagined",
                             "control: shuffled-time experience")):
        M = np.array(S[key], float)
        M = M / max(M.sum(), 1e-9)
        ax.imshow(M, cmap="magma", interpolation="nearest")
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("next concept cell")
    axes[0].set_ylabel("current concept cell")
    fig.suptitle(f"Test 2 — sequential imagination.  KL(imagined‖real) "
                 f"{R['sequences']['kl_imagined_to_real']:.3f}   vs   "
                 f"shuffled control {R['sequences']['kl_shuffled_control_to_real']:.3f}",
                 fontsize=11)
    fig.tight_layout()
    p = f"{FIGDIR}/03_transitions.png"
    fig.savefig(p, dpi=125)
    plt.close(fig)
    paths.append(p)

    # 4 -- rollout against persistence
    ro = R["rollout"]
    hs = [int(k.split("=")[1]) for k in ro]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(hs, [ro[f"h={h}"]["model"] for h in hs], "o-", c="#27ae60",
            label="world model, rolled forward")
    ax.plot(hs, [ro[f"h={h}"]["persistence"] for h in hs], "s--", c="#7f8c8d",
            label="persistence (nothing moved)")
    ax.set_xlabel("horizon (frames ahead)")
    ax.set_ylabel("next-state accuracy")
    ax.set_title("Test 3 — imagining forward.  The dashed line is the arm to beat",
                 fontsize=11)
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    p = f"{FIGDIR}/04_rollout.png"
    fig.savefig(p, dpi=125)
    plt.close(fig)
    paths.append(p)

    # 5 -- surprise distributions
    su = R["surprise"]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    bins = np.linspace(0, 1, 26)
    ax.hist(su["_within"], bins=bins, alpha=.62, label="within a sequence",
            color="#2980b9", density=True)
    ax.hist(su["_spliced"], bins=bins, alpha=.62,
            label="spliced from another video", color="#c0392b", density=True)
    ax.set_xlabel("surprise")
    ax.set_ylabel("density")
    ax.set_title(f"Test 4 — surprise.  difference {su['difference']:+.3f}, "
                 f"d = {su['cohens_d']:+.2f}", fontsize=11)
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    p = f"{FIGDIR}/05_surprise.png"
    fig.savefig(p, dpi=125)
    plt.close(fig)
    paths.append(p)

    # 6 -- cross-modal imagery
    cm = R["cross_modal"]
    S = np.array(cm["_scores"], float)
    lab = np.array(cm["_labels"], int)
    M = np.stack([S[lab == c].mean(0) for c in range(len(bn.NAMES))])
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im2 = ax.imshow(M, cmap="viridis")
    ax.set_xticks(range(len(bn.NAMES)))
    ax.set_xticklabels(bn.NAMES, rotation=40, ha="right")
    ax.set_yticks(range(len(bn.NAMES)))
    ax.set_yticklabels([f"{n} heard" for n in bn.NAMES])
    ax.set_xlabel("visual category the imagined sight matches")
    for i in range(len(bn.NAMES)):
        for j in range(len(bn.NAMES)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=7.5,
                    color="w" if M[i, j] < M.max() * .65 else "k")
    fig.colorbar(im2, ax=ax, shrink=.82)
    ax.set_title(f"Test 5 — hear a held-out recording, imagine the sight\n"
                 f"names its own class {cm['imagined_sight_names_its_class']:.3f} "
                 f"(chance {cm['chance']:.3f}); the diagonal is the claim",
                 fontsize=10.5)
    fig.tight_layout()
    p = f"{FIGDIR}/06_cross_modal.png"
    fig.savefig(p, dpi=125)
    plt.close(fig)
    paths.append(p)
    return paths


# --------------------------------------------------------------------- main
def boot(diff, n=20000, seed=7):
    """Bootstrap CI of a paired mean difference over independent units."""
    d = np.asarray(diff, float)
    r = np.random.default_rng(seed)
    b = d[r.integers(0, len(d), (n, len(d)))].mean(1)
    return (round(float(d.mean()), 4),
            [round(float(x), 4) for x in np.percentile(b, [2.5, 97.5])])


def main():
    rng = np.random.default_rng(SEED)
    print("building the real brain ...", flush=True)
    D = build(rng)                       # the codes: computed once, shared
    names = [f"c{c}" for c in range(N_CELL)]

    per_seed = []
    for sd in SEEDS:
        r = np.random.default_rng(sd)
        assoc, cells = bind_all(D, r, sd)
        D["brain"].assoc = assoc
        D["brain"].concepts = names
        mind = Mind(D["brain"], list(names), {})
        one = {"imagery": test_imagery(D, assoc, r),
               "sequences": test_sequences(D, mind, cells, r),
               "rollout": test_rollout(D, mind, cells),
               "surprise": test_surprise(D, mind, cells, r),
               "cross_modal": test_cross_modal(D, assoc)}
        per_seed.append(one)
        print(f"  seed {sd}: {len(set(cells.tolist()))}/{N_CELL} cells, "
              f"KL {one['sequences']['kl_imagined_to_real']:.3f}, "
              f"cross-modal "
              f"{one['cross_modal']['imagined_sight_names_its_class']:.3f}",
              flush=True)

    # -- aggregate: every headline number is a mean over SEEDS, with spread --
    R = {"seeds": list(SEEDS)}
    arms = [k for k in per_seed[0]["imagery"] if not k.startswith("_")]
    R["imagery"] = {}
    for a in arms:
        R["imagery"][a] = {
            m: round(float(np.mean([p["imagery"][a][m] for p in per_seed])), 4)
            for m in ("novelty", "coherence", "verbatim")}
        R["imagery"][a]["novelty_sd"] = round(float(np.std(
            [p["imagery"][a]["novelty"] for p in per_seed])), 4)
    R["imagery"]["_diag"] = per_seed[0]["imagery"]["_diag"]

    def avg(sec, key):
        return round(float(np.mean([p[sec][key] for p in per_seed])), 4)

    s0 = per_seed[0]["sequences"]
    R["sequences"] = {k: avg("sequences", k) for k in s0 if not k.startswith("_")}
    R["sequences"]["_T"] = s0["_T"]
    R["surprise"] = {k: avg("surprise", k) for k in per_seed[0]["surprise"]
                     if not k.startswith("_")}
    R["surprise"]["_within"] = per_seed[0]["surprise"]["_within"]
    R["surprise"]["_spliced"] = per_seed[0]["surprise"]["_spliced"]
    R["cross_modal"] = {k: avg("cross_modal", k)
                        for k in per_seed[0]["cross_modal"]
                        if not k.startswith("_")}
    R["cross_modal"]["_scores"] = per_seed[0]["cross_modal"]["_scores"]
    R["cross_modal"]["_labels"] = per_seed[0]["cross_modal"]["_labels"]

    # Rollout: paired over SEQUENCES. The seeds are averaged into each
    # sequence first and are NOT extra units -- five seeds re-measure the same
    # thirteen videos, so concatenating them would bootstrap 65 correlated
    # values as if independent and buy a ~2.2x too narrow interval for free.
    # The independent unit is the video.
    R["rollout"] = {}
    for h in HORIZONS:
        per = np.stack([np.array(p["rollout"][f"h={h}"]["per_seq_model"])
                        - np.array(p["rollout"][f"h={h}"]
                                   ["per_seq_persistence"])
                        for p in per_seed])              # (seeds, sequences)
        d = per.mean(0)                                  # -> one per sequence
        m, ci = boot(d)
        R["rollout"][f"h={h}"] = {
            "model": avg("rollout", f"h={h}") if False else round(float(
                np.mean([np.mean(p["rollout"][f"h={h}"]["per_seq_model"])
                         for p in per_seed])), 4),
            "persistence": round(float(np.mean(
                [np.mean(p["rollout"][f"h={h}"]["per_seq_persistence"])
                 for p in per_seed])), 4),
            "skill": m, "skill_ci95": ci,
            "separated": bool(ci[0] > 0),
            "n_sequences": len(d)}

    print("figures ...", flush=True)
    figs = figures(R, D)

    im = R["imagery"]
    real = im["a real unseen crop"]
    ctl = im[f"{N_COMPOSE} stored crops, averaged"]
    gen = [k for k in im if k.startswith(("sampled", "composed"))]
    # Closest arm on the plane, by distance to where real data sits. Ranking by
    # novelty alone picks whichever arm collapsed coherence hardest -- the exact
    # degeneracy §9.19 built the pair to prevent, and the first version of this
    # verdict walked straight into it and named T=8 (coherence 0.36) "best".
    best = min(gen, key=lambda k: (im[k]["novelty"] - real["novelty"]) ** 2
               + (im[k]["coherence"] - real["coherence"]) ** 2)
    clears = [k for k in gen if im[k]["coherence"] >= real["coherence"]]
    R["verdict"] = {
        "imagery_closest_arm": best,
        "imagery_arms_at_least_as_coherent_as_real": clears,
        "imagery_reaches_real_data": bool(
            any(im[k]["novelty"] >= real["novelty"] for k in clears)),
        # a comparison against the control is only meaningful between arms that
        # are BOTH at least as coherent as real data
        "imagery_beats_the_no_concept_control": bool(
            ctl["coherence"] >= real["coherence"]
            and any(im[k]["novelty"] > ctl["novelty"] for k in clears)),
        "no_concept_control_reaches_real_data": bool(
            ctl["coherence"] >= real["coherence"]
            and ctl["novelty"] >= real["novelty"]),
        "sequences_beat_shuffled_control": bool(
            R["sequences"]["kl_imagined_to_real"]
            < R["sequences"]["kl_shuffled_control_to_real"]),
        # a bare `skill > 0` is satisfied by +0.003; the CI has to clear zero
        "rollout_separated_from_persistence": {
            f"h={h}": R["rollout"][f"h={h}"]["separated"] for h in HORIZONS},
        "surprise_rises_on_spliced": bool(R["surprise"]["difference"] > 0),
        "cross_modal_above_chance": bool(
            R["cross_modal"]["imagined_sight_names_its_class"]
            > R["cross_modal"]["chance"]),
    }
    R["figures"] = figs
    R["caveats"] = [
        "H23 measured cross-scene concept naming at chance; every number here "
        "runs on concepts reliable within a scene only",
        f"verbatim is 0.000 everywhere because {N_CELL} cells over "
        f"{len(D['V'])} crops leaves no singleton "
        f"({R['imagery']['_diag']['singletons']} of "
        f"{R['imagery']['_diag']['cells_used']} used cells won exactly once). "
        "§9.19's 0.368 came from 112 cells over 216 pairs. This is a "
        "configuration difference, not a repair of that failure",
        "the visual prototypes in test 5 are built on all crops; only the "
        "recordings are held out, so it measures hearing->sight and not "
        "generalisation of the sight",
    ]
    json.dump(R, open(OUT, "w"), indent=1)

    print("\n" + "=" * 66)
    print(f"{'arm':<32}{'novelty':>9}{'coherence':>11}{'verbatim':>10}")
    for k, v in im.items():
        if k.startswith("_"):
            continue
        mark = "  <- target" if k == "a real unseen crop" else ""
        print(f"{k:<32}{v['novelty']:>9.3f}±{v['novelty_sd']:<5.3f}"
              f"{v['coherence']:>9.3f}{v['verbatim']:>10.3f}{mark}")
    s = R["sequences"]
    print(f"\nsequences  KL(imagined‖real) {s['kl_imagined_to_real']:.3f}  "
          f"vs shuffled control {s['kl_shuffled_control_to_real']:.3f}")
    print(f"           self-loop real {s['self_loop_real']:.3f} / imagined "
          f"{s['self_loop_imagined']:.3f}   collapsed {s['collapsed_imagined']:.3f}")
    print("\nrollout (paired over sequences, CI must clear 0)")
    for h in HORIZONS:
        r = R["rollout"][f"h={h}"]
        print(f"  h={h}: model {r['model']:.3f}  persistence "
              f"{r['persistence']:.3f}  skill {r['skill']:+.4f} "
              f"CI [{r['skill_ci95'][0]:+.4f}, {r['skill_ci95'][1]:+.4f}]"
              f"{'  SEPARATED' if r['separated'] else ''}")
    print(f"\nsurprise   within {R['surprise']['within_sequence']:.3f}  "
          f"spliced {R['surprise']['spliced_across']:.3f}  "
          f"d {R['surprise']['cohens_d']:+.2f}")
    print(f"cross-modal {R['cross_modal']['imagined_sight_names_its_class']:.3f} "
          f"(chance {R['cross_modal']['chance']:.3f})")
    print("\n" + json.dumps(R["verdict"], indent=1))
    print(f"wrote {OUT} and {len(figs)} figures in {FIGDIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
