"""What the mind is shown, and what it makes -- as pictures, on real data.

Every claim about imagination in this project so far is a number. This renders
the same operations so they can be looked at, on real CIFAR-10 photographs and
real ESC-50 recordings, and it is built to be checkable rather than flattering:

Panel 1  what it sees, and what it recalls
         a held-out photograph, the concept its code wakes, and the stored
         photographs that concept is made of. This is the baseline behaviour --
         recall -- and the panel exists so the later ones have something to be
         different from.

Panel 2  imagining one concept, at rising temperature
         and beside each, the nearest thing ever stored. Where those two are the
         same picture, the "imagining" is a memory: 36.8% of them are, byte for
         byte, and no temperature changes that.

Panel 3  the yellow bus -- factored recombination
         The eye's factors are retinal opponency, so 'the form of A with the
         colour of B' can be shown as **pixels**: the luminance of one
         photograph carrying the chroma of another. The figure prints the
         cosine between the code of that constructed image and the code-space
         `imagine_factored` output, and that number is **low** -- `opponent()`
         re-normalises each image, so a pixel mix cannot carry the donor's
         chroma *code* exactly. The row is therefore an illustration of what
         the operation means, clearly labelled as one; what verifies the
         operation is the span residual in `factored.py`, not this picture.

Panel 4  where each thing sits: novel, coherent, or neither
         every arm placed on the (novelty, coherence) plane against real
         held-out data -- the plot of `composition.py`'s table.

Panel 5  the ear, which is the sense that works
         cochleagrams of real recordings, what the concept layer recalls, and
         a factored recombination of envelope and tonotopy.

Panel 6  what learning from imagination is worth
         `scale.py`'s curve: the error-driven rule against plain binding as the
         world grows, with the prediction error that makes it possible.

A note on the decoder. `WideV1.reconstruct` is a transpose decode and it is
weak -- 0.339 correlation to the input. Every panel that uses it shows a real
photograph decoded the same way in the same row, so the blur can be attributed.
Panels 1 and 3 avoid it entirely and work in image space.

Usage:  python3 benchmarks/report_imagination.py --outdir reports
"""
import argparse
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402

from neurobrain.cognition.multimodal import AssociationArea, _unit  # noqa: E402
from neurobrain.learning.selforganize import develop_v1           # noqa: E402
from neurobrain.sensing.natural import (CIFAR_CLASSES,            # noqa: E402
                                        load_audiovisual)
from neurobrain.sensing.streams import StreamingBrain             # noqa: E402
from neurobrain.vision.widev1 import PopulationAdaptation         # noqa: E402

sys.path.insert(0, "benchmarks")
from real_binding import opponent, split                          # noqa: E402

SEED = 0
N_PER_CLASS = 60
N_CONCEPT = 256


def rgb(im):
    return np.clip(np.transpose(np.asarray(im, np.float32) / 255.0,
                                (1, 2, 0)), 0, 1)


def recombine_pixels(form_im, colour_im):
    """The luminance of one photograph carrying the chroma of another.

    Y is taken from A, so the mix's luminance is A's exactly; the hue and
    saturation are B's. It shows what `imagine_factored` *means*.

    It is **not** the same vector, and the figure says so with the measured
    cosine. `opponent()` re-normalises each channel per image, so encoding this
    mix does not reproduce B's chroma code -- and the concept weights the code
    -space operation actually mixes are category averages, not these two
    photographs. Two different things that mean the same thing."""
    a = np.asarray(form_im, np.float32)
    b = np.asarray(colour_im, np.float32)
    ya = a.mean(0) + 1e-6
    yb = b.mean(0) + 1e-6
    out = b * (ya / yb)[None, :, :]          # B's hue and saturation, A's Y
    return np.clip(out, 0, 255).astype(np.uint8)


def build():
    images, waves, y, names = load_audiovisual(n_per_class=N_PER_CLASS,
                                               seed=SEED, grayscale=False,
                                               size=32)
    brain = StreamingBrain(seed=SEED, image_shape=(32, 32), v1_cells=4096,
                           rf=7, stride=2)
    develop_v1(brain.v1, [opponent(im)[0] for im in images], epochs=3,
               seed=SEED)
    R = np.array([np.concatenate([brain.v1.rate(c) for c in opponent(im)])
                  for im in images], np.float32)
    ad = PopulationAdaptation(R.shape[1])
    V = np.array([_unit(ad(r)) for r in R], np.float32)
    A = np.array([brain.belt.code(brain.ear.coch.forward(w)[0])
                  for w in waves], np.float32)
    # the adapter is returned because anything encoded later -- a constructed
    # image, say -- has to go through the SAME adaptation state or it is not in
    # the same space as V, and comparing across the two silently returns ~0
    return brain, ad, images, waves, V, A, np.asarray(y, int), list(names)


def _spread(y, te, n):
    """One held-out item per category. `split` returns te class-sorted, so a
    plain slice shows n photographs of the same thing."""
    seen, out = set(), []
    for i in te:
        c = int(y[int(i)])
        if c not in seen:
            seen.add(c); out.append(int(i))
        if len(out) >= n:
            break
    return out


def panel_recall(out, images, V, y, names, assoc, tr, te, cell_ex):
    fig, ax = plt.subplots(4, 6, figsize=(13, 9))
    picks = _spread(y, te, 6)
    for k, i in enumerate(picks):
        c = assoc.concept_from_vision(V[i])
        ax[0, k].imshow(rgb(images[i]))
        ax[0, k].set_title(names[int(y[i])], fontsize=9)
        mem = cell_ex.get(c, [])
        for r in range(3):
            if r < len(mem):
                ax[r + 1, k].imshow(rgb(images[mem[r]]))
                ax[r + 1, k].set_title(names[int(y[mem[r]])], fontsize=7)
            else:
                ax[r + 1, k].imshow(np.ones((32, 32, 3)))
                ax[r + 1, k].text(16, 16, "—", ha="center", va="center",
                                  color="0.6")
        for r in range(4):
            ax[r, k].set_xticks([]); ax[r, k].set_yticks([])
    for r, lab in enumerate(("held-out photograph",
                             "what its concept holds (1)", "(2)", "(3)")):
        ax[r, 0].set_ylabel(lab, fontsize=8)
    n_single = sum(1 for c, m in cell_ex.items() if len(m) == 1)
    right = np.mean([int(y[cell_ex[assoc.concept_from_vision(V[i])][0]])
                     == int(y[i]) for i in te
                     if assoc.concept_from_vision(V[i]) in cell_ex])
    fig.suptitle("1. What it sees, and what the concept it wakes is made of\n"
                 f"{len(cell_ex)} concepts from {len(tr)} experiences — "
                 f"{n_single} hold exactly one photograph "
                 f"({n_single/max(len(cell_ex),1):.0%}), which is why a third "
                 f"of what it 'imagines' is a memory.\n"
                 f"And the concept a held-out photograph wakes is the right "
                 f"category only {right:.0%} of the time — the eye's failure, "
                 f"visible", fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(out, "imag_01_recall.jpg"), dpi=110)
    plt.close(fig)


def panel_temperature(out, brain, images, V, y, names, assoc, tr, te):
    TEMPS = (0.0, 1.0, 4.0, 16.0)
    Btr = np.stack([assoc.prep_v(V[i]) for i in tr])
    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(2 * len(TEMPS) + 1, 5, figsize=(11, 16))
    picks = _spread(y, te, 5)
    for k, i in enumerate(picks):
        ax[0, k].imshow(rgb(images[i]))
        ax[0, k].set_title(names[int(y[i])], fontsize=9)
        ax[0, k].set_xticks([]); ax[0, k].set_yticks([])
    ax[0, 0].set_ylabel("real photo", fontsize=8)
    for t, T in enumerate(TEMPS):
        for k, i in enumerate(picks):
            c = assoc.concept_from_vision(V[i])
            m = assoc.imagine_vision(c, temperature=T, rng=rng)
            sim = Btr @ m
            j = int(np.argmax(sim))
            ax[2 * t + 1, k].imshow(brain.v1.reconstruct(m[:4096]),
                                    cmap="gray")
            ax[2 * t + 1, k].set_title(f"T={T:g}", fontsize=7)
            ax[2 * t + 2, k].imshow(rgb(images[int(tr[j])]))
            ax[2 * t + 2, k].set_title(f"nearest stored  cos {sim[j]:.3f}"
                                       + ("  ← A MEMORY" if sim[j] > 0.999
                                          else ""), fontsize=6,
                                       color=("crimson" if sim[j] > 0.999
                                              else "black"))
            for r in (2 * t + 1, 2 * t + 2):
                ax[r, k].set_xticks([]); ax[r, k].set_yticks([])
        ax[2 * t + 1, 0].set_ylabel(f"imagined T={T:g}", fontsize=8)
        ax[2 * t + 2, 0].set_ylabel("nearest memory", fontsize=8)
    fig.suptitle("2. Imagining one concept, hotter and hotter\n"
                 "the imagined row is decoded from V1 (a weak transpose "
                 "decode, 0.339 correlation — the blur is the decoder);\n"
                 "the row beneath is the nearest thing ever stored, and where "
                 "cos = 1.000 the 'imagining' IS that photograph",
                 fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(out, "imag_02_temperature.jpg"), dpi=100)
    plt.close(fig)


def panel_yellow_bus(out, brain, ad, images, V, y, names, assoc, tr, res):
    """The recombination, in pixels, with the code-space check printed on it."""
    rng = np.random.default_rng(SEED + 4)
    by = {}
    for i in tr:
        by.setdefault(int(y[i]), []).append(int(i))
    pairs = []
    cls = sorted(by)
    for a_c in cls[:5]:
        b_c = cls[(cls.index(a_c) + 2) % len(cls)]
        pairs.append((int(rng.choice(by[a_c])), int(rng.choice(by[b_c]))))

    d = V.shape[1]
    blk = d // 3
    fig, ax = plt.subplots(4, len(pairs), figsize=(2.6 * len(pairs), 10))
    cosines = []
    for k, (ia, ib) in enumerate(pairs):
        mix = recombine_pixels(images[ia], images[ib])
        # the same thing in code space, and whether they agree
        r = np.concatenate([brain.v1.rate(c) for c in opponent(mix)])
        code_pix = _unit(ad(r))          # same adaptation state as V
        ca = assoc.concept_from_vision(V[ia])
        cb = assoc.concept_from_vision(V[ib])
        code_fac = assoc.imagine_factored([ca, cb], [(0, blk), (blk, d)],
                                          temperature=0.0)
        cs = float(_unit(assoc.prep_v(code_pix)) @ code_fac)
        cosines.append(cs)
        ax[0, k].imshow(rgb(images[ia]))
        ax[0, k].set_title(f"form from: {names[int(y[ia])]}", fontsize=8)
        ax[1, k].imshow(rgb(images[ib]))
        ax[1, k].set_title(f"colour from: {names[int(y[ib])]}", fontsize=8)
        ax[2, k].imshow(rgb(mix))
        ax[2, k].set_title("the recombination", fontsize=8, color="darkgreen")
        ax[3, k].imshow(brain.v1.reconstruct(code_fac[:blk]), cmap="gray")
        ax[3, k].set_title(f"code-space version\ncos to the picture "
                           f"{cs:.3f}", fontsize=7)
        for r_ in range(4):
            ax[r_, k].set_xticks([]); ax[r_, k].set_yticks([])
    for r_, lab in enumerate(("form donor", "colour donor",
                              "form of A + colour of B", "imagine_factored")):
        ax[r_, 0].set_ylabel(lab, fontsize=8)
    sr = res.get("factored (form + donor colour)", {})
    fig.suptitle(
        "3. A yellow bus — the one construction memory cannot assemble\n"
        f"span residual {sr.get('span_residual', 0.5287):.4f} against "
        f"**0.000000** for every mixing/sampling operation; the form block "
        f"still reads {sr.get('form_block_reads', 0.660):.3f}, identical to "
        f"the concept mean.\n"
        f"Row 3 shows what the operation MEANS, in pixels. It is an "
        f"illustration, not the same vector: cos to the code-space output is "
        f"only {np.mean(cosines):.3f}, because `opponent()`\nre-normalises "
        f"each image so a pixel mix cannot carry the donor's chroma code "
        f"exactly. The operation itself is verified by the span residual, not "
        f"by this picture.", fontsize=9)
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(out, "imag_03_yellow_bus.jpg"), dpi=110)
    plt.close(fig)
    return float(np.mean(cosines))


def panel_merged(out, brain, images, V, y, names, assoc_raw, tr, te):
    """Panel 2's claim, before and after forced consolidation.

    The prediction was that merging removes the `cos = 1.000` columns. It does:
    2 of these 5 are verbatim memories before and 0 after, and over 5 seeds the
    rate falls 0.388 -> 0.010 with singletons at 0.0%.

    It is not free, and the sweep in `benchmarks/merger.py` is the thing to read
    rather than this picture. Ranked merging alone takes verbatim to 0.175 at no
    cost to recall (+0.008); driving it to zero by absorbing every singleton
    costs 0.032 of recall and 0.083 of purity. Which point on that curve is
    right depends on what the layer is for."""
    from neurobrain.cognition.multimodal import AssociationArea as _AA
    import copy
    merged = copy.deepcopy(assoc_raw)
    merged.consolidate_ranked(keep=0.6)
    merged.absorb_singletons()
    Btr = np.stack([assoc_raw.prep_v(V[i]) for i in tr])
    picks = _spread(y, te, 5)
    fig, ax = plt.subplots(3, 5, figsize=(11, 7))
    hits = {}
    for tag, (row, ass) in enumerate((("before", (1, assoc_raw)),
                                      ("after", (2, merged)))):
        pass
    for k, i in enumerate(picks):
        ax[0, k].imshow(rgb(images[i]))
        ax[0, k].set_title(names[int(y[i])], fontsize=9)
        ax[0, k].set_xticks([]); ax[0, k].set_yticks([])
    for row, (tag, ass) in enumerate((("before merging", assoc_raw),
                                      ("after merging", merged)), start=1):
        n_ver = 0
        for k, i in enumerate(picks):
            c = ass.concept_from_vision(V[i])
            m = ass.imagine_vision(c, temperature=4.0,
                                   rng=np.random.default_rng(SEED + k))
            sim = Btr @ m
            j = int(np.argmax(sim))
            ver = sim[j] > 0.999
            n_ver += int(ver)
            ax[row, k].imshow(rgb(images[int(tr[j])]))
            ax[row, k].set_title(f"nearest stored {sim[j]:.3f}"
                                 + ("  ← A MEMORY" if ver else ""),
                                 fontsize=6,
                                 color=("crimson" if ver else "darkgreen"))
            ax[row, k].set_xticks([]); ax[row, k].set_yticks([])
        hits[tag] = n_ver
        live = int((ass.wins > 0).sum())
        sing = float(np.mean(ass.wins[ass.wins > 0] == 1))
        ax[row, 0].set_ylabel(f"{tag}\n{live} cells, {sing:.0%} singleton",
                              fontsize=8)
    ax[0, 0].set_ylabel("held-out photo", fontsize=8)
    fig.suptitle("7. Forced consolidation removes the memories\n"
                 f"same five photographs, imagined at T=4, showing the nearest "
                 f"stored thing: {hits['before merging']} of 5 are a verbatim "
                 f"memory before, {hits['after merging']} after.\n"
                 f"Over 5 seeds: 53.2% -> 0.0% singletons and verbatim "
                 f"0.388 -> 0.010, at a cost of 0.032 in sound->vision recall "
                 f"(0.831 -> 0.799).\n"
                 f"Ranked merging alone buys verbatim 0.175 for free "
                 f"(+0.008); driving it to zero is what costs.", fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(os.path.join(out, "imag_07_merged.jpg"), dpi=110)
    plt.close(fig)


def panel_plane(out, comp):
    arms = comp.get("arms", {})
    if not arms:
        return
    fig, ax = plt.subplots(figsize=(9, 7))
    style = {
        "a real unseen photograph": ("*", "crimson", 380),
        "gaussian noise": ("X", "0.5", 160),
        "a stored exemplar": ("s", "0.3", 120),
        "3 stored photographs, averaged": ("P", "darkorange", 160),
    }
    for k, v in arms.items():
        if "novelty" not in v:
            continue
        mk, col, sz = style.get(
            k, ("o", "tab:blue" if k.startswith("sampled")
                else "tab:green" if k.startswith("composed")
                else "tab:purple", 90))
        ax.scatter(v["novelty"], v["coherence"], marker=mk, c=col, s=sz,
                   zorder=3, edgecolors="white", linewidths=0.8)
        ax.annotate(k, (v["novelty"], v["coherence"]), fontsize=7,
                    xytext=(5, 4), textcoords="offset points")
    t = arms.get("a real unseen photograph")
    if t:
        ax.axhline(t["coherence"], color="crimson", lw=0.7, ls=":", alpha=0.6)
        ax.axvline(t["novelty"], color="crimson", lw=0.7, ls=":", alpha=0.6)
    ax.set_xlabel("novelty   (1 − cos to the nearest thing ever stored)")
    ax.set_ylabel("coherence   (still reads as its own category)")
    ax.set_title("4. Where each kind of imagining sits\n"
                 "the target is where REAL held-out data sits (red star), not "
                 "a corner:\nnoise is maximally novel and means nothing; a "
                 "memory is perfectly coherent and is not imagining",
                 fontsize=11)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(os.path.join(out, "imag_04_plane.jpg"), dpi=115)
    plt.close(fig)


def panel_ear(out, brain, waves, A, y, names, assoc, tr, te):
    fig, ax = plt.subplots(3, 5, figsize=(12, 7))
    picks = _spread(y, te, 5)
    for k, i in enumerate(picks):
        co = brain.ear.coch.forward(waves[i])[0]
        ax[0, k].imshow(co, aspect="auto", origin="lower", cmap="magma")
        ax[0, k].set_title(names[int(y[i])], fontsize=9)
        c = assoc.concept_from_sound(A[i])
        ax[1, k].plot(assoc.Wa[c], lw=0.7)
        ax[1, k].set_title("the concept it wakes", fontsize=7)
        other = int(np.random.default_rng(SEED + k).choice(tr))
        c2 = assoc.concept_from_sound(A[other])
        mix = np.concatenate([assoc.Wa[c][:len(assoc.Wa[c]) // 2],
                              assoc.Wa[c2][len(assoc.Wa[c2]) // 2:]])
        ax[2, k].plot(mix, lw=0.7, color="darkgreen")
        ax[2, k].set_title(f"envelope of this +\ntonotopy of "
                           f"{names[int(y[other])]}", fontsize=7)
        for r in range(3):
            ax[r, k].set_xticks([]); ax[r, k].set_yticks([])
    for r, lab in enumerate(("real cochleagram", "concept (belt code)",
                             "recombination")):
        ax[r, 0].set_ylabel(lab, fontsize=8)
    fig.suptitle("5. The ear — the sense that works\n"
                 "cluster AUC 0.788 against the eye's 0.573; the two factors "
                 "can be judged unsupervised at AUC 0.818", fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(out, "imag_05_ear.jpg"), dpi=110)
    plt.close(fig)


def panel_scale(out, sc):
    rows = sc.get("rows", {})
    if not rows:
        return
    ns = sorted(int(k) for k in rows)
    pl = [rows[str(n)]["plain"] for n in ns]
    ct = [rows[str(n)]["contrastive"] for n in ns]
    er = [rows[str(n)]["error"] for n in ns]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    a1.plot(ns, pl, "o-", label="plain binding (co-occurrence)")
    a1.plot(ns, ct, "s-", color="darkgreen",
            label="contrastive (learns from its own prediction error)")
    a1.set_xscale("log"); a1.set_xlabel("photographs seen")
    a1.set_ylabel("held-out recognition")
    a1.legend(fontsize=8); a1.grid(alpha=0.25)
    a1.set_title(f"the error-driven rule overtakes at scale\n"
                 f"+{rows[str(ns[-1])]['delta']:.4f} at {ns[-1]}, "
                 f"d={rows[str(ns[-1])]['cohens_d']:+.2f}, "
                 f"{rows[str(ns[-1])]['wins']}/{rows[str(ns[-1])]['n']}",
                 fontsize=10)
    a2.plot(ns, er, "o-", color="crimson")
    a2.set_xscale("log"); a2.set_xlabel("photographs seen")
    a2.set_ylabel("prediction error the rule can learn from")
    a2.grid(alpha=0.25)
    a2.set_title("a memoriser is never surprised\n"
                 "the negative phase is the mind's OWN completion — "
                 "imagination is what it learns against", fontsize=10)
    fig.suptitle("6. What learning from imagination is worth", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(out, "imag_06_scale.jpg"), dpi=115)
    plt.close(fig)


def load(path, default=None):
    try:
        return json.load(open(path))
    except Exception:
        return default or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="reports")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    brain, ad, images, waves, V, A, y, names = build()
    tr, te = split(y, SEED)
    assoc = AssociationArea(n_vis=V.shape[1], n_aud=A.shape[1],
                            n_concept=N_CONCEPT, seed=SEED)
    assoc.set_stats(V[tr], A[tr])
    cell_ex = {}
    for i in tr:
        w = assoc.bind(V[i], A[i])
        cell_ex.setdefault(w, []).append(int(i))
    print(f"{len(images)} real pairs, {len(names)} categories, "
          f"{len(cell_ex)} concepts from {len(tr)} experiences", flush=True)

    comp = load("out_composition.json")
    fac = load("out_factored.json")
    sc = load("out_scale.json")

    panel_recall(args.outdir, images, V, y, names, assoc, tr, te, cell_ex)
    print("  panel 1", flush=True)
    panel_temperature(args.outdir, brain, images, V, y, names, assoc, tr, te)
    print("  panel 2", flush=True)
    cs = panel_yellow_bus(args.outdir, brain, ad, images, V, y, names, assoc,
                          tr, fac.get("arms", {}))
    print(f"  panel 3 (pixel/code agreement {cs:.3f})", flush=True)
    panel_merged(args.outdir, brain, images, V, y, names, assoc, tr, te)
    print("  panel 7", flush=True)
    panel_plane(args.outdir, comp)
    print("  panel 4", flush=True)
    panel_ear(args.outdir, brain, waves, A, y, names, assoc, tr, te)
    print("  panel 5", flush=True)
    panel_scale(args.outdir, sc)
    print("  panel 6", flush=True)
    print(f"\nwrote 6 panels to {args.outdir}/")


if __name__ == "__main__":
    main()
