"""Visual JPG report: what was learned, what gets detected, and what it is called.

Panels produced (into --outdir):
  01_learned_prototypes.jpg  what the system learned, per class, plus V1 filters
  02_multi_detection.jpg     one scene, every saccade tagged / classified / labelled
  03_attention_ior.jpg       the saliency map and what inhibition-of-return does
  04_pursuit.jpg             a moving object tracked across frames
  05_confusion.jpg           which classes it confuses, detected vs missed

Usage:  python3 benchmarks/report_vision.py --dataset mnist --outdir reports/
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch

import neurobrain as nb
from neurobrain.sensing.streams import (build_scene, SaccadicEye, StreamingBrain,
                                        MovingScene, _unit)
from neurobrain.vision.widev1 import WideV1, _nearest_prototype

DPI = 110
OK, BAD, DIM = "#2e9e4f", "#d14545", "#8899a6"


def _save(fig, path):
    fig.savefig(path, dpi=DPI, format="jpg", bbox_inches="tight",
                pil_kwargs={"quality": 92})
    plt.close(fig)
    print("wrote", path, flush=True)


# ---------------------------------------------------------------- panel 1 ---
def panel_learned(brain, trx, trY, classes, names, path):
    """What the system actually learned: the class prototypes it will match
    against, and a sample of the V1 receptive fields doing the seeing."""
    n = len(classes)
    fig = plt.figure(figsize=(1.15 * max(n, 8), 6.4))
    gs = fig.add_gridspec(3, max(n, 8), hspace=0.35, wspace=0.15)

    for j, c in enumerate(classes):
        idx = np.where(trY == c)[0]
        ex = trx[idx[0]]
        proto = trx[idx[:24]].mean(0)
        ax = fig.add_subplot(gs[0, j]); ax.imshow(ex, cmap="gray"); ax.axis("off")
        if j == 0:
            ax.set_ylabel("example", fontsize=9)
        ax.set_title(f"{names[c]}", fontsize=9)
        ax = fig.add_subplot(gs[1, j]); ax.imshow(proto, cmap="magma"); ax.axis("off")

    # V1 filters: what the eye's cells are tuned to
    W = getattr(brain.v1, "filters", None)
    if W is None:
        W = getattr(brain.v1, "W", None)
    k = max(n, 8)
    if W is not None:
        W = np.asarray(W)
        rf = brain.v1.rf
        picks = np.linspace(0, len(W) - 1, k).astype(int)
        for j, p in enumerate(picks):
            f = W[p][:rf * rf].reshape(rf, rf) if W[p].size >= rf * rf else None
            ax = fig.add_subplot(gs[2, j]); ax.axis("off")
            if f is not None:
                ax.imshow(f, cmap="RdBu_r")
    fig.text(0.085, 0.855, "one example", ha="right", va="center", fontsize=10)
    fig.text(0.085, 0.545, "class prototype\n(mean of 24)", ha="right",
             va="center", fontsize=10)
    fig.text(0.085, 0.235, "V1 receptive\nfields", ha="right", va="center",
             fontsize=10)
    fig.suptitle("What the system learned — class prototypes it matches against, "
                 "and the V1 filters doing the seeing", fontsize=13, y=0.99)
    _save(fig, path)


# ---------------------------------------------------------------- panel 2 ---
def panel_multi_detection(scene, fixations, preds, names, path, title):
    """Every saccade in one scene: tagged on-object or background, classified,
    and labelled — with the ground truth beside it."""
    fig, ax = plt.subplots(figsize=(11.5, 10.2))
    ax.imshow(scene.canvas, cmap="gray", interpolation="nearest")
    tile = scene.tile

    # ground truth boxes
    for (r, c), lab in zip(scene.positions, scene.labels):
        ax.add_patch(Rectangle((c - tile / 2, r - tile / 2), tile, tile,
                               fill=False, ec="#3d7fd1", lw=1.4, ls=":"))
        ax.text(c - tile / 2, r - tile / 2 - 3, f"gt:{names[lab]}",
                color="#3d7fd1", fontsize=7.5)

    # scan path
    pts = [(f.col, f.row) for f in fixations]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=8, lw=0.7, color="#ffd24d",
                                     alpha=0.75, shrinkA=6, shrinkB=6))

    n_hit = n_obj = 0
    for i, (f, p) in enumerate(zip(fixations, preds)):
        on_obj = f.true_label >= 0
        if on_obj:
            n_obj += 1
            good = (p == f.true_label)
            n_hit += good
            col = OK if good else BAD
        else:
            col = DIM
        ax.add_patch(Circle((f.col, f.row), tile * 0.5, fill=False, ec=col,
                            lw=1.9 if on_obj else 0.9,
                            alpha=1.0 if on_obj else 0.55))
        ax.text(f.col + tile * 0.52, f.row, f"{i}",
                color=col, fontsize=6.5, va="center")
        if on_obj:
            ax.text(f.col - tile / 2, f.row + tile / 2 + 9,
                    f"{names[p] if p >= 0 else '?'}", color=col, fontsize=8,
                    fontweight="bold")

    ax.set_xlim(0, scene.canvas.shape[1]); ax.set_ylim(scene.canvas.shape[0], 0)
    ax.axis("off")
    acc = n_hit / max(n_obj, 1)
    ax.set_title(
        f"{title}\n{len(fixations)} saccades → {n_obj} landed on an object "
        f"({n_obj / max(len(fixations),1):.0%}) → {n_hit} named correctly ({acc:.0%})\n"
        "dotted blue = ground truth · green = correct · red = wrong · grey = background",
        fontsize=11.5)
    _save(fig, path)
    return dict(saccades=len(fixations), on_object=n_obj,
                on_object_rate=round(n_obj / max(len(fixations), 1), 4),
                named_correct=n_hit, accuracy_given_hit=round(acc, 4))



# --------------------------------------------------------------- panel 2b ---
def panel_scene_spread(loader, brain, Xc, yc, names, path, seeds=(0, 1, 2),
                       n_sac=45):
    """The same detector on three different scenes. Detection rate swings
    enormously with scene layout, so one scene is never the result."""
    trx, trY, tex, teY = loader(n_train=1500, n_test=500)
    fig, axes = plt.subplots(1, len(seeds), figsize=(5.6 * len(seeds), 6.0))
    rows = []
    for ax, sd in zip(np.atleast_1d(axes), seeds):
        scene = build_scene(trx, trY, size=256, n_objects=12, seed=sd)
        fx = SaccadicEye(scene, seed=sd).free_view(n_saccades=n_sac, correct=True)
        X = np.array([_unit(brain.v1.rate_over(f.frames)) for f in fx], np.float32)
        pr = _nearest_prototype(Xc, yc, X, len(names)).astype(int)
        ax.imshow(scene.canvas, cmap="gray")
        tile = scene.tile
        for (r, c) in scene.positions:
            ax.add_patch(Rectangle((c - tile / 2, r - tile / 2), tile, tile,
                                   fill=False, ec="#3d7fd1", lw=1.0, ls=":"))
        hit = obj = 0
        for f, pp in zip(fx, pr):
            on = f.true_label >= 0
            if on:
                obj += 1
                good = pp == f.true_label
                hit += good
                col = OK if good else BAD
            else:
                col = DIM
            ax.add_patch(Circle((f.col, f.row), tile * 0.45, fill=False, ec=col,
                                lw=1.6 if on else 0.7, alpha=1 if on else 0.5))
        rate = obj / max(len(fx), 1)
        rows.append(dict(seed=int(sd), on_object_rate=round(rate, 4),
                         named_correct=int(hit), on_object=int(obj)))
        ax.set_title(f"scene seed {sd}\non-object {rate:.0%} · named "
                     f"{hit}/{max(obj,1)} ({hit/max(obj,1):.0%})", fontsize=11)
        ax.axis("off")
    lo = min(r["on_object_rate"] for r in rows)
    hi = max(r["on_object_rate"] for r in rows)
    fig.suptitle("The same detector, three scenes — on-object rate swings "
                 f"{lo:.0%} to {hi:.0%}. One scene is never the result.",
                 fontsize=13)
    _save(fig, path)
    return rows


# ---------------------------------------------------------------- panel 3 ---
def panel_attention(scene, eye_ior, eye_noior, path):
    """Attention: the saliency map the eye chooses from, and what happens to
    coverage when inhibition-of-return is removed."""
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6))
    sal = eye_ior._compute_saliency()
    axes[0].imshow(scene.canvas, cmap="gray"); axes[0].set_title("scene", fontsize=11)
    axes[1].imshow(sal, cmap="inferno")
    axes[1].set_title("peripheral saliency\n(local contrast — what attention is drawn to)",
                      fontsize=11)
    axes[2].imshow(scene.canvas, cmap="gray", alpha=0.55)
    for name, eye, col, mk in (("with IOR", eye_ior, "#2e9e4f", "o"),
                               ("no IOR", eye_noior, "#d14545", "x")):
        pts = np.array(eye.visited) if eye.visited else np.zeros((0, 2))
        if len(pts):
            axes[2].plot(pts[:, 1], pts[:, 0], mk, ms=5, color=col, alpha=0.8,
                         label=f"{name} — {len(set(map(tuple, pts)))} distinct spots")
    axes[2].legend(fontsize=9, loc="upper right")
    axes[2].set_title("inhibition of return spreads the search",
                      fontsize=11)
    for a in axes:
        a.axis("off")
    fig.suptitle("Attention in detection: saliency picks the target, "
                 "inhibition-of-return stops it staring", fontsize=13)
    _save(fig, path)


# ---------------------------------------------------------------- panel 4 ---
def panel_pursuit(trx, trY, names, path, seed=0, n_steps=8):
    """Pursuit: an object moves, the eye follows it, and the tag it carries
    persists across frames — that persistence is what makes it one object
    rather than a new detection each frame."""
    moving = MovingScene(trx, trY, size=200, n_objects=5, seed=seed)
    brain = StreamingBrain(seed=seed)
    Xc = np.array([_unit(brain.v1.rate(im)) for im in trx[:500]], np.float32)
    yc = trY[:500]

    frames, tracks = [], []
    tgt = None
    for t in range(n_steps):
        canvas = moving.render() if t == 0 else moving.step()
        sc = moving.scene_now()
        if tgt is None:
            tgt = sc.positions[0]
        near = moving.nearest(*tgt)
        if near is not None:
            tgt = (int(near.row), int(near.col))
        r, c = tgt
        h, w = canvas.shape
        r = int(np.clip(r, 14, h - 15)); c = int(np.clip(c, 14, w - 15))
        crop = canvas[r - 14:r + 14, c - 14:c + 14]
        pred = int(_nearest_prototype(Xc, yc, _unit(brain.v1.rate(crop))[None], 10)[0])
        frames.append((canvas.copy(), r, c, crop.copy(), pred,
                       sc.label_at(r, c)))
        tracks.append((r, c))

    fig = plt.figure(figsize=(2.15 * n_steps, 5.6))
    gs = fig.add_gridspec(2, n_steps, hspace=0.28, wspace=0.12,
                          height_ratios=[2.4, 1])
    for t, (canvas, r, c, crop, pred, truth) in enumerate(frames):
        ax = fig.add_subplot(gs[0, t]); ax.imshow(canvas, cmap="gray"); ax.axis("off")
        good = (pred == truth) and truth >= 0
        col = OK if good else (BAD if truth >= 0 else DIM)
        ax.add_patch(Circle((c, r), 15, fill=False, ec=col, lw=2))
        if t:
            pr, pc = tracks[t - 1]
            ax.add_patch(FancyArrowPatch((pc, pr), (c, r), arrowstyle="-|>",
                                         mutation_scale=9, lw=1.1, color="#ffd24d"))
        ax.set_title(f"t={t}", fontsize=9)
        ax = fig.add_subplot(gs[1, t]); ax.imshow(crop, cmap="gray"); ax.axis("off")
        ax.set_xlabel("")
        ax.set_title(f"→ {names[pred] if pred>=0 else '?'}"
                     + ("" if truth < 0 else f" (gt {names[truth]})"),
                     fontsize=8, color=col)
    kept = sum(1 for _, _, _, _, p, tr in frames if p == frames[0][4])
    fig.suptitle("Smooth pursuit — the eye follows one moving object; "
                 f"the tag it carries stays the same on {kept}/{n_steps} frames "
                 "(bottom row = what the fovea actually received)", fontsize=12.5)
    _save(fig, path)
    return dict(steps=n_steps, tag_stable_frames=int(kept),
                tag_stability=round(kept / n_steps, 4),
                correct_frames=int(sum(1 for f in frames if f[4] == f[5] and f[5] >= 0)))


# ---------------------------------------------------------------- panel 5 ---
def panel_confusion(y_true, y_pred, names, path, title):
    labs = sorted(set(int(v) for v in y_true))
    M = np.zeros((len(labs), len(labs)))
    for t, p in zip(y_true, y_pred):
        if int(p) in labs:
            M[labs.index(int(t)), labs.index(int(p))] += 1
    Mn = M / np.maximum(M.sum(1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    im = axes[0].imshow(Mn, cmap="viridis", vmin=0, vmax=1)
    axes[0].set_xticks(range(len(labs))); axes[0].set_yticks(range(len(labs)))
    axes[0].set_xticklabels([names[l] for l in labs], rotation=45, ha="right", fontsize=8)
    axes[0].set_yticklabels([names[l] for l in labs], fontsize=8)
    axes[0].set_xlabel("called"); axes[0].set_ylabel("actually was")
    axes[0].set_title("confusion (row-normalised)", fontsize=11)
    for i in range(len(labs)):
        for j in range(len(labs)):
            if Mn[i, j] > 0.02:
                axes[0].text(j, i, f"{Mn[i,j]:.2f}", ha="center", va="center",
                             fontsize=6.5, color="w" if Mn[i, j] < 0.6 else "k")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    rec = np.diag(Mn)
    order = np.argsort(rec)
    axes[1].barh([names[labs[i]] for i in order], rec[order],
                 color=[BAD if rec[i] < 0.5 else OK for i in order])
    axes[1].axvline(1 / len(labs), color="k", ls="--", lw=1,
                    label=f"chance {1/len(labs):.0%}")
    axes[1].set_xlim(0, 1); axes[1].legend(fontsize=9)
    axes[1].set_title("per-class recall — what it can and cannot see", fontsize=11)
    for i, v in enumerate(rec[order]):
        axes[1].text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=8)
    fig.suptitle(title, fontsize=13)
    _save(fig, path)
    return {names[labs[i]]: round(float(rec[i]), 4) for i in range(len(labs))}


# --------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist", choices=["mnist", "fashion"])
    ap.add_argument("--outdir", default="reports")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--saccades", type=int, default=45)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    tag = a.dataset

    if a.dataset == "mnist":
        loader, names = nb.load_mnist, [str(i) for i in range(10)]
    else:
        loader = nb.load_fashion_mnist
        names = ["tshirt", "trouser", "pullover", "dress", "coat",
                 "sandal", "shirt", "sneaker", "bag", "boot"]
    trx, trY, tex, teY = loader(n_train=1500, n_test=500)
    brain = StreamingBrain(seed=a.seed)
    res = {"dataset": a.dataset}

    panel_learned(brain, trx, trY, sorted(set(trY.tolist())), names,
                  f"{a.outdir}/{tag}_01_learned_prototypes.jpg")

    # the read-out: centred crops -> prototypes
    Xc = np.array([_unit(brain.v1.rate(im)) for im in trx[:700]], np.float32)
    yc = trY[:700]

    scene = build_scene(trx, trY, size=256, n_objects=12, seed=a.seed)
    eye = SaccadicEye(scene, seed=a.seed)
    fixations = eye.free_view(n_saccades=a.saccades, correct=True)
    Xs = np.array([_unit(brain.v1.rate_over(f.frames)) for f in fixations], np.float32)
    preds = _nearest_prototype(Xc, yc, Xs, len(names)).astype(int)
    res["multi_detection"] = panel_multi_detection(
        scene, fixations, preds, names,
        f"{a.outdir}/{tag}_02_multi_detection.jpg",
        f"Multi-object detection in one cluttered scene — {a.dataset}")

    res["scene_spread"] = panel_scene_spread(
        loader, brain, Xc, yc, names,
        f"{a.outdir}/{tag}_02b_scene_spread.jpg")

    eye2 = SaccadicEye(scene, seed=a.seed, ior_radius=0)
    eye2.free_view(n_saccades=a.saccades, correct=True)
    panel_attention(scene, eye, eye2, f"{a.outdir}/{tag}_03_attention_ior.jpg")

    try:
        res["pursuit"] = panel_pursuit(trx, trY, names,
                                       f"{a.outdir}/{tag}_04_pursuit.jpg",
                                       seed=a.seed)
    except Exception as e:
        res["pursuit"] = {"ERROR": f"{type(e).__name__}: {e}"}
        print("pursuit failed:", e, flush=True)

    on = [(f, p) for f, p in zip(fixations, preds) if f.true_label >= 0]
    if on:
        res["per_class_recall_streaming"] = panel_confusion(
            [f.true_label for f, _ in on], [p for _, p in on], names,
            f"{a.outdir}/{tag}_05_confusion.jpg",
            f"What gets detected and what it is called — {a.dataset}, "
            "through the moving eye")

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(x) for x in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return o
    res = _clean(res)
    json.dump(res, open(f"{a.outdir}/{tag}_vision.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
