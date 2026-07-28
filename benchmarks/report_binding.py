"""Binding JPG report: does hearing a sound actually recall what was seen?

Panels produced (into --outdir):
  binding_01_pairs.jpg      the audio-visual pairs that were bound
  binding_02_ablation.jpg   the test that matters: vision real vs noise vs zeros
  binding_03_concept_cells.jpg  what each concept cell learned in both senses
  binding_04_recall.jpg     hear a sound -> the visual code it recalls

The ablation panel is the whole argument. If replacing the visual code with
noise does not move the answer, the binding is not cross-modal.

Usage:  python3 benchmarks/report_binding.py --outdir reports/
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import neurobrain as nb
from neurobrain.sensing.streams import StreamingBrain, _unit
from neurobrain.audition.audio import sound_dataset

DPI = 110
OK, BAD, DIM = "#2e9e4f", "#d14545", "#8899a6"


def _save(fig, path):
    fig.savefig(path, dpi=DPI, format="jpg", bbox_inches="tight",
                pil_kwargs={"quality": 92})
    plt.close(fig)
    print("wrote", path, flush=True)


class Bench:
    def __init__(self, seed=0, n_concept=32, n_per_class=4):
        self.seed = seed
        self.n_concept = n_concept
        self.n_per_class = n_per_class
        self.trx, self.trY, _, _ = nb.load_mnist(n_train=1200, n_test=400)
        self.sigs, labels, self.names = sound_dataset(n_per_class=8, seed=3)
        self.labels = np.asarray(labels)
        self.ncls = len(self.names)

    def vis(self, brain, c, k=0):
        idx = np.where(self.trY == c % 10)[0]
        return _unit(brain.v1.rate(self.trx[idx[k % len(idx)]]))

    def img(self, c, k=0):
        idx = np.where(self.trY == c % 10)[0]
        return self.trx[idx[k % len(idx)]]

    def aud(self, brain, i):
        return brain.belt.code(brain.ear.coch.forward(self.sigs[i])[0])

    def split(self):
        tr, te = [], []
        for c in range(self.ncls):
            idx = np.where(self.labels == c)[0]
            tr += list(idx[:self.n_per_class]); te += list(idx[self.n_per_class:8])
        return tr, te

    def fit(self, vis_mode="real", shuffled=False):
        rng = np.random.default_rng(self.seed)
        brain = StreamingBrain(seed=self.seed, n_concept=self.n_concept)
        tr, te = self.split()
        pairs = []
        for k, i in enumerate(tr):
            c = int(self.labels[i])
            v = self.vis(brain, c, k)
            if vis_mode == "noise":
                v = _unit(rng.standard_normal(v.shape).astype(np.float32))
            elif vis_mode == "zeros":
                v = np.zeros_like(v)
            pairs.append((v, self.aud(brain, i), c))
        tgt = [p[2] for p in pairs]
        if shuffled:
            tgt = list(rng.permutation(tgt))
        brain.calibrate(np.array([p[0] for p in pairs], np.float32),
                        np.array([p[1] for p in pairs], np.float32))
        for (v, a, _), lab in zip(pairs, tgt):
            brain.bind(v, a, int(lab))
        return brain, te

    def label_acc(self, brain, te, via):
        ok = n = 0
        for i in te:
            got = brain.recall_visual_from_sound(self.aud(brain, i), via=via)
            if got is not None:
                n += 1; ok += int(got == int(self.labels[i]))
        return ok / max(n, 1)


def panel_pairs(b, path):
    brain, _ = b.fit()
    fig, axes = plt.subplots(3, b.ncls, figsize=(2.05 * b.ncls, 6.2))
    axes = np.atleast_2d(axes)
    for c in range(b.ncls):
        i = int(np.where(b.labels == c)[0][0])
        axes[0, c].imshow(b.img(c), cmap="gray")
        axes[0, c].set_title(f"{b.names[c]}", fontsize=9)
        coch = brain.ear.coch.forward(b.sigs[i])[0]
        axes[1, c].imshow(coch, aspect="auto", origin="lower", cmap="magma")
        axes[2, c].plot(b.aud(brain, i), lw=0.5, color="#8a4fd1")
        for r in range(3):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    for r, lab in enumerate(["seen\n(digit)", "heard\n(cochleagram)",
                             "sound code\nbound to it"]):
        axes[r, 0].set_ylabel(lab, fontsize=9)
    fig.suptitle("The pairs that were bound — each sound class co-occurred with "
                 "one digit class", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, path)


def panel_ablation(b, path):
    """The test that matters."""
    modes = ("real", "noise", "zeros")
    res = {}
    for via in ("nearest", "assoc"):
        res[via] = {}
        for m in modes:
            brain, te = b.fit(vis_mode=m)
            res[via][m] = b.label_acc(brain, te, via)
        brain, te = b.fit(shuffled=True)
        res[via]["shuffled_labels"] = b.label_acc(brain, te, via)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), sharey=True)
    for ax, via, title in zip(axes, ("nearest", "assoc"),
                              ("BEFORE — stored-label nearest neighbour",
                               "AFTER — through the concept cells")):
        keys = list(modes) + ["shuffled_labels"]
        vals = [res[via][k] for k in keys]
        cols = ["#3d7fd1", BAD, "#d18a45", DIM]
        ax.bar(range(len(keys)), vals, color=cols)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(["vision\nreal", "vision\nNOISE", "vision\nZEROS",
                            "labels\nshuffled"], fontsize=9)
        ax.axhline(1 / b.ncls, color="k", ls="--", lw=1)
        ax.text(len(keys) - 0.4, 1 / b.ncls + 0.015, f"chance {1/b.ncls:.0%}",
                fontsize=8, ha="right")
        spread = max(res[via][m] for m in modes) - min(res[via][m] for m in modes)
        ax.set_title(f"{title}\nvision ablation spread = {spread:.0%}"
                     + ("  →  vision is IGNORED" if spread < 0.05
                        else "  →  vision participates"),
                     fontsize=11,
                     color=BAD if spread < 0.05 else OK)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.015, f"{v:.0%}", ha="center", fontsize=10,
                    fontweight="bold")
        ax.set_ylim(0, 1.12); ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("held-out recall accuracy")
    fig.suptitle("Does the binding actually read vision?  Replace every visual "
                 "code with noise and with zeros, and see if the answer moves.",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, path)
    return {k: {m: round(float(v), 4) for m, v in d.items()} for k, d in res.items()}


def panel_concept_cells(b, path, max_cells=12):
    """What each concept cell holds in BOTH senses -- this is the thing that
    makes recall cross-modal."""
    brain, _ = b.fit()
    cells = sorted(brain._cell_votes.items(),
                   key=lambda kv: -sum(kv[1].values()))[:max_cells]
    if not cells:
        return {}
    n = len(cells)
    fig, axes = plt.subplots(2, n, figsize=(1.6 * n, 4.6))
    axes = np.atleast_2d(axes)
    for j, (cell, votes) in enumerate(cells):
        name = max(votes.items(), key=lambda kv: kv[1])[0]
        pure = len(votes) == 1
        axes[0, j].plot(brain.assoc.Wv[cell], lw=0.5, color="#3d7fd1")
        axes[1, j].plot(brain.assoc.Wa[cell], lw=0.5, color="#8a4fd1")
        axes[0, j].set_title(f"cell {cell}\n→ {b.names[name]}", fontsize=8,
                             color=OK if pure else "#d18a45")
        for r in range(2):
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
    axes[0, 0].set_ylabel("Wv\n(vision it\nexpects)", fontsize=8)
    axes[1, 0].set_ylabel("Wa\n(sound it\nexpects)", fontsize=8)
    npure = sum(1 for _, v in brain._cell_votes.items() if len(v) == 1)
    fig.suptitle(f"Concept cells after binding — {len(brain._cell_votes)} of "
                 f"{brain.assoc.n_concept} cells were ever used, {npure} pure "
                 "(one class only).\nEach cell carries BOTH senses: hearing "
                 "wakes it through Wa, and Wv is the vision it expects.",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    _save(fig, path)
    return dict(cells_used=len(brain._cell_votes), cells_total=brain.assoc.n_concept,
                pure_cells=int(npure))


def panel_recall(b, path):
    """Hear a held-out clip, recall a VISUAL code, and see which digit it
    matches. No label is read out anywhere in this path."""
    brain, te = b.fit()
    proto = {c: _unit(np.mean([b.vis(brain, c, k) for k in range(6)], 0))
             for c in range(b.ncls)}
    rows, ok, n = [], 0, 0
    for i in te:
        vc = brain.recall_visual_code_from_sound(b.aud(brain, i))
        if vc is None:
            continue
        sims = {c: float(_unit(vc) @ proto[c]) for c in proto}
        best = max(sims, key=sims.get)
        truth = int(b.labels[i])
        n += 1; ok += int(best == truth)
        rows.append((truth, best, sims))
    acc = ok / max(n, 1)

    show = rows[:b.ncls]
    fig, axes = plt.subplots(2, len(show), figsize=(1.85 * len(show), 5.0))
    axes = np.atleast_2d(axes)
    for j, (truth, best, sims) in enumerate(show):
        good = truth == best
        axes[0, j].imshow(b.img(truth), cmap="gray")
        axes[0, j].set_title(f"heard {b.names[truth]}", fontsize=8, color="#3d7fd1")
        axes[1, j].imshow(b.img(best), cmap="gray")
        axes[1, j].set_title(f"recalled {b.names[best]}", fontsize=8,
                             color=OK if good else BAD)
        for r in range(2):
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
        for sp in axes[1, j].spines.values():
            sp.set_color(OK if good else BAD); sp.set_linewidth(2.5)
    axes[0, 0].set_ylabel("what was\nsounding", fontsize=9)
    axes[1, 0].set_ylabel("what the\nsound recalled", fontsize=9)
    fig.suptitle("Cross-modal recall — a held-out sound returns a VISUAL CODE, "
                 f"matched against digit prototypes.\n{ok}/{n} correct "
                 f"({acc:.0%}) against {1/b.ncls:.0%} chance. No label is read "
                 "out anywhere in this path.", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    _save(fig, path)
    return dict(n=n, correct=ok, code_recall_accuracy=round(acc, 4),
                chance=round(1 / b.ncls, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="reports")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-concept", type=int, default=32)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    b = Bench(seed=a.seed, n_concept=a.n_concept)
    res = {}
    panel_pairs(b, f"{a.outdir}/binding_01_pairs.jpg")
    res["ablation"] = panel_ablation(b, f"{a.outdir}/binding_02_ablation.jpg")
    res["concept_cells"] = panel_concept_cells(
        b, f"{a.outdir}/binding_03_concept_cells.jpg")
    res["code_recall"] = panel_recall(b, f"{a.outdir}/binding_04_recall.jpg")
    json.dump(res, open(f"{a.outdir}/binding.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
