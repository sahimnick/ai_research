"""Audio JPG report: what was learned, what gets detected in a stream, what it is called.

Panels produced (into --outdir):
  01_learned_sounds.jpg    each sound class: waveform, cochleagram, belt code
  02_multi_detection.jpg   one soundscape, every onset found / tagged / labelled
  03_tracking.jpg          sliding-window label trace vs ground truth (audio "pursuit")
  04_threshold.jpg         the attention knob: detection threshold vs precision/recall
  05_confusion.jpg         which sounds it confuses

Usage:  python3 benchmarks/report_audio.py --outdir reports/
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import neurobrain as nb
from neurobrain.sensing.streams import (build_soundscape, ContinuousEar, StreamingBrain,
                                        onset_scores, _unit)
from neurobrain.vision.widev1 import _nearest_prototype
from neurobrain.audition.audio import sound_dataset

DPI = 110
OK, BAD, DIM = "#2e9e4f", "#d14545", "#8899a6"


def _save(fig, path):
    fig.savefig(path, dpi=DPI, format="jpg", bbox_inches="tight",
                pil_kwargs={"quality": 92})
    plt.close(fig)
    print("wrote", path, flush=True)


def panel_learned(brain, sigs, labels, names, path):
    """What the ear learned: for one clip of each class, the waveform it heard,
    the cochleagram it built, and the shift-invariant belt code it reduced to."""
    ncls = len(names)
    fig, axes = plt.subplots(3, ncls, figsize=(2.05 * ncls, 6.6))
    axes = np.atleast_2d(axes)
    for c in range(ncls):
        i = int(np.where(np.asarray(labels) == c)[0][0])
        w = sigs[i]
        coch = brain.ear.coch.forward(w)[0]
        code = brain.belt.code(coch)
        axes[0, c].plot(w, lw=0.4, color="#3d7fd1"); axes[0, c].set_title(names[c], fontsize=9)
        axes[1, c].imshow(coch, aspect="auto", origin="lower", cmap="magma")
        axes[2, c].plot(code, lw=0.5, color="#8a4fd1")
        for r in range(3):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    for r, lab in enumerate(["waveform\n(what arrived)", "cochleagram\n(freq x time)",
                             "belt code\n(shift-invariant)"]):
        axes[r, 0].set_ylabel(lab, fontsize=9)
    fig.suptitle("What the ear learned — one clip per class through the full "
                 "front end", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, path)


def panel_multi_detection(brain, scape, bank_X, bank_y, names, path, ncls):
    """Every event the ear found in one unsegmented soundscape: tagged as a real
    onset or a false alarm, then classified and labelled."""
    wave, sr = scape.wave, scape.sr
    t = np.arange(len(wave)) / sr
    energy, _ = brain.ear.energy_trace(wave)
    found = brain.ear.detect_onsets(wave)
    tol = 0.12 * sr

    fig, axes = plt.subplots(3, 1, figsize=(16.5, 9.2), sharex=True,
                             gridspec_kw={"height_ratios": [1.1, 1.5, 1.0]})
    axes[0].plot(t, wave, lw=0.3, color="#4a5b6a")
    for ev in scape.events:
        axes[0].axvspan(ev.start / sr, ev.end / sr, color="#3d7fd1", alpha=0.17)
    axes[0].set_ylabel("waveform")
    axes[0].set_title("blue bands = real events (never given to the ear)", fontsize=10)

    coch = brain.ear.coch.forward(wave)[0]
    axes[1].imshow(coch, aspect="auto", origin="lower", cmap="magma",
                   extent=[0, len(wave) / sr, 0, coch.shape[0]])
    axes[1].set_ylabel("cochleagram")

    rows, hit, fa = [], 0, 0
    for f in found:
        truth = [e for e in scape.events if abs(e.start - f) <= tol]
        code = brain.belt.code(brain.ear.listen(wave, f))
        pred = int(_nearest_prototype(bank_X, bank_y, code[None], ncls)[0])
        x = f / sr
        if truth:
            hit += 1
            good = pred == truth[0].label
            col = OK if good else BAD
            rows.append(dict(t=round(x, 3), pred=int(pred),
                             truth=int(truth[0].label), correct=bool(good)))
        else:
            fa += 1
            col = DIM
            rows.append(dict(t=round(x, 3), pred=int(pred), truth=None,
                             correct=None))
        for ax in axes[:2]:
            ax.axvline(x, color=col, lw=1.3, alpha=0.9)
        axes[2].axvline(x, color=col, lw=1.3)
        axes[2].text(x, 0.55, names[pred], rotation=90, fontsize=7.5, color=col,
                     ha="center", va="bottom")
        if truth:
            axes[2].text(x, 0.42, names[truth[0].label], rotation=90, fontsize=6.5,
                         color="#3d7fd1", ha="center", va="top")
    axes[2].set_ylim(0, 1); axes[2].set_yticks([])
    axes[2].set_xlabel("seconds")
    axes[2].set_ylabel("called / was")

    s = onset_scores(found, scape.events, sr=sr)
    ok = sum(1 for r in rows if r["correct"])
    fig.suptitle(
        f"Multi-event detection in one unsegmented soundscape — "
        f"{len(found)} onsets found for {len(scape.events)} real events "
        f"(precision {s['precision']:.0%}, recall {s['recall']:.0%}) · "
        f"{ok}/{max(hit,1)} named correctly ({ok/max(hit,1):.0%})\n"
        "green = found and named right · red = found, named wrong · grey = false alarm",
        fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, path)
    return dict(n_found=len(found), n_true=len(scape.events),
                precision=round(s["precision"], 4), recall=round(s["recall"], 4),
                f1=round(s["f1"], 4), true_positives=hit, false_alarms=fa,
                named_correct=ok,
                accuracy_given_detection=round(ok / max(hit, 1), 4), events=rows)


def panel_tracking(brain, scape, bank_X, bank_y, names, path, ncls, hop_s=0.05):
    """The audio analogue of pursuit: slide a window along the stream and read
    the label out continuously, against what was really sounding."""
    wave, sr = scape.wave, scape.sr
    win = int(0.40 * sr)
    hop = int(hop_s * sr)
    starts = list(range(0, max(len(wave) - win, 1), hop))
    preds, truths = [], []
    for s0 in starts:
        code = brain.belt.code(brain.ear.listen(wave, s0))
        preds.append(int(_nearest_prototype(bank_X, bank_y, code[None], ncls)[0]))
        cur = [e.label for e in scape.events if e.start <= s0 + win / 2 <= e.end]
        truths.append(int(cur[0]) if cur else -1)
    tt = np.array(starts) / sr
    preds, truths = np.array(preds), np.array(truths)
    live = truths >= 0
    acc = float((preds[live] == truths[live]).mean()) if live.any() else 0.0

    fig, ax = plt.subplots(figsize=(16.5, 5.2))
    ax.step(tt, preds, where="mid", lw=1.4, color="#8a4fd1", label="called")
    m = truths.astype(float); m[~live] = np.nan
    ax.step(tt, m, where="mid", lw=3.0, color="#3d7fd1", alpha=0.4,
            label="actually sounding")
    ok = live & (preds == truths)
    ax.plot(tt[ok], preds[ok], "o", ms=4, color=OK, label="agree")
    bad = live & (preds != truths)
    ax.plot(tt[bad], preds[bad], "x", ms=5, color=BAD, label="disagree")
    ax.set_yticks(range(ncls)); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("seconds"); ax.legend(fontsize=9, ncol=4, loc="upper right")
    ax.set_title(f"Continuous tracking — a 400 ms window slid every {hop_s*1000:.0f} ms, "
                 f"labelled without being told where anything starts. "
                 f"While something is sounding it agrees {acc:.0%} of the time.",
                 fontsize=12)
    fig.tight_layout()
    _save(fig, path)
    return dict(windows=len(starts), windows_with_sound=int(live.sum()),
                tracking_accuracy=round(acc, 4))


def panel_threshold(scape, path, thresholds=(1.0, 1.5, 2.0, 2.5, 3.0, 3.5)):
    """The one knob that decides what the ear even notices."""
    rows = []
    for th in thresholds:
        ear = ContinuousEar(threshold=th)
        s = onset_scores(ear.detect_onsets(scape.wave), scape.events, sr=scape.sr)
        rows.append((th, s["precision"], s["recall"], s["f1"]))
    th, pr, rc, f1 = map(np.array, zip(*rows))
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.plot(th, pr, "o-", label="precision", color="#3d7fd1")
    ax.plot(th, rc, "s-", label="recall", color="#d18a45")
    ax.plot(th, f1, "^-", label="F1", color="#2e9e4f", lw=2.4)
    best = th[int(np.argmax(f1))]
    ax.axvline(2.5, color=BAD, ls="--", lw=1.4, label="shipped default (2.5)")
    ax.axvline(best, color=OK, ls=":", lw=1.8, label=f"best F1 here ({best})")
    ax.set_xlabel("ContinuousEar.threshold"); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    ax.set_title("Detection threshold — what the ear pays attention to.\n"
                 f"The default trades {(1-rc[th==2.5][0]):.0%} of all events "
                 "for precision it does not need.", fontsize=12)
    fig.tight_layout()
    _save(fig, path)
    return [dict(threshold=float(a), precision=round(float(b), 4),
                 recall=round(float(c), 4), f1=round(float(d), 4))
            for a, b, c, d in rows]


def panel_confusion(events, names, path):
    real = [(e["truth"], e["pred"]) for e in events if e["truth"] is not None]
    if not real:
        return {}
    labs = sorted(set(t for t, _ in real))
    M = np.zeros((len(labs), len(labs)))
    for t, p in real:
        if p in labs:
            M[labs.index(t), labs.index(p)] += 1
    Mn = M / np.maximum(M.sum(1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    im = axes[0].imshow(Mn, cmap="viridis", vmin=0, vmax=1)
    axes[0].set_xticks(range(len(labs))); axes[0].set_yticks(range(len(labs)))
    axes[0].set_xticklabels([names[l] for l in labs], rotation=45, ha="right", fontsize=8)
    axes[0].set_yticklabels([names[l] for l in labs], fontsize=8)
    axes[0].set_xlabel("called"); axes[0].set_ylabel("actually was")
    axes[0].set_title("sound confusion (row-normalised)", fontsize=11)
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    rec = np.diag(Mn); order = np.argsort(rec)
    axes[1].barh([names[labs[i]] for i in order], rec[order],
                 color=[BAD if rec[i] < 0.5 else OK for i in order])
    axes[1].axvline(1 / len(labs), color="k", ls="--", lw=1,
                    label=f"chance {1/len(labs):.0%}")
    axes[1].set_xlim(0, 1); axes[1].legend(fontsize=9)
    axes[1].set_title("per-sound recall, on events the ear found itself", fontsize=11)
    fig.suptitle("What gets heard and what it is called", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, path)
    return {names[labs[i]]: round(float(rec[i]), 4) for i in range(len(labs))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="reports")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--events", type=int, default=24)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    brain = StreamingBrain(seed=a.seed)
    sigs, labels, names = sound_dataset(n_per_class=8, seed=a.seed + 5)
    labels = np.asarray(labels)
    ncls = len(names)
    X = np.array([brain.belt.code(brain.ear.coch.forward(s)[0]) for s in sigs],
                 np.float32)
    perm = np.random.default_rng(a.seed + 11).permutation(len(X))
    X, ylab = X[perm], labels[perm]
    half = len(X) // 2
    bank_X, bank_y = X[:half], ylab[:half]

    scape = build_soundscape(n_events=a.events, seed=a.seed)
    res = {}
    panel_learned(brain, sigs, labels, names, f"{a.outdir}/audio_01_learned_sounds.jpg")
    res["multi_detection"] = panel_multi_detection(
        brain, scape, bank_X, bank_y, names,
        f"{a.outdir}/audio_02_multi_detection.jpg", ncls)
    res["tracking"] = panel_tracking(
        brain, scape, bank_X, bank_y, names,
        f"{a.outdir}/audio_03_tracking.jpg", ncls)
    res["threshold_sweep"] = panel_threshold(
        scape, f"{a.outdir}/audio_04_threshold.jpg")
    res["per_sound_recall"] = panel_confusion(
        res["multi_detection"]["events"], names,
        f"{a.outdir}/audio_05_confusion.jpg")
    res["multi_detection"].pop("events", None)

    json.dump(res, open(f"{a.outdir}/audio.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
