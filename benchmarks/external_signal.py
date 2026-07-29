"""Does replay need an EXTERNAL teaching signal to improve perception?

The acceptance gate failed criterion 1: replaying a lived day into
`vision.cortex` never improved detection at any consolidation rate (0/5 seeds
across two orders of magnitude). The proposed reason was structural rather than
a tuning failure -- **replay into perception is self-training on its own
beliefs**, because the labels handed to `vision.cortex` were produced by
`vision.cortex`, so no information enters.

That explanation makes a sharp prediction: supply the label from somewhere the
visual system did not produce, and the same replay should start to help.

Five arms replay the SAME lived day into perception. They differ only in where
the consolidation label comes from:

  no_dream    watch, do not dream                       (control)
  self        the label the mind believed               (the failing baseline)
  teacher     ground truth                              (the ceiling)
  pred_error  believed label, but only episodes the world model failed to
              predict are replayed                     (selection, not new info)
  second_mod  the label a co-occurring SOUND is classified as, by an auditory
              bank trained separately -- a genuinely independent sense with its
              own error rate                           (the biological answer)

Reading the result:

* If `teacher` improves detection, "replay does not help perception" becomes
  **"replay needs an external teaching signal"** -- a far stronger claim.
* If `second_mod` also improves it, the signal need not be a supervisor: another
  sense is enough, which is the version that matters for a brain.
* If `pred_error` does not, selection alone is not information, as predicted.
* If even `teacher` fails, the structural story is wrong and the fault is in
  the consolidation mechanism itself.

Usage:  python3 benchmarks/external_signal.py out_external.json
"""
import json, sys
import numpy as np

import neurobrain as nb
from neurobrain.sensing.streams import build_scene, SaccadicEye, StreamingBrain, _unit
from neurobrain.audition.audio import sound_dataset
from neurobrain.vision.widev1 import _nearest_prototype

SEEDS = (0, 1, 2, 3, 4)
N_PALLIUM, N_SCENES, N_SAC = 300, 3, 60
# The rate sweep showed detection does not move at all below an effective
# consolidation rate of ~0.1, so at the default this experiment would compare
# label sources inside a dead zone and report five identical zeros. 5.0 puts it
# where self-labelling measurably HURTS (-0.043), which is the regime in which
# a better label source has something to prove.
PERCEPTION_LR = 5.0
ARMS = ("no_dream", "self", "teacher", "pred_error", "second_mod")


def detection(mind, trx, trY, seed):
    scene = build_scene(trx, trY, size=256, n_objects=12, seed=seed + 500)
    fix = SaccadicEye(scene, seed=seed + 500).free_view(n_saccades=45, correct=True)
    on = [f for f in fix if f.true_label >= 0]
    if not on:
        return 0.0
    ok = sum(int(mind.perceive(np.asarray(np.mean(f.frames, 0), np.float32),
                               remember=False) == str(int(f.true_label)))
             for f in on)
    return ok / len(on)


class Ear:
    """A second sense with TEN classes -- one tone per digit -- and its own
    error rate. `sound_dataset` only has 8 classes, so a 10-tone bank is built
    here: each digit is a distinct frequency, heard with noise, and named by
    nearest-prototype over belt codes. Nothing visual enters; the ear's answer
    is wrong at its own rate, which is the point."""

    FREQS = [220, 277, 330, 392, 466, 554, 659, 784, 880, 988]

    def __init__(self, seed=0, n_per_class=10, noise=0.35):
        self.b = StreamingBrain(seed=seed)
        rng = np.random.default_rng(seed)
        self.n = 10
        sigs, labs = [], []
        for d, f in enumerate(self.FREQS):
            for _ in range(n_per_class):
                w = nb.tone(float(f) * (1.0 + 0.03 * rng.standard_normal()))
                w = w + noise * rng.standard_normal(len(w)).astype(np.float32)
                sigs.append(w.astype(np.float32)); labs.append(d)
        self.sigs, self.labels = sigs, np.asarray(labs)
        X = np.array([self.b.belt.code(self.b.ear.coch.forward(s)[0])
                      for s in sigs], np.float32)
        perm = rng.permutation(len(X))
        X, y = X[perm], self.labels[perm]
        half = len(X) // 2
        self.bank_X, self.bank_y = X[:half], y[:half]
        self.rng = rng

    def hears(self, true_digit):
        """A tone for this digit sounds; what does the ear ALONE call it?"""
        d = int(true_digit) % self.n
        w = nb.tone(float(self.FREQS[d]) * (1.0 + 0.03 * self.rng.standard_normal()))
        w = (w + 0.35 * self.rng.standard_normal(len(w))).astype(np.float32)
        code = self.b.belt.code(self.b.ear.coch.forward(w)[0])
        return int(_nearest_prototype(self.bank_X, self.bank_y, code[None], self.n)[0])


def run_seed(sd, trx, trY, ear):
    out = {}
    audio_acc = []
    for arm in ARMS:
        mind = nb.build_unified_mind(n_pallium=N_PALLIUM)
        for s in range(N_SCENES):
            scene = build_scene(trx, trY, size=256, n_objects=12, seed=sd * 10 + s)
            n0 = len(mind.episodes)
            fix = mind.watch(scene, n_saccades=N_SAC, seed=sd * 10 + s)
            # attach what was really there, and what a second sense heard
            for ep, f in zip(mind.episodes.episodes[n0:], fix):
                ep.truth = str(int(f.true_label)) if f.true_label >= 0 else None
                if arm == "second_mod" and f.true_label >= 0:
                    h = ear.hears(f.true_label)
                    ep.heard = None if h is None else str(h)
                    if arm == "second_mod":
                        audio_acc.append(int(ep.heard == ep.truth))
                else:
                    ep.heard = None

        before = detection(mind, trx, trY, sd)
        if arm == "self":
            mind.dream(cycles=3, replays_per_cycle=300,
                       perception_lr=PERCEPTION_LR, relabel=lambda ep: ep.label)
        elif arm == "teacher":
            mind.dream(cycles=3, replays_per_cycle=300,
                       perception_lr=PERCEPTION_LR, relabel=lambda ep: getattr(ep, "truth", None))
        elif arm == "pred_error":
            mind.dream(cycles=3, replays_per_cycle=300,
                       perception_lr=PERCEPTION_LR, relabel=lambda ep: ep.label if ep.surprise >= 0.9 else None)
        elif arm == "second_mod":
            mind.dream(cycles=3, replays_per_cycle=300,
                       perception_lr=PERCEPTION_LR, relabel=lambda ep: getattr(ep, "heard", None))
        out[arm] = dict(before=before, after=detection(mind, trx, trY, sd))
    out["_audio_label_accuracy"] = (float(np.mean(audio_acc))
                                    if audio_acc else None)
    return out


def main():
    trx, trY, tex, teY = nb.load_mnist(n_train=4000, n_test=300)
    ear = Ear(seed=0)
    per_seed = []
    for sd in SEEDS:
        per_seed.append(run_seed(sd, trx, trY, ear))
        r = per_seed[-1]
        print(f"seed {sd}: " + "  ".join(
            f"{a}={r[a]['after']:.3f}" for a in ARMS), flush=True)

    aa = [s["_audio_label_accuracy"] for s in per_seed
          if s["_audio_label_accuracy"] is not None]
    out = {"per_seed": per_seed,
           "audio_label_accuracy": round(float(np.mean(aa)), 4) if aa else None,
           "vs_no_dream": {}}
    print(f"\nthe second modality's own label accuracy: "
          f"{out['audio_label_accuracy']:.1%}  (an independent, noisy signal)")

    print(f"\n{'arm':<12}{'detection after':>17}{'vs no_dream':>14}{'wins':>7}{'d':>8}   verdict")
    for arm in ARMS[1:]:
        x = np.array([s[arm]["after"] for s in per_seed])
        y = np.array([s["no_dream"]["after"] for s in per_seed])
        d = x - y
        sd_ = float(d.std(ddof=1))
        cd = float(d.mean() / (sd_ + 1e-12))
        rec = dict(after=round(float(x.mean()), 4), mean=round(float(d.mean()), 4),
                   sd=round(sd_, 4), wins=int((d > 0).sum()), n=len(d),
                   cohens_d=round(cd, 3))
        out["vs_no_dream"][arm] = rec
        verdict = ("IMPROVES" if cd >= 0.8 and rec["wins"] >= 4
                   else "HURTS" if cd <= -0.8 else "no effect")
        print(f"{arm:<12}{rec['after']:>17.4f}{rec['mean']:>+14.4f}"
              f"{rec['wins']:>4}/{rec['n']}{rec['cohens_d']:>8.2f}   {verdict}")

    t = out["vs_no_dream"]["teacher"]
    m = out["vs_no_dream"]["second_mod"]
    print("\n=== what this means ===")
    if t["cohens_d"] >= 0.8 and t["wins"] >= 4:
        print("  teacher IMPROVES detection -> the failure was the SIGNAL, not the")
        print("  mechanism. 'replay does not help perception' becomes")
        print("  'replay needs an external teaching signal'.")
        if m["cohens_d"] >= 0.8 and m["wins"] >= 4:
            print("  and a second modality is enough -- no supervisor required.")
        else:
            print("  but a second modality is NOT enough at its current accuracy.")
    else:
        print("  teacher does NOT improve detection -> the structural story is")
        print("  wrong. The fault is in the consolidation mechanism itself, not")
        print("  in where the label comes from.")

    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1
                        else "out_external.json", "w"), indent=1)


if __name__ == "__main__":
    main()
