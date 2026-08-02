"""Multi-step DPC on tracked position: can it imagine forward?

§9.22 measured one-step DPC skill at +0.05 and **rollout at -0.14 to -0.26** --
imagining three steps with no input was worse than assuming a frozen world. The
limiting factor was named there and not worked around: *the model is fit on
one-step transitions and nothing constrains its behaviour beyond one step.*

This attacks exactly that, and changes two things, both stated because the
second one changes what a success would mean.

1. Multi-horizon training, without backpropagation
--------------------------------------------------
The model is trained on horizons 1..K rather than 1 alone. Exact multi-step
optimisation would need credit assignment through the unrolled chain -- BPTT --
which this project does not have and does not want. Instead each horizon gets
its **own local delta-rule update**: the k-step prediction error drives a
correction with the *current* state as the presynaptic term.

    for k in 1..K:   err = r_{t+k} - predict_k(r_t);   A += lr * outer(err, r_t)

That is local, gradient-free, and an **approximation** of multi-step
optimisation rather than the real thing. Saying so matters: if it works, what
worked is horizon-spread training, not a solved credit-assignment problem.

2. The state is the tracked POSITION, not the whole code
--------------------------------------------------------
§9.22 predicted a 32-d PCA latent of the whole frame. Predicting *where the
tagged object will be* is far better posed, has ground truth from VOT boxes, and
is what a world model owes a planner.

**This is an easier task, and success here does not repair §9.22's negative.**
The two are not comparable and no claim is made that they are. What it can show
is whether a local rule can learn forward dynamics *at all* when the target is
well posed.

The control is the same one §9.22 lost to and `world/continuous.py` lost to
before it: **persistence**, "assume nothing moved", plus a constant-velocity
arm, because on smooth motion that is the baseline any dynamics model must beat
to have said anything. Skill is 1 - err_model/err_baseline; 0.000 means no
better than the baseline.

Usage:  python3 benchmarks/dpc_multistep.py out.json [vot_dir] [n_seq]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eye_vot import load_gt, resize                           # noqa: E402

VOT_DIR = sys.argv[2] if len(sys.argv) > 2 else "vot"
N_SEQ = int(sys.argv[3]) if len(sys.argv) > 3 else 40
SIZE = 224
K = 3                      # training horizons 1..K
LAGS = (1, 2, 3)
HIST = 3                   # frames of history in the state
SEEDS = (0, 1, 2, 3, 4)


def trajectory(d, name):
    """The target's (y, x, log-area) per frame, on a constant canvas."""
    p = os.path.join(d, name)
    jpgs = sorted(f for f in os.listdir(p) if f.endswith(".jpg"))
    gtp = os.path.join(p, "groundtruth.txt")
    if not jpgs or not os.path.exists(gtp):
        return None
    gt = load_gt(gtp)
    from PIL import Image
    a = np.asarray(Image.open(os.path.join(p, jpgs[0])))
    sy, sx = SIZE / a.shape[0], SIZE / a.shape[1]
    out = []
    for g in gt[:len(jpgs)]:
        if g is None:
            break
        y, x, h, w = g
        if h < 2 or w < 2:
            break
        out.append([(y + h / 2) * sy, (x + w / 2) * sx,
                    float(np.log(max(h * sy * w * sx, 1e-6)))])
    return np.asarray(out, np.float32) if len(out) >= 12 else None


def states(traj):
    """State = recent **displacements** plus a bias, not absolute coordinates.

    Absolute pixel coordinates (0..224) make the delta rule diverge outright --
    the first version of this overflowed to nan, because an update of
    ``lr * outer(err, s)`` scales with ||s||^2 and ||s|| was ~200. Displacements
    are small, and they are also the right features: what predicts where a thing
    goes next is how it has been moving, not where the origin is.
    """
    S, cur = [], []
    for t in range(HIST - 1, len(traj)):
        d = [traj[t - i] - traj[t - i - 1] for i in range(HIST - 1)]
        S.append(np.concatenate(d + [[traj[t][2]], [1.0]]))
        cur.append(traj[t])
    return np.asarray(S, np.float32), np.asarray(cur, np.float32)


class MultiDPC:
    """One matrix per horizon, each trained by its own local delta rule.

    The update is **normalised LMS** -- divided by ||s||^2 -- which keeps a
    local rule stable whatever the input scale. That is divisive normalisation
    of the update, not a gradient, and it is what inhibition does in a circuit
    that must not ring; the same reasoning `PredictiveStack` uses to derive its
    inference gain from the spectral norm rather than hand-picking it.
    """

    def __init__(self, d, k, lr=0.3, seed=0):
        rng = np.random.default_rng(seed)
        self.A = rng.normal(0, 0.01, (k, 3, d)).astype(np.float32)
        self.k, self.lr = k, lr

    def predict(self, s, h, cur):
        """h steps ahead = where it is now + the predicted displacement."""
        return cur + self.A[h - 1] @ s

    def learn(self, s, cur, targets):
        n = float(s @ s) + 1e-6
        for h, tgt in targets:
            err = tgt - self.predict(s, h, cur)
            self.A[h - 1] += (self.lr / n) * np.outer(err, s)


def skill(model_err, base_err):
    return 1.0 - float(np.mean(model_err)) / (float(np.mean(base_err)) + 1e-9)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_dpc_multistep.json"
    if not os.path.isdir(VOT_DIR):
        print(f"no frame directory at {VOT_DIR!r} -- nothing reported.")
        json.dump({"error": "no frames"}, open(out_path, "w"), indent=1)
        return
    names = sorted(d for d in os.listdir(VOT_DIR)
                   if os.path.isdir(os.path.join(VOT_DIR, d)))[:N_SEQ]
    trajs = {}
    for n in names:
        t = trajectory(VOT_DIR, n)
        if t is not None:
            trajs[n] = t
    if len(trajs) < 8:
        print(f"only {len(trajs)} usable sequences -- nothing reported.")
        json.dump({"error": "too few sequences", "n": len(trajs)},
                  open(out_path, "w"), indent=1)
        return
    keys = list(trajs)
    moved = float(np.mean([np.abs(np.diff(trajs[k][:, :2], axis=0)).mean()
                           for k in keys]))
    print(f"{len(trajs)} sequences; the target moves {moved:.2f}px per frame "
          f"on average", flush=True)
    if moved < 0.2:
        print("  the targets barely move -- persistence is unbeatable and this "
              "measures nothing. REFUSING.")
        json.dump({"error": "targets static", "moved": moved},
                  open(out_path, "w"), indent=1)
        return

    res = {"n_sequences": len(trajs), "horizons": list(LAGS), "hist": HIST,
           "px_per_frame": round(moved, 3), "skill": {}, "skill_sd": {}}
    acc = {f"h{h}": {"dpc": [], "cv": []} for h in LAGS}
    acc["rollout"] = {"dpc": [], "cv": []}

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        order = list(rng.permutation(len(keys)))
        cut = int(0.65 * len(order))
        tr = [keys[i] for i in order[:cut]]
        te = [keys[i] for i in order[cut:]]
        d = (HIST - 1) * 3 + 2      # displacements + log-area + bias
        m = MultiDPC(d, len(LAGS), seed=seed)
        pairs = []
        for k in tr:
            S, P = states(trajs[k])
            for t in range(len(S)):
                tg = [(h, P[t + h][:3]) for h in LAGS if t + h < len(P)]
                if tg:
                    pairs.append((S[t], P[t][:3], tg))
        for _ in range(12):
            for i in rng.permutation(len(pairs)):
                m.learn(*pairs[i])

        err = {f"h{h}": {"dpc": [], "persist": [], "cv": []} for h in LAGS}
        roll = {"dpc": [], "persist": [], "cv": []}
        for k in te:
            S, P = states(trajs[k])
            for t in range(len(S)):
                cur = P[t][:3]
                vel = S[t][:3]                     # the most recent step
                for h in LAGS:
                    if t + h >= len(P):
                        continue
                    tgt = P[t + h][:3]
                    err[f"h{h}"]["dpc"].append(
                        np.linalg.norm(m.predict(S[t], h, cur) - tgt))
                    err[f"h{h}"]["persist"].append(np.linalg.norm(cur - tgt))
                    err[f"h{h}"]["cv"].append(
                        np.linalg.norm(cur + vel * h - tgt))
                # free rollout: feed the model its OWN output, no new input
                if t + max(LAGS) < len(P):
                    s, pos = S[t].copy(), cur.copy()
                    for _ in range(max(LAGS)):
                        nxt = m.predict(s, 1, pos)   # its OWN output, no input
                        step = nxt - pos
                        s = np.concatenate([step, s[:3 * (HIST - 2)],
                                            [nxt[2]], [1.0]])
                        pos = nxt
                    tgt = P[t + max(LAGS)][:3]
                    roll["dpc"].append(np.linalg.norm(pos - tgt))
                    roll["persist"].append(np.linalg.norm(cur - tgt))
                    roll["cv"].append(
                        np.linalg.norm(cur + vel * max(LAGS) - tgt))
        for h in LAGS:
            e = err[f"h{h}"]
            acc[f"h{h}"]["dpc"].append(skill(e["dpc"], e["persist"]))
            acc[f"h{h}"]["cv"].append(skill(e["cv"], e["persist"]))
        acc["rollout"]["dpc"].append(skill(roll["dpc"], roll["persist"]))
        acc["rollout"]["cv"].append(skill(roll["cv"], roll["persist"]))
        print(f"  seed {seed}: h1 {acc['h1']['dpc'][-1]:+.3f}  "
              f"rollout {acc['rollout']['dpc'][-1]:+.3f}", flush=True)

    print(f"\n{'horizon':<12}{'DPC skill':>12}{'sd':>8}"
          f"{'const-velocity':>17}")
    for k in list(f"h{h}" for h in LAGS) + ["rollout"]:
        dm, ds = float(np.mean(acc[k]["dpc"])), float(np.std(acc[k]["dpc"]))
        cm = float(np.mean(acc[k]["cv"]))
        res["skill"][k] = round(dm, 4)
        res["skill_sd"][k] = round(ds, 4)
        res["skill"][k + "_cv"] = round(cm, 4)
        print(f"{k:<12}{dm:>12.4f}{ds:>8.4f}{cm:>17.4f}")

    r = res["skill"]["rollout"]
    res["can_imagine_forward"] = bool(r > 0.05)
    res["beats_constant_velocity"] = bool(r > res["skill"]["rollout_cv"])

    print(f"\n--- can it imagine forward? (§9.22 measured -0.14 here) ---")
    print(f"  rollout skill {r:+.4f} +/- {res['skill_sd']['rollout']:.4f}   "
          f"against persistence")
    print(f"  constant velocity {res['skill']['rollout_cv']:+.4f}   "
          f"<- the baseline that matters on smooth motion")
    if res["can_imagine_forward"]:
        print(f"  Forward imagination WORKS on this target: three steps driven "
              f"by the model's own output\n  beat assuming a frozen world.")
        if not res["beats_constant_velocity"]:
            print(f"  But it does NOT beat constant velocity "
                  f"({res['skill']['rollout_cv']:+.4f}), which is arithmetic "
                  f"rather than a\n  world model. The dynamics it learned are "
                  f"no better than 'keep going the way you were'.")
        else:
            print(f"  And it beats constant velocity, so it has learned "
                  f"something beyond linear extrapolation.")
    else:
        print(f"  Forward imagination still fails. Multi-horizon training did "
              f"not fix what §9.22 named,\n  even on a well-posed target with "
              f"real ground truth.")
    print("\n  NOTE: this target (tracked position) is far better posed than "
          "§9.22's whole-frame\n  latent. The two are not comparable and this "
          "does not repair that negative.")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
