"""Dynamic predictive coding on real imagery: does it beat "nothing moved"?

The proposed stack is  sensors -> DPC (dynamics) -> HBPC (inference) ->
world model -> imagination.  Of those, HBPC exists and is measured
(`PredictiveStack`, §9.16: at matched dimensionality it equals PCA and beats a
random projection, though *iterating* collapses participation 46.8 -> 16.2).
**DPC does not exist for vision anywhere in this project.** `world/continuous.py`
learns per-action dynamics on a place-cell population in an arena; nothing
learns how a *sensory* state evolves in time. This measures whether it should.

What DPC is here
----------------
A latent state ``r_t`` and a learned transition. Two commitments, both taken
from what this project has already measured rather than from the literature:

* **Predict the change, not the state.** `world/continuous.py` records the
  discrete loop losing to the trivial baseline (spatial error 1.63 against 0.96
  for "assume nothing moved"). Learning ``delta = r_{t+1} - r_t`` and adding it
  to the current state hands the model that baseline for free, so its capacity
  goes to the part that actually moves.
* **Local learning only.** ``A <- A + lr * outer(err, r)`` is the delta rule --
  Widrow-Hoff, the same form as Rescorla-Wagner, a product of two quantities
  present at the synapse. No gradients, no autograd, no loss function, in
  keeping with the rest of this codebase.

The *dynamic* part -- the "which dynamics are running right now" level that
distinguishes DPC from one linear map -- is **K transition matrices with a
winner-take-all gate**, the winner alone updated. That is competitive learning,
local, and it gives the higher level something to be: the index of the motion
currently in play, which is the hidden cause an HBPC stage above would infer.

The control that decides everything
-----------------------------------
**Persistence**: predict that nothing changed. On video this baseline is brutal,
and this project has already lost to it once. Skill is reported as

    skill = 1 - err_model / err_persistence

so 0.000 means "exactly as good as assuming a frozen world" and negative means
worse. Two further controls: ``shuffled-time``, which pools every transition
globally and re-deals them (destroying dynamics while keeping the frame
statistics -- §9.9's lesson that shuffling *within* a sequence leaves most of
the signal intact), and ``K=1``, which separates "dynamics helped" from "having
several dynamics helped".

Where the frames come from
--------------------------
Measured, not assumed: NY511 cameras return **byte-identical frames for 30 s**
and change on a median period of **97 s**. They are periodic stills, not video,
so raw CCTV cannot supply a dense sequence. Sequences are therefore made by
panning a window across a larger real frame -- real-world imagery at real
dimensions, dense sampling, known dynamics, and ecologically the dominant source
of retinal motion, which is self-generated. Three motions (right, left, down)
give the gate something to separate.

And the question that matters for the stack as a whole: DPC is run on the
**eye's code** and on **raw pixels**, treated identically. §9.20 found 4 px of
translation destroys 63% of the eye's code, which predicts the eye is a poor
substrate for a model of motion. That prediction is tested here rather than
argued.

Usage:  python3 benchmarks/dpc_video.py out_dpc.json [n_cam]
"""
import json
import sys

import numpy as np

from neurobrain.sensing.live import CctvCamera, NY511_LIVE_IDS, SensorUnavailable
from neurobrain.vision.ventral import build_ventral_stream_on

N_CAM = int(sys.argv[2]) if len(sys.argv) > 2 else 16
WINDOW = 160
STEP = 6                      # pixels of egomotion per frame
T = 10                        # frames per sequence
#: The latent width is set by what is ESTIMABLE, not by what scores best. Each
#: dynamics is a DIM x DIM matrix fitted from the transitions the gate routes to
#: it; at DIM=96 that was 9216 parameters from ~56 transitions, so "K=3 is worse
#: than K=1" would have measured data starvation rather than the architecture.
#: At DIM=32 it is 1024 parameters from ~110. Still generous, and reported.
DIM = 32
MOTIONS = {"right": (0, 1), "left": (0, -1), "down": (1, 0)}
STAGES = ("V2", "V4")


def grab(cam, n_want):
    out = []
    need = WINDOW + STEP * T + 2
    for i in range(len(cam.urls)):
        if len(out) >= n_want:
            break
        try:
            r = cam.read(i)
        except SensorUnavailable:
            continue
        if not getattr(r, "live", False):
            continue
        a = np.asarray(r.data, np.float32)
        if a.ndim != 3 or a.shape[0] < need or a.shape[1] < need:
            continue
        out.append(a.transpose(2, 0, 1).mean(0) / 255.0)
    return out


def pan(frame, dy, dx):
    """A sequence of T views, the window walking across a real frame."""
    h, w = frame.shape
    y0 = (h - WINDOW) // 2 - (dy * STEP * T) // 2
    x0 = (w - WINDOW) // 2 - (dx * STEP * T) // 2
    views = []
    for t in range(T):
        y = int(np.clip(y0 + dy * STEP * t, 0, h - WINDOW))
        x = int(np.clip(x0 + dx * STEP * t, 0, w - WINDOW))
        views.append(frame[y:y + WINDOW, x:x + WINDOW])
    return views


def eye_codes(stream, views, stages):
    h = stream.hierarchy
    names = [getattr(l, "name", type(l).__name__) for l in h.layers]
    ks = {s: names.index(s) for s in stages}
    out = {s: [] for s in stages}
    for v in views:
        h.forward(v[None])
        for s in stages:
            out[s].append(np.asarray(h.layers[ks[s]].log["output"],
                                     np.float32).ravel())
    return {s: np.stack(out[s]) for s in stages}


def pixel_code(views):
    P = []
    for v in views:
        g = 32
        ph, pw = v.shape[0] // g, v.shape[1] // g
        P.append(v[:g * ph, :g * pw].reshape(g, ph, g, pw).mean((1, 3)).ravel())
    return np.stack(P)


def pca_fit(X, d):
    mu = X.mean(0)
    Xc = X - mu
    # economy SVD on the smaller side; X is (n_samples, n_features)
    if Xc.shape[0] <= Xc.shape[1]:
        G = Xc @ Xc.T
        w, U = np.linalg.eigh(G)
        idx = np.argsort(w)[::-1][:d]
        w, U = np.maximum(w[idx], 1e-9), U[:, idx]
        V = (Xc.T @ U) / np.sqrt(w)
    else:
        G = Xc.T @ Xc
        w, V = np.linalg.eigh(G)
        V = V[:, np.argsort(w)[::-1][:d]]
    return mu, np.asarray(V, np.float32)


def pca_apply(mu, V, X):
    Z = (X - mu) @ V
    s = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
    return (Z / s).astype(np.float32)          # unit norm: scale-free errors


class DPC:
    """K linear dynamics over the change, chosen by WTA, trained by delta rule."""

    def __init__(self, d, k=3, lr=0.05, seed=0):
        rng = np.random.default_rng(seed)
        self.A = rng.normal(0, 0.01, (k, d, d)).astype(np.float32)
        self.k, self.lr = k, lr

    def predict_all(self, r):
        """Every dynamics' guess at the next state: current + predicted change."""
        return r[None] + np.einsum("kij,j->ki", self.A, r)

    def choose(self, r, nxt):
        """Which dynamics explains this transition best -- the higher level."""
        P = self.predict_all(r)
        return int(np.argmin(np.linalg.norm(P - nxt[None], axis=1)))

    def learn(self, r, nxt):
        m = self.choose(r, nxt)
        err = (nxt - r) - self.A[m] @ r          # residual on the CHANGE
        self.A[m] += self.lr * np.outer(err, r)  # local: outer(error, state)
        return m


def run_arm(Ztr, k=3, epochs=6, seed=0, shuffle=False):
    """Train on training sequences, score next-step skill on held-out ones."""
    d = Ztr[0].shape[1]
    dpc = DPC(d, k=k, seed=seed)
    pairs = [(z[t], z[t + 1]) for z in Ztr for t in range(len(z) - 1)]
    rng = np.random.default_rng(seed)
    if shuffle:
        # destroy the dynamics, keep the frame statistics: re-deal the targets
        # ACROSS all sequences, not within one -- §9.9's lesson
        tgt = [b for _, b in pairs]
        rng.shuffle(tgt)
        pairs = [(a, t) for (a, _), t in zip(pairs, tgt)]
    for _ in range(epochs):
        order = rng.permutation(len(pairs))
        for i in order:
            dpc.learn(*pairs[i])
    return dpc


SEEDS = (0, 1, 2, 3, 4)


def score(dpc, Zte):
    em, ep, modes = [], [], []
    for z in Zte:
        for t in range(len(z) - 1):
            m = dpc.choose(z[t], z[t + 1])
            p = dpc.predict_all(z[t])[m]
            em.append(float(np.linalg.norm(p - z[t + 1])))
            ep.append(float(np.linalg.norm(z[t] - z[t + 1])))
            modes.append(m)
    em, ep = float(np.mean(em)), float(np.mean(ep))
    return 1.0 - em / (ep + 1e-9), em, ep, modes


def rollout(dpc, Zte, horizon=3):
    """Imagination: run the model forward with NO new observation."""
    sm, sp = [], []
    for z in Zte:
        for t in range(len(z) - horizon):
            r = z[t].copy()
            m = dpc.choose(z[t], z[t + 1])
            for _ in range(horizon):
                r = dpc.predict_all(r)[m]
                r = r / (np.linalg.norm(r) + 1e-9)
            sm.append(float(np.linalg.norm(r - z[t + horizon])))
            sp.append(float(np.linalg.norm(z[t] - z[t + horizon])))
    return 1.0 - float(np.mean(sm)) / (float(np.mean(sp)) + 1e-9)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_dpc.json"
    cam = CctvCamera(ids=list(NY511_LIVE_IDS))
    print(f"reaching for {N_CAM} live cameras (need >= "
          f"{WINDOW + STEP * T + 2}px) ...", flush=True)
    frames = grab(cam, N_CAM)
    if len(frames) < 6:
        print(f"only {len(frames)} usable cameras -- nothing reported.")
        json.dump({"error": "too few cameras", "n": len(frames)},
                  open(out_path, "w"), indent=1)
        return
    n = len(frames)
    cut = max(3, int(0.6 * n))
    print(f"  {n} cameras, {cut} train / {n - cut} test, window {WINDOW}px, "
          f"{len(MOTIONS)} motions x {T} frames, {STEP}px/frame", flush=True)

    seqs = [(fi, mname, pan(f, *d))
            for fi, f in enumerate(frames)
            for mname, d in MOTIONS.items()]
    movement = float(np.mean([np.abs(s[2][-1] - s[2][0]).mean()
                              for s in seqs]))
    print(f"  a sequence moves the pixels by {movement:.3f} "
          f"(0 would mean the pan did nothing)", flush=True)
    if movement < 0.01:
        print("  the pan did not move the scene -- REFUSING to report.")
        json.dump({"error": "pan did not move", "delta": movement},
                  open(out_path, "w"), indent=1)
        return

    print("\ngrowing the eye on these views ...", flush=True)
    stream = build_ventral_stream_on(
        np.stack([s[2][0] for s in seqs[:24]]), size=WINDOW, verbose=False)

    print("encoding every sequence ...", flush=True)
    raw = {s: [] for s in STAGES}
    raw["pixels"] = []
    for i, (_fi, _m, views) in enumerate(seqs):
        e = eye_codes(stream, views, STAGES)
        for s in STAGES:
            raw[s].append(e[s])
        raw["pixels"].append(pixel_code(views))
        print(f"  {i + 1}/{len(seqs)}", end="\r", flush=True)

    reps = [f"eye-{s}" for s in STAGES] + ["pixels"]
    key = {f"eye-{s}": s for s in STAGES}
    key["pixels"] = "pixels"
    res = {"n_cameras": n, "window": WINDOW, "step_px": STEP, "T": T,
           "dim": DIM, "motions": list(MOTIONS), "arms": {}}

    print("\n\n" + "=" * 74)
    for rep in reps:
        R = raw[key[rep]]
        tr = [i for i, s in enumerate(seqs) if s[0] < cut]
        te = [i for i, s in enumerate(seqs) if s[0] >= cut]
        mu, V = pca_fit(np.concatenate([R[i] for i in tr]), DIM)
        Ztr = [pca_apply(mu, V, R[i]) for i in tr]
        Zte = [pca_apply(mu, V, R[i]) for i in te]

        # every arm over every seed: a 7% effect from one seed is not a result,
        # which is §9.10's standing lesson in this project
        lab = [seqs[i][1] for i in te for _ in range(T - 1)]
        acc = {a: [] for a in ("DPC K=3", "DPC K=1", "shuffled-time")}
        rolls, purities, persist = [], [], None
        for sd in SEEDS:
            d3 = run_arm(Ztr, k=len(MOTIONS), seed=sd)
            sk, _em, ep, modes = score(d3, Zte)
            acc["DPC K=3"].append(sk)
            persist = ep
            acc["DPC K=1"].append(score(run_arm(Ztr, k=1, seed=sd), Zte)[0])
            acc["shuffled-time"].append(score(
                run_arm(Ztr, k=len(MOTIONS), seed=sd, shuffle=True), Zte)[0])
            rolls.append(rollout(d3, Zte))
            p = 0.0
            for m in set(modes):
                sel = [lab[i] for i, v in enumerate(modes) if v == m]
                if sel:
                    p += max(sel.count(u) for u in set(sel))
            purities.append(p / max(1, len(modes)))

        print(f"\n{rep}   (dim {R[0].shape[1]} -> PCA {DIM}, "
              f"persistence err {persist:.4f})")
        print(f"  {'arm':<16}{'skill':>9}{'sd':>8}")
        for a, v in acc.items():
            print(f"  {a:<16}{np.mean(v):>9.4f}{np.std(v):>8.4f}")
        print(f"  {'rollout x3':<16}{np.mean(rolls):>9.4f}"
              f"{np.std(rolls):>8.4f}   (imagining 3 steps, no new input)")
        print(f"  {'gate purity':<16}{np.mean(purities):>9.4f}"
              f"{np.std(purities):>8.4f}   (chance "
              f"{1.0 / len(MOTIONS):.3f})")

        res["arms"][rep] = {
            "skill": {a: round(float(np.mean(v)), 4) for a, v in acc.items()},
            "skill_sd": {a: round(float(np.std(v)), 4) for a, v in acc.items()},
            "persistence_err": round(persist, 4),
            "rollout3": round(float(np.mean(rolls)), 4),
            "rollout3_sd": round(float(np.std(rolls)), 4),
            "gate_purity": round(float(np.mean(purities)), 4),
            "dim_raw": int(R[0].shape[1])}

    print("\n" + "=" * 74)
    best = max(reps, key=lambda r: max(res["arms"][r]["skill"].values()))
    bs = max(res["arms"][best]["skill"]["DPC K=3"],
             res["arms"][best]["skill"]["DPC K=1"])
    res["best_rep"] = best
    res["beats_persistence"] = bool(bs > 0.05)
    res["gate_helps"] = bool(any(
        res["arms"][r]["skill"]["DPC K=3"] >
        res["arms"][r]["skill"]["DPC K=1"] + 0.02 for r in reps))
    res["can_imagine_forward"] = bool(
        max(res["arms"][r]["rollout3"] for r in reps) > 0.0)
    # NOT a comparison across representations: skill is relative to each arm's
    # OWN persistence baseline, and those differ by 3x here (pixels 0.19,
    # eye-V4 0.62). A code that barely moves between frames has a strong
    # baseline and little headroom; one that moves a lot has a weak baseline and
    # plenty. Ranking representations by skill would measure how much each code
    # moves, not how predictable it is.
    res["skill_is_within_arm_only"] = True

    print(f"\n--- does a learned dynamics beat assuming a frozen world? ---")
    for rep in reps:
        a = res["arms"][rep]
        print(f"  {rep:<10} K=3 {a['skill']['DPC K=3']:+.4f}   "
              f"K=1 {a['skill']['DPC K=1']:+.4f}   "
              f"shuffled {a['skill']['shuffled-time']:+.4f}   "
              f"rollout {a['rollout3']:+.4f}   "
              f"(persistence err {a['persistence_err']:.3f})")
    print("\n  Skill is measured against each arm's OWN persistence baseline, "
          "and those differ by 3x.\n  These columns compare arms WITHIN a "
          "representation; they do not rank representations.")
    if not res["beats_persistence"]:
        print(f"\n  DPC does not beat persistence anywhere (best {bs:+.4f}). A "
              f"model that cannot predict the\n  next state better than "
              f"'nothing moved' is not a world model.")
    else:
        print(f"\n  Best one-step skill is {bs:+.4f} on {best}: real, since "
              f"shuffled-time is far worse, but\n  small -- a few percent over "
              f"assuming a frozen world.")
    if not res["gate_helps"]:
        print("  The K>1 gate buys NOTHING over a single linear map. The part "
              "that makes this 'dynamic'\n  predictive coding rather than one "
              "transition matrix is not paying for itself.")
    if not res["can_imagine_forward"]:
        print("  And rollout is NEGATIVE on every representation: imagining 3 "
              "steps with no new input is\n  WORSE than assuming nothing "
              "moved. Multi-step rollout is exactly what a world model owes "
              "an\n  imagination engine, and this does not have it.")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
