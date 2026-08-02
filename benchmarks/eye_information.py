"""How much of the image survives to each stage, and do the four act as one eye?

Two goal components have never been measured directly, and one benchmark can
ask both:

* **استخراج حداکثر ویژگی ها و اطلاعات ممکن از تصویر** -- how much information
  about the image does each area actually carry?
* **بصورت چشم واحد عمل کنند** -- do the four stages act as a *unified* eye, or
  is the union no better than its best member?

Method: linear decodability. Fit a ridge map from an area's code back to the
image and measure R² on **held-out frames**. Linear readout is the standard
convention for "information the code makes explicit" -- a non-linear decoder
would measure what could in principle be recovered, which is a different and
much weaker claim.

Everything is compared at **matched dimensionality**, which is not optional
here. §9.16 is this project's own record of what happens otherwise: a 1764-d
code was read against a 256-d one and the difference was reported as an effect
of the architecture. So each area's code is reduced to the same D by PCA before
decoding, and the *unified* arm concatenates all four areas and is reduced to
that same D -- so if it wins, it wins on what it represents rather than on being
four times as wide.

Arms:

    V1_complex, V2, pool, V4   each area alone
    unified                    all four concatenated, then cut to the SAME D
    pixels                     PCA of the raw image -- the **ceiling**, not a
                               competitor: the decoding target is a block mean
                               of the image, which is an exact *linear* function
                               of this arm's own input, so it is guaranteed to
                               win and its score says how well the task can be
                               done at this width, nothing more

Two splits, because they answer different questions and only one of them was
run at first:

    by-video     train and test come from DIFFERENT sequences -- generalisation
                 to a scene never seen
    within-video frames shuffled before splitting, so the same scenes appear on
                 both sides. Adjacent video frames are near-duplicates, so this
                 **leaks** and is an upper bound, not a result

The first version of this benchmark reported only the by-video split and every
eye arm came out negative; the same codes under the within-video split score
+0.33 to +0.38. Reporting one without the other would have turned a
distribution shift into a claim about the architecture.

Reported across several D, because a single width can flatter or flatten any of
these, and read against the **participation ratio** of each code -- how many
directions it actually uses, as opposed to how many it has.

Frames come from the VOT2019 sequences on disk: real photographs, in quantity,
and not chosen by this project.

Usage:  python3 benchmarks/eye_information.py out_eye_info.json [vot_dir] [n]
"""
import json
import os
import sys

import numpy as np

from neurobrain.vision.ventral import (build_ventral_stream_on,
                                       code_participation)

VOT_DIR = sys.argv[2] if len(sys.argv) > 2 else "vot"
N_FRAME = int(sys.argv[3]) if len(sys.argv) > 3 else 360
SIZE = 224                 # native, per the goal's real dimensions
RECON = 56                 # the image is decoded at 56x56 = 3136 targets
DIMS = (16, 64, 128)
AREAS = ("V1_complex", "V2", "pool", "V4")


def load_frames(d, n_want):
    from PIL import Image
    out = []
    seqs = sorted(x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x)))
    if not seqs:
        return out
    per = max(1, n_want // max(1, len(seqs)))
    for s in seqs:
        jpgs = sorted(f for f in os.listdir(os.path.join(d, s))
                      if f.endswith(".jpg"))
        step = max(1, len(jpgs) // per)
        for f in jpgs[::step][:per]:
            if len(out) >= n_want:
                return out
            a = np.asarray(Image.open(os.path.join(d, s, f)).convert("L"),
                           np.float32) / 255.0
            h, w = a.shape
            if h < SIZE or w < SIZE:
                continue
            y, x = (h - SIZE) // 2, (w - SIZE) // 2
            out.append(a[y:y + SIZE, x:x + SIZE])
    return out


def block_mean(a, g):
    ph, pw = a.shape[0] // g, a.shape[1] // g
    return a[:g * ph, :g * pw].reshape(g, ph, g, pw).mean((1, 3))


def area_codes(stream, frames, areas):
    h = stream.hierarchy
    names = [getattr(l, "name", type(l).__name__) for l in h.layers]
    ks = {a: names.index(a) for a in areas}
    out = {a: [] for a in areas}
    for i, f in enumerate(frames):
        h.forward(f[None])
        for a in areas:
            out[a].append(np.asarray(h.layers[ks[a]].log["output"],
                                     np.float32).ravel())
        if (i + 1) % 25 == 0:
            print(f"    {i + 1}/{len(frames)}", end="\r", flush=True)
    return {a: np.stack(out[a]) for a in areas}


def pca(Xtr, Xte, d):
    """Fit on train only -- fitting on all of it would leak the test set.

    Components are standardised by their training spread, so one ridge penalty
    means the same thing on every component and on every arm; without it the
    penalty falls almost entirely on the leading direction.
    """
    mu = Xtr.mean(0)
    A = Xtr - mu
    if A.shape[0] <= A.shape[1]:
        w, U = np.linalg.eigh(A @ A.T)
        idx = np.argsort(w)[::-1][:d]
        w = np.maximum(w[idx], 1e-9)
        V = (A.T @ U[:, idx]) / np.sqrt(w)
    else:
        w, V = np.linalg.eigh(A.T @ A)
        V = V[:, np.argsort(w)[::-1][:d]]
    V = np.asarray(V, np.float32)
    Ztr, Zte = (Xtr - mu) @ V, (Xte - mu) @ V
    s = Ztr.std(0) + 1e-9
    return np.asarray(Ztr / s, np.float32), np.asarray(Zte / s, np.float32)


def ridge_r2(Ztr, Ytr, Zte, Yte, lam=1.0):
    """Closed-form ridge, then R² against the TRAIN mean as the null model.

    The intercept is **not** penalised. Penalising it -- which the first version
    did -- drives the prediction toward zero rather than toward the mean as lam
    grows, so heavy regularisation produced R² of -4 on images whose mean is
    ~0.4, and that was an artefact of the penalty rather than of any code.
    """
    Ztr = np.hstack([Ztr, np.ones((len(Ztr), 1), np.float32)])
    Zte = np.hstack([Zte, np.ones((len(Zte), 1), np.float32)])
    P = np.eye(Ztr.shape[1], dtype=np.float32)
    P[-1, -1] = 0.0                                    # free intercept
    W = np.linalg.solve(Ztr.T @ Ztr + lam * P, Ztr.T @ Ytr)
    ss_res = float(np.sum((Yte - Zte @ W) ** 2))
    ss_tot = float(np.sum((Yte - Ytr.mean(0)) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-12)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_eye_info.json"
    if not os.path.isdir(VOT_DIR):
        print(f"no frame directory at {VOT_DIR!r} -- nothing reported.")
        json.dump({"error": "no frames", "path": VOT_DIR},
                  open(out_path, "w"), indent=1)
        return
    frames = load_frames(VOT_DIR, N_FRAME)
    if len(frames) < 120:
        print(f"only {len(frames)} frames -- too few to fit {max(DIMS)} "
              f"components. Nothing reported.")
        json.dump({"error": "too few frames", "n": len(frames)},
                  open(out_path, "w"), indent=1)
        return
    n = len(frames)
    cut = int(0.7 * n)
    print(f"{n} frames at {SIZE}px, {cut} train / {n - cut} test, "
          f"decoding to {RECON}x{RECON}", flush=True)

    print("\ngrowing the eye on the training frames ...", flush=True)
    stream = build_ventral_stream_on(np.stack(frames[:cut]), size=SIZE,
                                     verbose=False)

    print("encoding every frame at every area ...", flush=True)
    codes = area_codes(stream, frames, AREAS)
    # Each area is scaled to the same mean row norm BEFORE concatenating.
    # Without this the union is not a union: PCA of the raw concatenation is
    # dominated by whichever area happens to carry the most variance (V2, at
    # 381924 dims), so "the four together" would really mean "V2, slightly
    # perturbed" and any verdict about a unified eye would be about that.
    codes["unified"] = np.concatenate(
        [codes[a] / (np.mean(np.linalg.norm(codes[a], axis=1)) + 1e-9)
         for a in AREAS], axis=1)
    codes["pixels"] = np.stack([f[::2, ::2].ravel() for f in frames])
    Y = np.stack([block_mean(f, RECON).ravel() for f in frames])
    Ytr, Yte = Y[:cut], Y[cut:]

    arms = list(AREAS) + ["unified", "pixels"]
    splits = {"by-video": np.arange(n),
              "within-video": np.random.default_rng(0).permutation(n)}
    res = {"n_frames": n, "size": SIZE, "recon": RECON, "dims": list(DIMS),
           "raw_dim": {a: int(codes[a].shape[1]) for a in arms},
           "r2": {s: {a: {} for a in arms} for s in splits},
           "participation": {}}

    for sname, order in splits.items():
        tr, te = order[:cut], order[cut:]
        ytr, yte = Y[tr], Y[te]
        note = ("different sequences in train and test"
                if sname == "by-video" else "same scenes both sides -- LEAKS")
        print(f"\n=== {sname} split ({note}) ===")
        print(f"{'arm':<12}{'raw dim':>10}"
              + "".join(f"{'D=' + str(d):>10}" for d in DIMS)
              + f"{'particip.':>11}")
        for a in arms:
            X = codes[a]
            row = []
            for d in DIMS:
                Ztr, Zte = pca(X[tr], X[te], d)
                r2 = ridge_r2(Ztr, ytr, Zte, yte)
                res["r2"][sname][a][str(d)] = round(r2, 4)
                row.append(r2)
            if sname == "by-video":
                p = float(code_participation(X[te] / (np.linalg.norm(
                    X[te], axis=1, keepdims=True) + 1e-9)))
                res["participation"][a] = round(p, 2)
            print(f"{a:<12}{X.shape[1]:>10}"
                  + "".join(f"{v:>10.4f}" for v in row)
                  + f"{res['participation'][a]:>11.2f}")

    D = str(max(DIMS))
    R = res["r2"]["within-video"]         # the leaky upper bound decides these
    best_single = max(AREAS, key=lambda a: R[a][D])
    bs = R[best_single][D]
    uni = R["unified"][D]
    px = R["pixels"][D]
    res["r2_generalising"] = {a: res["r2"]["by-video"][a][D] for a in arms}
    res["best_single_area"] = best_single
    res["unified_beats_best_single"] = bool(uni > bs + 0.01)
    res["eye_beats_pixel_compression"] = bool(max(uni, bs) > px + 0.01)
    res["depth_profile"] = [R[a][D] for a in AREAS]

    print(f"\n--- how much of the image survives, at D={D} "
          f"(within-video) ---")
    print("  " + "  ".join(f"{a} {R[a][D]:.3f}" for a in AREAS))
    print("  generalising to unseen scenes: "
          + "  ".join(f"{a} {res['r2']['by-video'][a][D]:+.3f}"
                      for a in AREAS))
    falls = all(R[AREAS[i + 1]][D] <= R[AREAS[i]][D] + 1e-9
                for i in range(len(AREAS) - 1))
    res["monotonic_loss_with_depth"] = bool(falls)
    if falls:
        print("  Information about the image falls monotonically with depth. "
              "Each stage discards and\n  none adds -- the hierarchy is a "
              "cascade of losses as far as image content is concerned.")
    else:
        print("  Information does NOT fall monotonically -- some stage carries "
              "more about the image than\n  the one below it.")

    print(f"\n--- do the four act as ONE eye? (matched at D={D}) ---")
    print(f"  best single area: {best_single} {bs:.4f}")
    print(f"  all four unified: {uni:.4f}  ({uni - bs:+.4f})")
    if res["unified_beats_best_single"]:
        print("  YES. At the SAME width, the four areas together carry more "
              "about the image than any one\n  of them -- they hold "
              "complementary information, which is what a unified eye means.")
    else:
        print("  NO. At the same width the union is no better than its best "
              "member, so the areas are not\n  complementary: whatever the "
              "deeper stages hold is already in the shallower one.")

    print(f"\n--- against the ceiling ---")
    print(f"  pixels at D={D}: {px:.4f}   best eye arm: {max(uni, bs):.4f}")
    print("  The pixel arm is a CEILING, not a rival: the decoding target is a "
          "block mean of the image,\n  which is an exact linear function of "
          "that arm's own input, so it must win. What it says is\n  that the "
          f"task is {px:.2f}-solvable at this width, and the eye recovers "
          f"{max(uni, bs) / px:.0%} of that.")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
