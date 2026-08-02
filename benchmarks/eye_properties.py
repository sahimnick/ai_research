"""Distance and shadow: the goal's 'ویژگیهای خاص' , which nothing has measured.

The goal asks for **درک رنگ و ویژگیهای خاص مانند فاصله، عمق، سایه، ارتباط**.
Colour was measured in §9.18 and falsified, with the cause located in
`ComplexCellLayer`'s single-channel quadrature. Distance, depth and shadow have
never been measured at all. Two of them can be, with data already on disk and
without inventing ground truth.

1. Distance, from a real cue with real labels
---------------------------------------------
As a tracked object approaches or recedes its **annotated box grows or
shrinks**. VOT ground truth therefore contains a genuine relative-distance
signal, produced by the world rather than by this benchmark. The question is
whether the eye's code carries it:

    can a linear probe read log(target area) off the code?

Scored as R² on **held-out sequences**, against raw pixels through the identical
pipeline. Frames are rescaled to a constant frame size, so what is being
predicted is the object's *angular* size and not how big the source video was.

The confound worth naming: the target's area covaries with how much of the frame
it occupies, so a code that merely measured "how much stuff is in the middle"
would score. That is why the pixel arm matters -- it is exactly such a code, and
the eye has to beat it to have said anything.

2. Shadow, as the thing it is easy to confuse an object with
------------------------------------------------------------
A shadow changes **illumination** while leaving reflectance -- the object --
unchanged. Any visual system that does not separate the two treats a shadow edge
as an object edge. Each real frame is perturbed two ways in the same region:

    shadow    the region is multiplied by a factor: illumination changes,
              content does not
    object    the region is replaced by a different image patch: content changes

Both are normalised by how much they moved the **pixels**, so the measure is
*code change per unit pixel change*, and the statistic is the ratio

    shadow-sensitivity / object-sensitivity

Below 1 means the code discounts illumination relative to content -- some shadow
invariance. At 1 it treats a shadow exactly like a new object. Raw pixels give
the null by construction: pixels *are* the pixel change, so their ratio is 1.

Usage:  python3 benchmarks/eye_properties.py out.json [vot_dir] [n_seq]
"""
import json
import os
import sys

import numpy as np

from neurobrain.vision.ventral import build_ventral_stream_on

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eye_information import pca, ridge_r2                     # noqa: E402
from eye_vot import load_gt, resize                           # noqa: E402

VOT_DIR = sys.argv[2] if len(sys.argv) > 2 else "vot"
N_SEQ = int(sys.argv[3]) if len(sys.argv) > 3 else 20
SIZE = 224
DIM = 48
AREAS = ("V2", "pool", "V4")
SHADOW_FACTOR = 0.45
PATCH = 64


def frames_with_area(d, name, n_want=8):
    """Frames at a constant canvas, paired with the target's log area."""
    from PIL import Image
    p = os.path.join(d, name)
    jpgs = sorted(f for f in os.listdir(p) if f.endswith(".jpg"))
    gtp = os.path.join(p, "groundtruth.txt")
    if not jpgs or not os.path.exists(gtp):
        return []
    gt = load_gt(gtp)
    step = max(1, len(jpgs) // n_want)
    out = []
    for i in range(0, min(len(jpgs), len(gt)), step):
        if gt[i] is None:
            continue
        y, x, h, w = gt[i]
        if h < 3 or w < 3:
            continue
        a = np.asarray(Image.open(os.path.join(p, jpgs[i])).convert("L"),
                       np.float32) / 255.0
        sy, sx = SIZE / a.shape[0], SIZE / a.shape[1]
        f = resize(a, sy, sx)
        if f.shape != (SIZE, SIZE):
            continue
        # angular size on the constant canvas, not the source video's pixels
        out.append((f, float(np.log((h * sy) * (w * sx)))))
        if len(out) >= n_want:
            break
    return out


def codes_of(stream, frames, areas):
    h = stream.hierarchy
    names = [getattr(l, "name", type(l).__name__) for l in h.layers]
    ks = {a: names.index(a) for a in areas}
    out = {a: [] for a in areas}
    for i, f in enumerate(frames):
        h.forward(f[None])
        for a in areas:
            out[a].append(np.asarray(h.layers[ks[a]].log["output"],
                                     np.float32).ravel())
        if (i + 1) % 40 == 0:
            print(f"    {i + 1}/{len(frames)}", end="\r", flush=True)
    return {a: np.stack(out[a]) for a in areas}


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_eye_properties.json"
    if not os.path.isdir(VOT_DIR):
        print(f"no frame directory at {VOT_DIR!r} -- nothing reported.")
        json.dump({"error": "no frames"}, open(out_path, "w"), indent=1)
        return
    names = sorted(d for d in os.listdir(VOT_DIR)
                   if os.path.isdir(os.path.join(VOT_DIR, d)))[:N_SEQ]
    per_seq = {}
    for n in names:
        fa = frames_with_area(VOT_DIR, n)
        if len(fa) >= 5:
            per_seq[n] = fa
    if len(per_seq) < 6:
        print(f"only {len(per_seq)} usable sequences -- nothing reported.")
        json.dump({"error": "too few sequences", "n": len(per_seq)},
                  open(out_path, "w"), indent=1)
        return
    seq_names = list(per_seq)
    cut = int(0.65 * len(seq_names))
    tr_seq, te_seq = seq_names[:cut], seq_names[cut:]
    tr = [fa for s in tr_seq for fa in per_seq[s]]
    te = [fa for s in te_seq for fa in per_seq[s]]
    frames = [f for f, _ in tr] + [f for f, _ in te]
    y = np.array([[a] for _, a in tr] + [[a] for _, a in te], np.float32)
    ntr = len(tr)
    print(f"{len(per_seq)} sequences -> {ntr} train / {len(te)} test frames "
          f"from DISJOINT sequences", flush=True)
    print(f"  log-area spread: train {y[:ntr].std():.2f}, "
          f"test {y[ntr:].std():.2f} (a flat target cannot be predicted)",
          flush=True)

    print("\ngrowing the eye ...", flush=True)
    stream = build_ventral_stream_on(np.stack(frames[:ntr]), size=SIZE,
                                     verbose=False)
    print("encoding ...", flush=True)
    codes = codes_of(stream, frames, AREAS)
    codes["pixels"] = np.stack([f[::2, ::2].ravel() for f in frames])

    res = {"n_sequences": len(per_seq), "train_frames": ntr,
           "test_frames": len(te), "dim": DIM,
           "distance_r2": {}, "distance_r2_within": {}}
    # Two splits, for the reason §9.25 had to learn: this code carries ~nothing
    # across unseen scenes about ANYTHING (§9.27), so a by-sequence null cannot
    # by itself say that *distance* is missing. The within-sequence split asks
    # whether the information is there at all; the by-sequence one asks whether
    # it transfers. Only both together separate "no distance signal" from "no
    # transfer".
    n = len(frames)
    shuf = np.random.default_rng(0).permutation(n)
    wtr, wte = shuf[:ntr], shuf[ntr:]
    print(f"\n--- 1. distance: can the code read the target's angular size? ---")
    print(f"  {'arm':<10}{'held-out sequences':>21}{'within-sequence':>18}")
    for a in list(AREAS) + ["pixels"]:
        X = codes[a]
        Ztr, Zte = pca(X[:ntr], X[ntr:], DIM)
        r2 = ridge_r2(Ztr, y[:ntr], Zte, y[ntr:])
        Ztr2, Zte2 = pca(X[wtr], X[wte], DIM)
        r2w = ridge_r2(Ztr2, y[wtr], Zte2, y[wte])
        res["distance_r2"][a] = round(r2, 4)
        res["distance_r2_within"][a] = round(r2w, 4)
        print(f"  {a:<10}{r2:>21.4f}{r2w:>18.4f}")

    # --- 2. shadow vs object, on the same region of the same frames ---
    print(f"\n--- 2. shadow: is an illumination change treated like an object"
          f" change? ---")
    rng = np.random.default_rng(0)
    base = frames[ntr:][:40]
    donors = frames[:ntr]
    sens = {a: {"shadow": [], "object": []} for a in list(AREAS) + ["pixels"]}
    for i, f in enumerate(base):
        y0 = int(rng.integers(20, SIZE - PATCH - 20))
        x0 = int(rng.integers(20, SIZE - PATCH - 20))
        sh = f.copy()
        sh[y0:y0 + PATCH, x0:x0 + PATCH] *= SHADOW_FACTOR
        ob = f.copy()
        d = donors[int(rng.integers(0, len(donors)))]
        ob[y0:y0 + PATCH, x0:x0 + PATCH] = d[y0:y0 + PATCH, x0:x0 + PATCH]
        trio = codes_of(stream, [f, sh, ob], AREAS)
        trio["pixels"] = np.stack([v[::2, ::2].ravel() for v in (f, sh, ob)])
        dpx = {"shadow": float(np.abs(sh - f).mean()),
               "object": float(np.abs(ob - f).mean())}
        for a in sens:
            C = trio[a]
            for k, j in (("shadow", 1), ("object", 2)):
                if dpx[k] < 1e-6:
                    continue
                dc = float(np.linalg.norm(C[j] - C[0])
                           / (np.linalg.norm(C[0]) + 1e-9))
                sens[a][k].append(dc / dpx[k])       # per unit pixel change
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(base)}", end="\r", flush=True)

    res["shadow"] = {}
    print(f"\n  {'arm':<10}{'shadow':>10}{'object':>10}{'ratio':>9}")
    for a in sens:
        s = float(np.mean(sens[a]["shadow"]))
        o = float(np.mean(sens[a]["object"]))
        res["shadow"][a] = {"shadow_sens": round(s, 4),
                            "object_sens": round(o, 4),
                            "ratio": round(s / (o + 1e-9), 4)}
        print(f"  {a:<10}{s:>10.4f}{o:>10.4f}"
              f"{res['shadow'][a]['ratio']:>9.3f}")

    best_d = max(AREAS, key=lambda a: res["distance_r2"][a])
    px_d = res["distance_r2"]["pixels"]
    res["reads_distance"] = bool(res["distance_r2"][best_d] > px_d + 0.05)
    ratios = {a: res["shadow"][a]["ratio"] for a in AREAS}
    best_s = min(ratios, key=ratios.get)
    res["discounts_shadow"] = bool(ratios[best_s] <
                                   res["shadow"]["pixels"]["ratio"] - 0.10)

    bw = max(AREAS, key=lambda a: res["distance_r2_within"][a])
    res["distance_within_best"] = bw
    res["has_distance_signal_at_all"] = bool(
        res["distance_r2_within"][bw] > 0.10)

    print(f"\n--- what the eye knows about these properties ---")
    print(f"  distance: best area {best_d} R2 {res['distance_r2'][best_d]:+.3f}"
          f"   raw pixels {px_d:+.3f}")
    if res["reads_distance"]:
        print(f"    The code carries the object's angular size beyond what "
              f"a plain image compression does.")
    elif res["distance_r2"][best_d] > 0.05:
        print(f"    The code carries some distance signal, but no more than "
              f"raw pixels do -- and pixels\n    carry it trivially, since a "
              f"nearer object simply fills more of the frame.")
    else:
        print(f"    The code does not read the object's angular size on "
              f"held-out sequences at all.")
    print(f"    within-sequence: {bw} {res['distance_r2_within'][bw]:+.3f}, "
          f"pixels {res['distance_r2_within']['pixels']:+.3f}")
    if res["has_distance_signal_at_all"]:
        print(f"    So the signal IS in the code and does not transfer -- the "
              f"same shape as §9.27, not a\n    property-specific failure.")
    else:
        print(f"    And it is absent WITHIN a sequence too, so this is not "
              f"§9.27's transfer problem: the\n    code does not represent the "
              f"object's angular size at all, even on scenes it developed on.")
    print(f"  shadow: best area {best_s} ratio {ratios[best_s]:.3f}   "
          f"pixels {res['shadow']['pixels']['ratio']:.3f} (1.0 = no "
          f"discounting)")
    if res["discounts_shadow"]:
        print(f"    The hierarchy discounts an illumination change relative "
              f"to a content change. Some\n    shadow invariance is present.")
    else:
        print(f"    The hierarchy responds to a shadow as strongly as to a "
              f"new object, per unit of pixel\n    change. Nothing here "
              f"separates illumination from reflectance -- a shadow edge and "
              f"an\n    object edge are the same event to this eye.")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
