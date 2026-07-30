"""Does it actually imagine, or does it recall an average? And what shape is it?

`cross_modal_dream.py` showed that replaying a sound, imagining the sight, and
binding the pair improves real cross-modal recall in 6 of 6 seeds while a random
imagined sight does nothing. That is a real effect. It is *not* evidence of
imagination, and the difference matters enough to measure rather than assume.

The imagined sight is `Wv[concept_from_sound(a)]` -- a stored weight vector. A
concept cell's `Wv` is the running average of everything bound to it. So the
mind may simply be **recalling a category average**, which would make "imagine"
a flattering name for a lookup. Four measurements decide it, all on held-out
sounds:

    fidelity      cosine from the imagined sight to the nearest sight the mind
                  has actually seen. At ~1.0 it is replaying a stored image
                  verbatim and nothing was imagined
    novelty       the same distance, compared against what a *real* unseen
                  photograph scores. This is the sharp one: if opening its eyes
                  produces something further from memory than imagining does,
                  then imagination is the LESS novel of the two and the word is
                  wrong
    vocabulary    how many distinct things it can imagine at all. An "infinite
                  inner world" that resolves to a handful of attractors is a
                  finite one
    position      is the imagined code a class *prototype*, an *exemplar*, or
                  neither -- and does it lie inside the convex hull of
                  experience (interpolation) or outside it (extrapolation)?
                  Only the second is imagination in the strong sense

The honest prediction from the architecture: fidelity high, novelty *below* real
perception, vocabulary equal to the number of concept cells, position squarely
inside the hull. If all four come out that way, what the dream benchmark
measured was consolidation toward class means -- useful, real, and not
imagination.

Usage:  python3 benchmarks/imagination_shape.py out_imagination_shape.json
"""
import json
import sys

import numpy as np

from neurobrain.cognition.multimodal import AssociationArea
from neurobrain.sensing.natural import load_audiovisual
from neurobrain.sensing.streams import StreamingBrain, _unit

sys.path.insert(0, "benchmarks")
from real_binding import encode, split                       # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
N_CONCEPT = 256


def hull_residual(v, B, iters=300, lr=0.5):
    """How much of ``v`` cannot be built from a convex mix of the rows of ``B``.

    Projected gradient onto the simplex -- non-negative weights summing to one,
    which is exactly "a blend of things I have seen". The residual is what is
    left over, so 0 means the imagined code sits inside the convex hull of
    experience and could have been produced by interpolation alone."""
    w = np.full(len(B), 1.0 / len(B))
    for _ in range(iters):
        r = w @ B - v
        g = B @ r
        w -= lr * g
        w = np.maximum(w, 0.0)
        s = w.sum()
        w = w / s if s > 1e-9 else np.full(len(B), 1.0 / len(B))
    r = w @ B - v
    return float(np.linalg.norm(r) / (np.linalg.norm(v) + 1e-9))


def run_seed(V, A, y, n_cls, seed):
    tr, te = split(y, seed)
    assoc = AssociationArea(n_vis=V.shape[1], n_aud=A.shape[1],
                            n_concept=N_CONCEPT, seed=seed)
    assoc.set_stats(V[tr], A[tr])
    for i in tr:
        assoc.bind(V[i], A[i])

    Btr = V[tr]                                   # everything it has ever seen
    protos = np.stack([_unit(Btr[y[tr] == c].mean(0)) for c in range(n_cls)])

    imagined, cells = [], []
    for i in te:
        c = assoc.concept_from_sound(A[i])
        cells.append(c)
        imagined.append(_unit(assoc.Wv[c]))
    imagined = np.asarray(imagined, np.float32)

    # -- fidelity: how close is the imagined sight to a remembered one? -----
    fid = (imagined @ Btr.T).max(1)
    # -- the same question asked of REAL unseen photographs ----------------
    real = (V[te] @ Btr.T).max(1)
    # -- is it a prototype or an exemplar? ---------------------------------
    to_proto = np.array([float(imagined[k] @ protos[int(y[i])])
                         for k, i in enumerate(te)])
    real_to_proto = np.array([float(V[i] @ protos[int(y[i])]) for i in te])
    # -- could interpolation alone have produced it? -----------------------
    sub = np.random.default_rng(seed).permutation(len(Btr))[:120]
    hull = np.array([hull_residual(imagined[k], Btr[sub])
                     for k in range(0, len(imagined), 5)])
    hull_real = np.array([hull_residual(V[i], Btr[sub])
                          for i in te[::5]])

    uniq = len(set(cells))
    p = np.bincount(cells, minlength=N_CONCEPT).astype(np.float64)
    p = p[p > 0] / p.sum()
    return dict(
        fidelity=float(fid.mean()), fidelity_sd=float(fid.std()),
        real_fidelity=float(real.mean()),
        to_prototype=float(to_proto.mean()),
        real_to_prototype=float(real_to_proto.mean()),
        hull_residual=float(hull.mean()),
        real_hull_residual=float(hull_real.mean()),
        distinct=uniq, cells_live=int((assoc.wins > 0).sum()),
        entropy_bits=float(-(p * np.log2(p)).sum()),
        effective=float(2 ** (-(p * np.log2(p)).sum())),
        n_test=int(len(te)))


def main():
    out_path = (sys.argv[1] if len(sys.argv) > 1
                else "out_imagination_shape.json")
    images, waves, y, names = load_audiovisual(n_per_class=60, seed=0,
                                               grayscale=False, size=32)
    n_cls = len(names)
    brain = StreamingBrain(seed=0, image_shape=(32, 32), v1_cells=4096,
                           rf=7, stride=2)
    V, A = encode(brain, images, waves, colour=True, adapt=True, develop=True)
    print(f"{len(images)} real pairs, {n_cls} categories\n", flush=True)

    rows = [run_seed(V, A, y, n_cls, sd) for sd in SEEDS]
    m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    res = {"per_seed": rows, "mean": {k: round(v, 4) for k, v in m.items()},
           "n_class": n_cls}

    print(f"{'measurement':<36}{'imagined':>10}{'real sight':>12}")
    print(f"{'closeness to a remembered sight':<36}{m['fidelity']:>10.3f}"
          f"{m['real_fidelity']:>12.3f}")
    print(f"{'closeness to its class prototype':<36}{m['to_prototype']:>10.3f}"
          f"{m['real_to_prototype']:>12.3f}")
    print(f"{'outside the hull of experience':<36}{m['hull_residual']:>10.3f}"
          f"{m['real_hull_residual']:>12.3f}")
    print(f"\n{'vocabulary':<36}{m['distinct']:>10.1f} distinct imaginings "
          f"from {m['n_test']:.0f} sounds")
    print(f"{'':<36}{m['effective']:>10.1f} effective "
          f"({m['entropy_bits']:.2f} bits) over {m['cells_live']:.0f} "
          f"concept cells")

    print("\n=== does it imagine? ===")
    verdicts = []
    if m["fidelity"] > 0.95:
        verdicts.append("no -- it replays a stored sight almost verbatim")
    elif m["fidelity"] > m["real_fidelity"]:
        verdicts.append(
            f"no -- what it imagines is CLOSER to memory ({m['fidelity']:.3f}) "
            f"than a real unseen photograph is ({m['real_fidelity']:.3f}). "
            f"Opening its eyes is more novel than imagining.")
    else:
        verdicts.append(
            f"partly -- what it imagines is further from memory "
            f"({m['fidelity']:.3f}) than a real photograph is "
            f"({m['real_fidelity']:.3f})")
    if m["to_prototype"] > m["real_to_prototype"] + 0.05:
        verdicts.append(
            f"what it produces is a class AVERAGE: closer to the prototype "
            f"({m['to_prototype']:.3f}) than a real member of the class is "
            f"({m['real_to_prototype']:.3f})")
    if m["hull_residual"] < 0.2:
        verdicts.append(
            f"it stays inside the convex hull of what it has seen "
            f"(residual {m['hull_residual']:.3f}) -- interpolation, never "
            f"extrapolation")
    verdicts.append(
        f"its whole inner world is {m['effective']:.0f} distinct things "
        f"({m['entropy_bits']:.2f} bits). Not infinite.")
    for v in verdicts:
        print(f"  - {v}")

    res["verdicts"] = verdicts
    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
