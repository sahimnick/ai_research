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

Corrected after the fact
------------------------
The first version of this benchmark compared imagined codes against the stored
bank in **raw code space**, while `Wv` and everything `imagine_vision` returns
live in ``prep_v`` space. The same photograph sits at cosine 0.899 to itself
across those two representations, so every fidelity here was understated. The
reported 0.886 is really **0.979**, and `composition.py` found why: of ~112
concept cells recruited from 216 pairs, 59 win exactly once, and a cell that
wins once holds its training photograph verbatim with `mode_var` exactly zero.
For those cells this file's temperature sweep is measuring a constant. The
conclusion below did not change -- it got stronger.

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
# 0.0 is the old behaviour exactly -- read the cell mean back. Above that the
# cell samples from the subspace it has learned it varies in.
TEMPERATURES = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0)


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

    # Everything it has ever seen -- in ``prep_v`` space, which is where `Wv`
    # and everything `imagine_vision` returns actually live. Comparing an
    # imagined code against the *raw* bank was this benchmark's original error:
    # the same photograph sits at cosine 0.899 to itself across the two
    # representations, so every distance was read through a 0.1 fog and the
    # headline fidelity came out 0.886 when the truth is 0.979. See
    # `composition.py`, which found it -- most of that 0.979 is cells that hold
    # a stored photograph verbatim.
    Btr = np.stack([assoc.prep_v(V[i]) for i in tr])
    Bte = np.stack([assoc.prep_v(V[i]) for i in te])
    protos = np.stack([_unit(Btr[y[tr] == c].mean(0)) for c in range(n_cls)])

    rng = np.random.default_rng(seed + 31)
    cells = [assoc.concept_from_sound(A[i]) for i in te]

    # coherence: does the imagined sight still read as its own category?
    protos_all = np.stack([_unit(Btr[y[tr] == c].mean(0)) for c in range(n_cls)])
    per_temp = {}
    for T in TEMPERATURES:
        M = np.asarray([assoc.imagine_vision(c, temperature=T, rng=rng)
                        for c in cells], np.float32)
        fid_T = (M @ Btr.T).max(1)
        coh_T = np.array([int(np.argmax(protos_all @ M[k])) == int(y[i])
                          for k, i in enumerate(te)], float)
        # The mean alone would say "almost verbatim" about a population that is
        # really two populations. Split it: the fraction that is bit-identical
        # to something stored, and where the rest actually sit.
        ex = fid_T > 0.999
        per_temp[T] = dict(fidelity=float(fid_T.mean()),
                           coherence=float(coh_T.mean()),
                           verbatim=float(ex.mean()),
                           fidelity_rest=float(fid_T[~ex].mean())
                           if (~ex).any() else float("nan"))
    imagined = np.asarray([assoc.imagine_vision(c, temperature=0.0)
                           for c in cells], np.float32)

    # -- fidelity: how close is the imagined sight to a remembered one? -----
    fid = (imagined @ Btr.T).max(1)
    # -- the same question asked of REAL unseen photographs ----------------
    real = (Bte @ Btr.T).max(1)
    # -- is it a prototype or an exemplar? ---------------------------------
    to_proto = np.array([float(imagined[k] @ protos[int(y[i])])
                         for k, i in enumerate(te)])
    real_to_proto = np.array([float(Bte[k] @ protos[int(y[i])])
                              for k, i in enumerate(te)])
    # -- could interpolation alone have produced it? -----------------------
    sub = np.random.default_rng(seed).permutation(len(Btr))[:120]
    hull = np.array([hull_residual(imagined[k], Btr[sub])
                     for k in range(0, len(imagined), 5)])
    hull_real = np.array([hull_residual(Bte[k], Btr[sub])
                          for k in range(0, len(Bte), 5)])

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
        per_temp=per_temp,
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
    m = {k: float(np.mean([r[k] for r in rows]))
         for k in rows[0] if k != 'per_temp'}
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

    # -- can sampling the concept's own variation make it novel AND right? --
    print(f"\n{'temperature':<12}{'fidelity':>10}{'coherence':>11}"
          f"{'verbatim':>10}{'rest':>8}   "
          f"(real photograph: fidelity {m['real_fidelity']:.3f})")
    tt = {}
    for T in TEMPERATURES:
        f = float(np.mean([r["per_temp"][T]["fidelity"] for r in rows]))
        c = float(np.mean([r["per_temp"][T]["coherence"] for r in rows]))
        v = float(np.mean([r["per_temp"][T]["verbatim"] for r in rows]))
        rst = float(np.mean([r["per_temp"][T]["fidelity_rest"] for r in rows]))
        tt[T] = dict(fidelity=round(f, 4), coherence=round(c, 4),
                     verbatim=round(v, 4), fidelity_rest=round(rst, 4))
        mark = "  <- novel as a real sight" if f <= m["real_fidelity"] else ""
        print(f"{T:<12.1f}{f:>10.3f}{c:>11.3f}{v:>10.3f}{rst:>8.3f}{mark}")
    res["temperature"] = tt
    ok = [T for T in TEMPERATURES
          if tt[T]["fidelity"] <= m["real_fidelity"] and tt[T]["coherence"] >= 0.5]
    if ok:
        T = min(ok)
        print(f"\n  at temperature {T}: fidelity {tt[T]['fidelity']:.3f} "
              f"(<= {m['real_fidelity']:.3f}) AND still names its own category "
              f"{tt[T]['coherence']:.0%} of the time.")
        print("  novel and coherent at once -- that is imagining.")
    else:
        best = min(TEMPERATURES, key=lambda T: tt[T]["fidelity"])
        print(f"\n  no temperature is both novel and coherent. The furthest "
              f"from memory is T={best} at {tt[best]['fidelity']:.3f}, "
              f"coherence {tt[best]['coherence']:.3f}.")
        print("  Sampling the learned subspace does not escape the span of "
              "what is stored.")
    res["imagination_passes"] = bool(ok)

    print("\n=== does it imagine? ===")
    verdicts = []
    vb = float(np.mean([r["per_temp"][0.0]["verbatim"] for r in rows]))
    rest = float(np.mean([r["per_temp"][0.0]["fidelity_rest"] for r in rows]))
    res["verbatim_fraction"] = round(vb, 4)
    res["fidelity_excluding_verbatim"] = round(rest, 4)
    if m["fidelity"] > 0.95:
        # Not "it replays a stored sight almost verbatim" -- 0.979 is a mean
        # over two populations and that phrasing overstates what it covers.
        # Say which is which.
        verdicts.append(
            f"no -- {vb:.1%} of what it produces is bit-identical to a stored "
            f"photograph (cosine > 0.999); the rest sits at {rest:.3f}. The "
            f"mean {m['fidelity']:.3f} is that mixture, not a uniform near-copy. "
            f"Either way the generated representation stays far closer to "
            f"memory than a real unseen observation does "
            f"({m['real_fidelity']:.3f}).")
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
        # "is a class average" is one step further than the measurement goes.
        # What is measured is that the output sits closer to the class centre
        # than real members of the class do -- which a centroid, a manifold
        # centre or a learned attractor all satisfy. And it cannot be an average
        # for the cells that won once: those hold a single exemplar. So the
        # population is prototype-LIKE in aggregate and literally bimodal
        # underneath. Claim the behaviour, not the identity.
        verdicts.append(
            f"it behaves like a learned class PROTOTYPE: {m['to_prototype']:.3f} "
            f"to its class centre against {m['real_to_prototype']:.3f} for a "
            f"real member -- prototype-like, though not necessarily an average "
            f"(a cell that won once holds one exemplar, not a mean of several)")
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
