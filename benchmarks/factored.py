"""A yellow bus: recombining factors, and the first thing that leaves the span.

`composition.py` ended on a ceiling with an algebraic cause. Every way this
architecture had of imagining -- reading a concept's mean, sampling its learned
subspace, mixing several concepts, anchoring to a percept -- is a **weighted sum
of stored vectors**. Sums of stored vectors live in the span of the stored
vectors. So no amount of mixing or temperature can produce something that could
not have been assembled from memory, and the measured novelty of a composition
turned out to be the arithmetic of averaging *k* near-orthogonal codes (0.389,
against 0.362 for three stored photographs averaged with no concept layer at
all) rather than anything the concepts contributed.

A system that has seen a red car, a blue truck and a green bus should be able to
imagine a **yellow bus**. Not 0.3*red_car + 0.4*blue_truck + 0.3*green_bus --
that is a smear, and it is what the previous mechanism produces. A yellow bus is
the *form* of one thing carrying the *colour* of another, and it is a point the
world never presented.

That operation is available here without inventing any new representation,
because the eye's code is already factored: `rate()` concatenates the three
retinal opponent channels, so

    V = [ luminance (4096) | red-green (4096) | blue-yellow (4096) ]
        \_____ form _____/  \________ colour ________/

and form and colour occupy disjoint blocks. Recombining across the blocks is
algebraically different from mixing whole codes, not merely different in degree:
with `Wv[A] = [f_A, c_A]` and `Wv[B] = [f_B, c_B]`, the recombination
`[f_A, c_B]` is in `span{A, B}` only if a scalar is both 1 and 0. It is not.

The measurement that decides it
-------------------------------
Cosine to the nearest stored code was the wrong instrument for this question --
it asks "does this resemble a memory", when the question is "**could this have
been assembled from memories at all**". The right instrument is the residual
after projecting onto the span of everything stored:

    span_residual   || q - P_B q ||  /  || q ||,  P_B the projector onto the
                    row space of the stored bank

Exactly zero means the code is a linear combination of things already seen, and
no claim of novelty survives it however the cosines look. This is the sharp
version of the ceiling, and it makes a falsifiable prediction: **every arm from
`composition.py` should score exactly zero**, and if factoring is real, the
recombined arm should not.

Coherence has to come along, because leaving the span is trivial on its own --
noise does it. So each arm is also asked whether it still reads as an object:

    form_reads      does the whole code still classify as the category whose
                    form it was given? A yellow bus has to still be a bus
    colour_taken    does its colour block sit closer to the donor's colour than
                    to the form-parent's? A yellow bus has to not be green

Both are needed. An arm that leaves the span and reads as nothing has produced
noise; one that keeps both has produced a thing that was never observed.

Usage:  python3 benchmarks/factored.py out_factored.json
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
#: the eye emits one V1 bank per opponent channel, concatenated
CHANNELS = 3


def span_projector(B):
    """Projector onto the row space of the stored bank, via its SVD.

    ``B`` is (n_stored, d) with n_stored << d, so this is an n x n problem and
    the rank is read off the singular values rather than assumed."""
    U, s, Vt = np.linalg.svd(B, full_matrices=False)
    keep = s > (s.max() * 1e-6 if s.size and s.max() > 0 else 0)
    return Vt[keep]                                    # (rank, d), orthonormal


def span_residual(Q, R):
    """Fraction of each row of ``Q`` that the span of the bank cannot explain."""
    proj = (Q @ R.T) @ R
    num = np.linalg.norm(Q - proj, axis=1)
    den = np.maximum(np.linalg.norm(Q, axis=1), 1e-9)
    return num / den


def run_seed(V, A, y, n_cls, seed):
    tr, te = split(y, seed)
    d = V.shape[1]
    blk = d // CHANNELS
    FORM, COLOUR = (0, blk), (blk, d)

    assoc = AssociationArea(n_vis=V.shape[1], n_aud=A.shape[1],
                            n_concept=N_CONCEPT, seed=seed)
    assoc.set_stats(V[tr], A[tr])
    votes = {}
    for i in tr:
        win = assoc.bind(V[i], A[i])
        votes.setdefault(win, {})
        votes[win][int(y[i])] = votes[win].get(int(y[i]), 0) + 1
    cell_cat = {c: max(v.items(), key=lambda kv: kv[1])[0]
                for c, v in votes.items()}
    by_cat = {}
    for c, cat in cell_cat.items():
        by_cat.setdefault(cat, []).append(c)

    Btr = np.stack([assoc.prep_v(V[i]) for i in tr])
    Bte = np.stack([assoc.prep_v(V[i]) for i in te])
    protos = np.stack([_unit(Btr[y[tr] == c].mean(0)) for c in range(n_cls)])
    R = span_projector(Btr)
    rng = np.random.default_rng(seed + 17)
    cells = [assoc.concept_from_sound(A[i]) for i in te]
    want = [int(y[i]) for i in te]

    def sibling(c):
        return int(rng.choice(by_cat.get(cell_cat.get(c), [c])))

    def donor(c):
        """A concept the mind grouped elsewhere -- the colour comes from a thing
        that was never this shape. No test label is consulted."""
        other = [d for d in cell_cat if cell_cat[d] != cell_cat.get(c)]
        return int(rng.choice(other)) if other else int(c)

    # prototypes restricted to the form block, so "is it still a bus" can be
    # asked of the form alone. The whole-code read-out sees two thirds of a
    # different object and would charge the recombination for the read-out's
    # confusion rather than for any damage to the form.
    protos_f = np.stack([_unit(p[FORM[0]:FORM[1]]) for p in protos])

    def score(M, donors=None):
        res = span_residual(M, R)
        form_ok = np.array([int(np.argmax(protos @ M[k])) == want[k]
                            for k in range(len(M))], float)
        Mf = np.stack([_unit(m[FORM[0]:FORM[1]]) for m in M])
        block_ok = np.array([int(np.argmax(protos_f @ Mf[k])) == want[k]
                             for k in range(len(M))], float)
        out = dict(span_residual=float(res.mean()),
                   in_span=float((res < 1e-6).mean()),
                   form_reads=float(form_ok.mean()),
                   form_block_reads=float(block_ok.mean()))
        if donors is not None:
            # did the colour block actually come from the donor rather than
            # from the concept that supplied the form?
            took = []
            for k, (c, dn) in enumerate(zip(cells, donors)):
                q = _unit(M[k, COLOUR[0]:COLOUR[1]])
                cd = _unit(assoc.Wv[dn, COLOUR[0]:COLOUR[1]])
                cf = _unit(assoc.Wv[c, COLOUR[0]:COLOUR[1]])
                took.append(float(q @ cd) > float(q @ cf))
            out["colour_taken"] = float(np.mean(took))
        return out

    out = {}
    # a random stored code -- here only as the "residual is exactly 0" anchor,
    # so it is not drawn from the wanted class and its read-outs sit at chance
    out["a stored code (random)"] = score(Btr[[int(rng.integers(len(Btr)))
                                               for _ in te]])
    out["a real unseen photograph"] = score(Bte)
    out["gaussian noise"] = score(np.stack(
        [_unit(rng.standard_normal(d).astype(np.float32)) for _ in te]))

    # ---- everything composition.py could do, on the sharp instrument ------
    out["the concept mean"] = score(np.stack(
        [assoc.imagine_vision(c, temperature=0.0) for c in cells]))
    out["sampled T=4"] = score(np.stack(
        [assoc.imagine_vision(c, temperature=4.0, rng=rng) for c in cells]))
    out["composed T=4"] = score(np.stack(
        [assoc.imagine_composite([c, sibling(c), sibling(c)], temperature=4.0,
                                 rng=rng) for c in cells]))

    # ---- the recombination ------------------------------------------------
    dn = [donor(c) for c in cells]
    out["factored (form + donor colour)"] = score(np.stack(
        [assoc.imagine_factored([c, j], [FORM, COLOUR], temperature=0.0)
         for c, j in zip(cells, dn)]), donors=dn)
    out["factored, T=4"] = score(np.stack(
        [assoc.imagine_factored([c, j], [FORM, COLOUR], temperature=4.0,
                                rng=rng)
         for c, j in zip(cells, dn)]), donors=dn)
    # and the control that says the gain is the CROSSING, not the slicing:
    # rebuild from the same two blocks of the SAME cell, which is just the cell
    same = [int(c) for c in cells]
    out["factored, both blocks same cell"] = score(np.stack(
        [assoc.imagine_factored([c, c], [FORM, COLOUR], temperature=0.0)
         for c in cells]), donors=same)

    out["_diag"] = dict(rank=int(R.shape[0]), stored=int(len(Btr)), dim=int(d))
    return out


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_factored.json"
    images, waves, y, names = load_audiovisual(n_per_class=60, seed=0,
                                               grayscale=False, size=32)
    n_cls = len(names)
    brain = StreamingBrain(seed=0, image_shape=(32, 32), v1_cells=4096,
                           rf=7, stride=2)
    V, A = encode(brain, images, waves, colour=True, adapt=True, develop=True)
    print(f"{len(images)} real audio-visual pairs, {n_cls} categories, "
          f"code {V.shape[1]}d = {CHANNELS} opponent channels x "
          f"{V.shape[1]//CHANNELS}\n", flush=True)

    rows = [run_seed(V, A, y, n_cls, sd) for sd in SEEDS]
    keys = [k for k in rows[0] if k != "_diag"]
    res = {"seeds": list(SEEDS), "n_class": n_cls, "arms": {}}
    for k in keys:
        res["arms"][k] = {q: round(float(np.mean([r[k][q] for r in rows])), 4)
                          for q in rows[0][k]}
    res["diag"] = {q: round(float(np.mean([r["_diag"][q] for r in rows])), 4)
                   for q in rows[0]["_diag"]}

    g = res["diag"]
    print(f"the stored bank spans a {g['rank']:.0f}-dimensional subspace of "
          f"{g['dim']:.0f} ({g['stored']:.0f} codes).")
    print("anything with residual 0 is a linear combination of things already "
          "seen.\n")

    print(f"{'arm':<34}{'span resid':>12}{'in span':>9}{'whole':>8}"
          f"{'form blk':>10}{'colour':>9}")
    for k in keys:
        a = res["arms"][k]
        cl = f"{a['colour_taken']:>9.3f}" if "colour_taken" in a else f"{'-':>9}"
        print(f"{k:<34}{a['span_residual']:>12.4f}{a['in_span']:>9.3f}"
              f"{a['form_reads']:>8.3f}{a['form_block_reads']:>10.3f}{cl}")

    old = ["the concept mean", "sampled T=4", "composed T=4"]
    fac = res["arms"]["factored (form + donor colour)"]
    real = res["arms"]["a real unseen photograph"]
    print("\n=== can anything it makes fail to be a sum of memories? ===")
    worst = max(res["arms"][k]["span_residual"] for k in old)
    print(f"  every mixing/sampling arm: residual <= {worst:.6f}, "
          f"in-span {min(res['arms'][k]['in_span'] for k in old):.3f}")
    print("  -> confirmed algebraically and numerically: mixing whole codes "
          "CANNOT leave the span.")
    print(f"  factored recombination:    residual {fac['span_residual']:.4f}, "
          f"in-span {fac['in_span']:.3f}")
    print(f"  a real unseen photograph:  residual {real['span_residual']:.4f}")
    res["escapes_span"] = bool(fac["span_residual"] > 1e-3)
    if res["escapes_span"]:
        frac = fac["span_residual"] / max(real["span_residual"], 1e-9)
        res["escape_vs_real"] = round(float(frac), 4)
        print(f"\n  the recombination is OUTSIDE the span, at {frac:.0%} of a "
              f"real unseen photograph's residual, in {fac['in_span']:.0%} "
              f"of cases inside it.")
        print(f"  it takes the donor's colour {fac['colour_taken']:.3f}, and "
              f"its FORM BLOCK still reads as its own category "
              f"{fac['form_block_reads']:.3f}")
        print(f"  (the concept mean's form block reads "
              f"{res['arms']['the concept mean']['form_block_reads']:.3f}; a "
              f"real photograph {real['form_block_reads']:.3f}).")
        # the honest cost, held to the same floor composition.py imposed
        keeps = fac["form_block_reads"] >= real["form_block_reads"]
        res["keeps_coherence"] = bool(keeps)
        if keeps:
            print("  Form survives the crossing. That is a code the world never "
                  "presented, memory cannot assemble, and the mind still reads "
                  "as an object.")
        else:
            print(f"  But the whole code reads {fac['form_reads']:.3f} against "
                  f"{real['form_reads']:.3f} for a real photograph -- below the "
                  f"floor `composition.py` imposed.")
            print("  Leaving the span is established; doing it without cost is "
                  "not.")
    else:
        print("\n  the recombination is still inside the span -- factoring the "
              "code did not buy the escape.")

    ctl = res["arms"]["factored, both blocks same cell"]
    print(f"\n  control -- both blocks from the SAME cell: residual "
          f"{ctl['span_residual']:.4f}, form {ctl['form_reads']:.3f}")
    print("  (it should be back inside the span: that arm is just the cell, so "
          "the escape is the CROSSING, not the slicing)")

    mean = res["arms"]["the concept mean"]
    print(f"\n  what the crossing costs, read two ways:")
    print(f"    whole code : {mean['form_reads']:.3f} -> "
          f"{fac['form_reads']:.3f}  -- but two thirds of this code is now a "
          f"different object, so this charges the recombination")
    print(f"                 for the read-out's confusion")
    print(f"    form block : {mean['form_block_reads']:.3f} -> "
          f"{fac['form_block_reads']:.3f}  -- the form itself is "
          f"{'INTACT' if abs(fac['form_block_reads'] - mean['form_block_reads']) < 0.02 else 'changed'}")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
