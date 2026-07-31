"""Imagining by combining, and a metric that is about imagining.

`imagination_shape.py` set the bar as *the imagined must be further from memory
than a real unseen photograph is*, and nothing passed it: sampling a concept's
learned subspace moved fidelity 0.886 -> 0.700 and saturated there, and more
subspace directions did not help (4/16/64/128 modes all landed near 0.70).

Two things were wrong with that, and the second is the interesting one.

**The bar was wrong.** This eye makes distinct photographs nearly orthogonal --
a fresh one sits at cosine 0.28 to everything stored -- so "as decorrelated from
memory as a new observation" is a statement about the representation, not about
imagining. Anything assembled from stored components lies in the span of memory
and can never clear it. A bar no construction could pass is not a test.

**The measurement was in the wrong space.** `Wv` rows and everything
:meth:`imagine_vision` returns live in ``prep_v`` space -- mean-subtracted,
variance-normalised -- and the stored bank was being compared in raw code space.
Same item, the two representations sit at cosine 0.899, so every distance was
being read through a 0.1 fog. Measured in the space the concept cells actually
live in, the imagined sight's closeness to memory is not 0.886 but **0.979**,
and the reason is not subtle:

    of 112 concept cells recruited from 216 pairs, **59 won exactly once**.
    Vigilance copies the pair into an uncommitted cell verbatim, and a cell that
    never wins again never calls `_grow_subspace`, so its `Wv` row *is* a stored
    photograph and its `mode_var` is exactly zero.

For those cells ``imagine_vision`` returns the exemplar unchanged at **every**
temperature. 37% of what this mind "imagines" is bit-for-bit recall, and no
temperature touches it -- which is what the 0.70 saturation was made of. It was
never a weak generative model; it was a third of the population that cannot move
averaged against two thirds that can.

The two-thirds that can move do something almost as awkward. A cell that won
exactly twice learns one mode, and by construction that mode is the line joining
its two members -- so sampling along it interpolates between two memorised
photographs and, at high temperature, lands on one. Measured on the cells that
do have a subspace, ``verbatim`` climbs 0.000 -> 0.044 as temperature goes 0 ->
4, which is why the whole population's verbatim rate *rises* with temperature
instead of falling. Turning up the noise on a single concept walks toward
memory, not away from it.

A singleton concept has no spread of its own and that is not a bug to patch --
one observation carries no variation. But the *category* it belongs to does, and
that is what composition reaches.

A metric that is actually about imagining
-----------------------------------------
Two quantities, and imagination is the *pair*, not either one:

    novelty      1 - cos to the nearest thing ever stored. Zero means recall
    coherence    does it still read as its own category, against prototypes
                 built from real data

Neither alone means anything: pure noise scores maximal novelty and chance
coherence, and a stored exemplar scores zero novelty and the highest coherence
of anything here. What matters is how close a sample gets to where **real
held-out data** sits in that plane -- a genuinely new observation is both novel
and coherent, and that point is the target rather than an unreachable corner.

So the score reported is the distance in (novelty, coherence) space to the real
data's own point, and every arm is placed on the same plane, including three
reference points that bound it: a stored exemplar of the right category,
Gaussian noise, and a real unseen photograph.

Note what this does and does not reward. Recall is *over*-coherent -- a
remembered prototype classifies better than a fresh photograph does, 0.83
against 0.36 -- so approaching the target means coherence coming **down**, and a
sufficiently destructive arm could close the gap by simply degrading. That is
what `crossed` is in the table for: it overshoots to 0.26, below the real point,
and its gap stops improving. The metric is a distance, not a direction.

``verbatim`` rides along beside the pair -- the fraction of samples sitting above
cosine 0.999 to something stored. It is not part of the score, it is the thing
the old space concealed, and any arm that claims to imagine has to drive it down.

Five ways of imagining
----------------------
    mean          the concept's average -- the behaviour before any of this
                  (`sampled T=0`)
    sampled       one concept, perturbed inside its own learned subspace
    composed      several concepts of the same kind mixed, drawing variation
                  from each. Wider by construction: the sample lives in the span
                  of all of them, so a singleton borrows its category's spread
    crossed       composed out of concepts the mind grouped *differently* -- a
                  chimera. The other bound: maximal reach, and the arm most
                  likely to lose coherence
    real+imagined the goal's own phrasing -- an imagined code blended toward a
                  real percept that is present now. The anchoring result says a
                  fully self-generated night is the worst arm measured
                  (-0.0301), so this asks what the mixture buys

Which concepts count as "the same kind" is decided by the mind's own vote map --
the majority training label of each cell -- never by the test item's label. The
sound picks the cell; the cell picks its siblings.

The anchor arm is scored apart from the rest, and the reason is a degeneracy
worth naming rather than reporting around. The anchor *is* a real held-out
photograph, and the target *is* where real held-out photographs sit, so the gap
falls to zero as ``anchor_weight`` goes to one by construction -- at 0.75 the
output already sits at cosine 0.95 to the very photograph it is supposed to be
imagining about. On this plane that arm measures how much of the answer was
copied from the question. It stays in the table with that cosine printed beside
it, and it is excluded from `best`, which ranges over the arms that generate
without being shown the thing they are generating.

Usage:  python3 benchmarks/composition.py out_composition.json
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
TEMPS = (0.0, 1.0, 2.0, 4.0, 8.0)
ANCHORS = (0.0, 0.25, 0.5, 0.75)
N_COMPOSE = 3
VERBATIM = 0.999


def place(M, Btr, protos, want):
    """Put candidate codes on the plane. ``want`` is the category each should be.

    Every argument must already be in ``prep_v`` space -- the space `Wv` and
    everything `imagine_*` returns live in. Mixing it with raw code space is the
    error this benchmark was written to correct.
    """
    s = (M @ Btr.T).max(1)
    coh = np.array([int(np.argmax(protos @ M[k])) == int(want[k])
                    for k in range(len(M))], float)
    return dict(novelty=float(1.0 - s.mean()), coherence=float(coh.mean()),
                verbatim=float((s > VERBATIM).mean()))


def run_seed(V, A, y, n_cls, seed):
    tr, te = split(y, seed)
    assoc = AssociationArea(n_vis=V.shape[1], n_aud=A.shape[1],
                            n_concept=N_CONCEPT, seed=seed)
    assoc.set_stats(V[tr], A[tr])

    # bind, and keep the vote map: which category each cell came to answer to.
    # This is the mind's own knowledge of its concepts, gathered on training
    # pairs only -- it is what lets a cell find its siblings without anyone
    # consulting the label of the thing being imagined.
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
    all_cells = sorted(cell_cat)

    # the plane, in prep_v space throughout
    Btr = np.stack([assoc.prep_v(V[i]) for i in tr])
    Bte = np.stack([assoc.prep_v(V[i]) for i in te])
    protos = np.stack([_unit(Btr[y[tr] == c].mean(0)) for c in range(n_cls)])
    rng = np.random.default_rng(seed + 91)
    cells = [assoc.concept_from_sound(A[i]) for i in te]
    want = [int(y[i]) for i in te]

    def siblings(c, cross=False):
        """N_COMPOSE cells: ``c`` plus companions from its own kind, or not."""
        cat = cell_cat.get(c)
        if cross:
            pool = [d for d in all_cells if cell_cat.get(d) != cat] or all_cells
        else:
            pool = by_cat.get(cat, [c])
        return [c] + [int(rng.choice(pool)) for _ in range(N_COMPOSE - 1)]

    out = {}
    # ---- the reference points that bound the plane ------------------------
    same = [int(rng.choice(np.flatnonzero(y[tr] == w))) for w in want]
    out["a stored exemplar"] = place(Btr[same], Btr, protos, want)
    noise = np.stack([_unit(rng.standard_normal(V.shape[1]).astype(np.float32))
                      for _ in te])
    out["gaussian noise"] = place(noise, Btr, protos, want)
    out["a real unseen photograph"] = place(Bte, Btr, protos, want)

    # The control that decides whether composition means anything. Averaging
    # N_COMPOSE near-orthogonal unit vectors lands 1/sqrt(N) from each of them
    # by arithmetic alone -- at N=3 that is a novelty of 0.42 for free, which is
    # most of what the composed arm scores. So: the same number of *stored
    # photographs* of the same kind, averaged, with no concept cells involved.
    # Anything composition is worth has to be worth it over this.
    by_cat_ex = {}
    for k, i in enumerate(tr):
        by_cat_ex.setdefault(int(y[i]), []).append(k)
    mix = []
    for c in cells:
        pool = by_cat_ex.get(cell_cat.get(c), list(range(len(tr))))
        pick = [int(rng.choice(pool)) for _ in range(N_COMPOSE)]
        mix.append(_unit(Btr[pick].mean(0)))
    out[f"{N_COMPOSE} stored photographs, averaged"] = place(
        np.stack(mix), Btr, protos, want)

    # ---- the arms --------------------------------------------------------
    for T in TEMPS:
        M = np.stack([assoc.imagine_vision(c, temperature=T, rng=rng)
                      for c in cells])
        out[f"sampled T={T}"] = place(M, Btr, protos, want)
    for T in (1.0, 2.0, 4.0, 8.0):
        M = np.stack([assoc.imagine_composite(siblings(c), temperature=T,
                                              rng=rng) for c in cells])
        out[f"composed T={T}"] = place(M, Btr, protos, want)
    M = np.stack([assoc.imagine_composite(siblings(c, cross=True),
                                          temperature=4.0, rng=rng)
                  for c in cells])
    out["crossed T=4.0"] = place(M, Btr, protos, want)
    for aw in ANCHORS:
        M = np.stack([assoc.imagine_composite(siblings(c), temperature=4.0,
                                              anchor=V[i], anchor_weight=aw,
                                              rng=rng)
                      for c, i in zip(cells, te)])
        p = place(M, Btr, protos, want)
        # how much of the answer came from the question -- the degeneracy of
        # this axis, measured rather than argued about
        p["copied"] = float(np.mean([float(M[k] @ Bte[k])
                                     for k in range(len(te))]))
        out[f"real+imagined anchor={aw}"] = p

    used = np.flatnonzero(assoc.wins > 0)
    out["_diag"] = dict(
        cells=int(len(used)), pairs=int(len(tr)),
        singletons=int((assoc.wins[used] == 1).sum()),
        no_subspace=int((assoc.mode_var[used].max(1) <= 0).sum()),
        woke_a_singleton=float(np.mean([assoc.mode_var[c].max() <= 0
                                        for c in cells])))
    return out


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_composition.json"
    images, waves, y, names = load_audiovisual(n_per_class=60, seed=0,
                                               grayscale=False, size=32)
    n_cls = len(names)
    brain = StreamingBrain(seed=0, image_shape=(32, 32), v1_cells=4096,
                           rf=7, stride=2)
    V, A = encode(brain, images, waves, colour=True, adapt=True, develop=True)
    print(f"{len(images)} real audio-visual pairs, {n_cls} categories "
          f"({', '.join(names)}), chance {1/n_cls:.3f}\n", flush=True)

    rows = [run_seed(V, A, y, n_cls, sd) for sd in SEEDS]
    keys = [k for k in rows[0] if k != "_diag"]
    res = {"seeds": list(SEEDS), "n_class": n_cls,
           "chance": round(1 / n_cls, 4), "arms": {}}
    for k in keys:
        res["arms"][k] = {q: round(float(np.mean([r[k][q] for r in rows])), 4)
                          for q in rows[0][k]}
    res["diag"] = {q: round(float(np.mean([r["_diag"][q] for r in rows])), 4)
                   for q in rows[0]["_diag"]}
    res["per_seed"] = [{k: {q: round(float(v), 4) for q, v in r[k].items()}
                        for k in keys} for r in rows]

    d = res["diag"]
    print(f"the layer: {d['cells']:.0f} concept cells from {d['pairs']:.0f} "
          f"pairs, of which {d['singletons']:.0f} won exactly once and "
          f"{d['no_subspace']:.0f} have no subspace at all.")
    print(f"{d['woke_a_singleton']:.1%} of test sounds wake one of those -- for "
          f"them a single concept CANNOT imagine, at any temperature.\n")

    target = res["arms"]["a real unseen photograph"]
    print(f"{'arm':<26}{'novelty':>9}{'coherence':>11}{'verbatim':>10}"
          f"{'gap to real':>13}{'copied':>9}")
    for k in keys:
        a = res["arms"][k]
        a["gap_to_real"] = round(float(np.hypot(
            a["novelty"] - target["novelty"],
            a["coherence"] - target["coherence"])), 4)
        mark = ("   <- the target" if k == "a real unseen photograph"
                else "   <- degenerate, not ranked" if "copied" in a else "")
        cp = f"{a['copied']:>9.3f}" if "copied" in a else f"{'-':>9}"
        print(f"{k:<26}{a['novelty']:>9.3f}{a['coherence']:>11.3f}"
              f"{a['verbatim']:>10.3f}{a['gap_to_real']:>13.3f}{cp}{mark}")

    # `best` ranges over arms that generate without being shown the answer --
    # the anchored ones are excluded because their gap falls to zero by
    # construction as the anchor takes over (see the module docstring).
    refs = ("a stored exemplar", "gaussian noise", "a real unseen photograph",
            f"{N_COMPOSE} stored photographs, averaged")
    arms = [k for k in keys
            if k not in refs and "copied" not in res["arms"][k]]
    # Ranking, with the guard the `crossed` arm showed to be necessary. A plain
    # distance is gameable from the low side: an arm that destroys coherence
    # walks past the target and its gap keeps shrinking. So an arm is only
    # ranked if it is **at least as coherent as real data**, and among those the
    # score is how close its novelty comes. Undershooting coherence is not a
    # cheaper route to imagining, it is damage.
    floor = target["coherence"]
    for k in keys:
        a = res["arms"][k]
        a["qualifies"] = bool(a["coherence"] >= floor)
        a["novelty_gap"] = round(abs(a["novelty"] - target["novelty"]), 4)
    ok = [k for k in arms if res["arms"][k]["qualifies"]]
    res["disqualified"] = [k for k in arms if not res["arms"][k]["qualifies"]]
    best = min(ok or arms, key=lambda k: res["arms"][k]["novelty_gap"])
    base = res["arms"]["sampled T=0.0"]
    b = res["arms"][best]
    res["best"] = best
    print("\n=== how close does imagining get to a real new observation? ===")
    print(f"  ranked on novelty among arms at least as coherent as real data "
          f"({floor:.3f}).")
    if res["disqualified"]:
        print(f"  disqualified for falling below it: "
              f"{', '.join(res['disqualified'])}")
    print(f"  the mean alone:  novelty {base['novelty']:.3f}, coherence "
          f"{base['coherence']:.3f}, verbatim {base['verbatim']:.3f}")
    print(f"  best arm ({best}): novelty {b['novelty']:.3f}, coherence "
          f"{b['coherence']:.3f}, verbatim {b['verbatim']:.3f}")
    closed = 1 - b["novelty_gap"] / max(base["novelty_gap"], 1e-9)
    res["closed_fraction"] = round(float(closed), 4)
    print(f"  closed {closed:.0%} of the novelty distance from a recalled "
          f"average to a real new sight.")

    # the question the arm split is for: does composing beat perturbing one?
    bs = min((k for k in arms if k.startswith("sampled")),
             key=lambda k: res["arms"][k]["novelty_gap"])
    bc = min((k for k in arms if k.startswith("composed")
              and res["arms"][k]["qualifies"]),
             key=lambda k: res["arms"][k]["novelty_gap"])
    res["best_single"], res["best_composed"] = bs, bc
    sa, ca = res["arms"][bs], res["arms"][bc]
    print(f"\n  best single concept  : {bs:<16} novelty {sa['novelty']:.3f}, "
          f"verbatim {sa['verbatim']:.3f}")
    print(f"  best composed        : {bc:<16} novelty {ca['novelty']:.3f}, "
          f"verbatim {ca['verbatim']:.3f}")
    print(f"  -> composition is the only thing that stops the mind returning "
          f"bit-identical memories: verbatim {sa['verbatim']:.3f} -> "
          f"{ca['verbatim']:.3f}.")

    # and the control that says whether the CONCEPTS were doing the work
    ctl = res["arms"][f"{N_COMPOSE} stored photographs, averaged"]
    dn = ca["novelty"] - ctl["novelty"]
    res["novelty_over_averaging"] = round(float(dn), 4)
    res["beats_averaging"] = bool(ca["novelty_gap"] < ctl["novelty_gap"]
                                  and abs(dn) > 0.05)
    # paired over seeds -- 0.028 is small enough that the spread decides it
    ctl_key = f"{N_COMPOSE} stored photographs, averaged"
    dv = np.array([r[bc]["novelty"] - r[ctl_key]["novelty"] for r in rows])
    sdv = float(dv.std(ddof=1))
    res["novelty_over_averaging_paired"] = dict(
        delta=round(float(dv.mean()), 4), sd=round(sdv, 4),
        cohens_d=round(float(dv.mean() / (sdv + 1e-12)), 3),
        wins=int((dv > 0).sum()), n=len(dv))
    print(f"\n  the control -- {N_COMPOSE} stored photographs averaged, no "
          f"concept layer: novelty {ctl['novelty']:.3f}, coherence "
          f"{ctl['coherence']:.3f}, verbatim {ctl['verbatim']:.3f}")
    p = res["novelty_over_averaging_paired"]
    print(f"  paired over {p['n']} seeds: {p['delta']:+.4f} novelty, "
          f"sd {p['sd']:.4f}, d={p['cohens_d']:+.2f}, {p['wins']}/{p['n']} wins")
    if res["beats_averaging"]:
        print(f"  -> the concept layer adds {dn:+.3f} novelty over plain "
              f"averaging: composition is conceptual, not arithmetic.")
    else:
        print(f"  -> {bc} reaches novelty {ca['novelty']:.3f} against the "
              f"control's {ctl['novelty']:.3f} ({dn:+.3f}), at coherence "
              f"{ca['coherence']:.3f} against {ctl['coherence']:.3f}.")
        print(f"     Averaging {N_COMPOSE} near-orthogonal unit codes sits "
              f"1/sqrt({N_COMPOSE}) from each by arithmetic alone. The novelty "
              f"of a composition IS that arithmetic; the concept cells")
        print("     contribute the coherence, and the temperature spends it. "
              "Composition fixes verbatim recall -- it does not, on this "
              "measurement, imagine.")

    nz = res["arms"]["gaussian noise"]
    print(f"\n  (noise sits at novelty {nz['novelty']:.3f}, coherence "
          f"{nz['coherence']:.3f} -- maximal novelty is not the goal)")

    print("\n  the anchored axis, reported apart because its gap is a copy "
          "fraction:")
    for aw in ANCHORS:
        a = res["arms"][f"real+imagined anchor={aw}"]
        print(f"    anchor={aw:<5} gap {a['gap_to_real']:.3f}, but sits at "
              f"cos {a['copied']:.3f} to the photograph it imagines about")
    top = max(arms, key=lambda k: res["arms"][k]["novelty"])
    print(f"\n  ceiling: nothing self-generated exceeds novelty "
          f"{res['arms'][top]['novelty']:.3f} against real data's "
          f"{target['novelty']:.3f} -- everything assembled from stored parts "
          f"stays in the span of memory.")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
