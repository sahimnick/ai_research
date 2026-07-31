"""What has to be true before a constraint solver is worth building.

`inner_world.py` located the gap: every route from an imagining back into the
mind runs through `bind`, which treats what it is handed as an observation, so
an unjudged crossing is replayed as fact and measures the same as a random
sight. The obvious next piece is the judgement -- something that asks whether a
crossing is *possible* before the concepts absorb it.

This measures whether that piece can exist here, before building it, because a
selector can only be as good as the constraint it has to work with.

The judgement, built the project's way
--------------------------------------
:class:`FactorCompatibility` is a Hebbian association between the two factors,
accumulated from the pairs the world presented -- one outer product, online, no
gradients. Blocks that co-occurred drive each other; blocks that never did, do
not. It is read as a compatibility rather than a recall, and it is tried two
ways, because the projection it lives in turns out to matter more than the rule:

    random      random directions per factor -- the cerebellar / mushroom-body
                plan, sparse random connectivity feeding an associative layer
    concepts    the mind's own concept cells, read one factor at a time. The
                reduced space the layer already built, rather than an arbitrary
                one

And the ceiling
---------------
A compatibility over (form, colour) can only work if **form constrains colour**.
The only route from one to the other here is through the category, so the most
any such model could know is how much colour a category determines -- measured
directly as how often a same-category thing's colour is closer than a
different-category thing's. That number is the ceiling, and reporting a model's
score without it would make a weak result look like a weak model.

Both are reported as AUC, on the same held-out data, so they are comparable:

    available   same-category colour closer than different-category colour
    achieved    a real pairing scoring above a crossing

Usage:  python3 benchmarks/constraint.py out_constraint.json
"""
import json
import sys

import numpy as np

from neurobrain.cognition.multimodal import (AssociationArea,
                                             FactorCompatibility, _unit)
from neurobrain.sensing.natural import load_audiovisual
from neurobrain.sensing.streams import StreamingBrain

sys.path.insert(0, "benchmarks")
from real_binding import encode, split                       # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
N_CONCEPT = 256
CHANNELS = 3
RANKS = (16, 64, 256)


def auc(pos, neg):
    """How often a positive outscores a negative. 0.5 is nothing."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return 0.5
    return float((pos[:, None] > neg[None, :]).mean()
                 + 0.5 * (pos[:, None] == neg[None, :]).mean())


class ConceptCompatibility(FactorCompatibility):
    """The same Hebbian rule, projected onto the concept cells rather than noise.

    The layer has already built a reduced representation of this world; using it
    instead of random directions costs nothing and is the more obvious choice
    once asked. Whether it helps is the measurement."""

    def __init__(self, assoc, bounds, **kw):
        live = np.flatnonzero(assoc.wins > 0)
        super().__init__(bounds, rank=max(len(live), 1), seed=0)
        self.P = [np.stack([_unit(assoc.Wv[c, lo:hi]) for c in live])
                  for lo, hi in self.bounds]


def run_seed(V, A, y, n_cls, seed):
    tr, te = split(y, seed)
    d = V.shape[1]
    blk = d // CHANNELS
    bounds = [(0, blk), (blk, d)]

    assoc = AssociationArea(n_vis=d, n_aud=A.shape[1], n_concept=N_CONCEPT,
                            seed=seed)
    assoc.set_stats(V[tr], A[tr])
    for i in tr:
        assoc.bind(V[i], A[i])
    P = {int(i): assoc.prep_v(V[i]) for i in list(tr) + list(te)}
    rng = np.random.default_rng(seed + 23)

    # ---- the ceiling: how much colour does a category actually determine? --
    same, diff = [], []
    for k in te:
        c = int(y[k])
        pool_s = [j for j in tr if int(y[j]) == c]
        pool_d = [j for j in tr if int(y[j]) != c]
        q = _unit(P[int(k)][blk:])
        same.append(float(q @ _unit(P[int(rng.choice(pool_s))][blk:])))
        diff.append(float(q @ _unit(P[int(rng.choice(pool_d))][blk:])))
    out = {"available": auc(same, diff),
           "colour_gap": float(np.mean(same) - np.mean(diff))}

    # ---- the judgement, both projections -----------------------------------
    def crossed(k):
        v = P[int(k)].copy()
        v[blk:] = P[int(rng.choice(tr))][blk:]
        return v

    models = {f"random rank={r}": FactorCompatibility(bounds, rank=r, seed=seed)
              for r in RANKS}
    models["concepts"] = ConceptCompatibility(assoc, bounds)
    for tag, fc in models.items():
        for i in tr:
            fc.observe(P[int(i)])
        real = [fc.score(P[int(k)]) for k in te]
        cross = [fc.score(crossed(k)) for k in te]
        out[tag] = auc(real, cross)
    out["cells"] = int((assoc.wins > 0).sum())

    # ---- the same question of a factorisation that SHOULD be constrained ---
    # Sight and sound are two factors of one concept, and unlike form and
    # colour the world genuinely answers whether they go together: a bark goes
    # with a dog. If a compatibility cannot find that either, the rule is not
    # what is failing.
    nv, na = d, A.shape[1]
    b2 = [(0, nv), (nv, nv + na)]

    def pair(i, j=None):
        return np.concatenate([assoc.prep_v(V[int(i)]),
                               assoc.prep_a(A[int(i if j is None else j)])])

    fc2 = FactorCompatibility(b2, rank=256, seed=seed)
    for i in tr:
        fc2.observe(pair(i))
    real2 = [fc2.score(pair(k)) for k in te]
    bad2 = [fc2.score(pair(k, int(rng.choice(
        [j for j in tr if int(y[j]) != int(y[k])])))) for k in te]
    out["sight x sound"] = auc(real2, bad2)

    # ...and the control that decides between "weak rule" and "weak
    # representation": hand the SAME rule a visual code that actually clusters
    # by category -- the class prototype -- and ask again. Nothing about the
    # rule changes; only whether similar things are similar in what it is given.
    protoV = {c: _unit(np.mean([assoc.prep_v(V[int(i)]) for i in tr
                                if int(y[i]) == c], axis=0))
              for c in range(n_cls)}

    def oracle(i, j=None):
        return np.concatenate([protoV[int(y[int(i)])],
                               assoc.prep_a(A[int(i if j is None else j)])])

    fc3 = FactorCompatibility(b2, rank=256, seed=seed)
    for i in tr:
        fc3.observe(oracle(i))
    real3 = [fc3.score(oracle(k)) for k in te]
    bad3 = [fc3.score(oracle(k, int(rng.choice(
        [j for j in tr if int(y[j]) != int(y[k])])))) for k in te]
    out["sight x sound, clustered sight"] = auc(real3, bad3)
    return out


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_constraint.json"
    images, waves, y, names = load_audiovisual(n_per_class=60, seed=0,
                                               grayscale=False, size=32)
    n_cls = len(names)
    brain = StreamingBrain(seed=0, image_shape=(32, 32), v1_cells=4096,
                           rf=7, stride=2)
    V, A = encode(brain, images, waves, colour=True, adapt=True, develop=True)
    print(f"{len(images)} real pairs, {n_cls} categories, code {V.shape[1]}d "
          f"= form {V.shape[1]//CHANNELS} | colour "
          f"{V.shape[1] - V.shape[1]//CHANNELS}\n", flush=True)

    rows = [run_seed(V, A, y, n_cls, sd) for sd in SEEDS]
    keys = [k for k in rows[0] if k not in ("cells", "colour_gap")]
    res = {"seeds": list(SEEDS),
           "mean": {k: round(float(np.mean([r[k] for r in rows])), 4)
                    for k in keys},
           "sd": {k: round(float(np.std([r[k] for r in rows], ddof=1)), 4)
                  for k in keys},
           "colour_gap": round(float(np.mean([r["colour_gap"] for r in rows])),
                               4),
           "per_seed": rows}

    ceil = res["mean"]["available"]
    print(f"{'':<32}{'AUC':>8}{'sd':>8}   (0.500 is nothing)")
    print(f"{'the CEILING':<32}{ceil:>8.3f}{res['sd']['available']:>8.3f}"
          f"   how much colour a category determines")
    print(f"{'':<22}{'':>8}{'':>8}   (same-category colour gap "
          f"{res['colour_gap']:+.4f})\n")
    # Fraction of the *available* signal captured, not a ratio of AUCs. A model
    # sitting at chance scores 0.509/0.545 = 93% on the ratio and 20% on this,
    # and only the second means anything: both numbers have a floor of 0.5 that
    # has to come out before they are compared.
    head = ceil - 0.5
    OTHER = ("sight x sound", "sight x sound, clustered sight")
    for k in keys:
        if k == "available":
            continue
        v, s = res["mean"][k], res["sd"][k]
        if k in OTHER:
            # a different factorisation: the form->colour ceiling does not
            # govern it, and dividing by it anyway printed "844%"
            print(f"{k:<32}{v:>8.3f}{s:>8.3f}   (other factorisation)")
            continue
        got = (v - 0.5) / head if head > 1e-9 else 0.0
        res.setdefault("captured", {})[k] = round(float(got), 4)
        print(f"{k:<32}{v:>8.3f}{s:>8.3f}   {got:>5.0%} of the available "
              f"signal")

    sxs = res["mean"]["sight x sound"]
    orc = res["mean"]["sight x sound, clustered sight"]
    print(f"\n  the same rule on a factorisation the world DOES constrain:")
    print(f"    sight x sound, as the eye codes it   {sxs:.3f}")
    print(f"    sight x sound, sight that CLUSTERS   {orc:.3f}"
          f"   <- the rule, given similar things that are similar")
    res["rule_is_adequate"] = bool(orc > 0.75)

    best = max((k for k in keys if k not in
                ("available", "sight x sound", "sight x sound, clustered sight")),
               key=lambda k: res["mean"][k])
    res["best"] = best
    print(f"\n=== can a constraint solver be built on these factors? ===")
    print(f"  the most any (form -> colour) model could know is AUC "
          f"{ceil:.3f}; the best one here reaches {res['mean'][best]:.3f} "
          f"({best}) -- {res['captured'][best]:.0%} of the available signal.")
    reachable = ceil - 0.5
    res["headroom"] = round(float(reachable), 4)
    if reachable < 0.10:
        print(f"\n  The model is not the weak part -- the CONSTRAINT is. "
              f"Form carries only {reachable:.3f} of AUC above chance about "
              f"colour,")
        print("  so a plausibility model over these two factors has almost "
              "nothing to say, and 'a yellow bus' is not in fact implausible in")
        print("  a world of cars and buses of every colour.")
        print("\n  This is a property of the FACTORISATION, not of the rule. "
              "The eye handed form and colour over already separated, and they "
              "are")
        print("  close to independent -- which is exactly why crossing them "
              "produces a coherent object (form survives intact) and exactly "
              "why")
        print("  nothing objects to the crossing. **The same independence that "
              "makes factored recombination work makes it unconstrainable.**")
        print("  A constraint solver needs factors that constrain each other; "
              "these were chosen for the opposite property.")
        if res["rule_is_adequate"]:
            print(f"\n  And the rule is not the limit anywhere here. Given a "
                  f"visual code in which similar things are actually similar, "
                  f"the SAME Hebbian outer")
            print(f"  product reaches {orc:.3f} on sight x sound, against "
                  f"{sxs:.3f} on the code the eye actually produces. The "
                  f"judgement machinery works; what it is")
            print("  being handed does not. That is the same upstream defect "
                  "that has blocked every downstream result in section 7.8 -- "
                  "the eye names")
            print("  photographs at 0.134 against 0.559 for digits, and a "
                  "representation that does not cluster cannot support a "
                  "generative model, a")
            print("  constraint solver, or a causal model, however each is "
                  "built.")
    else:
        print(f"\n  There is {reachable:.3f} of AUC to work with, and the best "
              f"model captures {(res['mean'][best]-0.5)/reachable:.0%} of it. "
              f"A selector is worth building on this factorisation.")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
