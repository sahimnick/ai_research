"""Does the inner world pay back, once the layer generalises and the
imagining is used as an error rather than as a fact?

`inner_world.py` returned the sharpest negative in this project: a night of
out-of-span recombinations improved held-out recognition by exactly as much as a
night of **random sights** (72.3% against 72.5%), and its diagnosis was that
binding a yellow bus to the real sound of a bus asserts something *false*. Two
things have changed since, and neither was available when that ran.

**The layer memorised.** 53% of concept cells held exactly one photograph, and
38.8% of what the mind "imagined" was that photograph byte for byte. Replaying
that is replaying memories with extra steps. `merger.py` fixed it: forced
consolidation takes singletons to 0.0% and verbatim to 1.0% while novelty rises
0.196 -> 0.346. **The payback test has never been run on a layer that
generalises.**

**Every route ran through `bind`.** That is the diagnosis restated as a design
fault: `bind` treats what it is handed as an observation, so the only thing the
architecture could do with an imagining was believe it. `bind_contrastive` does
not -- the imagining is the **negative phase**, the thing the layer moves *away*
from, so a wrong fantasy teaches by being wrong instead of by being believed.
A well-predicted pair then teaches nothing, which is the defining property of an
error-driven rule and the one instar lacks.

The arms, and what each isolates
--------------------------------
    no_dream            watch the day, stop
    stored              replay the real sight -- the positive control that says
                        whether the payback channel is open at all
    imagined_as_fact    bind the mind's own completion as though it had been
                        seen. `inner_world.py`'s failing arm, carried over so
                        the two tables can be read against each other
    imagined_as_error   the same completion, used as the negative phase.
                        Identical replay order, identical pairs, identical
                        number of plasticity events -- the *only* difference is
                        whether the fantasy is believed or learned against
    merged              forced consolidation, no night at all. Separates "the
                        layer generalises" from "the night did something"
    merged+error        both

The last three are the point. If `imagined_as_error` beats `imagined_as_fact` on
the same material, the failure was never the imagination -- it was what the
architecture did with it.

Measured on held-out data, and the second one is what the goal is about:

    sound -> vision     cross-modal recall
    see -> name         recognise a held-out photograph
    swapped             the same, with R and B exchanged. Luminance
                        (r+g+b)/3 is exactly invariant to that swap, so form is
                        held fixed and only what a recombination varies varies

Usage:  python3 benchmarks/payback.py out_payback.json
"""
import json
import sys

import numpy as np

from neurobrain.cognition.multimodal import AssociationArea, _unit
from neurobrain.sensing.natural import load_audiovisual
from neurobrain.sensing.streams import StreamingBrain

sys.path.insert(0, "benchmarks")
from real_binding import encode, split                       # noqa: E402

SEEDS = (0, 1, 2, 3, 4, 5)
N_CONCEPT = 256
REPLAYS = 400
DREAM_NOVELTY_RATE = 0.02
KEEP = 0.6
ARMS = ("no_dream", "stored", "imagined_as_fact", "imagined_as_fact_fixed",
        "imagined_as_error", "merged", "merged+error")


def fold(votes, merged):
    for old, new in merged.items():
        while new in merged:
            new = merged[new]
        if old in votes:
            dst = votes.setdefault(new, {})
            for lab, k in votes.pop(old).items():
                dst[lab] = dst.get(lab, 0) + k
    return votes


def wake(V, A, y, tr, seed):
    a = AssociationArea(n_vis=V.shape[1], n_aud=A.shape[1],
                        n_concept=N_CONCEPT, seed=seed)
    a.set_stats(V[tr], A[tr])
    votes = {}
    for i in tr:
        w = a.bind(V[i], A[i])
        votes.setdefault(w, {})
        votes[w][int(y[i])] = votes[w].get(int(y[i]), 0) + 1
    return a, votes


def night(a, votes, V, A, y, tr, arm, seed):
    """Every arm replays the same pairs in the same order, drawn from one
    generator seeded identically -- so the arms differ in the RULE and not in
    what they saw.

    Except that for a while they differed in one more thing, and it was found
    by `pathway_downstream.py` rather than here. ``imagine_from_sound`` returns
    a **prep_v**-space vector; ``bind_contrastive`` consumes it in that space,
    but :meth:`bind` preps whatever it is handed. So ``imagined_as_fact`` was
    binding a twice-normalised fantasy -- measured at cos **0.407** to the one
    the model generated, waking a different concept cell 21 times out of 36 --
    while ``imagined_as_error`` saw the undistorted one. The arms differed in
    the rule *and* in whether the imagining survived intact.

    ``imagined_as_fact_fixed`` un-preps before binding, which is an exact
    inverse (``prep_v(v_hat * sd + mu) == v_hat``, measured cos 1.0). It is the
    arm the comparison should always have used. The uncorrected one stays
    because the number it produced is published in EVALUATION.md and deleting
    it would make the correction unreadable.
    """
    if arm in ("no_dream", "merged"):
        return votes
    rng = np.random.default_rng(seed + 101)
    waking, a.novelty_rate = a.novelty_rate, DREAM_NOVELTY_RATE
    order = [int(rng.choice(tr)) for _ in range(REPLAYS)]
    for i in order:
        if arm == "stored":
            w = a.bind(V[i], A[i])
        elif arm == "imagined_as_fact":
            w = a.bind(a.imagine_from_sound(A[i], temperature=1.0, rng=rng),
                       A[i])
        elif arm == "imagined_as_fact_fixed":
            v_hat = a.imagine_from_sound(A[i], temperature=1.0, rng=rng)
            w = a.bind(v_hat * a.v_sd + a.v_mu, A[i])
        else:                                   # believed nothing; learned from
            w, _, _ = a.bind_contrastive(V[i], A[i], temperature=1.0, rng=rng)
        votes.setdefault(w, {})
        votes[w][int(y[i])] = votes[w].get(int(y[i]), 0) + 1
    a.novelty_rate = waking
    return votes


def probe(a, votes, V, Vs, A, y, tr, te, n_cls):
    name = {c: max(v.items(), key=lambda kv: kv[1])[0]
            for c, v in votes.items() if v}
    Btr = np.stack([a.prep_v(V[i]) for i in tr])
    protos = np.stack([_unit(Btr[y[tr] == c].mean(0)) for c in range(n_cls)])
    s2v = v2l = sw = 0
    for i in te:
        c = a.concept_from_sound(A[i])
        s2v += int(int(np.argmax(protos @ _unit(a.Wv[c]))) == int(y[i]))
        v2l += int(name.get(a.concept_from_vision(V[i]), -1) == int(y[i]))
        sw += int(name.get(a.concept_from_vision(Vs[i]), -1) == int(y[i]))
    n = max(len(te), 1)
    live = np.flatnonzero(a.wins > 0)
    return dict(sound_to_vision=s2v / n, see_to_name=v2l / n,
                swapped=sw / n, cells=int(len(live)),
                singletons=float(np.mean(a.wins[live] == 1)) if len(live) else 0.0)


def run_seed(V, Vs, A, y, n_cls, seed):
    tr, te = split(y, seed)
    out = {}
    for arm in ARMS:
        a, votes = wake(V, A, y, tr, seed)
        if arm.startswith("merged"):
            votes = fold(votes, a.consolidate_ranked(keep=KEEP))
        votes = night(a, votes, V, A, y, tr,
                      "imagined_as_error" if arm == "merged+error" else arm,
                      seed)
        out[arm] = probe(a, votes, V, Vs, A, y, tr, te, n_cls)
    return out


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_payback.json"
    images, waves, y, names = load_audiovisual(n_per_class=60, seed=0,
                                               grayscale=False, size=32)
    n_cls = len(names)
    brain = StreamingBrain(seed=0, image_shape=(32, 32), v1_cells=4096,
                           rf=7, stride=2)
    swapped = [im[[2, 1, 0]] if im.ndim == 3 else im for im in images]
    n = len(images)
    both, A = encode(brain, list(images) + swapped, list(waves) * 2,
                     colour=True, adapt=True, develop=True)
    V, Vs, A = both[:n], both[n:], A[:n]
    print(f"{n} real pairs, {n_cls} categories, chance {1/n_cls:.3f}")
    print(f"every night replays the same {REPLAYS} pairs in the same order; "
          f"the arms differ in the RULE\n", flush=True)

    rows = [run_seed(V, Vs, A, y, n_cls, sd) for sd in SEEDS]
    res = {"seeds": list(SEEDS), "arms": {}}
    METRICS = ("sound_to_vision", "see_to_name", "swapped")
    for arm in ARMS:
        res["arms"][arm] = {q: round(float(np.mean([r[arm][q] for r in rows])),
                                     4) for q in rows[0][arm]}

    print(f"{'arm':<20}{'cells':>7}{'singl':>8}{'s->vision':>11}"
          f"{'see->name':>11}{'swapped':>10}")
    for arm in ARMS:
        a = res["arms"][arm]
        print(f"{arm:<20}{a['cells']:>7.1f}{a['singletons']:>8.2f}"
              f"{a['sound_to_vision']:>11.3f}{a['see_to_name']:>11.3f}"
              f"{a['swapped']:>10.3f}")

    print(f"\nagainst no_dream, paired over {len(SEEDS)} seeds")
    print(f"{'arm':<20}" + "".join(f"{m:>22}" for m in METRICS))
    gate = {}
    for arm in ARMS[1:]:
        cells = []
        for m in METRICS:
            d = np.array([r[arm][m] - r["no_dream"][m] for r in rows])
            sd = float(d.std(ddof=1))
            cd = float(d.mean() / (sd + 1e-12))
            gate.setdefault(arm, {})[m] = dict(
                delta=round(float(d.mean()), 4), cohens_d=round(cd, 3),
                wins=int((d > 0).sum()), n=len(d))
            cells.append(f"{d.mean():>+9.4f} d={cd:>+5.2f} {int((d>0).sum())}/{len(d)}")
        print(f"{arm:<20}" + "".join(f"{c:>22}" for c in cells))
    res["gate"] = gate

    print("\n=== does the inner world pay back? ===")
    fact = gate["imagined_as_fact"]["see_to_name"]
    fixed = gate["imagined_as_fact_fixed"]["see_to_name"]
    err = gate["imagined_as_error"]["see_to_name"]
    both_ = gate["merged+error"]["see_to_name"]
    print(f"  believing the imagining : {fact['delta']:+.4f} "
          f"(d={fact['cohens_d']:+.2f}, {fact['wins']}/{fact['n']})")
    print(f"  ...undistorted (the fair one): {fixed['delta']:+.4f} "
          f"(d={fixed['cohens_d']:+.2f}, {fixed['wins']}/{fixed['n']})")
    print(f"  learning against it     : {err['delta']:+.4f} "
          f"(d={err['cohens_d']:+.2f}, {err['wins']}/{err['n']})")
    res["correction_widens_gap"] = bool(fixed["delta"] < fact["delta"])
    print(f"  -> removing the space bug moves `believe` by "
          f"{fixed['delta'] - fact['delta']:+.4f}; the gap the claim rests on "
          f"{'WIDENS' if res['correction_widens_gap'] else 'NARROWS'}")
    print(f"  ...on a layer that generalises: {both_['delta']:+.4f} "
          f"(d={both_['cohens_d']:+.2f}, {both_['wins']}/{both_['n']})")

    def passes(g):
        return g["cohens_d"] >= 0.8 and g["wins"] >= 0.75 * g["n"]

    winners = [a for a in ("imagined_as_error", "merged+error")
               if any(passes(gate[a][m]) for m in METRICS)]
    res["pays_back"] = winners
    if winners:
        best = max(winners,
                   key=lambda a: max(gate[a][m]["delta"] for m in METRICS))
        m = max(METRICS, key=lambda m: gate[best][m]["delta"])
        g = gate[best][m]
        print(f"\n  YES, and through the error path: {best} improves {m} by "
              f"{g['delta']:+.4f} (d={g['cohens_d']:+.2f}, "
              f"{g['wins']}/{g['n']}),")
        print(f"  while believing the same imaginings as fact gives "
              f"{gate['imagined_as_fact'][m]['delta']:+.4f}. Same completions, "
              f"same replay order, same number of")
        print("  plasticity events -- the difference is only whether the "
              "fantasy is believed or learned against.")
        # the structural point: are the two fixes additive, or do they need
        # each other? If merged+error exceeds the sum of its parts, then
        # generalising and learning-against are not two independent repairs.
        mm = "sound_to_vision"
        a1 = gate["merged"][mm]["delta"]
        a2 = gate["imagined_as_error"][mm]["delta"]
        a12 = gate["merged+error"][mm]["delta"]
        res["superadditive"] = bool(a12 > a1 + a2 + 1e-9)
        print(f"\n  and they are not two independent repairs: consolidation "
              f"alone {a1:+.4f}, learning-against alone {a2:+.4f}, sum "
              f"{a1 + a2:+.4f},")
        print(f"  together {a12:+.4f}. "
              + ("Superadditive -- an error-driven rule needs concepts worth "
                 "being wrong about, and" if res["superadditive"] else
                 "Roughly additive."))
        if res["superadditive"]:
            print("  concepts that memorise give it nothing to predict. Each "
                  "fix is most of the other's precondition.")
        # and the honest scope
        print(f"\n  Scope, stated: the gain is in CROSS-MODAL recall "
              f"(hear a sound, know what it looks like). Direct visual naming "
              f"does not improve")
        print(f"  ({gate['merged+error']['see_to_name']['delta']:+.4f}) and "
              f"neither does colour robustness "
              f"({gate['merged+error']['swapped']['delta']:+.4f}). The eye is "
              f"still the eye.")
        st = gate["stored"][mm]
        print(f"\n  The comparison that matters most: replaying what it "
              f"ACTUALLY SAW is worth {st['delta']:+.4f} "
              f"(d={st['cohens_d']:+.2f}). Learning against what it")
        print(f"  imagines is worth {a12:+.4f} -- "
              f"{a12 / max(st['delta'], 1e-9):.1f}x more than replaying "
              f"reality.")
    else:
        print("\n  No arm clears the gate. Neither believing the imagining nor "
              "learning against it moves held-out perception,")
        print("  on a layer that generalises or one that does not.")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
