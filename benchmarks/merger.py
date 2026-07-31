"""Forced consolidation: does making the layer generalise fix the imagining?

The single fact behind almost every negative result in this project is that the
concept layer **memorises**. 52% of its cells hold exactly one photograph, which
is why 36.8% of what it "imagines" is that photograph byte for byte at every
temperature, and why the panel-2 figure has two of five columns marked A MEMORY
at cosine 1.000 all the way to T=16. Vigilance says "if it is new, make a cell",
and nothing ever says "these two were the same thing".

So: a merger that runs periodically and is *forced* -- it merges whether or not
any pair clears a bar, because on this data no bar works.

Why by rank and not by threshold
--------------------------------
Measured over 111 cells from 216 real pairs, the visual similarity between
concept cells has median −0.009 and a 99th percentile of 0.206. A threshold of
0.85 merges **zero** pairs; 0.75 merges zero; 0.1 admits 189 at once. There is
no setting that merges the right amount, because the scale belongs to the code
-- distinct photographs are near-orthogonal in it -- rather than to how alike
two concepts are.

The *ordering* is fine, though: the top 100 pairs by summed similarity are
**82% same-category**. So the merger holds a rate, not a similarity, which is
the same correction the vigilance controller needed.

What is measured, and why each one is here
------------------------------------------
    singletons      cells holding exactly one experience -- the thing being
                    attacked, and the user-facing prediction (52% -> under 20%)
    verbatim        fraction of imaginings bit-identical to a stored photograph.
                    This is what singletons *cause*, and the real target
    novelty         at fixed coherence -- merging must buy imagination, not just
                    a smaller table
    sound->vision   the recall the layer exists for. Merging 82%-pure pairs
                    still merges 18% wrong ones, so this is the cost side and it
                    has to be reported at every compression level
    purity          how mixed the surviving cells' labels are

A sweep over compression rather than a single setting, because the interesting
result is the *shape* -- where recall starts paying for generalisation.

Usage:  python3 benchmarks/merger.py out_merger.json
"""
import json
import sys

import numpy as np

from neurobrain.cognition.multimodal import AssociationArea, _unit
from neurobrain.sensing.natural import load_audiovisual
from neurobrain.sensing.streams import StreamingBrain

sys.path.insert(0, "benchmarks")
from real_binding import encode, split                       # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
N_CONCEPT = 256
KEEPS = (1.0, 0.8, 0.6, 0.4, 0.25, 0.15)
TEMPS = (0.0, 4.0)
VERBATIM = 0.999


def measure(assoc, votes, V, A, y, tr, te, n_cls, rng):
    live = np.flatnonzero(assoc.wins > 0)
    Btr = np.stack([assoc.prep_v(V[i]) for i in tr])
    protos = np.stack([_unit(Btr[y[tr] == c].mean(0)) for c in range(n_cls)])
    name = {c: max(v.items(), key=lambda kv: kv[1])[0]
            for c, v in votes.items() if v}

    s2l = s2v = 0
    for i in te:
        c = assoc.concept_from_sound(A[i])
        s2l += int(name.get(c, -1) == int(y[i]))
        s2v += int(int(np.argmax(protos @ _unit(assoc.Wv[c]))) == int(y[i]))
    n = max(len(te), 1)

    out = dict(cells=int(len(live)),
               singletons=float(np.mean(assoc.wins[live] == 1)),
               sound_to_label=s2l / n, sound_to_vision=s2v / n)
    tot = sum(sum(v.values()) for v in votes.values() if v)
    top = sum(max(v.values()) for v in votes.values() if v)
    out["purity"] = top / max(tot, 1)

    cells = [assoc.concept_from_sound(A[i]) for i in te]
    want = [int(y[i]) for i in te]
    for T in TEMPS:
        M = np.stack([assoc.imagine_vision(c, temperature=T, rng=rng)
                      for c in cells])
        s = (M @ Btr.T).max(1)
        coh = np.mean([int(np.argmax(protos @ M[k])) == want[k]
                       for k in range(len(M))])
        out[f"verbatim_T{T:g}"] = float((s > VERBATIM).mean())
        out[f"novelty_T{T:g}"] = float(1.0 - s.mean())
        out[f"coherence_T{T:g}"] = float(coh)
    return out


def fold(votes, merged):
    """A merge moves evidence, so the vote map has to move with it."""
    for old, new in merged.items():
        while new in merged:
            new = merged[new]
        if old in votes:
            dst = votes.setdefault(new, {})
            for lab, k in votes.pop(old).items():
                dst[lab] = dst.get(lab, 0) + k
    return votes


def run_seed(V, A, y, n_cls, seed):
    tr, te = split(y, seed)
    out = {}
    for keep in KEEPS:
        assoc = AssociationArea(n_vis=V.shape[1], n_aud=A.shape[1],
                                n_concept=N_CONCEPT, seed=seed)
        assoc.set_stats(V[tr], A[tr])
        votes = {}
        for i in tr:
            w = assoc.bind(V[i], A[i])
            votes.setdefault(w, {})
            votes[w][int(y[i])] = votes[w].get(int(y[i]), 0) + 1
        if keep < 1.0:
            votes = fold(votes, assoc.consolidate_ranked(keep=keep))
        out[f"keep={keep}"] = measure(assoc, votes, V, A, y, tr, te, n_cls,
                                      np.random.default_rng(seed + 61))

    # the targeted pass: ranked merging leaves ~31% singletons however hard it
    # compresses, because the survivors are genuinely dissimilar. This one goes
    # after the singletons themselves, which are the cells that cannot imagine.
    for pre in (1.0, 0.6):
        assoc = AssociationArea(n_vis=V.shape[1], n_aud=A.shape[1],
                                n_concept=N_CONCEPT, seed=seed)
        assoc.set_stats(V[tr], A[tr])
        votes = {}
        for i in tr:
            w = assoc.bind(V[i], A[i])
            votes.setdefault(w, {})
            votes[w][int(y[i])] = votes[w].get(int(y[i]), 0) + 1
        if pre < 1.0:
            votes = fold(votes, assoc.consolidate_ranked(keep=pre))
        votes = fold(votes, assoc.absorb_singletons())
        tag = "absorb" if pre == 1.0 else f"keep={pre}+absorb"
        out[tag] = measure(assoc, votes, V, A, y, tr, te, n_cls,
                           np.random.default_rng(seed + 61))
    return out


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_merger.json"
    images, waves, y, names = load_audiovisual(n_per_class=60, seed=0,
                                               grayscale=False, size=32)
    n_cls = len(names)
    brain = StreamingBrain(seed=0, image_shape=(32, 32), v1_cells=4096,
                           rf=7, stride=2)
    V, A = encode(brain, images, waves, colour=True, adapt=True, develop=True)
    print(f"{len(images)} real pairs, {n_cls} categories, chance "
          f"{1/n_cls:.3f}\n", flush=True)

    rows = [run_seed(V, A, y, n_cls, sd) for sd in SEEDS]
    keys = list(rows[0])
    res = {"seeds": list(SEEDS), "arms": {}}
    for k in keys:
        res["arms"][k] = {q: round(float(np.mean([r[k][q] for r in rows])), 4)
                          for q in rows[0][k]}

    print(f"{'compression':<14}{'cells':>7}{'singleton':>11}{'verbatim':>10}"
          f"{'novelty':>9}{'coherence':>11}{'s->vision':>11}{'purity':>8}")
    for k in keys:
        a = res["arms"][k]
        print(f"{k:<14}{a['cells']:>7.1f}{a['singletons']:>11.3f}"
              f"{a['verbatim_T4']:>10.3f}{a['novelty_T4']:>9.3f}"
              f"{a['coherence_T4']:>11.3f}{a['sound_to_vision']:>11.3f}"
              f"{a['purity']:>8.3f}")

    base, ref = res["arms"]["keep=1.0"], None
    print("\n=== does forcing the layer to generalise fix the imagining? ===")
    print(f"  before: {base['cells']:.0f} cells, "
          f"{base['singletons']:.1%} singletons, "
          f"{base['verbatim_T4']:.1%} of imaginings are a stored photograph")
    hit = [k for k in keys if res["arms"][k]["singletons"] < 0.20]
    res["reaches_20pct"] = hit[0] if hit else None
    if hit:
        a = res["arms"][hit[0]]
        print(f"  singletons fall under 20% at {hit[0]}: "
              f"{a['singletons']:.1%} over {a['cells']:.0f} cells")
    else:
        best = min(keys, key=lambda k: res["arms"][k]["singletons"])
        print(f"  singletons never reach 20%; the lowest is "
              f"{res['arms'][best]['singletons']:.1%} at {best}")

    # the question the whole phase is for
    print(f"\n  {'compression':<14}{'verbatim':>10}{'change':>10}"
          f"{'s->vision':>11}{'change':>10}")
    for k in keys:
        a = res["arms"][k]
        print(f"  {k:<14}{a['verbatim_T4']:>10.3f}"
              f"{a['verbatim_T4'] - base['verbatim_T4']:>+10.3f}"
              f"{a['sound_to_vision']:>11.3f}"
              f"{a['sound_to_vision'] - base['sound_to_vision']:>+10.3f}")

    # where does generalising stop being free?
    free = [k for k in keys
            if res["arms"][k]["sound_to_vision"] >= base["sound_to_vision"]
            - 0.02]
    res["free_compression"] = free[-1] if free else "keep=1.0"
    a = res["arms"][res["free_compression"]]
    print(f"\n  recall holds (within 0.02) down to {res['free_compression']}: "
          f"{a['cells']:.0f} cells, {a['singletons']:.1%} singletons, "
          f"verbatim {a['verbatim_T4']:.3f}")
    if a["verbatim_T4"] < base["verbatim_T4"] - 0.05:
        print("  -> generalising is what stops the mind returning memories, "
              "and it is nearly free.")
    else:
        print("  -> merging shrinks the table without changing what is "
              "imagined; the singletons were not the cause.")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
