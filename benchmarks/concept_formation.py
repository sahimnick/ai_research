"""Do concepts form on real data, or does the mind keep one cell per experience?

This is the gate on everything else in the goal. A mind that stores one concept
cell per thing it has met has an exemplar memory, not concepts: replay is a
structural no-op (a replayed pair re-selects its own cell), consolidation has
nothing to merge, and every "hear a sound, recall X" probe collapses into
"find the nearest stored sound and read off what sat beside it". Measured on
360 real CIFAR-10 / ESC-50 pairs, that is exactly what happened -- **216 cells
for 216 training pairs, purity 1.00**.

The cause was not the vigilance value. It was the scale `match` is measured on.
Averaging a cell's drive across the two senses puts that scale at the mercy of
the weaker one: the visual code's between-exemplar similarity tops out near 0
on photographs while audio reaches 0.95, so the averaged best match over a whole
day peaked at **0.740** and a vigilance of 0.80 could never be satisfied by
anything. Every pair recruited. No value in (0.74, 1] would have worked, and any
lower one would need re-choosing whenever either sense changed quality.

Taking the **max** instead -- a cell is as awake as its best-driving sense, which
is what multisensory neurons do (Stein & Meredith's inverse effectiveness) --
puts the same day in 0.648-0.954, a range a threshold can live inside.

Taking the **max** instead has the opposite failure on the synthetic bank --
almost every match clears the bar, nothing recruits, and the layer collapses to
two cells. So the sweep here is not over vigilance at all. Vigilance is driven
by a homeostatic controller toward a target `novelty_rate` -- what fraction of
experience becomes something new -- and what is swept is the combination rule
against that rate, reporting the four things that decide whether a concept
exists:

    cells/pair   below 1.0 or it is not a concept layer
    purity       does each cell still answer to one real category
    s->label     held-out recall, against the 1-NN floor the audio code
                 already reaches on its own
    s->vision    held-out cross-modal retrieval, the read-out no lookup can do

and then what a night's consolidation does to whatever survived.

Usage:  python3 benchmarks/concept_formation.py out_concept_formation.json
"""
import copy
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
RULES = ("mean", "max", "reliability")
RATES = (None, 0.35, 0.50, 0.65)
MERGE = (0.15, 0.25, 0.40, 0.60)


def build(V, A, y, tr, rule, rate, seed):
    a = AssociationArea(n_vis=V.shape[1], n_aud=A.shape[1],
                        n_concept=N_CONCEPT, vigilance=0.80,
                        match_rule=rule, novelty_rate=rate, seed=seed)
    a.set_stats(V[tr], A[tr])
    votes = {}
    for i in tr:
        w = a.bind(V[i], A[i])
        votes.setdefault(w, {})
        votes[w][int(y[i])] = votes[w].get(int(y[i]), 0) + 1
    return a, votes


def fold(votes, merged):
    for old, new in merged.items():
        if old in votes:
            dst = votes.setdefault(new, {})
            for lab, n in votes.pop(old).items():
                dst[lab] = dst.get(lab, 0) + n
    return votes


def probe(a, votes, V, A, y, tr, te, n_cls):
    name = {c: max(v.items(), key=lambda kv: kv[1])[0] for c, v in votes.items()}
    protoV = np.stack([_unit(V[tr][y[tr] == c].mean(0)) for c in range(n_cls)])
    s2l = s2v = 0
    for i in te:
        cs = a.concept_from_sound(A[i])
        s2l += int(name.get(cs, -1) == int(y[i]))
        s2v += int(int(np.argmax(protoV @ _unit(a.Wv[cs]))) == int(y[i]))
    tot = sum(sum(v.values()) for v in votes.values())
    top = sum(max(v.values()) for v in votes.values())
    return dict(sound_to_label=s2l / len(te), sound_to_vision=s2v / len(te),
                cells=int((a.wins > 0).sum()), purity=top / max(tot, 1))


def main():
    out_path = (sys.argv[1] if len(sys.argv) > 1
                else "out_concept_formation.json")
    images, waves, y, names = load_audiovisual(n_per_class=60, seed=0,
                                               grayscale=False)
    n_cls = len(names)
    brain = StreamingBrain(seed=0)
    V, A = encode(brain, images, waves, colour=True, adapt=True, develop=True)
    print(f"{len(images)} real pairs, {n_cls} categories, chance "
          f"{1/n_cls:.3f}\n", flush=True)

    # the floor the concept layer has to beat: nearest stored sound, no concepts
    floors = []
    for sd in SEEDS:
        tr, te = split(y, sd)
        S = A[te] @ A[tr].T
        floors.append(float(np.mean(y[tr][S.argmax(1)] == y[te])))
    floor = float(np.mean(floors))
    print(f"floor -- 1-NN on the sound code alone, no concept layer: "
          f"{floor:.3f}\n", flush=True)

    res = {"chance": round(1 / n_cls, 4), "floor_1nn": round(floor, 4),
           "grid": []}
    print(f"{'rule':<13}{'rate':>7}{'cells':>8}{'/pair':>8}{'purity':>8}"
          f"{'s->label':>10}{'s->vision':>11}")
    for rule in RULES:
        for vig in RATES:
            rows = []
            for sd in SEEDS:
                tr, te = split(y, sd)
                a, votes = build(V, A, y, tr, rule, vig, sd)
                r = probe(a, votes, V, A, y, tr, te, n_cls)
                r["per_pair"] = r["cells"] / len(tr)
                rows.append(r)
            m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
            rec = dict(rule=rule, novelty_rate=vig, **{k: round(v, 4)
                                                    for k, v in m.items()})
            res["grid"].append(rec)
            print(f"{rule:<13}{str(vig):>7}{m['cells']:>8.1f}"
                  f"{m['per_pair']:>8.2f}{m['purity']:>8.3f}"
                  f"{m['sound_to_label']:>10.3f}{m['sound_to_vision']:>11.3f}",
                  flush=True)

    # ------------------------------------------------- and then a night ----
    print(f"\n--- consolidation, from the best compressing setting that still "
          f"beats the floor ---")
    ok = [r for r in res["grid"]
          if r["sound_to_label"] >= floor - 0.02 and r["purity"] >= 0.9]
    pick = min(ok, key=lambda r: r["per_pair"]) if ok else None
    if pick is None:
        print("  no setting both compresses and holds recall.")
        res["chosen"] = None
    else:
        print(f"  waking rule: {pick['rule']}, novelty_rate {pick['novelty_rate']} "
              f"-> {pick['per_pair']:.2f} cells per pair, "
              f"s->label {pick['sound_to_label']:.3f}")
        res["chosen"] = pick
        print(f"\n{'merge':>7}{'cells':>8}{'/pair':>8}{'compress':>10}"
              f"{'purity':>8}{'s->label':>10}{'s->vision':>11}")
        for th in MERGE:
            rows = []
            for sd in SEEDS:
                tr, te = split(y, sd)
                a, votes = build(V, A, y, tr, pick["rule"], pick["novelty_rate"], sd)
                before = int((a.wins > 0).sum())
                votes = fold(copy.deepcopy(votes), a.consolidate(threshold=th))
                r = probe(a, votes, V, A, y, tr, te, n_cls)
                r["per_pair"] = r["cells"] / len(tr)
                r["compress"] = before / max(r["cells"], 1)
                rows.append(r)
            m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
            res.setdefault("consolidation", {})[str(th)] = {
                k: round(v, 4) for k, v in m.items()}
            print(f"{th:>7.2f}{m['cells']:>8.1f}{m['per_pair']:>8.2f}"
                  f"{m['compress']:>9.1f}x{m['purity']:>8.3f}"
                  f"{m['sound_to_label']:>10.3f}{m['sound_to_vision']:>11.3f}",
                  flush=True)

    print("\n=== do concepts exist yet? ===")
    base = next(r for r in res["grid"]
                if r["rule"] == "mean" and r["novelty_rate"] is None)
    print(f"  as it was (mean, fixed vigilance): {base['per_pair']:.2f} cells per pair "
          f"-- one per experience, so no.")
    if pick:
        print(f"  best compressing setting: {pick['per_pair']:.2f} cells per "
              f"pair at s->label {pick['sound_to_label']:.3f} "
              f"(floor {floor:.3f})")
        cons = res.get("consolidation", {})
        if cons:
            # the most compression that still HOLDS recall -- picking the most
            # compression outright rewards a threshold that merges a dog into a
            # cat, and 0.15 does exactly that (1.8x, but purity 0.994 -> 0.842
            # and recall 0.946 -> 0.797)
            held = {k: v for k, v in cons.items()
                    if v["sound_to_label"] >= floor - 0.03
                    and v["purity"] >= 0.95}
            b = (min(held.items(), key=lambda kv: kv[1]["per_pair"])
                 if held else min(cons.items(),
                                  key=lambda kv: kv[1]["per_pair"]))
            print(f"  after a night at merge {b[0]}: {b[1]['per_pair']:.2f} "
                  f"per pair, {b[1]['compress']:.1f}x compression, "
                  f"s->label {b[1]['sound_to_label']:.3f}, "
                  f"purity {b[1]['purity']:.3f}, "
                  f"s->vision {b[1]['sound_to_vision']:.3f}")
            over = {k: v for k, v in cons.items() if k not in held}
            if over:
                print(f"  (merging harder does compress more -- "
                      + ", ".join(f"{k}: {v['compress']:.1f}x at purity "
                                  f"{v['purity']:.3f}" for k, v in over.items())
                      + " -- but that is categories collapsing, not forming)")
            verdict = (b[1]["per_pair"] < 0.5
                       and b[1]["sound_to_label"] >= floor - 0.03)
            print(f"  -> {'YES' if verdict else 'not yet'}: fewer cells than "
                  f"experiences, recall held.")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
