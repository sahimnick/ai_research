"""Do concept cells form at all? Sensor fusion, measured on held-out queries.

`AssociationArea` is where the two senses are supposed to become one thing: a
pool of concept cells wired to both, so that either sense alone can wake the
concept and read out what the other expects. That is the mechanism the whole
"perception and concepts from sensor fusion" goal rests on.

It was not working. Binding 48 audio-visual pairs from 8 well-separated classes
woke **one concept cell out of 32**, and cross-modal recall sat at exactly
chance -- while the sound codes alone were 96% separable by nearest prototype.
The information was there; the layer had one category and could not express a
second. The cause was a runaway in the competition: the first cell to win tunes
toward the data, so it wins the next pair too, and the homeostatic term meant to
prevent that was scaled at 0.1 against a drive gap of ~1.0.

Two guards were added -- ART vigilance (a poor match recruits an uncommitted
cell instead of dragging the incumbent) and a conscience term (DeSieno 1988)
scaled to the drive. Both have a free parameter, so both are swept here.

**The queries are held out.** Binding uses 5 sounds per class; recall is probed
with 3 *different* ones the association area has never seen. This matters more
than it sounds: querying with a code that was bound is a lookup, and a lookup
scores well even on shuffled labels. On held-out sounds the shuffled control has
nothing to retrieve and must fall to chance -- which is what makes the real-minus
-shuffled spread mean something.

Usage:  python3 benchmarks/concept_cells.py out_concept_cells.json
"""
import json
import sys

import numpy as np

import neurobrain as nb
from neurobrain.audition.audio import sound_dataset
from neurobrain.sensing.streams import StreamingBrain, _unit

SEEDS = (0, 1, 2, 3, 4)
N_BIND, N_TEST = 5, 3          # sounds per class, disjoint
VIGILANCE = (0.0, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95)
CONSCIENCE = (0.0, 0.5, 1.0, 2.0)


def codes(seed):
    """Visual and auditory codes for every class, computed once per seed."""
    trx, trY, _, _ = nb.load_mnist(n_train=1200, n_test=400)
    sigs, labels, names = sound_dataset(n_per_class=N_BIND + N_TEST, seed=3)
    b = StreamingBrain(seed=seed)
    labels = np.asarray(labels)
    per = {}
    for c in range(len(names)):
        vi = np.flatnonzero(trY == c)[:N_BIND]
        ai = np.flatnonzero(labels == c)[:N_BIND + N_TEST]
        per[c] = (
            [_unit(b.v1.rate(trx[i])) for i in vi],
            [b.belt.code(b.ear.coch.forward(sigs[i])[0]) for i in ai])
    return b, per, len(names)


def run(b, per, n_cls, vig, con, seed, shuffled):
    """Bind, then recall from sounds that were never bound."""
    from neurobrain.cognition.multimodal import AssociationArea

    rng = np.random.default_rng(seed)
    v0, a0 = per[0][0][0], per[0][1][0]
    b.assoc = AssociationArea(n_vis=len(v0), n_aud=len(a0), n_concept=32,
                              vigilance=vig, conscience=con, seed=seed)
    b._cell_votes = {}
    b.bindings = []

    # The control shuffles labels PER PAIR, not per class. A per-class
    # permutation is a bijection -- it renames the classes and preserves every
    # bit of the mapping, so real and shuffled come out bit-identical, which is
    # exactly what the first version of this benchmark reported (spread
    # +0.000 at all 28 settings). Shuffling pair by pair is what actually
    # destroys the correspondence the concept cells are supposed to learn.
    pairs = [(per[c][0][k], per[c][1][k], c)
             for c in range(n_cls) for k in range(N_BIND)]
    labs = [p[2] for p in pairs]
    if shuffled:
        labs = list(rng.permutation(labs))
    for (v, a, _), lab in zip(pairs, labs):
        b.bind(v, a, int(lab))

    ok = n = 0
    for c in range(n_cls):
        for k in range(N_BIND, N_BIND + N_TEST):      # held out
            got = b.recall_visual_from_sound(per[c][1][k])
            if got is None:
                continue
            n += 1
            ok += int(int(got) == c)
    return ok / max(n, 1), int((b.assoc.wins > 0).sum()), n


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_concept_cells.json"
    prepared = [codes(sd) for sd in SEEDS]
    n_cls = prepared[0][2]
    chance = 1.0 / n_cls
    print(f"{n_cls} classes, {N_BIND} sounds bound + {N_TEST} held out per "
          f"class, {len(SEEDS)} seeds, chance {chance:.3f}\n", flush=True)

    res = {"chance": round(chance, 4), "n_class": n_cls, "seeds": list(SEEDS),
           "grid": []}
    print(f"{'vig':>6}{'consc':>7}{'real':>8}{'shuf':>8}{'spread':>9}"
          f"{'+/-':>7}{'wins':>7}{'cells':>7}")
    for vig in VIGILANCE:
        for con in CONSCIENCE:
            real, shuf, cells = [], [], []
            for (b, per, k), sd in zip(prepared, SEEDS):
                r, cr, _ = run(b, per, k, vig, con, sd, shuffled=False)
                s, _, _ = run(b, per, k, vig, con, sd, shuffled=True)
                real.append(r)
                shuf.append(s)
                cells.append(cr)
            d = np.array(real) - np.array(shuf)
            sd_ = float(d.std(ddof=1))
            rec = dict(vigilance=vig, conscience=con,
                       real=round(float(np.mean(real)), 4),
                       shuffled=round(float(np.mean(shuf)), 4),
                       spread=round(float(d.mean()), 4), sd=round(sd_, 4),
                       wins=int((d > 0).sum()), n=len(d),
                       cells=round(float(np.mean(cells)), 1),
                       cohens_d=round(float(d.mean() / (sd_ + 1e-12)), 3))
            res["grid"].append(rec)
            print(f"{vig:>6.2f}{con:>7.1f}{rec['real']:>8.3f}"
                  f"{rec['shuffled']:>8.3f}{rec['spread']:>+9.3f}"
                  f"{sd_:>7.3f}{rec['wins']:>4}/{rec['n']}"
                  f"{rec['cells']:>7.1f}", flush=True)

    # -------------------------------------------------------- the choice ---
    good = [r for r in res["grid"]
            if r["cohens_d"] >= 0.8 and r["wins"] >= 0.8 * r["n"]]
    best = max(good, key=lambda r: r["real"]) if good else None
    base = next(r for r in res["grid"]
                if r["vigilance"] == 0.0 and r["conscience"] == 0.0)
    print(f"\nthe layer as it was (vigilance 0, conscience 0): "
          f"real {base['real']:.3f}, {base['cells']:.1f} concept cells "
          f"for {n_cls} classes")
    if best is None:
        print("NO setting separates real from shuffled -- the fault is not in "
              "the competition but somewhere upstream of it.")
    else:
        print(f"best: vigilance {best['vigilance']}, conscience "
              f"{best['conscience']} -> real {best['real']:.3f} vs shuffled "
              f"{best['shuffled']:.3f} (spread {best['spread']:+.3f}, "
              f"d={best['cohens_d']:+.2f}, {best['wins']}/{best['n']}), "
              f"{best['cells']:.1f} cells")
        print(f"      {base['real']:.3f} -> {best['real']:.3f} is "
              f"{best['real']/max(base['real'],1e-9):.1f}x, and "
              f"{best['real']/chance:.1f}x chance")
    res["chosen"] = best
    res["baseline"] = base

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
