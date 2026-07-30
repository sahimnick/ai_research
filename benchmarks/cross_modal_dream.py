"""Can the mind learn from sights it never saw -- only imagined?

This is the goal's hardest claim made testable. The mind is supposed to imagine
from its concepts, form *new* concepts out of what it imagined, and be better in
the real world for it. Each of those three is a place the claim can fail, and
the third is the only one that matters: an inner world that changes nothing
outside it is decoration.

The setup is a night's sleep in which **only sound is replayed**. For each
replayed recording the association area is asked what it expects to *see*
(`vision_from_sound`), and that imagined sight is bound to the sound as if it
had been experienced. Nothing visual enters from outside. Whatever the visual
side of those new concepts becomes, the mind made up.

Four arms, differing only in what the night contains:

    no_dream    wake, then stop                                  (control)
    stored      replay the real (sight, sound) pairs it lived    (the ceiling
                -- ordinary hippocampal replay, with real sensory content)
    imagined    replay sound, imagine the sight, bind the pair   (the claim)
    confabulated  replay sound, bind it to a RANDOM visual code  (the control
                that decides whether "imagined" means anything: if a random
                sight helps as much, the imagination is doing no work and the
                gain is just extra binding events)
    consolidated  no replay at all -- only merge concept cells that turned out
                to be the same thing. Waking vigilance is deliberately eager
                and on real data it recruits about one cell per example, which
                is an exemplar memory rather than a set of concepts. This is
                the other thing a night is for
    imagined_consolidated   both: imagine, then integrate

Then the same held-out probes as `real_binding.py`, on recordings the mind has
never heard:

    sound -> label     name what would be seen
    sound -> vision    retrieve a visual code and identify it. This is the one
                       the dream can only help by imagining *well*
    concepts           how many cells exist, and how purely each maps to a
                       real category. A night that invents concepts should
                       raise the count; a night that blurs them lowers purity

The prediction, if imagination is doing real work: `imagined` beats `no_dream`
and `confabulated`, and does not reach `stored`. If `imagined` matches
`confabulated`, the mechanism is bookkeeping. If it matches `stored`, the
imagined sights are as good as the real ones, which would be a strong claim and
needs the purity number to back it.

There is a **sparse condition** too, because that is where generative replay is
supposed to earn its keep (van de Ven et al. 2020, brain-inspired replay):
a few categories are given only a handful of waking examples, so the mind has
concepts it has barely seen. If imagining fills those in, the gain should be
concentrated there rather than spread evenly.

Usage:  python3 benchmarks/cross_modal_dream.py out_cross_modal_dream.json [n_per]
"""
import json
import sys

import numpy as np

from neurobrain.cognition.multimodal import AssociationArea
from neurobrain.sensing.streams import _unit
from neurobrain.vision.widev1 import _nearest_prototype

SEEDS = (0, 1, 2, 3, 4, 5)
TRAIN_FRAC = 0.6
# Big enough that recruitment is decided by the data and not by the array.
# Measured on this exact material: a pool of 32 or 64 is used to the last cell,
# and only at 128+ does the number settle -- 77 cells, identical at 128, 256 and
# 512. Held-out recall rises 0.560 -> 0.600 across that boundary, so a saturated
# pool is not a neutral choice. It also makes the whole experiment impossible:
# a night cannot create a new concept when every cell is already spoken for.
N_CONCEPT = 256
REPLAYS = 400
ARMS = ("no_dream", "stored", "imagined", "confabulated", "consolidated",
        "imagined_consolidated")
SPARSE_FRAC = 0.4          # this share of categories gets a starved day
SPARSE_KEEP = 3            # ...this many waking examples each


def split(y, seed, sparse=False):
    """Stratified, and optionally starved for a random subset of categories."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    starved = set()
    if sparse:
        n = max(1, int(round(SPARSE_FRAC * len(classes))))
        starved = set(rng.permutation(classes)[:n].tolist())
    tr, te = [], []
    for c in classes:
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        cut = max(1, int(round(len(idx) * TRAIN_FRAC)))
        keep = idx[:cut]
        if int(c) in starved:
            keep = keep[:SPARSE_KEEP]
        tr.extend(keep.tolist())
        te.extend(idx[cut:].tolist())
    return np.array(tr), np.array(te), sorted(starved)


def wake(V, A, y, tr, seed):
    """Live the day: bind what was seen and heard together."""
    assoc = AssociationArea(n_vis=V.shape[1], n_aud=A.shape[1],
                            n_concept=N_CONCEPT, seed=seed)
    assoc.set_stats(V[tr], A[tr])
    votes = {}
    for i in tr:
        win = assoc.bind(V[i], A[i])
        votes.setdefault(win, {})
        votes[win][int(y[i])] = votes[win].get(int(y[i]), 0) + 1
    return assoc, votes


def dream(assoc, votes, V, A, y, tr, arm, seed):
    """A night. Only `stored` has any real sensory content in it."""
    if arm == "no_dream":
        return
    rng = np.random.default_rng(seed + 101)
    if arm in ("imagined", "stored", "confabulated", "imagined_consolidated"):
        for _ in range(REPLAYS):
            i = int(rng.choice(tr))
            if arm == "stored":
                v = V[i]
            elif arm == "confabulated":
                v = _unit(rng.standard_normal(V.shape[1]).astype(np.float32))
            else:
                # hear it again, and see what the concept expects. Nothing
                # visual arrives from outside; this is the mind's own picture.
                v = assoc.vision_from_sound(A[i])
            win = assoc.bind(v, A[i])
            votes.setdefault(win, {})
            votes[win][int(y[i])] = votes[win].get(int(y[i]), 0) + 1
    if arm in ("consolidated", "imagined_consolidated"):
        # the other thing a night is for: exemplars that turned out to be the
        # same thing become one concept, and their evidence is pooled
        for old, new in assoc.consolidate().items():
            if old in votes:
                dst = votes.setdefault(new, {})
                for lab, n in votes.pop(old).items():
                    dst[lab] = dst.get(lab, 0) + n


def probe(assoc, votes, V, A, y, tr, te, n_cls):
    name = {c: max(v.items(), key=lambda kv: kv[1])[0] for c, v in votes.items()}
    protoV = np.stack([_unit(V[tr][y[tr] == c].mean(0))
                       if (y[tr] == c).any() else np.zeros(V.shape[1], np.float32)
                       for c in range(n_cls)])
    s2l = s2v = n = 0
    for i in te:
        n += 1
        cs = assoc.concept_from_sound(A[i])
        s2l += int(name.get(cs, -1) == int(y[i]))
        s2v += int(int(np.argmax(protoV @ _unit(assoc.Wv[cs]))) == int(y[i]))
    tot = sum(sum(v.values()) for v in votes.values())
    top = sum(max(v.values()) for v in votes.values())
    return dict(sound_to_label=s2l / max(n, 1), sound_to_vision=s2v / max(n, 1),
                cells=int((assoc.wins > 0).sum()), purity=top / max(tot, 1))


def run_seed(V, A, y, n_cls, seed, sparse):
    tr, te, starved = split(y, seed, sparse=sparse)
    out = {"n_train": int(len(tr)), "n_test": int(len(te)), "starved": starved}
    for arm in ARMS:
        assoc, votes = wake(V, A, y, tr, seed)
        dream(assoc, votes, V, A, y, tr, arm, seed)
        out[arm] = probe(assoc, votes, V, A, y, tr, te, n_cls)
        if starved:
            # was the gain where the day was thin?
            m = np.isin(y[te], starved)
            out[arm]["starved_sound_to_vision"] = probe(
                assoc, votes, V, A, y, tr, te[m], n_cls)["sound_to_vision"]
            out[arm]["rich_sound_to_vision"] = probe(
                assoc, votes, V, A, y, tr, te[~m], n_cls)["sound_to_vision"]
    return out


def report(res, per_seed, chance, tag):
    print(f"\n--- {tag} ---")
    print(f"{'arm':<14}{'s->label':>10}{'s->vision':>11}{'vs no_dream':>13}"
          f"{'d':>8}{'wins':>7}{'cells':>8}{'purity':>8}")
    base = np.array([s["no_dream"]["sound_to_vision"] for s in per_seed])
    for arm in ARMS:
        sv = np.array([s[arm]["sound_to_vision"] for s in per_seed])
        sl = np.array([s[arm]["sound_to_label"] for s in per_seed])
        d = sv - base
        sd_ = float(d.std(ddof=1))
        cd = float(d.mean() / (sd_ + 1e-12)) if arm != "no_dream" else 0.0
        rec = dict(sound_to_label=round(float(sl.mean()), 4),
                   sound_to_vision=round(float(sv.mean()), 4),
                   delta=round(float(d.mean()), 4), cohens_d=round(cd, 3),
                   wins=int((d > 0).sum()), n=len(d),
                   cells=round(float(np.mean([s[arm]["cells"] for s in per_seed])), 1),
                   purity=round(float(np.mean([s[arm]["purity"] for s in per_seed])), 4))
        res.setdefault(tag, {})[arm] = rec
        print(f"{arm:<14}{rec['sound_to_label']:>10.3f}"
              f"{rec['sound_to_vision']:>11.3f}{rec['delta']:>+13.4f}"
              f"{cd:>8.2f}{rec['wins']:>4}/{rec['n']}{rec['cells']:>8.1f}"
              f"{rec['purity']:>8.3f}")
    if per_seed[0].get("starved"):
        print(f"\n{'arm':<14}{'starved':>10}{'rich':>9}   "
              f"(categories with only {SPARSE_KEEP} waking examples)")
        for arm in ARMS:
            st = float(np.mean([s[arm]["starved_sound_to_vision"] for s in per_seed]))
            ri = float(np.mean([s[arm]["rich_sound_to_vision"] for s in per_seed]))
            res[tag][arm]["starved"] = round(st, 4)
            res[tag][arm]["rich"] = round(ri, 4)
            print(f"{arm:<14}{st:>10.3f}{ri:>9.3f}")
    return res


def verdict(res, tag):
    r = res[tag]
    imag, conf, stor = r["imagined"], r["confabulated"], r["stored"]
    print(f"\n=== does imagining during sleep help in the waking world? "
          f"({tag}) ===")
    beats_none = imag["cohens_d"] >= 0.8 and imag["wins"] >= 0.8 * imag["n"]
    beats_conf = imag["sound_to_vision"] > conf["sound_to_vision"] + 0.02
    if not beats_none:
        print("  no. Imagined replay does not move cross-modal recall.")
    elif not beats_conf:
        print(f"  it moves ({imag['delta']:+.4f}) -- but a RANDOM visual code "
              f"moves it as much ({conf['delta']:+.4f}). The gain is extra "
              f"binding events, not imagination.")
    else:
        print(f"  yes: {imag['delta']:+.4f} (d={imag['cohens_d']:+.2f}, "
              f"{imag['wins']}/{imag['n']}), against confabulation "
              f"{conf['delta']:+.4f} and stored replay {stor['delta']:+.4f}.")
        print(f"  the imagined sights are worth "
              f"{imag['delta'] / max(stor['delta'], 1e-9):.0%} of real ones.")


def main():
    out_path = (sys.argv[1] if len(sys.argv) > 1
                else "out_cross_modal_dream.json")
    n_per = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    from neurobrain.sensing.natural import load_audiovisual
    from neurobrain.sensing.streams import StreamingBrain
    images, waves, y, names = load_audiovisual(n_per_class=n_per, seed=0)
    n_cls = len(names)
    brain = StreamingBrain(seed=0)
    V = np.array([_unit(brain.v1.rate(im)) for im in images], np.float32)
    A = np.array([brain.belt.code(brain.ear.coch.forward(w)[0])
                  for w in waves], np.float32)
    print(f"{len(images)} audiovisual samples, {n_cls} categories "
          f"({', '.join(names)}), chance {1/n_cls:.3f}", flush=True)

    res = {"n_class": n_cls, "chance": round(1 / n_cls, 4),
           "replays": REPLAYS, "seeds": list(SEEDS)}
    for tag, sparse in (("balanced day", False), ("sparse day", True)):
        per_seed = [run_seed(V, A, y, n_cls, sd, sparse) for sd in SEEDS]
        res.setdefault("_raw", {})[tag] = per_seed
        report(res, per_seed, 1 / n_cls, tag)
        verdict(res, tag)

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
