"""Does a night made of things memory cannot assemble help the waking world?

`factored.py` produced the first codes in this project that are outside the span
of everything stored -- residual 0.5287 where every mixing and sampling arm
scores exactly 0.000 -- and that were still read as their own object. But a
sight nothing consumes is a curiosity. The goal is the *re-creation of the
outside world inside*, and the only evidence that anything has been re-created
is that the mind gets better at the world it actually has to live in.

So this asks the question `cross_modal_dream.py` asked, of content that is
categorically different from anything that benchmark could produce.

That benchmark established the shape of the answer for **in-span** content:

    imagined      keep the real recording, let the mind supply the sight
                  +0.0231, 6/6 -- the headline
    confabulated  the same binding operations on a random sight
                  +0.0000 exactly -- so the gain is the content, not plasticity
    dreamt        the mind invents BOTH halves, no real signal in the night
                  -0.0301, the worst arm measured

Two lessons carried forward. The gain is in the content, and the content has to
be **anchored** -- a night with nothing real in it degrades the concepts it is
made of. So every self-generated arm here keeps the **real recording** and
invents only the sight, which is also the goal's own phrasing: combining what is
real with what is imagined.

The arms, and what each one isolates
------------------------------------
    stored        replay the real sight. Real sensory content; the reference
    imagined      the concept's expected sight -- IN span, the known +0.0231
    factored      form from the concept the sound woke, colour from a concept
                  the mind grouped **elsewhere**. OUT of span. A yellow bus,
                  bound to the real sound of a bus
    factored_own  the same operation with the colour donor drawn from the
                  **same** category. This is the control that decides what the
                  crossing is worth: it is equally a recombination and equally
                  out of span, but it invents nothing the category did not
                  already contain
    confabulated  a random sight -- the +0.0000 floor, carried over so this
                  table can be read against the other one

Both `factored` arms leave the span, so a difference between them is not about
novelty in the algebraic sense: it is about whether crossing *categories* buys
anything over recombining inside one. That is the selection question in its
smallest form.

Measured on a normal day and on a **starved** day, because recombination should
pay where observations are thin -- that is what it is for, and the sparse day is
where `cross_modal_dream.py` found the only signal for its self-generated arms.

Usage:  python3 benchmarks/inner_world.py out_inner_world.json
"""
import json
import sys

import numpy as np

from neurobrain.sensing.natural import load_audiovisual
from neurobrain.sensing.streams import StreamingBrain, _unit

sys.path.insert(0, "benchmarks")
from cross_modal_dream import (DREAM_NOISE, DREAM_NOVELTY_RATE,  # noqa: E402
                               N_CONCEPT, REPLAYS, SPARSE_KEEP, probe, split,
                               wake)
from factored import span_projector, span_residual              # noqa: E402
from real_binding import encode                                 # noqa: E402

SEEDS = (0, 1, 2, 3, 4, 5)
CHANNELS = 3
ARMS = ("no_dream", "stored", "imagined", "factored", "factored_own",
        "confabulated")


def night(assoc, votes, V, A, y, tr, arm, seed, blocks):
    """One night. Every arm below keeps the real recording; only the sight
    differs, so the comparison is about what is imagined and nothing else."""
    if arm == "no_dream":
        return
    rng = np.random.default_rng(seed + 101)
    waking, assoc.novelty_rate = assoc.novelty_rate, DREAM_NOVELTY_RATE
    lab = {c: max(v.items(), key=lambda kv: kv[1])[0]
           for c, v in votes.items() if v}

    def donor(c, same):
        pool = [d for d in lab
                if d != c and ((lab[d] == lab.get(c)) == same)]
        return int(rng.choice(pool)) if pool else int(c)

    for _ in range(REPLAYS):
        i = int(rng.choice(tr))
        if arm == "stored":
            v = V[i]
        elif arm == "confabulated":
            v = _unit(rng.standard_normal(V.shape[1]).astype(np.float32))
        elif arm == "imagined":
            v = assoc.vision_from_sound(A[i])
        else:
            c = assoc.concept_from_sound(A[i])
            v = assoc.imagine_factored(
                [c, donor(c, same=(arm == "factored_own"))], blocks,
                temperature=0.0)
        if arm != "stored":
            v = _unit(v + DREAM_NOISE
                      * rng.standard_normal(len(v)).astype(np.float32))
        win = assoc.bind(v, A[i])
        votes.setdefault(win, {})
        votes[win][int(y[i])] = votes[win].get(int(y[i]), 0) + 1
    assoc.novelty_rate = waking                                   # morning


def dream_residual(assoc, votes, V, A, tr, arm, seed, blocks, R):
    """What was actually replayed -- in the span, or outside it?

    Reported rather than assumed. An arm that claims to dream new things and
    turns out to be replaying linear combinations would look identical in the
    accuracy table."""
    if arm in ("no_dream", "stored"):
        return 0.0
    rng = np.random.default_rng(seed + 101)
    lab = {c: max(v.items(), key=lambda kv: kv[1])[0]
           for c, v in votes.items() if v}
    out = []
    for _ in range(64):
        i = int(rng.choice(tr))
        if arm == "confabulated":
            v = _unit(rng.standard_normal(V.shape[1]).astype(np.float32))
        elif arm == "imagined":
            v = assoc.vision_from_sound(A[i])
        else:
            c = assoc.concept_from_sound(A[i])
            pool = [d for d in lab if d != c
                    and ((lab[d] == lab.get(c)) == (arm == "factored_own"))]
            d = int(rng.choice(pool)) if pool else int(c)
            v = assoc.imagine_factored([c, d], blocks, temperature=0.0)
        out.append(_unit(v))
    return float(span_residual(np.stack(out), R).mean())


def probe_robust(assoc, votes, V, Vs, y, te):
    """Look at a photograph, name it -- with its colours as they were, and with
    red and blue exchanged.

    This is the probe the whole benchmark turns on, and it exists because the
    accuracy table above cannot answer the question. Binding a yellow bus to the
    real sound of a bus asserts something **false** about the world, so a night
    of recombinations should not be expected to raise overall accuracy, and does
    not. What a recombination is actually evidence for is that *form survives a
    change of colour* -- so the thing to measure is whether the mind that
    dreamed colour variations it never saw holds up when the world shows it one.

    Swapping the R and B channels leaves luminance ``(r+g+b)/3`` **exactly**
    unchanged and moves both chromatic channels, so form is held fixed by
    construction and only the thing the recombination varied is varied.
    """
    name = {c: max(v.items(), key=lambda kv: kv[1])[0] for c, v in votes.items()}
    ok = swapped = 0
    for i in te:
        ok += int(name.get(assoc.concept_from_vision(V[i]), -1) == int(y[i]))
        swapped += int(name.get(assoc.concept_from_vision(Vs[i]), -1)
                       == int(y[i]))
    n = max(len(te), 1)
    return ok / n, swapped / n


def run_seed(V, Vs, A, y, n_cls, seed, sparse):
    tr, te, starved = split(y, seed, sparse=sparse)
    d = V.shape[1]
    blk = d // CHANNELS
    blocks = [(0, blk), (blk, d)]                     # form | colour
    R = span_projector(np.stack([_unit(V[i]) for i in tr]))
    out = {"n_train": int(len(tr)), "n_test": int(len(te)), "starved": starved}
    for arm in ARMS:
        assoc, votes = wake(V, A, y, tr, seed)
        Rp = span_projector(np.stack([assoc.prep_v(V[i]) for i in tr]))
        res = dream_residual(assoc, votes, V, A, tr, arm, seed, blocks, Rp)
        night(assoc, votes, V, A, y, tr, arm, seed, blocks)
        out[arm] = probe(assoc, votes, V, A, y, tr, te, n_cls)
        out[arm]["dream_residual"] = res
        v2l, v2l_sw = probe_robust(assoc, votes, V, Vs, y, te)
        out[arm]["vision_to_label"] = v2l
        out[arm]["vision_to_label_swapped"] = v2l_sw
        if starved:
            m = np.isin(y[te], starved)
            out[arm]["starved"] = probe(assoc, votes, V, A, y, tr, te[m],
                                        n_cls)["sound_to_vision"]
            out[arm]["rich"] = probe(assoc, votes, V, A, y, tr, te[~m],
                                     n_cls)["sound_to_vision"]
    return out


def report(res, per_seed, tag):
    print(f"\n--- {tag} ---")
    print(f"{'arm':<15}{'s->label':>10}{'s->vision':>11}{'vs no_dream':>13}"
          f"{'d':>7}{'wins':>7}{'cells':>7}{'purity':>8}{'replayed':>10}")
    base = np.array([s["no_dream"]["sound_to_vision"] for s in per_seed])
    for arm in ARMS:
        sv = np.array([s[arm]["sound_to_vision"] for s in per_seed])
        sl = np.array([s[arm]["sound_to_label"] for s in per_seed])
        dd = sv - base
        sd_ = float(dd.std(ddof=1))
        cd = 0.0 if arm == "no_dream" else float(dd.mean() / (sd_ + 1e-12))
        rec = dict(sound_to_label=round(float(sl.mean()), 4),
                   sound_to_vision=round(float(sv.mean()), 4),
                   delta=round(float(dd.mean()), 4), cohens_d=round(cd, 3),
                   wins=int((dd > 0).sum()), n=len(dd),
                   purity=round(float(np.mean(
                       [s[arm]["purity"] for s in per_seed])), 4),
                   cells=round(float(np.mean(
                       [s[arm]["cells"] for s in per_seed])), 1),
                   dream_residual=round(float(np.mean(
                       [s[arm]["dream_residual"] for s in per_seed])), 4))
        res.setdefault(tag, {})[arm] = rec
        print(f"{arm:<15}{rec['sound_to_label']:>10.3f}"
              f"{rec['sound_to_vision']:>11.3f}{rec['delta']:>+13.4f}"
              f"{cd:>7.2f}{rec['wins']:>4}/{rec['n']}{rec['cells']:>7.1f}"
              f"{rec['purity']:>8.3f}{rec['dream_residual']:>10.4f}")
    # the probe the recombination is actually evidence for
    print(f"\n{'arm':<15}{'see->name':>11}{'colours swapped':>17}"
          f"{'kept':>8}{'vs no_dream':>13}{'d':>7}{'wins':>7}")
    bsw = np.array([s["no_dream"]["vision_to_label_swapped"]
                    for s in per_seed])
    for arm in ARMS:
        v = float(np.mean([s[arm]["vision_to_label"] for s in per_seed]))
        w = np.array([s[arm]["vision_to_label_swapped"] for s in per_seed])
        dd = w - bsw
        sd_ = float(dd.std(ddof=1))
        cd = 0.0 if arm == "no_dream" else float(dd.mean() / (sd_ + 1e-12))
        kept = float(w.mean()) / max(v, 1e-9)
        res[tag][arm].update(
            vision_to_label=round(v, 4),
            vision_to_label_swapped=round(float(w.mean()), 4),
            colour_robustness=round(kept, 4),
            swapped_delta=round(float(dd.mean()), 4),
            swapped_d=round(cd, 3), swapped_wins=int((dd > 0).sum()))
        print(f"{arm:<15}{v:>11.3f}{float(w.mean()):>17.3f}{kept:>8.3f}"
              f"{dd.mean():>+13.4f}{cd:>7.2f}{int((dd > 0).sum()):>4}/{len(dd)}")
    if per_seed[0].get("starved"):
        print(f"\n{'arm':<15}{'starved':>10}{'rich':>9}   "
              f"(categories with only {SPARSE_KEEP} waking examples)")
        for arm in ARMS:
            st = float(np.mean([s[arm]["starved"] for s in per_seed]))
            ri = float(np.mean([s[arm]["rich"] for s in per_seed]))
            res[tag][arm]["starved"] = round(st, 4)
            res[tag][arm]["rich"] = round(ri, 4)
            print(f"{arm:<15}{st:>10.3f}{ri:>9.3f}")
    return res


def verdict(res, tag):
    r = res[tag]
    print(f"\n  === does out-of-span content help the waking world? ({tag}) ===")
    print(f"    what each night was made of, by span residual:")
    for arm in ARMS:
        if arm == "no_dream":
            continue
        inout = ("outside the span" if r[arm]["dream_residual"] > 1e-3
                 else "INSIDE the span")
        print(f"      {arm:<14}{r[arm]['dream_residual']:>8.4f}  {inout}")
    im, fa = r["imagined"], r["factored"]
    own, cf = r["factored_own"], r["confabulated"]
    print(f"\n    in-span imagining   {im['delta']:+.4f} "
          f"(d={im['cohens_d']:+.2f}, {im['wins']}/{im['n']})")
    print(f"    out-of-span, crossed{fa['delta']:+.4f} "
          f"(d={fa['cohens_d']:+.2f}, {fa['wins']}/{fa['n']})")
    print(f"    out-of-span, own cat{own['delta']:+.4f} "
          f"(d={own['cohens_d']:+.2f}, {own['wins']}/{own['n']})")
    print(f"    random sight        {cf['delta']:+.4f} "
          f"(d={cf['cohens_d']:+.2f}, {cf['wins']}/{cf['n']})")
    best = max(("imagined", "factored", "factored_own"),
               key=lambda k: r[k]["delta"])
    r_best = r[best]
    ok = r_best["cohens_d"] >= 0.8 and r_best["wins"] >= 0.75 * r_best["n"]
    res.setdefault("verdict", {})[tag] = dict(best=best, passes=bool(ok))
    if best in ("factored", "factored_own") and ok:
        print(f"\n    -> the best night is made of codes memory CANNOT "
              f"assemble ({best}, {r_best['delta']:+.4f}).")
        print("       Imagining outside the span is not a curiosity; it is "
              "worth more than replaying what was seen.")
    elif ok:
        print(f"\n    -> the best night is still the IN-span one "
              f"({best}, {r_best['delta']:+.4f}). Leaving the span is real and "
              f"does not yet pay:")
        print(f"       novelty was never the thing that was missing from the "
              f"night.")
    else:
        print(f"\n    -> no arm clears the gate on this day "
              f"(best {best} at {r_best['delta']:+.4f}, "
              f"d={r_best['cohens_d']:+.2f}).")
    if r["factored"]["purity"] < r["imagined"]["purity"] - 0.02:
        print(f"       cost: purity {r['imagined']['purity']:.3f} -> "
              f"{r['factored']['purity']:.3f} -- crossing categories drags "
              f"them together, the same way `dreamt` did.")

    # the question the accuracy table cannot answer
    print(f"\n    and the thing a recombination is EVIDENCE for -- that form "
          f"survives a change of colour:")
    nd = r["no_dream"]
    print(f"      a mind that never dreamt keeps "
          f"{nd['colour_robustness']:.1%} of its naming when red and blue are "
          f"exchanged")
    for arm in ("stored", "imagined", "factored", "factored_own"):
        a = r[arm]
        print(f"      {arm:<14}{a['vision_to_label_swapped']:>7.3f} "
              f"({a['colour_robustness']:.1%} kept)  "
              f"{a['swapped_delta']:+.4f}  d={a['swapped_d']:+.2f}  "
              f"{a['swapped_wins']}/{a['n']}")
    fa = r["factored"]
    gate = fa["swapped_d"] >= 0.8 and fa["swapped_wins"] >= 0.75 * fa["n"]
    beats = fa["swapped_delta"] > r["imagined"]["swapped_delta"]
    res["verdict"][tag]["colour_robustness_passes"] = bool(gate and beats)
    if gate and beats:
        print(f"\n      -> dreaming colour combinations the world never showed "
              f"makes the mind better at colours the world DOES show "
              f"({fa['swapped_delta']:+.4f}, d={fa['swapped_d']:+.2f}).")
        print("         That is the outside world re-created inside and paid "
              "back out.")
    elif gate:
        print(f"\n      -> the factored night helps ({fa['swapped_delta']:+.4f}"
              f", d={fa['swapped_d']:+.2f}) but no more than the in-span one "
              f"({r['imagined']['swapped_delta']:+.4f}).")
    else:
        print(f"\n      -> it does not ({fa['swapped_delta']:+.4f}, "
              f"d={fa['swapped_d']:+.2f}, {fa['swapped_wins']}/{fa['n']}). "
              f"Out-of-span content is real and still has no consumer that "
              f"benefits from it.")
        st = r["stored"]
        print(f"\n      What DID move colour robustness is replaying the real "
              f"sight ({st['swapped_delta']:+.4f}, d={st['swapped_d']:+.2f}) --"
              f" so the robustness came from")
        print(f"      SHARPENING the concept, not from imagined variation. And "
              f"the recombination is not being ignored: it recruits only "
              f"{r['factored']['cells'] - r['no_dream']['cells']:.0f} new cells"
              f", so it")
        print(f"      updates existing concepts, drifting each one's colour "
              f"block toward the average of all of them. In THIS world that is "
              f"a loss, not an invariance:")
        print(f"      colour is signal here, worth +0.171 to the read-out "
              f"(`factored.py`: whole code 0.831, form block alone 0.660). "
              f"Imagining a yellow bus is sound; concluding")
        print(f"      'colour does not matter' from it is not, and binding it "
              f"as a fact is the only thing this architecture knows how to do "
              f"with an imagining.")
    return res


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_inner_world.json"
    images, waves, y, names = load_audiovisual(n_per_class=60, seed=0,
                                               grayscale=False, size=32)
    n_cls = len(names)
    brain = StreamingBrain(seed=0, image_shape=(32, 32), v1_cells=4096,
                           rf=7, stride=2)
    # Both worlds encoded in ONE pass, so they share the developed receptive
    # fields and the same adaptation state -- otherwise the swapped set would
    # differ by which eye looked at it as well as by its colours.
    swapped = [im[[2, 1, 0]] if im.ndim == 3 else im for im in images]
    n = len(images)
    both, A = encode(brain, list(images) + swapped, list(waves) * 2,
                     colour=True, adapt=True, develop=True)
    V, Vs, A = both[:n], both[n:], A[:n]
    print(f"{len(images)} real audio-visual pairs, {n_cls} categories, "
          f"chance {1/n_cls:.3f}", flush=True)
    print("every night keeps the REAL recording and invents only the sight")
    print("robustness probe: R and B exchanged -- luminance exactly unchanged, "
          "both chromatic channels moved\n", flush=True)

    res = {"seeds": list(SEEDS), "n_class": n_cls}
    for tag, sparse in (("a full day", False), ("a starved day", True)):
        per = [run_seed(V, Vs, A, y, n_cls, sd, sparse) for sd in SEEDS]
        res.setdefault("per_seed", {})[tag] = per
        res = report(res, per, tag)
        res = verdict(res, tag)

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
