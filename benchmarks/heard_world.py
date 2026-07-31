"""The imagination engine where perception works: judged dreaming, on real sound.

Three benchmarks built the pieces and each one ended on the same wall.
`factored.py` produced codes outside the span of memory. `inner_world.py` found
they pay back nothing -- a night of recombinations measures the same as a night
of random sights. `constraint.py` built the judgement that was supposed to fix
that, showed it captures 92% of the signal available to it, and showed there is
almost no signal available: over (form, colour) the ceiling is AUC 0.545.

Every one of those was measured **through the eye**, and the eye names
photographs at 0.134 against 0.559 for digits. The suspicion has been that the
imagination machinery is fine and the representation under it is not. A control
with class prototypes reached 0.879, but it used labels, so it shows the rule is
adequate rather than that such a code is reachable.

The ear settles it without labels. Hearing in this project **works** -- 1-NN
0.914 over six real ESC-50 categories, 17.5x chance on the 50-way task -- and
the belt already emits two factors, `[spectral envelope | tonotopic profile]`,
exactly as the eye emits form and colour. Same rule, same code path, real field
recordings, and nothing supervised anywhere:

    does the front end cluster?     eye 0.573      ear 0.758
    can the factors be judged?      eye 0.541      ear 0.862   (oracle 0.879)

So the constraint solver is buildable. It was never buildable on vision.

What this benchmark then asks
-----------------------------
Whether **judging** an imagining is what was missing, which is the one thing
none of the earlier runs could test -- there was nothing worth judging.

    no_dream        the day, then stop
    stored          replay a real recording. The reference: real sensory content
    crossed         the envelope of one concept with the tonotopic profile of
                    another, chosen at random. Out of span, and unjudged --
                    `inner_world.py`'s arm, in the modality that can see
    crossed_judged  N_CAND candidates generated, the one the compatibility rates
                    **most plausible** replayed
    crossed_anti    the same generation, the **least** plausible replayed

That last arm is not decoration. `crossed_judged` draws N_CAND samples and keeps
one, so it differs from `crossed` in two ways at once -- selection *and* extra
sampling. `crossed_anti` draws exactly the same candidates and inverts only the
choice, so the gap between judged and anti is selection alone.

Usage:  python3 benchmarks/heard_world.py out_heard_world.json
"""
import json
import sys

import numpy as np

from neurobrain.audition.audio import Cochleagram
from neurobrain.cognition.multimodal import (AssociationArea,
                                             FactorCompatibility, _unit)
from neurobrain.sensing.natural import load_esc50
from neurobrain.sensing.streams import AuditoryBelt
from neurobrain.vision.widev1 import PopulationAdaptation

SEEDS = (0, 1, 2, 3, 4, 5)
SR, DUR, N_FREQ = 8000, 2.0, 36
N_CONCEPT = 256
TRAIN_FRAC = 0.6
#: How many classes the world has. The full 50-way task was tried first and the
#: benchmark could not answer its own question there: with ~7 training clips per
#: class **every** night hurt, including `stored` -- replaying the real
#: recording. When the positive control fails, the payback channel is inert and
#: a null result for selection means nothing. Restricting to the classes with
#: the most cached clips gives each one enough examples for replay to have
#: something to sharpen, which is what makes the selection arms readable.
N_CLASS = 6
REPLAYS = 400
DREAM_NOVELTY_RATE = 0.02
N_CAND = 12
RANK = 256
ARMS = ("no_dream", "stored", "crossed", "crossed_judged", "crossed_anti")


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return 0.5
    return float((pos[:, None] > neg[None, :]).mean())


def build(seed=0):
    """Real ESC-50 recordings, split into the belt's own two factors.

    ``PopulationAdaptation`` is not optional and the reason is measured
    (`real_time.py`): belt codes are non-negative and share a large component
    across every clip, so without it same-class cosine is 0.986 against
    different-class 0.976 and every downstream layer sees one category."""
    waves, y, _ = load_esc50(n_shards=6, sr=SR, dur_s=DUR, seed=seed)
    y = np.asarray(y, int)
    cls, cnt = np.unique(y, return_counts=True)
    keep = set(cls[np.argsort(-cnt)][:N_CLASS].tolist())
    idx = [i for i, v in enumerate(y) if int(v) in keep]
    remap = {c: k for k, c in enumerate(sorted(keep))}
    waves = [waves[i] for i in idx]
    y = np.array([remap[int(y[i])] for i in idx], int)
    coch = Cochleagram(sr=SR, n_freq=N_FREQ)
    belt = AuditoryBelt(n_freq=N_FREQ, seed=seed)
    pairs = [belt.code_channels(coch.forward(w)[0]) for w in waves]
    E = np.array([p[0] for p in pairs], np.float32)
    T = np.array([p[1] for p in pairs], np.float32)
    for M in (E, T):
        ad = PopulationAdaptation(M.shape[1])
        M[:] = np.array([_unit(ad(r)) for r in M], np.float32)
    return E, T, y


def split(y, seed):
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        cut = max(1, int(round(len(idx) * TRAIN_FRAC)))
        tr.extend(idx[:cut].tolist())
        te.extend(idx[cut:].tolist())
    return np.array(tr), np.array(te)


def wake(E, T, y, tr, seed):
    a = AssociationArea(n_vis=E.shape[1], n_aud=T.shape[1],
                        n_concept=N_CONCEPT, seed=seed)
    a.set_stats(E[tr], T[tr])
    votes = {}
    for i in tr:
        w = a.bind(E[i], T[i])
        votes.setdefault(w, {})
        votes[w][int(y[i])] = votes[w].get(int(y[i]), 0) + 1
    return a, votes


def probe(a, votes, E, T, y, te):
    """Hear a clip, name it -- through the concept layer, held out."""
    name = {c: max(v.items(), key=lambda kv: kv[1])[0] for c, v in votes.items()}
    ok = 0
    for i in te:
        m = a.match(a.prep_v(E[i]), a.prep_a(T[i]))
        ok += int(name.get(int(np.argmax(m)), -1) == int(y[i]))
    return ok / max(len(te), 1)


def night(a, votes, E, T, y, tr, arm, fc, seed):
    if arm == "no_dream":
        return
    rng = np.random.default_rng(seed + 101)
    waking, a.novelty_rate = a.novelty_rate, DREAM_NOVELTY_RATE
    for _ in range(REPLAYS):
        i = int(rng.choice(tr))
        if arm == "stored":
            e, t = E[i], T[i]
        elif arm == "crossed":
            e, t = E[i], T[int(rng.choice(tr))]
        else:
            cand = [int(rng.choice(tr)) for _ in range(N_CAND)]
            s = [fc.score(np.concatenate([E[i], T[j]])) for j in cand]
            pick = cand[int(np.argmax(s) if arm == "crossed_judged"
                            else np.argmin(s))]
            e, t = E[i], T[pick]
        w = a.bind(e, t)
        votes.setdefault(w, {})
        votes[w][int(y[i])] = votes[w].get(int(y[i]), 0) + 1
    a.novelty_rate = waking


def span_residual(Q, B):
    _, s, Vt = np.linalg.svd(B, full_matrices=False)
    R = Vt[s > s.max() * 1e-6]
    r = Q - (Q @ R.T) @ R
    return float(np.mean(np.linalg.norm(r, axis=1)
                         / np.maximum(np.linalg.norm(Q, axis=1), 1e-9)))


def run_seed(E, T, y, seed):
    tr, te = split(y, seed)
    rng = np.random.default_rng(seed + 7)
    X = np.concatenate([E, T], 1)
    nE = E.shape[1]
    bounds = [(0, nE), (nE, X.shape[1])]

    # --- is this front end one a constraint can live on? -------------------
    same, diff = [], []
    for k in te:
        ps = [j for j in tr if y[j] == y[k]]
        pd = [j for j in tr if y[j] != y[k]]
        if not ps or not pd:
            continue
        same.append(float(_unit(X[k]) @ _unit(X[int(rng.choice(ps))])))
        diff.append(float(_unit(X[k]) @ _unit(X[int(rng.choice(pd))])))
    out = {"cluster_auc": auc(same, diff)}

    fc = FactorCompatibility(bounds, rank=RANK, seed=seed)
    for i in tr:
        fc.observe(X[i])
    real = [fc.score(X[k]) for k in te]
    bad = []
    for k in te:
        j = int(rng.choice([q for q in tr if y[q] != y[k]]))
        bad.append(fc.score(np.concatenate([E[k], T[j]])))
    out["compat_auc"] = auc(real, bad)

    # --- and does a crossing leave the span, as it did for the eye? --------
    B = np.stack([_unit(X[i]) for i in tr])
    cross = np.stack([_unit(np.concatenate([E[k], T[int(rng.choice(tr))]]))
                      for k in te[:64]])
    out["cross_residual"] = span_residual(cross, B)
    out["stored_residual"] = span_residual(np.stack([_unit(X[k])
                                                     for k in tr[:64]]), B)

    # --- the night ---------------------------------------------------------
    for arm in ARMS:
        a, votes = wake(E, T, y, tr, seed)
        night(a, votes, E, T, y, tr, arm, fc, seed)
        out[arm] = probe(a, votes, E, T, y, te)
        out[arm + "_cells"] = int((a.wins > 0).sum())
    return out


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_heard_world.json"
    E, T, y = build()
    n_cls = len(np.unique(y))
    print(f"{len(E)} real ESC-50 clips, {n_cls} classes, chance "
          f"{1/n_cls:.3f}")
    print(f"the belt's own factors: envelope {E.shape[1]}d | tonotopic "
          f"{T.shape[1]}d\n", flush=True)

    rows = [run_seed(E, T, y, sd) for sd in SEEDS]
    res = {"seeds": list(SEEDS), "n_class": n_cls,
           "chance": round(1 / n_cls, 4), "per_seed": rows}

    def m(k):
        return float(np.mean([r[k] for r in rows]))

    print(f"does the front end cluster?          {m('cluster_auc'):.3f}   "
          f"(the eye: 0.573)")
    print(f"can the two factors be judged?       {m('compat_auc'):.3f}   "
          f"(the eye: 0.541; a LABELLED oracle: 0.879)")
    print(f"a crossing's span residual           {m('cross_residual'):.4f}   "
          f"(a stored code: {m('stored_residual'):.4f})\n")
    for k in ("cluster_auc", "compat_auc", "cross_residual",
              "stored_residual"):
        res[k] = round(m(k), 4)

    base = np.array([r["no_dream"] for r in rows])
    print(f"{'arm':<16}{'names it':>10}{'vs no_dream':>13}{'d':>7}{'wins':>7}"
          f"{'cells':>7}")
    for arm in ARMS:
        v = np.array([r[arm] for r in rows])
        d = v - base
        sd = float(d.std(ddof=1))
        cd = 0.0 if arm == "no_dream" else float(d.mean() / (sd + 1e-12))
        rec = dict(accuracy=round(float(v.mean()), 4),
                   delta=round(float(d.mean()), 4), sd=round(sd, 4),
                   cohens_d=round(cd, 3), wins=int((d > 0).sum()), n=len(d),
                   cells=round(float(np.mean([r[arm + "_cells"]
                                              for r in rows])), 1))
        res.setdefault("arms", {})[arm] = rec
        print(f"{arm:<16}{rec['accuracy']:>10.3f}{rec['delta']:>+13.4f}"
              f"{cd:>7.2f}{rec['wins']:>4}/{rec['n']}{rec['cells']:>7.1f}")

    # The positive control decides whether anything below is readable. If
    # replaying a REAL recording does not help, the route from a night into
    # perception is inert and a null for selection is a null for the channel.
    st = res["arms"]["stored"]
    res["channel_open"] = bool(st["delta"] > 0 and st["cohens_d"] >= 0.5)
    print(f"\n  positive control -- replaying the real recording: "
          f"{st['delta']:+.4f}, d={st['cohens_d']:+.2f}, "
          f"{st['wins']}/{st['n']}")
    if not res["channel_open"]:
        print("  the night does not help even with real content, so the "
              "payback channel is INERT here and nothing below is readable.")

    j, c, an = (res["arms"]["crossed_judged"], res["arms"]["crossed"],
                res["arms"]["crossed_anti"])
    sel = np.array([r["crossed_judged"] for r in rows]) - \
        np.array([r["crossed_anti"] for r in rows])
    ssd = float(sel.std(ddof=1))
    res["selection_effect"] = dict(
        delta=round(float(sel.mean()), 4), sd=round(ssd, 4),
        cohens_d=round(float(sel.mean() / (ssd + 1e-12)), 3),
        wins=int((sel > 0).sum()), n=len(sel))
    s = res["selection_effect"]

    print("\n=== was JUDGING the imagining what was missing? ===")
    print(f"  unjudged crossings      {c['delta']:+.4f} (d={c['cohens_d']:+.2f},"
          f" {c['wins']}/{c['n']})")
    print(f"  judged, most plausible  {j['delta']:+.4f} "
          f"(d={j['cohens_d']:+.2f}, {j['wins']}/{j['n']})")
    print(f"  judged, LEAST plausible {an['delta']:+.4f} "
          f"(d={an['cohens_d']:+.2f}, {an['wins']}/{an['n']})")
    print(f"\n  selection alone (most minus least, same candidates): "
          f"{s['delta']:+.4f}, d={s['cohens_d']:+.2f}, {s['wins']}/{s['n']}")
    ok = s["cohens_d"] >= 0.8 and s["wins"] >= 0.75 * s["n"]
    res["selection_works"] = bool(ok)
    if ok:
        print("\n  -> judging an imagining changes what it is worth. The two "
              "arms draw the SAME candidates and differ only in which one is")
        print("     kept, so this is the selection and nothing else. What was "
              "missing from the night was never novelty -- it was a reason to")
        print("     prefer one imagining over another, and a front end clear "
              "enough to supply one.")
    else:
        print(f"\n  -> selection does not change the outcome "
              f"({s['delta']:+.4f}, d={s['cohens_d']:+.2f}). The judgement "
              f"discriminates at AUC {m('compat_auc'):.3f} and still does not "
              f"buy anything when used to choose.")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
