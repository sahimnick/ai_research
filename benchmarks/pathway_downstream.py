"""What the pathways do once a concept layer, a night and an imagination sit on
top of them.

`pathways.py` measured the representations: cluster AUC, same-object invariance,
and a prototype read-out that adds no capacity of its own. It rejected the
object-centred architecture (H2 not met) and found, in the control that removes
the object frame, the largest movement of clustering in this project -- local
receptive fields with **frame-absolute** position, 0.630 against the current
pathway's 0.571, d = 2.18, 4 of 4.

Requirements 4, 10d and 10e are the rest: feed these codes to a **concept
layer**, and measure **replay** and **imagination** on top of each.

Hypotheses, before the run
--------------------------
**H6 (the concept layer tracks the representation).** Naming accuracy through
the concept layer will order the pathways the way cluster AUC does -- so
local-absolute first -- and **not** the way invariance does. If instead it tracks
invariance, or orders them differently from both, then cluster AUC is not
measuring what the layer consumes and `pathways.py`'s rejection was decided on
the wrong quantity.

**H7 (replay is a property of the rule, not the code).** `payback.py` found that
*learning against* an imagining beats *believing* it (+0.0833 against −0.0174 on
archives, +0.0236 against +0.0024 on live cameras). That is a claim about the
learning rule, so it should reproduce on **every** pathway. If it appears on some
and not others it was a property of the old representation all along.

**H8 (leaving the span is algebraic).** `factored.py` showed mixing whole codes
cannot escape the span of stored codes while crossing *factors* can, because one
scalar would have to be both 1 and 0. That argument does not mention the eye, so
the span residual should be ~0 for mixing and >0 for crossing on every pathway.
The new pathway makes the factors *spatial* -- upper bins against lower bins --
so a crossing is the top of one object carrying the bottom of another, which is
a part recombination rather than a colour swap.

**H9 (the rejection stands downstream).** The architecture rejected on its
representation should not be rescued here. If `local-spatial` beats
`local-absolute` downstream despite clustering worse, that is requirement 11's
exact failure mode -- an architecture that changes downstream learning without
improving the representation -- and it would have to be reported as such rather
than as a win.

Usage:  python3 benchmarks/pathway_downstream.py out_pathway_downstream.json
"""
import json
import sys

import numpy as np

from neurobrain.cognition.multimodal import AssociationArea, _unit
from neurobrain.learning.selforganize import develop_v1
from neurobrain.sensing.natural import load_audiovisual
from neurobrain.sensing.streams import StreamingBrain
from neurobrain.vision.localspatial import LocalSpatialEye
from neurobrain.vision.widev1 import PopulationAdaptation, WideV1

sys.path.insert(0, "benchmarks")
from pathways import FRAME, cluster_auc, place                # noqa: E402
from real_binding import split                                # noqa: E402

SEEDS = (0, 1, 2, 3)
N_IMAGES = 240
N_CONCEPT = 256
REPLAYS = 300
DREAM_NOVELTY_RATE = 0.02
PATHWAYS = ("rate", "relational", "local-spatial", "local-absolute")
NIGHTS = ("no_dream", "stored", "imagined_as_fact", "imagined_as_fact_fixed",
          "imagined_as_error")


_CACHE = {}


def encoder(path, frames, seed):
    """One coder per pathway, and the factor split each one offers.

    The split is what an imagined *crossing* recombines, and it is **not the
    same kind of thing** on every pathway -- which is itself part of what H8
    tests, so it is stated exactly rather than described as one split:

    * **local-spatial / local-absolute** -- position bins, row-major. The top
      row of bins comes from one concept and the middle and bottom rows from
      the other, so the crossing is the top of one object carried on the rest
      of another. A genuine part recombination.
    * **rate** -- the retinotopic grid by `cell_pos`, upper half against lower.
      The same spatial meaning, at whole-image resolution.
    * **relational** -- by *relative displacement*, because that code has no
      absolute spatial layout to split; summing over position is exactly what
      makes it invariant. Its crossing takes some co-occurrence offsets from
      one concept and the rest from another, which is a different operation
      from a part swap and cannot be reported as the same one.

    Banks are cached across the pathways that build them identically, for the
    reason `pathways._bank_key` gives: `rate` and `relational` read the same
    global bank, `local-spatial` and `local-absolute` the same patch bank, and
    growing each its own would vary the random draw as well as the component
    under test.
    """
    if path in ("rate", "relational"):
        gk = ("global", seed)
        if gk in _CACHE:
            v1 = _CACHE[gk]
        else:
            v1 = WideV1(n_cells=4096, window_ms=50, rf=7, stride=2,
                        image_shape=(FRAME, FRAME), seed=seed)
            develop_v1(v1, list(frames), epochs=3, tie=True, seed=seed)
            _CACHE[gk] = v1
        if path == "rate":
            code = lambda f: _unit(v1.drive(f))               # noqa: E731
            top = np.flatnonzero(v1.cell_pos < v1.n_pos // 2)
            dim = v1.n_cells
        else:
            code = lambda f: v1.relational_code(f, n_feat=12)  # noqa: E731
            probe = code(frames[0])
            dim = probe.shape[0]
            top = np.arange(dim // 2)
        return code, dim, top
    eye = LocalSpatialEye(image_shape=(FRAME, FRAME), seed=seed, spiking=False,
                          centred=(path == "local-spatial"))
    lk = ("local", seed)
    if lk in _CACHE:
        eye.banks = _CACHE[lk]               # the identical developed bank
    else:
        eye.develop(list(frames), epochs=3, seed=seed)
        _CACHE[lk] = eye.banks
    dim = eye.dim
    # `code` is (n_feat, bins*bins) flattened row-major, and `position_code`
    # lays bins out as `yy * bins + xx`. So bin b sits in row b // bins, and
    # the top rows are the ones with `b // bins < bins // 2`. Taking the first
    # (bins*bins)//2 indices instead -- the obvious thing -- would be the top
    # row plus one cell of the middle row at bins=3, which is not the split
    # the crossing is supposed to be.
    nb = eye.bins * eye.bins
    keep = np.array([f * nb + b for f in range(eye.n_feat)
                     for b in range(nb) if b // eye.bins < eye.bins // 2])
    return eye.code, dim, keep


def codes(coder, frames):
    R = np.array([coder(f) for f in frames], np.float32)
    ad = PopulationAdaptation(R.shape[1])
    return np.array([_unit(ad(r)) for r in R], np.float32)


def night(a, votes, V, A, y, tr, arm, seed):
    """The payback protocol. Every arm replays the same pairs in the same
    order; only the rule differs.

    ``imagined_as_fact_fixed`` is a correction, not a variant, and it has to be
    here because the arm it corrects carries a published number.

    ``imagine_from_sound`` returns a vector in **prep_v** space -- that is the
    space ``Wv`` lives in, and ``bind_contrastive`` uses it directly and
    correctly. But :meth:`bind` preps whatever it is handed, so
    ``bind(imagine_from_sound(...), a)`` normalises an already-normalised
    vector. Measured on this data: what gets bound sits at **cos 0.407** to the
    fantasy the model actually generated, and it wakes a *different* concept
    cell 21 times out of 36.

    So `payback.py`'s ``imagined_as_fact`` was not believing the imagining; it
    was believing a distorted version of it, while ``imagined_as_error`` saw
    the undistorted one. The gap between them -- the project's headline replay
    result -- was confounded by that difference. ``_fixed`` un-preps before
    binding, which is an exact inverse (``prep_v(v_hat * sd + mu) == v_hat``,
    measured cos 1.0), so the two arms finally differ only in what the rule
    does with the fantasy. Both are reported; the uncorrected one stays so the
    old number remains readable.
    """
    if arm == "no_dream":
        return votes
    rng = np.random.default_rng(seed + 101)
    waking, a.novelty_rate = a.novelty_rate, DREAM_NOVELTY_RATE
    for i in [int(rng.choice(tr)) for _ in range(REPLAYS)]:
        if arm == "stored":
            w = a.bind(V[i], A[i])
        elif arm == "imagined_as_fact":
            w = a.bind(a.imagine_from_sound(A[i], temperature=1.0, rng=rng),
                       A[i])
        elif arm == "imagined_as_fact_fixed":
            v_hat = a.imagine_from_sound(A[i], temperature=1.0, rng=rng)
            w = a.bind(v_hat * a.v_sd + a.v_mu, A[i])
        else:
            w, _, _ = a.bind_contrastive(V[i], A[i], temperature=1.0, rng=rng)
        votes.setdefault(w, {})
        votes[w][int(y[i])] = votes[w].get(int(y[i]), 0) + 1
    a.novelty_rate = waking
    return votes


def span_residual(Q, B):
    U, s, Vt = np.linalg.svd(B, full_matrices=False)
    R = Vt[s > s.max() * 1e-6]
    r = Q - (Q @ R.T) @ R
    return (np.linalg.norm(r, axis=1)
            / np.maximum(np.linalg.norm(Q, axis=1), 1e-9))


def run_seed(images, waves, y, n_cls, seed):
    tr, te = split(y, seed)
    frames = [place(im) for im in images]
    brain = StreamingBrain(seed=seed, image_shape=(FRAME, FRAME))
    A = np.array([brain.belt.code(brain.ear.coch.forward(w)[0])
                  for w in waves], np.float32)
    out = {}
    for path in PATHWAYS:
        coder, dim, keep = encoder(path, frames, seed)
        V = codes(coder, frames)
        rec = {"dim": int(V.shape[1])}
        rng = np.random.default_rng(seed)
        rec["cluster_auc"] = cluster_auc(V[te], y[te], V[tr], y[tr], rng)
        # The floor the concept layer has to clear. A 1-NN over the training
        # codes uses the representation and adds nothing else. If the layer's
        # naming ties this, the layer is an exemplar store and `see->name` is
        # reading the representation, not the layer -- which is the fact the
        # cells-per-pair number states directly.
        S = V[te] @ V[tr].T
        rec["1nn"] = float((y[tr][np.argmax(S, 1)] == y[te]).mean())

        # ---- requirement 4: a concept layer on top ------------------------
        for arm in NIGHTS:
            a = AssociationArea(n_vis=V.shape[1], n_aud=A.shape[1],
                                n_concept=N_CONCEPT, seed=seed)
            a.set_stats(V[tr], A[tr])
            votes = {}
            for i in tr:
                w = a.bind(V[i], A[i])
                votes.setdefault(w, {})
                votes[w][int(y[i])] = votes[w].get(int(y[i]), 0) + 1
            votes = night(a, votes, V, A, y, tr, arm, seed)
            name = {c: max(v.items(), key=lambda kv: kv[1])[0]
                    for c, v in votes.items() if v}
            # `a.Wv` rows live in **prep_v** space -- mean-subtracted and
            # variance-normalised -- so the prototypes they are scored against
            # have to be built there too. Built from raw `V` instead, the same
            # read-out scored 0.375 where the correct space gives 0.542: the
            # space mismatch this project has now hit four times, and it does
            # not announce itself because both numbers are above chance.
            protos = np.stack([
                _unit(np.stack([a.prep_v(V[i]) for i in tr[y[tr] == c]]).mean(0))
                for c in range(n_cls)])
            s2v = sum(int(int(np.argmax(protos @ _unit(a.Wv[
                a.concept_from_sound(A[i])]))) == int(y[i])) for i in te)
            v2n = sum(int(name.get(a.concept_from_vision(V[i]), -1)
                          == int(y[i])) for i in te)
            rec[f"{arm}: sound->vision"] = s2v / max(len(te), 1)
            rec[f"{arm}: see->name"] = v2n / max(len(te), 1)
            if arm == "no_dream":
                rec["cells"] = int((a.wins > 0).sum())
                rec["cells per pair"] = rec["cells"] / max(len(tr), 1)
                # ---- requirement 10e: imagination on this pathway ---------
                live = [int(c) for c in np.flatnonzero(a.wins > 0)]
                B = np.stack([a.prep_v(V[i]) for i in tr])
                if len(live) >= 2:
                    mix = np.stack([
                        a.imagine_composite([live[k % len(live)],
                                             live[(k + 1) % len(live)]],
                                            temperature=4.0,
                                            rng=np.random.default_rng(seed + k))
                        for k in range(24)])
                    other = np.setdiff1d(np.arange(dim), keep)
                    cross = []
                    for k in range(24):
                        c1 = a.Wv[live[k % len(live)]]
                        c2 = a.Wv[live[(k + 1) % len(live)]]
                        v = np.zeros(dim, np.float32)
                        v[keep] = c1[keep]
                        v[other] = c2[other]
                        cross.append(_unit(v))
                    rec["mix span residual"] = float(
                        span_residual(mix, B).mean())
                    rec["cross span residual"] = float(
                        span_residual(np.stack(cross), B).mean())
        out[path] = rec
        print(f"    {path:<16} AUC {rec['cluster_auc']:.3f}  see->name "
              f"{rec['no_dream: see->name']:.3f}  cross-span "
              f"{rec.get('cross span residual', float('nan')):.3f}", flush=True)
    return out


def main():
    out_path = (sys.argv[1] if len(sys.argv) > 1
                else "out_pathway_downstream.json")
    images, waves, y, names = load_audiovisual(n_per_class=N_IMAGES // 6,
                                               seed=0, grayscale=True, size=32)
    y = np.asarray(y, int)
    n_cls = len(names)
    print(f"{len(images)} photographs + real ESC-50 recordings, {n_cls} "
          f"categories, chance {1/n_cls:.3f}\n", flush=True)

    rows = []
    for sd in SEEDS:
        print(f"  seed {sd}", flush=True)
        rows.append(run_seed(images, waves, y, n_cls, sd))

    KEYS = (["cluster_auc", "1nn", "cells", "cells per pair",
             "mix span residual", "cross span residual"]
            + [f"{a}: {m}" for a in NIGHTS
               for m in ("sound->vision", "see->name")])
    res = {"seeds": list(SEEDS), "per_seed": rows, "arms": {}}
    for p in PATHWAYS:
        res["arms"][p] = {k: round(float(np.mean([r[p][k] for r in rows])), 4)
                          for k in KEYS if k in rows[0][p]}

    print(f"\n{'pathway':<16}{'AUC':>8}{'1-NN':>8}{'cells':>8}{'/pair':>8}"
          f"{'see->name':>11}{'sound->vis':>12}")
    for p in PATHWAYS:
        v = res["arms"][p]
        print(f"{p:<16}{v['cluster_auc']:>8.3f}{v['1nn']:>8.3f}"
              f"{v['cells']:>8.1f}{v['cells per pair']:>8.2f}"
              f"{v['no_dream: see->name']:>11.3f}"
              f"{v['no_dream: sound->vision']:>12.3f}")

    print(f"\n--- H6: does the concept layer track the REPRESENTATION? ---")
    by_auc = sorted(PATHWAYS, key=lambda p: -res["arms"][p]["cluster_auc"])
    by_nn = sorted(PATHWAYS, key=lambda p: -res["arms"][p]["1nn"])
    by_name = sorted(PATHWAYS,
                     key=lambda p: -res["arms"][p]["no_dream: see->name"])
    print(f"  ordered by cluster AUC : {' > '.join(by_auc)}")
    print(f"  ordered by 1-NN        : {' > '.join(by_nn)}")
    print(f"  ordered by see->name   : {' > '.join(by_name)}")
    res["H6_orders_agree"] = bool(by_auc[0] == by_name[0])
    print(f"  same pathway on top as cluster AUC: {res['H6_orders_agree']}")
    # The layer has to beat the representation read directly, or the ordering
    # it produces is the representation's ordering and H6 is untestable rather
    # than confirmed.
    gain = float(np.mean([res["arms"][p]["no_dream: see->name"]
                          - res["arms"][p]["1nn"] for p in PATHWAYS]))
    res["H6_layer_beats_1nn"] = bool(gain > 0.02)
    print(f"  concept layer minus 1-NN, averaged over pathways: {gain:+.4f}"
          f"  -> layer adds capacity: {res['H6_layer_beats_1nn']}")

    print(f"\n--- H7: does replay behave the same on every pathway? ---")
    print(f"  sound->vision deltas against no_dream. `stored` is the positive "
          f"control that says whether the payback channel is open at all;")
    print(f"  `believe (fixed)` is the arm that binds the SAME fantasy in the "
          f"space the model generated it in -- see night()'s docstring.")
    print(f"{'pathway':<16}{'stored':>10}{'believe':>10}"
          f"{'believe(fixed)':>16}{'learn against':>15}")
    for p in PATHWAYS:
        v = res["arms"][p]
        base = v["no_dream: sound->vision"]
        s_ = v["stored: sound->vision"] - base
        f_ = v["imagined_as_fact: sound->vision"] - base
        x_ = v["imagined_as_fact_fixed: sound->vision"] - base
        e_ = v["imagined_as_error: sound->vision"] - base
        res["arms"][p]["replay stored"] = round(float(s_), 4)
        res["arms"][p]["replay believe"] = round(float(f_), 4)
        res["arms"][p]["replay believe fixed"] = round(float(x_), 4)
        res["arms"][p]["replay against"] = round(float(e_), 4)
        print(f"{p:<16}{s_:>+10.4f}{f_:>+10.4f}{x_:>+16.4f}{e_:>+15.4f}")
    agree = all(res["arms"][p]["replay against"]
                >= res["arms"][p]["replay believe"] for p in PATHWAYS)
    fair = all(res["arms"][p]["replay against"]
               >= res["arms"][p]["replay believe fixed"] for p in PATHWAYS)
    res["H7_holds"] = bool(agree)
    res["H7_holds_against_fixed"] = bool(fair)
    print(f"  learning-against >= believing, on every pathway ....... {agree}")
    print(f"  ... and >= believing the UNDISTORTED fantasy .......... {fair}")
    if agree and not fair:
        print("  The claim survives only against the arm with the space bug in "
              "it. `payback.py`'s replay result is confounded and has to be "
              "re-reported against the corrected arm.")

    print(f"\n--- H8: is leaving the span algebraic? ---")
    print(f"{'pathway':<16}{'mix':>10}{'crossing':>12}")
    for p in PATHWAYS:
        v = res["arms"][p]
        print(f"{p:<16}{v.get('mix span residual', float('nan')):>10.4f}"
              f"{v.get('cross span residual', float('nan')):>12.4f}")
    ok = all(res["arms"][p].get("mix span residual", 1) < 1e-3
             and res["arms"][p].get("cross span residual", 0) > 1e-3
             for p in PATHWAYS)
    res["H8_holds"] = bool(ok)
    print(f"  mixing stays inside and crossing escapes, on every pathway: {ok}")

    print(f"\n--- H9: does the rejected architecture get rescued downstream? ---")
    spec, best = res["arms"]["local-spatial"], res["arms"]["local-absolute"]
    d = np.array([r["local-spatial"]["no_dream: see->name"]
                  - r["local-absolute"]["no_dream: see->name"] for r in rows])
    res["H9_rescued"] = bool(d.mean() > 0.02 and (d > 0).sum() >= 3)
    print(f"  local-spatial (rejected, AUC {spec['cluster_auc']:.3f}) "
          f"see->name {spec['no_dream: see->name']:.3f}")
    print(f"  local-absolute (kept,    AUC {best['cluster_auc']:.3f}) "
          f"see->name {best['no_dream: see->name']:.3f}")
    if res["H9_rescued"]:
        print("  The rejected one wins downstream while clustering worse -- "
              "requirement 11's exact failure mode, and it has to be reported "
              "as that rather than as a win.")
    else:
        print("  It is not rescued. The rejection stands on both the "
              "representation and what sits on top of it.")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
