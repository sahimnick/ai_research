"""Is StreamingBrain.bind actually binding vision to sound?

Three probes, each with a control:

1. **exact-code query** -- bind a code, query with the *same* code. This is a
   lookup, and the shuffled-label control proves it: both score 100%.
2. **held-out query** -- query with a different clip of the same class. The
   honest test.
3. **visual ablation** -- replace every visual code with noise, then with
   zeros. If the answer does not move, vision is not participating.

Probe 3 is the one that matters. Everything runs twice: `via="nearest"` is the
original stored-label path, `via="assoc"` routes through the concept cells.
"""
import json, sys
import numpy as np
import neurobrain as nb
from neurobrain.sensing.streams import StreamingBrain, _unit
from neurobrain.audition.audio import sound_dataset

VIAS = ("nearest", "assoc")
out = {}
trx, trY, tex, teY = nb.load_mnist(n_train=1200, n_test=400)
sigs, labels, names = sound_dataset(n_per_class=8, seed=3)
labels = np.asarray(labels)
ncls = len(names)


def _vis_for(brain, c, k=0):
    idx = np.where(trY == c % 10)[0]
    return _unit(brain.v1.rate(trx[idx[k % len(idx)]]))


def _aud(brain, i):
    return brain.belt.code(brain.ear.coch.forward(sigs[i])[0])


def _train_test_split():
    tr, te = [], []
    for c in range(ncls):
        idx = np.where(labels == c)[0]
        tr += list(idx[:4]); te += list(idx[4:8])
    return tr, te


def _fit(brain, tr_idx, shuffled=False, vis_mode="real", seed=0):
    rng = np.random.default_rng(seed)
    pairs = []
    for k, i in enumerate(tr_idx):
        c = int(labels[i])
        v = _vis_for(brain, c, k)
        if vis_mode == "noise":
            v = _unit(rng.standard_normal(v.shape).astype(np.float32))
        elif vis_mode == "zeros":
            v = np.zeros_like(v)
        pairs.append((v, _aud(brain, i), c))
    tgt = [p[2] for p in pairs]
    if shuffled:
        tgt = list(rng.permutation(tgt))
    brain.calibrate(np.array([p[0] for p in pairs], np.float32),
                    np.array([p[1] for p in pairs], np.float32))
    for (v, a, _), lab in zip(pairs, tgt):
        brain.bind(v, a, int(lab))
    return pairs, tgt


def score(query, brain, via, pairs=None, tgt=None, te_idx=None):
    ok = n = 0
    if query == "exact":
        for (_, a, _), lab in zip(pairs, tgt):
            got = brain.recall_visual_from_sound(a, via=via)
            if got is not None:
                n += 1; ok += int(got == lab)
    else:
        for i in te_idx:
            got = brain.recall_visual_from_sound(_aud(brain, i), via=via)
            if got is not None:
                n += 1; ok += int(got == int(labels[i]))
    return round(ok / max(n, 1), 4), n


tr_idx, te_idx = _train_test_split()

for via in VIAS:
    r = {}
    b = StreamingBrain(seed=0); p, t = _fit(b, tr_idx, shuffled=False)
    r["exact_real"], _ = score("exact", b, via, pairs=p, tgt=t)
    b = StreamingBrain(seed=0); p, t = _fit(b, tr_idx, shuffled=True)
    r["exact_shuffled"], _ = score("exact", b, via, pairs=p, tgt=t)

    b = StreamingBrain(seed=0); _fit(b, tr_idx, shuffled=False)
    r["heldout_real"], r["n_queries"] = score("heldout", b, via, te_idx=te_idx)
    b = StreamingBrain(seed=0); _fit(b, tr_idx, shuffled=True)
    r["heldout_shuffled"], _ = score("heldout", b, via, te_idx=te_idx)

    for mode in ("real", "noise", "zeros"):
        b = StreamingBrain(seed=0); _fit(b, tr_idx, vis_mode=mode)
        r[f"ablation_{mode}"], _ = score("heldout", b, via, te_idx=te_idx)
    vals = [r["ablation_real"], r["ablation_noise"], r["ablation_zeros"]]
    r["ablation_spread"] = round(max(vals) - min(vals), 4)
    r["vision_participates"] = bool(r["ablation_spread"] > 0.05)
    r["chance"] = round(1 / ncls, 4)
    out[via] = r
    print(via, json.dumps(r), flush=True)

# cross-modal CODE recall -- only the rewritten path can express this
b = StreamingBrain(seed=0)
_fit(b, tr_idx)
proto_v = {c: _vis_for(b, c) for c in range(ncls)}
ok = n = 0
for i in te_idx:
    vc = b.recall_visual_code_from_sound(_aud(b, i))
    if vc is None:
        continue
    best = max(proto_v, key=lambda c: float(_unit(vc) @ _unit(proto_v[c])))
    n += 1; ok += int(best == int(labels[i]))
out["code_recall"] = dict(
    accuracy=round(ok / max(n, 1), 4), n=n, chance=round(1 / ncls, 4),
    note="sound -> visual CODE -> nearest visual prototype; no label read out")
out["reverse_direction_available"] = b.recall_sound_code_from_vision(
    _vis_for(b, 0)) is not None
print("code_recall", json.dumps(out["code_recall"]), flush=True)

json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "out_binding.json", "w"),
          indent=1)
print(json.dumps(out, indent=1))
