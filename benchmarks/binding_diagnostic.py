"""Is StreamingBrain.bind actually binding anything?"""
import json, sys
import numpy as np
import neurobrain as nb
from neurobrain.sensing.streams import StreamingBrain, _unit
from neurobrain.audition.audio import sound_dataset

out = {}
trx, trY, tex, teY = nb.load_mnist(n_train=1200, n_test=400)
sigs, labels, names = sound_dataset(n_per_class=8, seed=3)
labels = np.asarray(labels)
ncls = len(names)


def codes(brain, idx):
    return [brain.belt.code(brain.ear.coch.forward(sigs[i])[0]) for i in idx]


# ---- 1. exact-code query (what the shipped test does) ---------------------
def exact(shuffled, seed=0):
    rng = np.random.default_rng(seed)
    brain = StreamingBrain(seed=seed)
    train = [np.where(labels == c)[0][:4] for c in range(ncls)]
    pairs = []
    for c in range(ncls):
        for i in train[c]:
            v = _unit(brain.v1.rate(trx[np.where(trY == c % 10)[0][0]]))
            pairs.append((v, brain.belt.code(brain.ear.coch.forward(sigs[i])[0]), c))
    tgt = [p[2] for p in pairs]
    if shuffled:
        tgt = list(rng.permutation(tgt))
    for (v, a, _), lab in zip(pairs, tgt):
        brain.bind(v, a, int(lab))
    ok = sum(int(brain.recall_visual_from_sound(a) == lab)
             for (_, a, _), lab in zip(pairs, tgt))
    return round(ok / len(pairs), 4)


# ---- 2. held-out query: a DIFFERENT clip of the same class ---------------
def heldout(shuffled, seed=0):
    """The honest test. If binding learned the class, a new clip of that class
    should still recall it. If it only memorised codes, this collapses."""
    rng = np.random.default_rng(seed)
    brain = StreamingBrain(seed=seed)
    tr_idx, te_idx = [], []
    for c in range(ncls):
        idx = np.where(labels == c)[0]
        tr_idx += list(idx[:4]); te_idx += list(idx[4:8])
    pairs = []
    for i in tr_idx:
        c = int(labels[i])
        v = _unit(brain.v1.rate(trx[np.where(trY == c % 10)[0][0]]))
        pairs.append((v, brain.belt.code(brain.ear.coch.forward(sigs[i])[0]), c))
    tgt = [p[2] for p in pairs]
    if shuffled:
        tgt = list(rng.permutation(tgt))
    for (v, a, _), lab in zip(pairs, tgt):
        brain.bind(v, a, int(lab))
    ok = n = 0
    for i in te_idx:
        c = int(labels[i])
        a = brain.belt.code(brain.ear.coch.forward(sigs[i])[0])
        got = brain.recall_visual_from_sound(a)
        if got is None:
            continue
        n += 1
        ok += int(got == c)
    return round(ok / max(n, 1), 4), n


# ---- 3. is v_code used at all? -------------------------------------------
def vcode_matters(seed=0):
    """Bind with real visual codes, then with pure noise in the visual slot.
    If the answer is identical, the visual code is dead weight."""
    res = {}
    for mode in ("real", "noise", "zeros"):
        rng = np.random.default_rng(7)
        brain = StreamingBrain(seed=seed)
        tr, te = [], []
        for c in range(ncls):
            idx = np.where(labels == c)[0]
            tr += list(idx[:4]); te += list(idx[4:8])
        for i in tr:
            c = int(labels[i])
            v = _unit(brain.v1.rate(trx[np.where(trY == c % 10)[0][0]]))
            if mode == "noise":
                v = rng.standard_normal(v.shape).astype(np.float32)
            elif mode == "zeros":
                v = np.zeros_like(v)
            brain.bind(v, brain.belt.code(brain.ear.coch.forward(sigs[i])[0]), c)
        ok = n = 0
        for i in te:
            a = brain.belt.code(brain.ear.coch.forward(sigs[i])[0])
            got = brain.recall_visual_from_sound(a)
            if got is not None:
                n += 1
                ok += int(got == int(labels[i]))
        res[mode] = round(ok / max(n, 1), 4)
    return res


# ---- 4. does bind() touch the workspace it claims to use? ----------------
def touches_workspace():
    brain = StreamingBrain(seed=0)
    before = int(brain.ws.n_concepts)
    v = _unit(brain.v1.rate(trx[0]))
    a = brain.belt.code(brain.ear.coch.forward(sigs[0])[0])
    for _ in range(50):
        brain.bind(v, a, 0)
    return dict(ws_concepts_before=before, ws_concepts_after=int(brain.ws.n_concepts),
                bindings_stored=len(brain.bindings),
                storage="python list, linear scan per recall, never pruned")


out["exact_query"] = {"real": exact(False), "shuffled": exact(True)}
ho_r, n_r = heldout(False); ho_s, _ = heldout(True)
out["heldout_query"] = {"real": ho_r, "shuffled": ho_s, "n_queries": n_r,
                        "chance": round(1 / ncls, 4)}
out["vcode_ablation"] = vcode_matters()
out["workspace"] = touches_workspace()
print(json.dumps(out, indent=1))
json.dump(out, open(sys.argv[1], "w"), indent=1)
