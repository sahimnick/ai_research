"""Vision pathway on two real datasets, with controls."""
import json, time, sys
import numpy as np
import neurobrain as nb
from neurobrain.vision.widev1 import WideV1, _encode, grown_readout
from neurobrain.vision.spikingvision import SpikingCategoryMap
from neurobrain.learning.selforganize import develop_v1

N_TRAIN, N_TEST, N_FIT, N_CELLS = 2500, 1000, 400, 1024
out = {}


def run(name, loader, filters):
    """filters: 'designed' (seeded gabor-ish) or 'developed' (self-organised)."""
    t0 = time.time()
    trx, trY, tex, teY = loader(n_train=N_TRAIN, n_test=N_TEST)
    t_load = time.time() - t0

    t0 = time.time()
    layer = WideV1(n_cells=N_CELLS, window_ms=50, seed=0)
    if filters == "designed":
        layer.train(trx[:N_FIT], epochs=1)
    else:
        layer = develop_v1(layer, trx[:N_FIT], epochs=3, seed=0)
    t_fit = time.time() - t0

    t0 = time.time()
    Xtr, Xte = _encode(layer, trx), _encode(layer, tex)
    t_enc = time.time() - t0

    acc, ncat, vig = grown_readout(Xtr, trY, Xte, teY)
    cortex = SpikingCategoryMap(dim=Xtr.shape[1], vigilance=vig, max_cells=6000)
    for x in Xtr:
        cortex.learn(x)
    cortex.name_cells(Xtr, trY)

    pred = np.array([int(cortex.recognise(x[None])[0]) for x in Xte])
    hit = pred == teY
    # per-class recall tells us *what* it confuses, not just how often
    per_class = {int(c): round(float(hit[teY == c].mean()), 4)
                 for c in sorted(set(teY.tolist()))}
    # a "don't know" answer is -1
    dunno = float((pred < 0).mean())

    r = dict(
        dataset=name, filters=filters,
        accuracy=round(float(hit.mean()), 4),
        chance=round(1.0 / len(per_class), 4),
        grown_readout_acc=round(float(acc), 4),
        categories=int(ncat), vigilance=round(float(vig), 4),
        compression=round(N_TRAIN / max(ncat, 1), 2),
        silent_fraction=round(float(layer.sparsity(tex[:60])), 4),
        dont_know_rate=round(dunno, 4),
        per_class_recall=per_class,
        worst_class=min(per_class, key=per_class.get),
        worst_class_recall=min(per_class.values()),
        sec_load=round(t_load, 2), sec_fit=round(t_fit, 2),
        sec_encode=round(t_enc, 2),
        images_per_sec=round((N_TRAIN + N_TEST) / max(t_enc, 1e-9), 1),
    )
    print(json.dumps(r), flush=True)
    return r


jobs = [
    ("mnist",   nb.load_mnist,         "designed"),
    ("fashion", nb.load_fashion_mnist, "designed"),
    ("mnist",   nb.load_mnist,         "developed"),
    ("fashion", nb.load_fashion_mnist, "developed"),
]
res = []
for name, loader, f in jobs:
    try:
        res.append(run(name, loader, f))
    except Exception as e:
        import traceback
        res.append(dict(dataset=name, filters=f, ERROR=f"{type(e).__name__}: {e}",
                        tb=traceback.format_exc()[-600:]))
        print(json.dumps(res[-1]), flush=True)

json.dump(res, open(sys.argv[1], "w"), indent=1)
