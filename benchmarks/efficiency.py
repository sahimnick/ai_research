"""Where does the time and memory actually go?"""
import json, time, sys, os, gc, resource, traceback
import numpy as np
import neurobrain as nb

out = {}


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # KiB->MiB on Linux


def timeit(fn, n=1):
    gc.collect()
    t0 = time.perf_counter()
    for _ in range(n):
        r = fn()
    return (time.perf_counter() - t0) / n, r


# ------------------------------------------------- per-stage cost of one glance
def stage_costs():
    from neurobrain.vision.widev1 import WideV1
    from neurobrain.sensing.streams import (StreamingBrain, build_scene, SaccadicEye,
                                            build_soundscape, _unit)
    trx, trY, tex, teY = nb.load_mnist(n_train=600, n_test=200)
    b = StreamingBrain(seed=0)
    scene = build_scene(trx, trY, size=256, n_objects=12, seed=0)
    eye = SaccadicEye(scene, seed=0)
    scape = build_soundscape(n_events=12, seed=0)
    img = trx[0]
    fix = eye.fixate(*eye.next_target())

    rows = {}
    rows["v1_rate_single_image"], _ = timeit(lambda: b.v1.rate(img), 20)
    rows["v1_rate_over_fixation"], _ = timeit(lambda: b.v1.rate_over(fix.frames), 20)
    rows["eye_next_target"], _ = timeit(lambda: eye.next_target(), 20)
    rows["eye_fixate"], _ = timeit(lambda: eye.fixate(60, 60), 20)
    rows["cochleagram"], _ = timeit(lambda: b.ear.coch.forward(scape.wave[:8000])[0], 10)
    rows["ear_detect_onsets_full"], _ = timeit(lambda: b.ear.detect_onsets(scape.wave), 3)
    coch = b.ear.coch.forward(scape.wave[:8000])[0]
    rows["belt_code"], _ = timeit(lambda: b.belt.code(coch), 20)
    r = b.v1.rate(img)
    rows["workspace_encode"], _ = timeit(lambda: b.ws.encode("vision", _unit(r)), 50)
    return {k: round(v * 1000, 3) for k, v in rows.items()}   # ms


# ------------------------------------------------------------- V1 scaling
def v1_scaling(sizes=(256, 512, 1024, 2048, 4096)):
    from neurobrain.vision.widev1 import WideV1
    trx, trY, tex, teY = nb.load_mnist(n_train=200, n_test=50)
    rows = []
    for n in sizes:
        gc.collect(); base = rss_mb()
        t0 = time.perf_counter()
        layer = WideV1(n_cells=n, seed=0)
        t_build = time.perf_counter() - t0
        t, _ = timeit(lambda: layer.rate(trx[0]), 10)
        w = getattr(layer, "W", None)
        rows.append(dict(n_cells=n, sec_build=round(t_build, 3),
                         ms_per_image=round(t * 1000, 3),
                         images_per_sec=round(1 / max(t, 1e-9), 1),
                         filter_bytes_mb=round(w.nbytes / 1e6, 2) if w is not None else None,
                         rss_after_mb=round(rss_mb(), 1)))
        print("v1", json.dumps(rows[-1]), flush=True)
    return rows


# --------------------------------------------------- the neuron/synapse core
def substrate_scaling(sizes=(1000, 10000, 100000)):
    from neurobrain.core.neuron import Population
    from neurobrain.core.synapse import SynapseBundle
    rows = []
    for n in sizes:
        gc.collect()
        t0 = time.perf_counter(); pop = Population(n, "regular_spiking"); t_build = time.perf_counter() - t0
        I = np.zeros(n, np.float32)
        t, _ = timeit(lambda: pop.step(I, 1.0), 20)
        rows.append(dict(n_neurons=n, sec_build=round(t_build, 4),
                         ms_per_step=round(t * 1000, 4),
                         neuron_steps_per_sec=round(n / max(t, 1e-9), 0),
                         realtime_x=round(1.0 / max(t * 1000, 1e-9), 2),
                         rss_mb=round(rss_mb(), 1)))
        print("pop", json.dumps(rows[-1]), flush=True)
    return rows


# ------------------------------------------- known defect: is it still there?
def defect_probes():
    from neurobrain.core.neuron import Population
    from neurobrain.core.synapse import SynapseBundle
    p = {}
    # A4: does Population.step mutate the caller's array?
    pop = Population(50, "regular_spiking")
    I = np.full(50, 1e6, np.float32)
    before = I.copy()
    pop.step(I, 1.0)
    p["A4_step_mutates_caller_input"] = bool(not np.array_equal(I, before))

    # A5: is the test set read-only?
    trx, trY, tex, teY = nb.load_mnist(n_train=100, n_test=50)
    p["A5_train_writeable"] = bool(trx.flags.writeable)
    p["A5_test_writeable"] = bool(tex.flags.writeable)

    # A3: are random projections sampled with replacement?
    a, b = Population(200, "regular_spiking"), Population(200, "regular_spiking")
    try:
        sb = SynapseBundle.random(a, b, p=0.1, rng=np.random.default_rng(0))
        pre, post = np.asarray(sb.pre), np.asarray(sb.post)
        pairs = pre.astype(np.int64) * 200 + post
        uniq = len(np.unique(pairs))
        p["A3_edges"] = int(len(pairs))
        p["A3_distinct_pairs"] = int(uniq)
        p["A3_duplicate_frac"] = round(1 - uniq / max(len(pairs), 1), 4)
        p["A3_effective_p"] = round(uniq / (200.0 * 200.0), 5)
        p["A3_requested_p"] = 0.1
    except Exception as e:
        p["A3_error"] = f"{type(e).__name__}: {e}"

    # A2: does max_edges truncate silently?
    try:
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sb = SynapseBundle.random(Population(2000, "regular_spiking"), Population(2000, "regular_spiking"), p=0.5,
                                      max_edges=1000, rng=np.random.default_rng(0))
            p["A2_edges_returned"] = int(len(np.asarray(sb.pre)))
            p["A2_edges_requested"] = int(0.5 * 2000 * 2000)
            p["A2_warned"] = bool(len(w) > 0)
    except Exception as e:
        p["A2_error"] = f"{type(e).__name__}: {e}"

    # C3: bigbrain RSS on this platform
    try:
        from neurobrain.tools import bigbrain as bb
        fn = getattr(bb, "_rss_mb", None) or getattr(bb, "_rss", None)
        p["C3_bigbrain_rss_reads"] = round(float(fn()), 3) if fn else "probe-not-found"
    except Exception as e:
        p["C3_error"] = f"{type(e).__name__}: {e}"
    return p


for name, fn in [("stage_ms", stage_costs), ("v1_scaling", v1_scaling),
                 ("substrate_scaling", substrate_scaling), ("defects", defect_probes)]:
    try:
        out[name] = fn()
        print(name, "ok", flush=True)
    except Exception as e:
        out[name] = dict(ERROR=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-700:])
        print("FAIL", name, out[name]["ERROR"], flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
out["peak_rss_mb"] = round(rss_mb(), 1)
json.dump(out, open(sys.argv[1], "w"), indent=1)
print("DONE", flush=True)
