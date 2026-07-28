# NeuroBrain v0.40 — Audit

**Scope:** 62 modules, 21,249 lines, 957 KB. Static analysis (AST) plus live execution:
all 62 files compiled, the package imported, and 36 public entry points smoke-tested
against a synthetic MNIST-format dataset. Findings below are *reproduced*, not inferred,
unless marked "by inspection".

**Environment used:** Python 3.12.3, NumPy 2.4.4, matplotlib 3.10.8, no torch, no network.

---

## A. Critical bugs

### A1 — The package cannot be imported without PyTorch
`backend.py` catches the optional torch import and sets `torch = None`, then uses it as a
**class-body decorator**, which Python evaluates at import time:

```python
try:
    import torch
except Exception:
    torch = None          # "torch is optional"

class TorchPopulation:
    @torch.no_grad()      # line 112 — evaluated at import, torch is None
    def step(self, I, dt=1.0): ...
```

Reproduced:
```
AttributeError: 'NoneType' object has no attribute 'no_grad'
  File "neurobrain/__init__.py", line 220, in <module>
  File "neurobrain/backend.py", line 112, in TorchPopulation
```
Because `__init__.py` imports `backend`, **all 62 modules become unreachable** on any
machine without torch — including the 60 that are pure NumPy.

**Fix:** guard the class, or apply `no_grad` at call time.
```python
if _HAVE_TORCH:
    class TorchPopulation: ...
else:
    class TorchPopulation:
        def __init__(self, *a, **k):
            raise ImportError("TorchPopulation needs torch installed")
```

### A2 — `max_edges` silently truncates projections
`SynapseBundle.random(..., max_edges=60_000_000)` clamps without warning:
```python
n_edges = int(min(max(n_edges, 0), max_edges))
```
Reproduced: requesting `p=0.5` on 2000×2000 with `max_edges=1000` returned **1000 edges,
effective p = 0.00025** — a 2000× error, silently. Any large-brain scaling result may have
been measured on a network that was not wired as specified.

**Fix:** `warnings.warn` (or raise) when the cap binds, and record the effective `p` on the bundle.

### A3 — Random projections sample edges *with replacement*
```python
pre  = rng.integers(0, source.n, n_edges)
post = rng.integers(0, target.n, n_edges)
```
Duplicate `(pre, post)` pairs are never removed, so a fraction of pairs carry 2–3 synapses
and therefore 2–3× the intended weight.

Reproduced at p=0.1 on 200×200:
```
requested 4000 edges → 4000 materialised, 3820 DISTINCT pairs
duplicates: 180 (4.5% of edges), worst pair multiplicity 3
effective p = 0.0955, not 0.10
```
This is a modelling bug, not cosmetic: it biases excitatory drive and gets worse as `p` rises.
**Fix:** sample without replacement (e.g. `rng.choice(N*M, n_edges, replace=False)` decoded
to `(pre, post)`), or deduplicate and top up.

### A4 — `Population.step` mutates the caller's input array
```python
I = np.asarray(I, dtype=np.float32)      # no-op when already float32 → same object
np.clip(I, -self._I_CLIP, self._I_CLIP, out=I)   # writes through to the caller
```
Any caller reusing a float32 current buffer gets it silently clipped between steps.
**Fix:** `I = np.array(I, dtype=np.float32, copy=True)` or clip into the scratch buffer.

### A5 — `load_mnist` returns a writeable train set and a read-only test set
```python
return trx[idx], trY[idx], tex[:n_test], teY[:n_test]
#      ^ fancy index = copy       ^ slice of np.frombuffer = read-only view
```
Reproduced:
```
train writeable: True | test writeable: False | test labels writeable: False
teX[0,0,0] = 5  →  ValueError: assignment destination is read-only
```
The asymmetry means in-place test-time augmentation or normalisation crashes only on the
test path, which is exactly where it is hardest to notice.
**Fix:** `np.frombuffer(...).copy()` in both `imgs()` and `labs()`.

### A6 — `persistence.load_brain` is unauthenticated pickle
```python
with gzip.open(path, "rb") as f:
    blob = pickle.load(f)
```
Loading any `.nbz` file is arbitrary code execution. The `FORMAT_VERSION` check happens
*after* unpickling, so it provides no protection at all. If trained brains are ever shared,
downloaded, or accepted by a server, this is a remote code execution primitive.

**Fix:** serialise the numpy state explicitly (`np.savez` + a declared schema), or sign the
payload with HMAC and verify before unpickling.

---

## B. Crashes on legitimate inputs

### B1 — `psyche.build_reconstructive_mind` hard-codes 60 examples per digit
```python
ex = (trx[trY == d][int(rng.integers(60))].reshape(-1) / 255.0)   # psyche.py:273
```
With `n_train=200` there are ~27 examples of each digit.
Reproduced: `IndexError: index 49 is out of bounds for axis 0 with size 27`.
**Fix:** `pool = trx[trY == d]; ex = pool[rng.integers(len(pool))]`.

### B2 — `grounding.build_grounded_causal_mind` indexes an empty history
`CausalFeatureLearner.train` only appends every 500 episodes:
```python
if (i + 1) % 500 == 0:
    hist.append(float(np.mean(run[-500:])))
```
then the caller does `hist[0]` and `hist[-1]` unguarded.
Reproduced with `n_episodes=200`: `IndexError: list index out of range`.
**Fix:** always append a final bucket, or guard with `if hist:`.

### B3 — Small configurations fail deep in the stack
`develop_v1` raises `ValueError: a hypercolumn needs at least 2 cells to compete` five frames
below the public API. Reproduced from `build_v1_v2`, `predictive_coding_experiment`, and
`full_loop_experiment`. The guard is correct; the problem is that the public entry points
accept `n_v1` without validating it and give the user no idea what the minimum is.
**Fix:** validate at the entry point with the constraint stated in the message.

---

## C. Correctness and measurement

### C1 — `compare_backends` compares misaligned time series
```python
tpop.step(It)                                   # warm-up, NOT counted
rates_t = [float(tpop.step(It)...) for _ in range(ms)]
```
numpy is timed from step 1; torch from step 2. The `rate_abs_diff` therefore compares
torch millisecond *t+1* against numpy millisecond *t*. The function whose stated purpose is
to prove the backends agree is off by one step.

### C2 — The two backends are not the same model
| | numpy `Population` | `TorchPopulation` |
|---|---|---|
| jitter applied to | `a`, `d` only | `a`, `b`, `c`, `d` (all, via `par()`) |
| sub-steps | `max(1, round(dt/0.5))` | hardcoded `range(2)`, `h = dt/2` |
| entry clip on `v` | yes | no |
| `activation` decay | yes | absent |

The comment claims "same sub-stepping as numpy". True only for `dt == 1.0`.

### C3 — `bigbrain` memory measurement is Linux-only and fails silently
```python
try:
    with open("/proc/self/status") as f: ...
except Exception:
    pass
return 0.0
```
On macOS or Windows every RSS figure in `BigBrainReport` reads **0.0** — in the module whose
entire purpose is measuring scale. **Fix:** use `resource.getrusage(RUSAGE_SELF).ru_maxrss`
(with the platform's unit correction) and never return a plausible-looking zero.

### C4 — `orientation_selectivity` is defined twice and one copy is unreachable
`__init__.py` imports the name from both `vision` and `selforganize`:

| | signature | returns |
|---|---|---|
| `vision.orientation_selectivity` | `(layer: SpikingConvLayer, n_angles=16)` | `np.ndarray` |
| `selforganize.orientation_selectivity` | `(layer: WideV1, n_orient=16, n_phase=8)` | `Dict[str, float]` |

`neurobrain.orientation_selectivity` resolves to the `selforganize` one; the `vision` one
cannot be reached from the package root, and `__all__` lists the name once as if there were
no ambiguity. **Fix:** re-export one under an alias.

### C5 — 35 public symbols are missing from `__all__`
Verified by diffing `__all__` against the module namespace:
```
AssemblyReport, BanditTask, ContrastivePredictiveA1, DelayedRewardTask,
DopaminergicActionLoop, DopaminergicModulator, EligibilityTrace, IntegratedBrain,
InvariantContrastiveA1, LatentPredictiveA1, MultiScaleA1, OscillatoryWorkingMemory,
PredictiveA2, RewardPredictionError, RhythmReport, STDPContrastiveA1, SensoryBridge,
SequenceContrastiveA2, SpikingEpisodicMemory, TemporalPool, a1_states, adapted_code,
assembly_experiment, available_devices, best_device, compare_backends, compare_rules,
integration_experiment, linear_probe, nuisance_transform, random_episodes,
speaker_normalize, spectrotemporal_gabor_bank, vocal_tract_warp, working_memory_experiment
```
That is essentially the entire v0.36–v0.40 feature set: `dopamine`, `rhythm`, `assembly`,
`integrated`, `backend`, and the newer `auditorycortex` classes. `from neurobrain import *`
and every documentation tool will miss all of it.

### C6 — `Brain.top_down` lifecycle
`top_down` is read every step but never cleared by `Brain.step()`; once set it persists
until manually overwritten. It is also only ever written for a region literally named
`"memory"` (`runtime.py:134`). If a brain spec has no `"memory"` region, `top_down.get(name)`
returns `None` for every region and the entire top-down pathway becomes a **silent no-op**.

### C7 — Minor
- `Neuron.inject()` allocates float64 into a float32 pipeline; `Population.grow()` deletes
  `_manual_I`, discarding queued injections.
- `neuron.py` docstring says "int8 type id"; the code uses `int16`.
- `grounding.CausalFeatureLearner.train` keeps every episode's error in `run` while only
  reading `run[-500:]` — unbounded growth on long runs.
- One import cycle: `selforganize ↔ mapdebug`.
- `world_model.py` and `worldmodel.py` are different modules with near-identical names.
- `dashboard.py` — 495 lines, imported by nothing, superseded by `console.py`.
- 145 raw `print()` calls across 40+ modules; no logging, no way for a caller to silence them.
- 3 silent `except: pass` (`bigbrain.py:149`, `perceptloop.py:207`, `realworld.py:74`);
  the `perceptloop` one swallows every exception in the causal-expectation loop.

**Reproducibility is otherwise good.** `closed_loop_experiment`, `sleep_benefit_experiment`,
and the curriculum path are bit-identical across runs at a fixed seed, and change with the
seed. The only field that varies is `AssemblyReport.seconds` (wall clock), which is expected.

---

## D. Security

| Issue | Location | Risk |
|---|---|---|
| Unauthenticated `pickle.load` | `persistence.py:42` | **RCE** on any shared/downloaded brain file |
| State-mutating GET, no auth or CSRF token | `console.py` `/cmd` | Any visited webpage can `<img src="http://localhost:8080/cmd?do=stop">` |
| Unvalidated `hz` parameter | `console.py` `/cmd` | `?hz=abc` → 500 traceback; `?hz=0` → `ZeroDivisionError` silently kills the tick thread; `?hz=-1` → 100% CPU spin |
| Single-threaded `HTTPServer` | `console.py:424` | One slow tick stalls the whole UI; no request isolation |
| Unlocked read of live state | `console.BrainConsole.graph` | Torn reads while the tick thread writes |
| Plain **http://** download | `realworld._FASHION_BASE` | Fashion-MNIST arrives unencrypted and unverified |
| Non-atomic, unchecksummed cache write | `realworld._download` | An interrupted download leaves a truncated file that passes `getsize > 0` **forever** |
| Hardcoded developer path | `realworld.py:70` `/root/.ccr/ca-bundle.crt` | Dev-machine artefact shipped in library code |

**Minimum fixes:** sign or replace the pickle format; require a token for `/cmd` and move it
to POST; validate `hz` into a sane range; write downloads to `.part` then `os.replace`, and
verify a SHA-256 against a table of known hashes; switch Fashion-MNIST to HTTPS.

---

## E. Missed business opportunities

This is strong research engineering. It is a coherent cognitive architecture with real
ablations, real controls, and unusually honest reporting — including results that *don't*
flatter the approach (the discrete loop losing to a stay-baseline; a random-field control at
58.4%). That honesty is rare and is itself an asset. It is packaged as a private scratch
directory, and that is what is costing you.

### E1 — No packaging, no tests, no license (highest leverage by far)
Absent: `pyproject.toml`, `setup.py`, `requirements.txt`, `README`, `LICENSE`, `CHANGELOG`,
`tests/`, CI config, `.gitignore`, `Dockerfile`.

Consequences:
- Nobody can `pip install` it.
- **No license means all rights reserved by default** — no company, lab, or university can
  legally use, fork, or cite it, even if they want to.
- Nothing verifies that v0.41 doesn't silently break v0.40. With 21k lines and no test suite,
  every refactor is a gamble.

A weekend of work converts this from a folder into a citable, installable artifact. Start
with: `pyproject.toml` (hatchling, `dependencies = ["numpy>=1.24"]`, extras `viz`/`torch`),
`LICENSE` (Apache-2.0 or MIT), a `tests/` directory seeded with the 30 entry points that
already pass, and a GitHub Actions matrix on 3.10–3.13.

### E2 — The results are the product, and they are buried in docstrings
The module docstrings contain real, falsifiable numbers: 87.6% learned vs 85.6% hand-designed
Gabor bank vs 58.4% random fields for the contrastive A1; "+13% with no new data" for dream
consolidation; the width and window ablations; the sparsity sweep in `workspace.py`
(0.04→69.2%, 0.08→77.4%, 0.16→82.1%, 0.25→75.6%) showing an optimum rather than "more is
better". A reader must import the package to discover you measured anything.

Move them into a `RESULTS.md` with a table per claim (claim / control / number / how to
reproduce), and a `benchmarks/` runner that regenerates them. This is what makes the work
citable and what makes a reviewer take it seriously in the first thirty seconds.

### E3 — No offline mode
`load_mnist` "raises on no network so callers can skip". In practice that means a majority of
entry points fail in CI, in Docker, on a plane, and in any air-gapped evaluation. During this
audit I had to synthesise an MNIST-format cache before most of the package would run at all.

Ship a tiny bundled synthetic dataset (a few hundred KB) and a `NEUROBRAIN_OFFLINE=1` path so
`pip install neurobrain && python -c "import neurobrain; neurobrain.closed_loop_experiment()"`
works in five seconds with zero network. That single change is usually the difference between
someone trying a project and closing the tab.

### E4 — The demos are slow with no feedback and no caching
Measured in this audit, at parameters **10× smaller than the defaults**:
`build_mind()` = 113 s, `build_ventral_stream()` = 68 s, `teach_starter_curriculum` = 17 s.
At default parameters these run for many minutes. There is no progress bar, no ETA, no
checkpointing, and no way to interrupt and resume.

`persistence.py` already exists and nothing in the build pipeline uses it. Caching fitted
stages (developed V1 filters, trained category maps) keyed by their parameters would make
iteration tolerable and cost perhaps 50 lines.

### E5 — `console.py` is the actual product and is treated as an afterthought
A live browser dashboard showing a spiking brain saccading across a cluttered scene — fovea,
saliency map, cochleagram, workspace code, naming, running accuracy — is the demo that earns
attention. It is currently one function mentioned near the bottom of a 314-line `__init__`
docstring. It deserves the first paragraph of the README, a `python -m neurobrain.console`
one-liner, a hosted instance, and a 30-second screen recording.

### E6 — Structural drag
- **`dashboard.py`** — 495 lines of dead code shipping in the package. Delete it or wire it back.
- **`world_model.py` vs `worldmodel.py`** — will cost you and every future contributor real
  time. Rename one (e.g. `concepts.py`).
- **A flat 62-module namespace** — sub-packages (`core/`, `vision/`, `audio/`, `cognition/`,
  `experiments/`, `viz/`) would make the project legible at a glance and let you version the
  experiment layer separately from the substrate.
- **`__init__.py` imports all 62 modules eagerly.** Import cost is ~0.5 s today and grows with
  every module. Consider PEP 562 lazy `__getattr__` for the experiment-tier modules.

---

## F. Suggested order of work

1. **A1** — one-line class guard. Unblocks every torch-free machine. *(minutes)*
2. **E1** — `pyproject.toml` + `LICENSE` + `README`. Makes the project usable and legal. *(hours)*
3. **A2, A3** — connectivity correctness. Everything numeric downstream depends on it. *(hours)*
4. **A4, A5, B1, B2** — aliasing and index bugs. Small, localised, currently crash real usage. *(hours)*
5. **E3** — bundled offline dataset. Unlocks CI, which unlocks tests. *(hours)*
6. **`tests/`** seeded with the 30 passing entry points as regression locks. *(days)*
7. **C4, C5** — public API hygiene, so v0.36–v0.40 work becomes visible. *(hour)*
8. **D** — pickle format and console auth, before anything is shared or hosted. *(day)*
9. **E2** — `RESULTS.md` + `benchmarks/`. Turns the work into something citable. *(days)*
10. **E4, E5, E6** — caching, the console as the front door, structural cleanup. *(ongoing)*
