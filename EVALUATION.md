# NeuroBrain v0.40 — Evaluation

**What this is.** A measured pass over the perception stack (vision, audition,
detection, cross-modal binding), the assembled mind, and the cost of running
them — on real data, every number against a control. Reproduce with
[`benchmarks/`](benchmarks/).

**Environment.** Python 3.12, NumPy 2.4.6, no torch, no matplotlib, Linux.
MNIST and Fashion-MNIST downloaded live. `import neurobrain` required a
do-nothing torch shim throughout (defect A1); **A1 has since been fixed** and
the shim is gone — every measurement here reproduces without it.

**Headline.** The perception front end is real and the core scientific claim
holds up: receptive fields *discovered* from data beat hand-designed ones on
both datasets. Three things do not hold up — cross-modal binding never touches
vision, the dream/consolidation path is a permanent no-op, and imagination is a
ten-state loop reciting the number line. Those three are exactly the mechanisms
the "unified, constantly imaginative mind" goal rests on.

---

## 1. Vision — the core claim survives contact with harder data

2500 train / 1000 test, 1024 V1 cells, 50 ms window, chance 10%.

| dataset | filters | accuracy | categories grown | compression | worst class | fit |
|---|---|---|---|---|---|---|
| MNIST | designed | 84.1% | 417 | 6.0× | `4` @ 60% | 2.5 s |
| MNIST | **discovered** | **87.8%** | 723 | 3.5× | `4` @ 72% | 1.2 s |
| Fashion | designed | 64.8% | 506 | 4.9× | `4` @ **0%** | 2.5 s |
| Fashion | **discovered** | **70.2%** | 754 | 3.3× | `6` @ 32% | 1.2 s |

**Self-organised filters win on both datasets** — +3.7 points on MNIST, +5.4 on
Fashion — and they are *faster to fit* than the designed path (1.2 s vs 2.5 s).
This is the project's central claim and it replicates on a dataset the module
docstrings never mention. It is the strongest result here.

Two problems:

- **Fashion class 4 (`coat`) collapses to 0% recall with designed filters.** Not
  "low" — zero. The overall 64.8% hides one class the system cannot see at all.
  Discovered filters partially rescue it (worst class becomes `6` at 32%).
- **Discovered filters buy accuracy with categories**, not compression: 723
  categories for 2500 images (3.5×) against 417 (6.0×). The read-out is doing
  more of the work than the headline accuracy suggests.

## 2. Audition — the ear works, and its default threshold is mistuned

3 soundscapes × 24 events, chance 12.5%.

| measure | result |
|---|---|
| onset detection F1 | 85.7% (precision **100%**, recall **75%**) |
| raw cochleagram, pre-segmented | 32.3% |
| belt code, pre-segmented | 84.4% |
| belt code, streaming | 92.6% ⚠ *see caveat* |

The belt code is worth **+52 points over the raw cochleagram** (84.4% vs 32.3%)
— the shift-invariant representation is doing real work.

**The ear misses exactly 6 of 24 events on every seed** (18/24, three seeds,
identical). It is not the refractory period — no two events are closer than
573 ms against a 250 ms refractory. It is the detection threshold:

| threshold | precision | recall | F1 |
|---|---|---|---|
| **2.5** (shipped default) | 100% | 75% | 85.7% |
| 1.5 | 92% | **100%** | **96.2%** |

Dropping `ContinuousEar.threshold` from 2.5 to 1.5 is worth **+10 F1 points**
for one constant. The default is buying precision nobody asked for at the cost
of a quarter of all events.

> **Since changed**, on the strength of a 182-run sweep (5 seeds × 3 event
> densities × 4 noise levels, `benchmarks/ear_threshold.py`). The deciding
> metric is *named yield* — of the events that really happened, how many
> arrived both found and correctly named — not F1, because a false alarm still
> gets classified and injects a confident wrong percept downstream.
>
> | | recall | accuracy on detections | **named yield** |
> |---|---|---|---|
> | threshold 2.5 (old) | 75.0% | 92.6% | **69.4%** |
> | threshold 1.5 (new) | **100%** | 87.5% | **87.5%** |
>
> Accuracy-on-detections *falls* 5 points and that is the metric becoming
> honest, not the system regressing: it is now scored on every real event
> instead of only the easy 75% the old threshold happened to find. Named yield,
> the number that matters, gains **+18.1 points**.
>
> The gain is largest where the old default failed worst — dense soundscapes,
> events 0.05–0.20 s apart: **83.3% vs 18.1%**. Two things it does not fix: at
> noise 0.30 naming still collapses to ~21% *while recall stays 100%*, so that
> failure is in the belt code's classifier, not the onset detector; and
> precision is no longer perfect (0.95).

> ⚠ **The 92.6% streaming figure is not comparable to the 84.4%.** Streaming is
> scored only on events the ear successfully detected — the easy 75% — and
> against a different clip pool than the pre-segmented condition. It is
> conditioned on its own success. Reported here because the benchmark computes
> it, not as a win.

## 3. Detection in clutter — looking works, recognising what you looked at does not

256×256 scenes, 12 objects, 60 saccades × 3 seeds.

| | MNIST | Fashion |
|---|---|---|
| saccade lands on an object | **85.6%** | **90.0%** |
| chance (object area / scene area) | 14.4% | 14.4% |
| **lift over chance** | **5.96×** | **6.27×** |
| recognition — centred crops | 77.7% | 71.7% |
| recognition — what the eye actually got | 82.6% | **35.9%** |
| recognition — foveation correction **off** | 18.9% | 28.6% |

**Saliency-driven looking is the strongest single component in the system.** Six
times chance, and it holds up *better* on Fashion than MNIST — local-contrast
saliency likes textured objects.

**Foveation correction is load-bearing:** turning it off drops MNIST recognition
from 82.6% to 18.9%. Two corrective steps are the difference between a working
system and noise.

**The complex-data failure is here.** On MNIST, streaming recognition (82.6%)
edges out centred crops (77.7%) — drift integration over the 50 ms window
genuinely helps. On Fashion the same pipeline collapses: 71.7% → **35.9%**, a
36-point loss. A 28-pixel fovea that comfortably contains a digit clips a coat
or a pullover, and the classifier never recovers. **This is the single biggest
accuracy problem in the system on realistic data.**

## 4. Cross-modal binding — it does not bind

This is the mechanism the whole "unified mind" idea rests on, and it is the
weakest thing in the codebase.

| probe | real | control | chance |
|---|---|---|---|
| exact-code query (what the shipped test does) | 100% | **100%** (shuffled labels) | 12.5% |
| held-out clip of the same class | 78.1% | 6.25% (shuffled labels) | 12.5% |
| visual code = real / noise / zeros | 78.1% / **78.1%** / **78.1%** | — | — |
| workspace concepts after 50 `bind()` calls | **0** | — | — |

Read those rows in order:

1. **The exact-code test is vacuous.** Bind a code, query with the *same* code,
   and `argmax` finds it — so shuffled labels score 100% too. Any test that
   queries with a stored code measures nothing.
2. **With a held-out clip it does work** — 78.1% against a 6.25% shuffled
   control. So there is a real, generalising mechanism here.
3. **But it is a pure audio classifier.** Replace every visual code with
   Gaussian noise, or with zeros, and the answer does not move by a thousandth.
   `recall_visual_from_sound` reads `a_code`, does nearest-neighbour over stored
   audio codes, and returns a stored **integer label**. The visual code is
   appended and never read. It returns a label, not a visual code, despite the
   name.
4. **It never touches the workspace.** The class docstring says "binding is
   simply the two codes being in the workspace at the same moment" and `bind()`'s
   own docstring says "Hebbian binding at the level of the workspace". `bind()`
   is `self.bindings.append((v_code, a_code, label))`. Fifty binds leave
   `ws.n_concepts` at zero.

Also: `self.bindings` is an unpruned Python list scanned linearly on every
recall — O(n) per query, unbounded growth.

**Vision and hearing never actually meet.** The one shared code that the whole
architecture is organised around is bypassed by the one function whose job is to
join the modalities.

> **Since fixed.** `bind()` now routes through `AssociationArea`'s competitive
> Hebbian concept cells. Re-measured: the vision ablation spread went from
> **0% → 36%** (noise in the visual slot now costs 34 points), and a held-out
> sound recalls a **visual code** at 31–47% against 12.5% chance — a direction
> the old code could not express. See `reports/binding_02_ablation.jpg` and
> `benchmarks/binding_diagnostic.py`. The table above is the *pre-fix*
> measurement, kept because it is what the control caught.

## 5. The unified mind and its imagination

`build_unified_mind()` — 6.0 s, all metrics on its own 10-concept world.

| metric | value |
|---|---|
| perceive / comprehend | 90.2% |
| recall / analogy / causal | 100% |
| self-model | 91.0% |
| attention benefit | +14.5% |
| **concepts in the workspace** | **10** |
| code separation | 0.65 |

### Imagination is counting

40-step chains from 8 seeds, at three temperatures:

| temperature | unique concepts | self-loop rate | entropy |
|---|---|---|---|
| 0.3 | 9.5 / 10 | 1.6% | 3.02 bits |
| 0.6 | 9.5 / 10 | 1.6% | 3.02 bits |
| 1.0 | 9.75 / 10 | 0.9% | 3.10 bits |

Temperature does essentially nothing, which is the tell. The transition matrix
is peaked at **400:1** (20.05 against 0.05), so every reasonable temperature
gives the same near-deterministic walk. Printing the chains shows what it is:

```
temp=0.3 seed=1   0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 0 → 1 → 2 → 3 → 4 …
temp=1.0 seed=2   0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 6 → 7 → 8 → 9 → 6 …
temp=3.0 seed=0   0 → 4 → 4 → 0 → 0 → 6 → 8 → 9 → 7 → 8 → 9 → 8 → 0 → 7 → 0 …
```

**The train of thought is the number line.** The curriculum taught successor
relations over ten digits, so that is the only structure the world model has to
wander. It is not a bug — the mechanism does exactly what it was built to do —
but it is the whole distance between what exists and "constantly imaginative".
Only at temperature 3.0 does it leave the sequence, and then it is noise.

### The imagine → perceive round trip closes at 100%

Feed each imagined percept back into perception: **100% of 208 imagined percepts
are recognised as the concept that generated them** (chance 10%). That is higher
than the mind's 90.2% accuracy on *real* images — because the reconstructions
are class prototypes, cleaner than any real sample. The loop is consistent, but
it demonstrates fidelity, not generativity: it never imagines anything it has
not already averaged.

### Dreaming is a no-op

```
dream() → {"replays": 0, "concepts_merged": 0}   in 0.00 s
episodes after build:            0
episodes after 20 perceive():    0
```

`UnifiedMind.sleep()` returns 0 immediately when `self.episodes` is empty, and
**nothing in the unified mind ever writes to it** — `perceive()` does not record
episodes. The `EpisodicBuffer` is constructed and stays empty for the object's
whole life. The "+13% with no new data" sleep-consolidation result is real in
`development.SleepConsolidator`, but the assembled mind never reaches it.

> **Since fixed.** `perceive()` now lays down an episode, weighted by the world
> model's own `1 - P(concept | previous)` so replay is prioritised by prediction
> error. `remember=False` keeps probes out of the day, and the build's own
> measurement loops use it — scoring a held-out set must not become the night's
> experience. `benchmarks/dream.py`:
>
> | arm | replays | pallium recall before → after | vs no-sleep |
> |---|---|---|---|
> | **real day** | **1200** | 79.0% → **81.0%** | **+2.0** |
> | shuffled-label day | 1200 | 79.0% → 55.8% | −23.3 |
> | no sleep | 0 | 79.0% → 79.0% | — |
>
> `dream()` is a mechanism now (0 → 1200 replays), but the honest gain is
> **+2.0 points**, not the +13% quoted for `SleepConsolidator` — that figure
> comes from a purpose-built `SemanticCortex` experiment, not this pallium.
>
> The **shuffled control is the real result**: replaying a mislabelled day costs
> 23.3 points. Consolidation is writing content-specific information, not just
> churning. Real minus shuffled is a 25.3-point gap.
>
> **What it does not do yet.** On the *rare* half of the day replay made things
> slightly **worse** (70.4% vs 73.9% unslept) — the opposite of what prioritised
> replay exists for. The cause is visible in the priority signal: surprise takes
> only **two distinct values** across 150 episodes, because it comes from a world
> model that knows nothing but the counting order. Priority is tracking "did this
> digit follow its successor", not "is this rare". That is item 4 (`_T` from lived
> experience), and this is now the measurement that will show it working.

## 6. Efficiency

### Cost of one glance and one listen

| stage | ms | note |
|---|---|---|
| `belt.code` | **14.20** | dominant cost of the whole pipeline |
| `v1.rate_over` (fixation) | 7.49 | 6 drift frames |
| `v1.rate` (one image) | 6.00 | |
| `ear.detect_onsets` | 5.35 | for 18 s of audio — cheap |
| `cochleagram` | 2.68 | |
| `ws.encode` | 0.24 | the shared code is nearly free |
| `eye.fixate` | 0.11 | |
| `eye.next_target` | 0.03 | saliency + IOR, essentially free |

**The auditory belt costs more than all of vision put together.** 14.2 ms
against 7.5 ms for a full fixation. For a system meant to run both senses on one
clock, the ear is the bottleneck — and it is the cheapest thing to fix, since
`belt.code` is a fixed linear projection over slices.

### Scaling

| V1 cells | ms/image | img/s | | neurons | ms/step | × real-time |
|---|---|---|---|---|---|---|
| 256 | 3.85 | 260 | | 1 000 | 0.043 | 23.2× |
| 1024 | 5.57 | 180 | | 10 000 | 0.091 | 11.1× |
| 4096 | 11.79 | 85 | | 100 000 | 0.542 | 1.84× |

V1 scales **sub-linearly** — 16× the cells for 3× the time. Good.
The Izhikevich population sustains **100 k neurons at 1.84× real time** on one
core, ~185 M neuron-steps/s. Peak RSS for the entire evaluation: **150 MB**.

The substrate is not the problem. Encoding 3500 images takes 20 s while the
neurons themselves would take under a second — the cost is in the V1 projection
and the belt, both of which are dense matrix work that a batched or torch path
would flatten.

### Known defects — all four still live

Probed directly, all reproduce exactly as AUDIT.md describes:

| defect | probe result |
|---|---|
| **A2** silent truncation | requested 2 000 000 edges, got **1 000**, **no warning** |
| **A3** sampling with replacement | 4000 edges → 3820 distinct, 4.5% duplicates, effective p = **0.0955** not 0.10 |
| **A4** input mutation | `Population.step` **mutates the caller's array** |
| **A5** read-only test set | train writeable ✔, test writeable ✘ |

---

## 7. Problems, ranked by what they cost

| # | problem | evidence | cost |
|---|---|---|---|
| 1 | ~~**Binding ignores vision entirely**~~ | **Fixed** — ablation spread 0% → 36%; sound now recalls a visual code | the "unified" claim now has a mechanism |
| 2 | ~~**Dream/consolidation never runs**~~ | **Fixed** — 0 → 1200 replays; +2.0 pts, and −23.3 for a shuffled day | replay now feeds consolidation |
| 3 | **Imagination is a 10-state counting loop** | chains are `0→1→2→…→9`; temperature inert | "constantly imaginative" is absent |
| 4 | **Streaming vision collapses on complex data** | Fashion 71.7% → 35.9% through the eye | the flagship path fails on realistic input |
| 5 | ~~**`import neurobrain` fails without torch**~~ | **Fixed** — verified with no torch, with a *broken* torch, and with torch 2.13 present | was blocking CI, users, and this evaluation |
| 6 | ~~**Ear default threshold costs 25% recall**~~ | **Fixed** — default now 1.5 after a 182-run sweep; named yield 69.4% → 87.5% | one constant |
| 7 | **Fashion class 4 at 0% recall** | designed filters, complete blind spot | hidden inside a 64.8% average |
| 8 | **Belt is the throughput bottleneck** | 14.2 ms vs 7.5 ms for all of vision | halves the achievable tick rate |
| 9 | **A2/A3 wiring bugs** | effective p = 0.0955; 2000× silent truncation | every scaling number is suspect |
| 10 | **`bindings` grows unbounded, O(n) recall** | linear scan, never pruned | a long-running mind degrades |

---

## 8. Next steps toward a unified, constantly imaginative mind

Ordered by what unblocks the goal, not by difficulty.

### Immediate (hours) — stop working around bugs

1. ~~**Fix A1**~~ — **done.** Locked by `tests/test_import_without_torch.py`,
   which simulates torch's absence even where torch is installed.
2. ~~**Retune `ContinuousEar.threshold`**~~ — **done**, to 1.5, after the
   182-run sweep the original note asked for.
3. **Fix A2/A3** before trusting any scaling result — warn on truncation, sample
   without replacement.

### The unification gap (days) — make the modalities actually meet

4. **Rewrite `StreamingBrain.bind` to go through the workspace.** It should
   `ws.learn_concept` a joint code — e.g. the concatenation or sum of the
   vision and audio codes projected into the shared space — not append a tuple
   to a list. Then `recall_visual_from_sound` returns a *visual code* the rest
   of the brain can consume, which is what the name has always promised.
5. **Make the held-out query the test.** The exact-code test scores 100% with
   shuffled labels; it can only mislead. `benchmarks/binding_diagnostic.py` has
   the honest version — the noise/zeros ablation is what caught this and should
   ship as a regression lock.
6. **Grow the workspace past 10 concepts.** The unified mind's world is ten MNIST
   digits. Feed Fashion classes and sound classes into the *same* workspace and
   the "unified" claim starts meaning something. `code_separation` (0.65 today)
   is the number to watch as the vocabulary grows.

### The imagination gap (days–weeks) — give it something to imagine

7. ~~**Wire `perceive()` to record episodes**~~ — **done.** `dream()` now
   replays (0 → 1200) and gains +2.0 points, while a shuffled-label day costs
   23.3 — consolidation is content-specific. It does *not* yet rescue the rare
   tail, because the surprise signal it prioritises by takes only two distinct
   values; that is what step 8 fixes.
8. **Learn transitions from lived experience, not the curriculum.**
   `UnifiedMind.live()` already runs a `CognitiveLoop` on the mind's own
   workspace. Use *that* to populate the transition matrix instead of the digit
   successor lesson, and the train of thought stops being the number line.
9. **Then measure imagination properly.** `benchmarks/unified.py` already
   reports chain entropy, unique-concept count and the imagine→perceive round
   trip. Today: 3.02 bits over 10 concepts, round trip 100% because it only ever
   replays prototypes. A mind that is *actually* imaginative should show entropy
   rising with the concept count while the round trip stays high — novelty
   without incoherence. That pair of numbers is the goal made falsifiable.

### The perception gap (weeks)

10. **Scale-adaptive fovea.** The Fashion collapse (71.7% → 35.9%) is a fixed
    28 px window clipping large objects. A fovea sized from the saliency blob,
    or a two-scale fovea, is the obvious fix and directly targets the largest
    accuracy loss measured here.
11. **Batch or GPU the belt and the V1 projection.** Both are dense matrix
    products; together they are ~95% of the wall clock. `backend.py` exists for
    exactly this and is currently only a liability.

### Making it stick

12. **Turn `benchmarks/` into CI.** Every number in this document is a
    regression lock waiting to happen. The binding diagnostic in particular
    would have caught, at any point, the fact that vision was never being read.
