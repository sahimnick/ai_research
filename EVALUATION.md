# NeuroBrain v0.40 — Evaluation

**What this is.** A measured pass over the perception stack (vision, audition,
detection, cross-modal binding), the assembled mind, and the cost of running
them — on real data, every number against a control. Reproduce with
[`benchmarks/`](benchmarks/).

**Environment.** Python 3.12, NumPy 2.4.6, no torch, no matplotlib, Linux.
MNIST, Fashion-MNIST, **CIFAR-10 and ESC-50** downloaded live. `import
neurobrain` required a do-nothing torch shim throughout (defect A1); **A1 has
since been fixed** and the shim is gone — every measurement here reproduces
without it.

**§7.8 is the real-world pass** — 600 ESC-50 field recordings and CIFAR-10
photographs, two corpora that share real categories and nothing else. Read it
first if you want to know what works outside the pre-segmented datasets: the ear
does (50-way at 17.5× chance, 1-NN 0.914 on six real categories), the eye did
not (1-NN 0.185 on photographs against 0.818 on digits) and is now roughly
half-fixed (0.326, with digits improving to 0.920 in the process). Cross-modal
retrieval works and *beats a real percept* at naming its own category — but
while the concept layer stores one cell per experience, that is the only claim
it can support. The eye, and concept formation behind it, are what everything
else now waits on.

**Headline — the one result.** *Replay changes one of the three stores that hold
what the mind knows, and the faculties expected to improve read the other two.*

There are three stores: `vision.cortex` (how it perceives), `space.cortex` (what
it remembers), and `_T` (what it expects to happen next). `dream()` writes only
to the second. So detection cannot improve no matter how much the mind sleeps —
it is read from a store sleep never touches — and neither can world-model
prediction, because `sleep()` replays *patterns* and never replays
*transitions*. Both measured at **exactly +0.0000** across 5 seeds, which is the
signature of a disconnection rather than a weak effect.

This is an architectural claim, not "sleep doesn't work", and it is falsifiable
in a specific way: **if consolidation is ever routed into `vision.cortex` or
`_T`, those two exact zeros must move.** If they stay exactly zero, the routing
did not really happen. `benchmarks/scene_to_dream.py` is the standing re-test —
run it unchanged after any such change and it re-answers the question.

**Both wires were then built, and both zeros moved.** The world model gains
+14.6 points (5/5 seeds, d=1.95) from replaying the day's *order*. Perception
turned out to be conditional rather than immovable, and §7.7 states the
condition: replay improves perceptual representations **only where the day
carries material the store does not already cover AND an external signal
supplies the label** — +38.5 points on an unfamiliar day with a teacher, and
nothing at all if either factor is missing.

The perception front end is real and the core scientific claim holds up:
receptive fields *discovered* from data beat hand-designed ones on both
datasets. Three mechanisms that did not hold up when first measured —
cross-modal binding never touching vision, dream as a permanent no-op,
imagination as a ten-state counting loop — have since been fixed; each section
below keeps the original measurement and the correction beside it.

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

> **Since fixed.** `_record_episode` now writes every lived transition into the
> world model — predict first, then learn, so a transition can never explain
> itself away. The counting lesson survives as a *prior* (`teach_counting=`)
> rather than as the entire world. `benchmarks/imagination.py`, 3 seeds:
>
> | | taught only | lived | Δ |
> |---|---|---|---|
> | transition matrix populated | 9.0% | **98.3%** | +89.3 |
> | transition row entropy | 0.53 bits | **2.80 bits** | +2.27 |
> | distinct surprise values | **2** | **154** | +152 |
> | chain entropy @ t=0.6 | 3.02 | 3.16 | +0.14 |
> | temperature span | 0.086 | 0.114 | +0.03 |
> | **imagine → re-perceive** | 1.000 | **1.000** | 0.00 |
>
> ```
> taught only:  0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 9 → 8 → 9 → 8 → 9 → 7
> lived:        0 → 1 → 2 → 1 → 0 → 5 → 6 → 7 → 8 → 9 → 8 → 9 → 0 → 6 → 0 → 2
> ```
>
> The structural change is large and the behavioural one is visible in the
> trajectory — backtracking (`2→1→0`) and jumps (`0→5`, `9→0`) instead of a
> fixed recital. **Coherence is not paid for it**: the round trip stays at
> 1.000, which is the pair that matters — novelty without incoherence.
>
> **Two caveats.** Chain entropy is a weak instrument here: a deterministic
> cycle through all ten states already has near-maximal *visit* entropy, so
> +0.14 understates the change; row entropy of the transition matrix
> (0.53 → 2.80) is the honest measure. And the round trip is saturated at
> 1.000 in *both* arms because `imagine` reconstructs stored prototypes — it
> is a ceiling, not evidence of imagery quality.

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
> **Re-evaluated across 6 seeds**, because the numbers above are one run each:
>
> | comparison | Δ | wins | Cohen's d |
> |---|---|---|---|
> | replay vs no sleep | **+1.6 pts** ± 0.8 | **6/6** | **2.11** |
> | real vs shuffled day | **+25.9 pts** ± 2.7 | 6/6 | 9.55 |
> | **prioritised vs uniform** | +0.1 pts ± 0.6 | 3/6 | **0.21** |
> | rare classes, replay vs none | −1.3 pts ± 1.5 | 2/6 | −0.83 |
>
> The gain survives repetition at +1.6 (not the +2.0 of the single run) and the
> shuffled control is overwhelming. But **prioritisation was doing nothing** —
> 3/6 seeds, d=0.21, a coin flip against uniform replay. That is exactly what a
> surprise signal with two distinct values predicts. The rare-class regression
> is suggestive but not established (loses 4/6, spread crosses zero).
>
> This is what item 4 addresses: with lived transitions the surprise signal goes
> from 2 to 154 distinct values.
>
> **The follow-up did not vindicate the hypothesis.** Repeating the
> prioritised-vs-uniform comparison *after* item 4, with surprise now taking 154
> distinct values instead of 2:
>
> | world model | prioritised − uniform | wins | Cohen's d |
> |---|---|---|---|
> | taught only (2 values) | +0.12 pts ± 0.61 | 3/6 | 0.21 |
> | lived (154 values) | +0.21 pts ± 0.58 | 4/6 | 0.36 |
>
> It moved in the predicted direction and is still **not meaningful** — a small
> effect swamped by its own spread. Making the priority signal informative was
> necessary but not sufficient.
>
> **The suspect was confirmed, directly.** Replaying one episode 200 times:
>
> ```
> cells: 210 -> 410   (+200 identical cells)
> recall: 0.7600 -> 0.7600   (moved by exactly nothing)
> the 200 added cells are identical: True
> ```
>
> And reconstruction was verbatim — every imagined percept scored cosine
> **1.000000** against a cell already in the store, so imagination could only
> replay experience. Both are now fixed; see §7.
>
> The remaining suspect is the consolidation *learner*, not the priority. Replay
> calls `space.remember(label, image=pattern)`, which appends an exemplar; the
> recall probe is nearest-neighbour over those exemplars, so replaying a
> surprising episode more often mostly adds near-duplicates that
> nearest-neighbour is insensitive to. With 400 episodes and 1200 replays
> everything is seen ~3× regardless, so prioritisation only reweights something
> the read-out does not measure. A consolidation step whose *strength* depends
> on replay frequency — the `SemanticCortex` prototype update that the +13%
> result actually uses — is where this would start to pay.

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
| 3 | ~~**Imagination is a 10-state counting loop**~~ | **Fixed** — transition matrix 9% → 98% populated, surprise 2 → 154 distinct values, chains wander and backtrack | the world model now learns from what is seen |
| 4 | **Streaming vision collapses on complex data** | Fashion 71.7% → 35.9% through the eye | the flagship path fails on realistic input |
| 5 | ~~**`import neurobrain` fails without torch**~~ | **Fixed** — verified with no torch, with a *broken* torch, and with torch 2.13 present | was blocking CI, users, and this evaluation |
| 6 | ~~**Ear default threshold costs 25% recall**~~ | **Fixed** — default now 1.5 after a 182-run sweep; named yield 69.4% → 87.5% | one constant |
| 7 | **Fashion class 4 at 0% recall** | designed filters, complete blind spot | hidden inside a 64.8% average |
| 8 | **Belt is the throughput bottleneck** | 14.2 ms vs 7.5 ms for all of vision | halves the achievable tick rate |
| 9 | **A2/A3 wiring bugs** | effective p = 0.0955; 2000× silent truncation | every scaling number is suspect |
| 10 | **`bindings` grows unbounded, O(n) recall** | linear scan, never pruned | a long-running mind degrades |

---

## 7. Two mechanisms built for two proven failures

Both failures were established by direct probe, not inference:

```
replaying ONE episode 200 times, with the old learner:
  cells: 210 -> 410      (+200 identical cells)
  recall: 0.7600 -> 0.7600        (moved by exactly nothing)

every imagined percept, with the old read-out:
  max cosine to a STORED cell = 1.000000     (verbatim replay)
```

`AssociativeCortex.consolidate` folds a replay into an existing trace by the
instar rule `w += lr·(x − w)` and counts its strength; `reconstruct_population`
/ `MentalSpace.blend` read out a Dirichlet-weighted population instead of one
winner. Both are local and Hebbian — no gradient enters.

### The blend works

| read-out | novelty (1 − max cosine to any stored cell) | imagine → re-perceive |
|---|---|---|
| verbatim (`blend=0`) | **0.00000** | 1.000 |
| `blend=3, conc 0.6` | 0.02291 | 1.000 |
| `blend=6, conc 0.9` | **0.04475** | **1.000** |
| `blend=12, conc 0.9` | 0.05376 | 0.962 |

The mind produces percepts that were never stored **while still recognising them
as what it meant** — up to `blend=6` at no coherence cost.

But cosine alone cannot tell "genuinely new" from "the same thing nudged", and
it is measured in pixel space, which is not where this system sees. Scored
again through the mind's own `WideV1`, and against real images as the yardstick
(`benchmarks/novelty.py`):

| | cosine | pixel L2 | **V1 perceptual** |
|---|---|---|---|
| a training image it saw | 0.166 | 0.556 | 0.166 |
| **a real held-out digit** | **0.223** | **0.655** | **0.198** |
| gaussian noise | 0.495 | 0.995 | 0.150 |
| shuffled-pixel digit | 0.762 | 1.234 | 0.374 |
| imagined, `blend=0` | −0.000 | 0.000 | 0.004 |
| imagined, `blend=6` | 0.047 | 0.295 | 0.075 |
| imagined, `blend=24` | 0.045 | 0.294 | 0.079 |

As a fraction of what a **real unseen digit** scores — the honest yardstick for
"new but valid":

| | cosine | pixel L2 | V1 perceptual |
|---|---|---|---|
| `blend=0` | 0.00× | 0.00× | 0.02× |
| `blend=6` | 0.21× | 0.45× | **0.38×** |
| `blend=12` | 0.23× | 0.48× | 0.41× |
| `blend=24` | 0.20× | 0.45× | 0.40× |

**The movement is real but partial.** Perceptual distance rises 19-fold from
`blend=0` (0.004) to `blend=6` (0.075), so this is not a rounding artefact in
one embedding — it moves in the space the system actually sees with. But it
reaches only **38–45% of the novelty a real unseen digit carries**, and it
**saturates**: `blend=24` is no further out than `blend=6`.

That saturation is structural, not a tuning failure. A Dirichlet-weighted sum
of stored traces is a convex combination, so it cannot leave the convex hull of
what was stored however many traces are mixed. **This interpolates rather than
replays — a real step past verbatim recall — but it does not extrapolate, and
extrapolation is what "imagining something new" finally requires.**

One caveat on the V1 metric: gaussian noise scores *lower* (0.150) than a real
held-out digit (0.198), because broadband noise drives V1 everywhere and lands
near the store. V1 distance is a good measure of perceptual novelty for
plausible images and a poor one for off-manifold junk; the shuffled-pixel
control (0.374) behaves as expected.

### The consolidator: the "recall loss" was the retriever, not the memory

First measurement said consolidation costs 1.2 points (0.821 → 0.809, losing
6/6 seeds). **That conclusion was wrong**, and it was wrong in an avoidable
way: it scored both memories with one retriever — plain nearest-neighbour,
which structurally rewards having more exemplars. It confounded *what the
memory knows* with *how it is read*.

Reading the same two memories five ways (`benchmarks/retrievers.py`, 5 seeds):

| retriever | append (1510 cells) | consolidate (315 cells) | append − consolidate |
|---|---|---|---|
| `nn` (the shipped one) | 0.822 ± 0.010 | 0.810 ± 0.010 | +0.012 · 5/5 · d=0.98 |
| **`knn5`** | 0.789 ± 0.028 | **0.824 ± 0.013** | **−0.035 · 0/5 · d=−1.78** |
| `proto` | 0.730 ± 0.027 | 0.722 ± 0.006 | +0.008 |
| `softmax` | 0.481 ± 0.087 | 0.789 ± 0.023 | −0.308 |
| `strength_nn` | 0.822 ± 0.010 | 0.331 ± 0.086 | +0.492 |

**Best against best: 0.824 (consolidate + knn5) vs 0.822 (append + nn) — a gap
of −0.002.** Consolidation destroyed no information. Under a retriever that
does not count duplicates, the 315-cell summary *beats* the 1510-cell log by
3.5 points, losing 0/5 seeds.

So the real result is **4.8× compression at no accuracy cost**, provided the
read-out is k-NN rather than a single nearest neighbour. The apparent loss was
an artefact of the measurement, exactly as separating the two effects was meant
to reveal.

Two retrievers fail informatively rather than randomly. `softmax` collapses on
the appending memory (0.481) because 1510 near-duplicates let one cluster
dominate a temperature-weighted vote. `strength_nn` collapses on the
consolidated memory (0.331) because a strength counter that reaches 201 swamps
similarity entirely — the most-replayed trace wins everything. Both are
retriever pathologies, not memory ones.

### Episode → dream now runs on lived looking, not hand-fed crops

Every replay result above was measured on `perceive(trx[i])` — a pre-cut,
centred 28×28 digit handed straight to the mind. `StreamingBrain` had **no
episodic buffer at all**, so lived sensory experience could not reach
consolidation even in principle, and a verdict on replay was a verdict about
hand-fed crops.

`UnifiedMind.watch(scene)` closes it: the saccadic eye free-views a real
256×256 scene, and each fixation — background ones included, because they were
part of the looking — is perceived, laid down as an episode with its surprise,
and its transition from the previous fixation written into the world model.

```
episodes after build   : 0
watched a real scene   : 40 saccades, 23 landed on an object
episodes after watching: 40
surprise from lived looking: 8 distinct values, mean 0.932
dream on lived experience  : {'replays': 480, 'concepts_merged': 0}
```

**Answer, measured** (`benchmarks/scene_to_dream.py`, 5 seeds, 180 episodes of
real looking per mind, 900 replays, paired against a no-dream arm):

| faculty | real dream − no dream | wins | d | |
|---|---|---|---|---|
| recall (`nn`) | **−0.0133** | 0/5 | −1.71 | **hurts** |
| recall (`knn5`) | −0.0053 | 1/5 | −0.51 | no effect |
| detection | **+0.0000** | 0/5 | 0.00 | no effect |
| rare-class recall | −0.0100 | 2/5 | −0.48 | no effect |
| world-model prediction | **+0.0000** | 0/5 | 0.00 | no effect |

**Nothing improves. One thing gets slightly worse.** The episode path is
demonstrably alive — 180 episodes laid down from real saccades, 900 replays —
so by elimination the fault is in replay/consolidation itself. Two of those
numbers are *exactly* zero, and that is structural rather than noisy:

- **Detection cannot move**, because `perceive()` reads `vision.cortex` (the
  DigitRecognizer) while `dream()` writes to `space.cortex` (the pallium). The
  mind perceives with one store and remembers with another, and sleep only
  touches the second. No amount of replay can reach perception.
- **The world model cannot move**, because `sleep()` replays *patterns* and
  never replays *transitions*. Consolidation is image-only; `_T` is never
  consolidated at all.
- Recall does move, and downward: consolidation folds distinct exemplars into
  shared traces, which `nn` punishes (d=−1.71) and `knn5` is nearly neutral to
  (−0.005). The shuffled-label day is worse still (−0.029), which confirms
  replay writes content-specific information — it is simply information that
  does not help.

So the perception → experience → sleep → improvement loop **does not yet close
in this architecture**, and the reason is now specific rather than suspected:
replay reaches one of the three stores that would have to change.

### Prioritised replay does not matter — at any budget

It was built to fix this, and it did not. Testing the obvious remaining
explanation — that 1200 replays over 400 episodes means everything is replayed
~3× and prioritisation has nothing left to choose:

| replay budget | prioritised − uniform | wins | Cohen's d |
|---|---|---|---|
| 1200 replays / 400 episodes (3.0×) | +0.0025 | 3/6 | +0.27 |
| 120 replays (0.3×) | −0.0004 | 4/6 | −0.05 |
| 39 replays (0.1×) | −0.0004 | 2/6 | −0.10 |

Scarcity does not rescue it either. **Three hypotheses have now been tested and
all three were wrong** — the two-valued surprise signal, the frequency-blind
learner, and the abundant budget. Prioritised replay provides no measurable
benefit in this architecture on this task.

The open question, stated as a question because the last two guesses failed:
surprise here is `1 − P(concept | previous)`, which measures how unexpected a
*transition* was. Whether an exemplar is hard to *recall* is a different
quantity. A priority signal built from reconstruction error rather than
transition surprise might behave differently — but that is untested.

---

## 7.5 The acceptance gate — does replay change behaviour?

Three criteria, fixed in advance so the gate cannot be argued into passing:
a criterion passes at |Cohen's d| ≥ 0.8 in the right direction on ≥ 4 of 5
seeds. `benchmarks/acceptance.py`, measured against a watch-but-do-not-dream
arm on the same lived day.

Two wires were added to make the criteria attemptable at all:
`sleep(to_perception=…)` routes replay into `vision.cortex` via the same
ART-style Hebbian rule waking perception uses, and `sleep(to_world=…)` replays
the day's **order** rather than isolated frames — hippocampal replay is
sequential, and the episodic buffer already holds the lived sequence.

| | memory-only night | **all-stores night** |
|---|---|---|
| detection | +0.0000 · 0/5 · d=0.00 | **+0.0000 · 0/5 · d=0.00** |
| world model | +0.0000 · 0/5 · d=0.00 | **+0.1455 · 5/5 · d=1.95** |
| recall (`nn`) | −0.0053 · d=−0.42 | −0.0053 · d=−0.42 |
| recall (`knn5`) | −0.0040 · d=−0.62 | −0.0040 · d=−0.62 |

| criterion | verdict |
|---|---|
| 1. detection changes meaningfully | **FAIL** |
| 2. world model changes meaningfully | **PASS** — +14.6 points, 5/5 seeds |
| 3. without a drop in recall | **PASS** |
| **all three** | **FAIL — 2 of 3** |

### What passed, and why

The world model moves decisively because the day's *order* is information it
genuinely did not have. The taught counting prior (20 passes of 0→1→…→9) was
simply wrong about this world, and replaying the lived sequence three times
outweighs it. Note the labels do not have to be *correct* for the ordering to
be informative — which is exactly why this works where detection does not.

### Why detection cannot be fixed by tuning

The wire is live: at the waking rate it moves detection by −3.4 points, so the
exact-zero disconnection is gone. But sweeping the consolidation rate over two
orders of magnitude finds **no beneficial window**:

| effective rate | detection Δ | wins |
|---|---|---|
| 0.000 | +0.0000 | 0/5 |
| 0.050 (the sleep rate) | +0.0000 | 0/5 |
| 0.100 | −0.0194 | 0/5 |
| 0.250 | **−0.0427** | 0/5 |
| 1.500 | −0.0361 | 0/5 |

Gentle does nothing; aggressive harms; **0/5 seeds at every rate**. Two
hypotheses were tested and rejected along the way. Background contamination:
45% of a free-viewing day lands on background and only 47% of episodes carry a
ground-truth-correct label, but a confidence gate separates background
perfectly (0.842 vs 0.154, keeping 33/33 objects and 0/27 background) — it
fixed criterion 3 and left detection untouched. Confidence predicting
correctness: it does not, r = +0.11, with correct fixations at 0.844 and wrong
ones at 0.830.

The structural reason is that **replay into perception is self-training on its
own beliefs**. The labels replayed into `vision.cortex` were produced *by*
`vision.cortex`. There is no external signal anywhere in the loop, so the best
it can do is sharpen what it already believes and the worst is amplify its own
error rate. It cannot add information it did not already have.

That asymmetry is the finding: **replay helps a store that was missing
structure the day contained, and cannot help a store that generated the very
labels being replayed.** Improving detection through sleep needs a signal
perception did not produce — a second modality, a prediction error, or a
teacher — not a better replay schedule.

---

## 7.6 Does replay need an external teaching signal?

The structural story from §7.5 — replay into perception is self-training on its
own beliefs, so no information enters — makes a sharp prediction: supply the
label from outside vision and the same replay should start to help. If it did,
"replay does not help perception" would become **"replay needs an external
teaching signal"**, which is a much stronger claim.

Tested with `benchmarks/external_signal.py`: the same lived day, replayed into
`vision.cortex`, differing only in where the consolidation label comes from —
the mind's own belief, ground truth, a prediction-error gate, or a genuinely
independent second sense (a ten-tone auditory bank with its own 34.8% error
rate).

**Two of my own bugs had to be found first, and the identical-numbers signature
found both.** The first run reported five identical zeros: it was executed at a
consolidation rate the earlier sweep had already shown to be a dead zone, so no
label source could have registered. The second reported `self`, `teacher` and
`second_mod` all at exactly 0.7819 — different labels, identical outcome, which
can only mean the label was never read. It was not: `_replay_into_perception`
picked the cell to move by *similarity*, making the update unsupervised by
construction. Consolidation is now label-directed (LVQ-style: the competition
runs within the taught class), which is what lets an outside signal teach at
all.

Re-run properly powered — 8 seeds, a detection probe of 184 fixations
(resolution 0.0054 rather than 0.030):

| rate | arm | vs no-dream | wins | d |
|---|---|---|---|---|
| eff 0.10 | self | −0.0069 | 2/8 | −0.23 |
| eff 0.10 | **teacher** | −0.0029 | 4/8 | −0.12 |
| eff 0.25 | self | −0.0182 | 2/8 | −0.41 |
| eff 0.25 | **teacher** | **+0.0007** | 4/8 | +0.04 |

| comparison | Δ | wins | d |
|---|---|---|---|
| teacher − self, eff 0.10 | +0.0040 | 3/8 | +0.64 |
| teacher − self, eff 0.25 | +0.0190 | 3/8 | +0.38 |

**The claim is not established.** The label source does matter directionally —
teacher beats self at both rates, consistently in sign — but a ground-truth
teacher only brings replay back to **parity** (+0.0007). It prevents replay
from damaging perception; it does not make replay improve it. Neither effect
clears the pre-registered bar.

Also worth correcting: the earlier "−0.0427, HURTS" was inflated by the coarse
probe. At proper resolution the self-label damage is −0.007 to −0.018.

### The likelier explanation, stated as a hypothesis

`vision.cortex` was already grown on 15,000 clean MNIST digits with this same
ART rule. A day of 180 foveal fixations cannot add to a store that saturated
long ago, no matter who supplies the labels — so a perfect teacher has nothing
left to teach. That predicts replay *should* help when the day contains
something the original training did not: a class the recogniser never saw, or a
real distribution shift. **That is the next experiment**, and it is written here
as a hypothesis rather than a result, because the last two structural stories in
this document were both wrong and were both caught by measurement rather than
by reasoning.

---

## 7.7 The condition under which replay improves perception

The question stopped being "does replay work" and became **"replay improves
latent sequential knowledge — under what conditions does it improve perceptual
representations?"** That is answerable, and the answer is a clean interaction.

The mind is always MNIST-trained. Only the *day it lives* varies, from material
its perceptual store already covers to material it has never seen:

| the day | no dream | self-labelled | **teacher-labelled** | teacher − no dream | wins | d |
|---|---|---|---|---|---|---|
| **familiar** (MNIST) | 0.8681 | 0.8606 | 0.8643 | −0.0038 | 2/6 | −0.40 |
| **mixed** (half Fashion) | 0.4751 | 0.4387 | **0.6880** | **+0.2130** | 6/6 | **+1.80** |
| **novel** (Fashion) | 0.0858 | 0.0764 | **0.4710** | **+0.3852** | 6/6 | **+7.70** |

`benchmarks/novel_day.py`, 6 seeds, a 176-fixation probe on held-out scenes
from the day's own world.

**Both factors are necessary and neither is sufficient.**

- *Familiar day + perfect teacher* → nothing (−0.004). A store already grown on
  15,000 clean digits has nothing left to be taught about digits.
- *Novel day + self-labelling* → nothing (−0.009). On Fashion the mind's own
  beliefs sit at chance, so replaying them reinforces noise. **Self-labelled
  replay never helps in any condition** (−0.008 to −0.036).
- *Novel day + teacher* → **+38.5 points, 6/6 seeds, d=7.70**, lifting detection
  from 8.6% (chance) to 47.1%.

And it is monotone in how unfamiliar the day is: −0.004 → +0.213 → +0.385.

So the standing hypothesis was right, and this is the form the claim should
take:

> **Replay improves perceptual representations only where the day carries
> material the perceptual store does not already cover, *and* an external
> signal supplies the label. Neither condition alone does anything.**

That is sharper than either earlier version — the teaching signal is necessary
but useless without novelty, and novelty is necessary but useless without the
signal.

One caveat kept in view: on the novel day `no_dream` sits at 8.6%, essentially
chance, so there is enormous headroom and part of +38.5 is a floor effect. The
mixed condition is the check on that — 47.5% → 68.8% with far less room, still
6/6 and d=1.80.

---

## 7.8 The real world: CIFAR-10 photographs and ESC-50 field recordings

Everything above this point was measured on MNIST, Fashion-MNIST, and synthetic
tones and chirps. Those are pre-segmented, contrast-normalised, single-object
gifts, and the goal explicitly asks for the other thing. This section replaces
them with **600 ESC-50 field recordings** (rain, sirens, engines, birds,
thunderstorms) and **CIFAR-10 photographs**.

### Three defects that were producing numbers rather than errors

Real data broke three things, and each returned a plausible-looking result
rather than an exception. That is the dangerous failure mode, so each is now
locked or re-derived.

**`_nearest_prototype` assumed labels were `0..n_class-1`.** ESC-50's *nature*
classes are labelled 10–19 and *urban* 40–49, so it built ten all-zero
prototypes and returned index 0 for every clip — **exactly 0.000**. The first
version of this section therefore read "all five auditory front ends collapse on
environmental sound", which is false. Two things gave it away: a linear probe on
the *same* features reached 0.55, and *animals* — the one subset whose labels
happen to fall inside the assumed range — worked normally. Fixed; pinned by
`tests/test_readout_labels.py`.

**`PredictiveA1.train` averaged per-frame `‖err‖/‖y‖`.** A field recording is
mostly silence, so `‖y‖ ≈ 0` was divided by the 1e-9 guard and a handful of
quiet frames set the epoch's figure: it reported **3,425,472 → 1,723,145**.
Pooled over frames (the standard normalised RMSE) the same run reads
**0.873 → 0.860**. Learning was never affected — the delta rule uses the raw
error — only the number was.

**`load_cifar10(grayscale=False)` cropped the channel axis.** Positional slicing
from axis 1 is correct for a grayscale stack and cuts the colour axis for a
colour one, silently returning a one-channel image of the wrong width.

### The auditory front end does hear the real world

Corrected, on 600 clips with stratified held-out splits (chance is 1/k):

| subset | classes | best | ×chance |
|---|---|---|---|
| all | 50 | 0.350 | **17.5×** |
| animals | 10 | 0.640 | 6.4× |
| nature | 10 | 0.638 | 6.4× |
| urban | 10 | 0.568 | 5.7× |
| human | 10 | 0.634 | 6.3× |
| interior | 10 | 0.655 | 6.6× |

The uncomfortable part: **the raw pooled cochleagram was the best front end**,
at 0.350 on the 50-way task, while the belt scored 0.198 — a 1024-cell spiking
layer losing to the spectrum it is built on.

### Why: the belt threw away frequency, and real sound *is* frequency

`AuditoryBelt` pooled across frequency *and* time, keeping only which filter
fired — a 36×497 cochleagram compressed to 38 numbers. That was measured and
correct for the designed 8-class bank, where classes differ by pattern and
absolute pitch is randomised across an octave and a half. Rain, wind, sea waves
and crickets invert the premise: they are stationary textures whose **spectral
profile is their identity**.

Adding a tonotopic channel alongside the invariant one, and sweeping its gain
over 8 paired splits (`benchmarks/belt_gain.py`):

| tono_gain | real (5 categories) | synthetic 8-class | d(real) | wins |
|---|---|---|---|---|
| 0.00 | 0.452 ± .023 | 0.870 ± .030 | — | — |
| **0.25** | **0.520 ± .036** | **0.879 ± .022** | +2.23 | 8/8 |
| 0.50 | 0.551 ± .032 | 0.846 ± .036 | +4.65 | 8/8 |
| 1.00 | 0.550 ± .021 | 0.778 ± .048 | +3.87 | 8/8 |

The two curves cross in a place with no tradeoff: at 0.25 **both** improve. Past
0.5 the designed bank's loss is geometric rather than informational — its linear
probe holds at 0.828 while its prototype read-out falls to 0.711.

### Cross-modal binding was collapsing to one concept cell

`AssociationArea.bind` chose its winner by pure argmax over the summed drive.
The first cell to win tuned toward the data, so it won the next pair too:
binding 48 pairs from 8 well-separated classes woke **1 concept cell out of 32**
and cross-modal recall sat at **exactly chance**, while the sound codes alone
were 96% separable. The homeostatic term meant to prevent this was scaled at 0.1
against a drive gap of ~1.0.

Fixed with ART vigilance (a poor match recruits an uncommitted cell instead of
dragging the incumbent) and a conscience term (DeSieno 1988) scaled to the
drive. On queries **held out** from binding, 5 seeds:

| vigilance | conscience | recall | shuffled | cells |
|---|---|---|---|---|
| 0.00 | 0.0 | 0.125 | 0.125 | 1.0 |
| **0.80** | **1.0** | **0.742** | 0.058 | 16.0 |
| 0.90 | 0.0 | 0.825 | 0.108 | 32.0 |

0.90/0.0 scores higher and is *not* the default: it was saturating the pool — 16
of 16, 32 of 32, becoming selective only at 40 cells once given 64. 0.80/1.0
recruits **15.3 cells whether the pool is 16, 32, 64 or 128**, so the number is
a property of the data rather than of the array.

### Concepts from two corpora that share nothing but meaning

CIFAR-10 and ESC-50 were built by different people for different tasks and
happen to contain the same real categories — airplane, automobile, bird, cat,
dog, frog. So the mind can be shown a *photograph* of a cat and played a *field
recording* of a cat, and **the only thing the two signals share is what they are
of**: no shared session, room, microphone or lighting, so no incidental
correlation to learn instead of the semantics.

360 pairs, 6 categories, 5 seeds, chance 0.167 (`benchmarks/real_binding.py`):

| path | real | control | which | d |
|---|---|---|---|---|
| sound → label | 0.931 | 0.156 | shuffled | 28.5 |
| **sound → vision** | **0.882** | 0.164 | mismatched | 16.1 |
| vision → label | 0.275 | 0.151 | shuffled | 6.9 |
| vision → sound | 0.236 | 0.143 | mismatched | 5.0 |

Two notes on honesty, both of which changed the reading:

*The controls had to be split.* Binding is unsupervised, so permuting labels
leaves `Wv` and `Wa` bit-identical and the two cross-modal read-outs reported
**+0.000** — correct by construction, and the same exact-zero signature as the
real bugs above. A `mismatched` control that re-pairs each sound with a
different category's sight is what actually moves them.

*The label path is mostly a nearest-neighbour index.* With vigilance high the
layer holds about one cell per training pair, so "hear a sound, find its cell,
read its name" is structurally 1-NN over the audio codes — which scores 0.924 on
its own. Fusion adds **+0.007** there, not the +0.40 a prototype baseline would
have suggested. **The genuine result is `sound → vision` at 0.882**: there is no
unimodal way to get a visual code from a recording at all.

### While the layer stores exemplars, every sound→X probe is one probe

`sound_to_vision` scored with a 1-NN read-out came out at **0.947 — exactly
`sound_to_label`**, to the last decimal. Not a coincidence, and worth stating
because it bounds every cross-modal number above: recruitment sets
`Wv[cell] = ` the pair's own visual code, so with one cell per pair the nearest
training image to a recalled sight *is* that pair's image, and its label is the
label the vote map holds. Every "hear a sound, get X" probe collapses to *find
the nearest stored sound and read off whatever was stored beside it*.

The one probe that does not collapse is the class-mean version, because it asks
where the retrieved code sits relative to the category rather than which
exemplar it is. That one reads 0.581–0.856 depending on the eye — and it beats
the **reference** (what the *true* visual code of the held-out item scores under
the identical probe, 0.333–0.367) by 174–233%. That is the one genuinely
pleasing result in this section: the recalled sight is a *cleaner* member of its
category than any real photograph, because a concept cell's `Wv` is an average
and a photograph is not. Recalling a better cat than you have ever seen is what
having a concept is for.

Until cells < pairs, that is the only cross-modal claim this architecture can
support, and it is why concept formation (§8, step 14) is the gate on everything
else.

### Vision is now the weak sense, and the reason is specific

Two recordings of the same source reach cosine 0.95; two photographs of the same
category reach **≈0** (median −0.006). Measured directly: `WideV1` scores 1-NN
**0.818 on MNIST and 0.185 on CIFAR-10**, where raw opponent pixels score 0.308.
A 1024-cell spiking layer doing worse than its own input is a coding fault, not
a hard-problem result.

The cause is a **common component**: 63.5% of cells are silent for a typical
photograph and the rest respond similarly to all of them, so cosine similarity
is dominated by the part every code shares. MNIST never showed it because a
digit is a high-contrast figure on an empty field — already mean-subtracted by
construction.

Three fixes, all things the biology has and this pipeline lacked, measured
cumulatively over 3 seeds (`benchmarks/natural_v1.py`):

| | CIFAR proto | CIFAR **1-NN** | MNIST 1-NN |
|---|---|---|---|
| grayscale, designed fields | 0.254 | 0.196 | 0.828 |
| colour opponency (L, R−G, B−Y) | 0.294 | 0.181 | 0.830 |
| + running-mean adaptation | 0.301 | 0.308 | 0.843 |
| + **discovered** receptive fields | 0.264 | **0.326** | **0.920** |

**1-NN on photographs 0.196 → 0.326, a 1.66× improvement, with no regression on
digits — MNIST gains 0.092.** The receptive fields in the last row were grown on
*CIFAR photographs* and applied to MNIST without retraining, so that row is
transfer: natural-image statistics make a better general-purpose V1 than
hand-written Gabors do, which is Olshausen & Field's result reproduced inside
this project's own spiking layer and consistent with its standing claim that
discovered fields beat designed ones.

Notes on what each part did, since they are not interchangeable:

* **Adaptation is where the 1-NN repair comes from** (0.181 → 0.308). It is the
  operation the belt and the workspace already had and vision did not, and it is
  kept **online** — a running mean updated per code, not batch statistics —
  because a mind meets its world one frame at a time.
* **Colour helps the class-mean read-out and not the exemplar one** (proto
  +0.040, 1-NN −0.015). Useful, but not the fix.
* **Discovered fields trade class-mean structure for exemplar structure**
  (proto 0.301 → 0.264, 1-NN 0.308 → 0.326). That trade is what made
  `sound → vision` read 0.856 with designed fields and 0.581 with discovered
  ones while vision itself got better by every other measure — the probe scores
  against class means, so changing the eye changes the ruler. Reporting the
  reference beside it is what makes the two comparable.
* **Spatial pooling, which was the belt's fix, is not the fix here**: pooled-only
  falls to 0.183 on CIFAR and 0.288 on MNIST. The two senses needed different
  repairs, which is worth knowing before assuming a fix generalises across them.

The eye is substantially better and **the gap to hearing is still open** — 0.326
against 0.914 for the ear on six real ESC-50 categories. That is the honest
statement; §8 step 13 is what remains.

### What a night does, and does not do, to concepts

`benchmarks/cross_modal_dream.py` replays **only sound** and asks the
association area what it expects to see, binding that imagined sight as if it
had been experienced. Nothing visual enters from outside.

| arm | sound → vision | vs no_dream | cells |
|---|---|---|---|
| no_dream | 0.714 | — | 216 |
| stored replay | 0.714 | +0.0000 | 216 |
| imagined | 0.721 | +0.0069 | 216 |
| confabulated (random sight) | 0.571 | −0.1435 | 256 |

Three findings, one of them structural:

1. **Ordinary replay is a no-op by construction here** — exactly +0.0000, the
   signature again, and this time it is real. With one cell per exemplar, a
   replayed pair re-selects its own cell and the instar step moves it toward
   where it already is.
2. **Confabulation is clearly destructive** (−0.14) and expands the pool to its
   limit, so the control behaves as a control should.
3. **Imagination helps only where the day was thin.** On a sparse day —
   categories given 3 waking examples — imagined replay moves the starved
   categories 0.524 → 0.552 while costing the rich ones 0.613 → 0.517. That is
   the generative-replay prediction (van de Ven et al. 2020) showing up with the
   right sign in the right place, and it is *not* yet a net win.

`AssociationArea.consolidate` was added to merge exemplars into concepts and
currently merges almost nothing, for a reason the measurement names: the joint
similarity criterion is limited by vision, whose between-exemplar similarity
tops out at 0.42. **Concept formation is blocked behind the visual front end**,
which is the same bottleneck as everything else in this section.

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
9. ~~**Then measure imagination properly.**~~ — **done**, and the blend read-out
   passes it: novelty 0.045 at a round trip of 1.000. Original note follows.
   **Then measure imagination properly.** `benchmarks/unified.py` already
   reports chain entropy, unique-concept count and the imagine→perceive round
   trip. Today: 3.02 bits over 10 concepts, round trip 100% because it only ever
   replays prototypes. A mind that is *actually* imaginative should show entropy
   rising with the concept count while the round trip stays high — novelty
   without incoherence. That pair of numbers is the goal made falsifiable.

### The real-world gap (now the critical path) — see §7.8

13. **Fix the visual front end on natural images.** Everything downstream is
    blocked behind it: concepts cannot consolidate because two photographs of
    the same category sit at cosine ≈0, so nothing ever merges. Colour opponency
    and online adaptation took 1-NN from 0.185 to 0.285 and are in; the gap to
    the auditory side (0.924) is still enormous. The next candidates are
    *learned* receptive fields (`selforganize.py` already has the machinery,
    and the project's own headline is that discovered fields beat designed
    ones) and keeping more than 28×28.
14. **Make the concept layer form concepts.** It currently holds one cell per
    experience — an exemplar memory with purity 1.00, which is why replay is a
    structural no-op. `AssociationArea.consolidate` exists and merges almost
    nothing while vision is this weak. Re-run it after step 13; the number to
    watch is cells-per-training-pair falling below 1.0 without recall dropping.
15. **Re-run the dream once concepts exist.** The sparse-day result already has
    the right shape — imagination helps starved categories (+0.028) and hurts
    rich ones (−0.096) — which is what prioritised replay is supposed to fix.
    Prioritise by `AssociationArea.novelty` rather than uniformly and the two
    halves should stop cancelling.

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
