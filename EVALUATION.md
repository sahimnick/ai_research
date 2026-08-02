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

**A correction to §7's headline, from §7.8.** The world-model gain reported
there — +14.6 points from replaying the day's order — is **repair, not
learning**. Replaying transitions adds a scaled copy of what `_T` already holds,
and a scaled copy cannot move an argmax; the gain existed only because
`build_unified_mind` seeds a counting curriculum that replay then dilutes.
Measured on real sound with the same machinery: no prior, +0.0000; a strong
wrong prior, +0.3464 — restoring exactly the value the model has with no prior
at all.

**§7.8 is the real-world pass** — 600 ESC-50 field recordings and CIFAR-10
photographs, two corpora that share real categories and nothing else, so any
association between them has to be semantic. Read it first if you want to know
what works outside the pre-segmented datasets.

Its headline, stated at the width the measurements support: **replaying a
concept's own average sight back into the layer that formed it improves real
cross-modal recall** — 6 of 6 seeds, d=1.79, worth 74% of replaying the sights
it actually saw, against a control that performs the identical binding
operations on a *random* sight and moves the result by exactly +0.0000. That
control is what makes it readable: the gain is the content, not the plasticity
events.

An earlier version of this line said the mind *imagines* a sight it has never
seen. That was measured and it is wrong: what it produces is **4.8× closer to
memory than a real unseen photograph is**, and closer to its class average than
a real member of the class. It recalls an average. The effect is real and the
word was not.

**A correction to that correction, from `composition.py`.** The ratio above
first read 3.2×, measured against the stored bank in raw code space while the
imagined codes live in `prep_v` space — the same photograph sits at cosine 0.899
to itself across the two, so every distance in that comparison was understated.
Corrected, it is worse than reported: **a third of what the mind "imagines" is a
stored photograph byte for byte**, because vigilance recruits by copying a pair
into a cell verbatim and 59 of 112 cells never win again. Composing several
concepts ends that (verbatim 0.397 → 0.001), but the novelty it creates is the
arithmetic of averaging near-orthogonal vectors, not the concepts: three stored
photographs averaged with no concept layer at all score 0.362 against
composition's 0.389.

Getting there meant repairing three things that had each produced a confident
wrong answer: the ear's read-out (a label-range bug reading as "the front end
collapses on environmental sound"), the eye (1-NN 0.185 on photographs against
0.818 on digits — now 0.326, with digits improving to 0.920 as a side effect),
and the concept layer, which held one cell per experience and made replay a
no-op by construction.

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

### Concepts now form — and the parameter that was wrong was not a value

The layer held **216 cells for 216 training pairs**. That was traced to
something more interesting than a mistuned threshold: `match` was the *average*
of the two senses' drive, which puts its scale at the mercy of the weaker one.
On real pairs the visual code's between-exemplar similarity sits near 0 while
audio reaches 0.95, so the averaged best match over a whole day peaked at
**0.740** — a vigilance of 0.80 was *unreachable*, every pair recruited, and
one cell per experience followed by construction. No value in (0.74, 1] could
have worked.

Taking the **max** instead — a cell is as awake as its best-driving sense, which
is what multisensory neurons do (Stein & Meredith's inverse effectiveness) —
fixes the real case and breaks the synthetic one, where both senses are strong,
almost every match clears the bar, nothing recruits, and the layer collapses to
**two cells**. Same code, same parameter, opposite failures. An absolute
threshold on a similarity cannot survive a change of data, because the
similarity *scale* is a property of the data.

Two changes, and neither is a tuned constant:

**Reliability-weighted combination.** Each sense is weighted by how *peaked* its
drive is across cells — a modality that returns the same value for every concept
has said nothing about which concept this is. That is reliability-weighted cue
combination (Ernst & Banks 2002), and it makes one rule cover both worlds
without being told which it is in.

**Homeostatic vigilance.** The fixed parameter is no longer a similarity but
`novelty_rate` — *what fraction of experience becomes something new* — with
vigilance driven to achieve it by an integral controller. This is the same move
`PredictiveA1` makes for sparsity and the belt makes for its per-band floor.
(Written with the error term inverted first. It did not read as a bug: the
synthetic bank *improved*, 16 cells → 32, because the sign happened to be
positive there and drove vigilance up until the pool ran out. A runaway wearing
the costume of a result; the giveaway was the other rule, frozen at two cells.)

At `novelty_rate=0.5`, over 5 seeds:

| rule | synthetic | real: cells/pair | s→label | s→vision |
|---|---|---|---|---|
| mean, fixed vigilance (as shipped) | 0.742 | **1.00** | 0.947 | 0.581 |
| mean | 0.792 | 0.57 | 0.943 | 0.660 |
| max | 0.458 | 0.48 | 0.942 | 0.681 |
| **reliability** | **0.742** | **0.53** | **0.947** | **0.671** |

`mean` is best on the synthetic bank and unusable on real data; `max` is the
reverse. Reliability is within a few points of the better of the two in each
world: it holds the synthetic number *exactly* (0.742), holds `sound → label`
exactly (0.947), raises cross-modal retrieval **0.581 → 0.671**, and takes the
layer from one cell per experience to roughly one per two. Below 1.0 there are
concepts; at 1.0 there is only a list.

**And consolidation finally merges something.** It had been merging exactly
nothing at every threshold, because the joint criterion was limited by vision.
With cells now grouping several exemplars each, a night at merge threshold 0.25
compresses 78 → 69 cells at purity 0.982, `sound → label` 0.925 against a floor
of 0.914, and `sound → vision` **unchanged at 0.696**. Merging harder does
compress more — 1.83× at 0.15 — but purity falls to 0.842 and recall to 0.797,
which is categories collapsing rather than forming, and the benchmark rejects it
on those grounds rather than reporting the bigger number.

### While the layer stored exemplars, every sound→X probe was one probe

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

That was the state before the two changes above. It is why concept formation was
the gate on everything else, and it is now cleared: at 0.32–0.53 cells per pair
the layer no longer has a private cell for each experience, so replay and
consolidation have something to act on and the cross-modal read-out is no longer
a lookup in disguise.

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

### A mind learns from a sight it never saw

This is the goal's hardest claim, and it now has a number. `cross_modal_dream.py`
replays **only sound**; for each replayed recording the association area is asked
what it expects to *see* and that imagined sight is bound as if it had been
experienced. Nothing visual enters from outside — whatever those concepts become
on the visual side, the mind made up.

Balanced day, 6 categories, 6 seeds, chance 0.167:

| arm | sound → vision | vs no_dream | d | wins | cells |
|---|---|---|---|---|---|
| no_dream | 0.676 | — | — | — | 115.0 |
| **stored** replay (real sights) | 0.707 | +0.0313 | 0.98 | 5/6 | 128.2 |
| **imagined** | **0.699** | **+0.0231** | **1.79** | **6/6** | 115.3 |
| confabulated (random sight) | 0.676 | +0.0000 | 0.00 | 2/6 | 115.8 |
| imagined, novelty-prioritised | 0.692 | +0.0162 | 1.19 | 4/6 | 115.3 |

**Imagining improves a real-world task in 6 of 6 seeds, and a random sight does
nothing at all.** The confabulation control is what makes this readable: the two
arms perform the identical number of binding operations on the identical
soundtrack, differing only in whether the visual half is imagined or noise, and
the noise arm moves the result by +0.0000. So the gain is not extra plasticity
events — it is the *content* of the imagining. Against replaying the sights the
mind actually saw, imagination is worth **74%**.

Two honest limits:

* **The effect is small** — 2.3 points on a 0.167-chance task. It is reliable
  (6/6, d=1.79) rather than large.
* **It does not rescue what was barely seen.** On a sparse day, where some
  categories get 3 waking examples, imagined replay does nothing (−0.012) while
  stored replay still helps (+0.015). A mind cannot imagine well from a category
  it has hardly met, which is sensible and is the opposite of what generative
  replay is usually proposed for (van de Ven et al. 2020). An earlier version of
  this table appeared to show the sparse-day gain; that was an artifact of the
  broken concept layer and does not survive its repair.

Two things had to be fixed before any of this could be read, and both are worth
recording because each produced a confident wrong answer first:

1. **With one cell per experience, replay is a no-op by construction.** The
   earlier table measured `stored` replay at exactly +0.0000 — a replayed pair
   re-selects its own cell and the instar step moves it to where it already is.
   Every other arm was equally unreadable.
2. **A night is not a day.** Left at the waking `novelty_rate`, the homeostatic
   controller kept minting categories during replay: 400 replays drove every
   dreaming arm to the 256-cell pool ceiling while `no_dream` sat at 115, so the
   arms differed in *how many cells they had* rather than in what they dreamt.
   With sleep's rate lowered to 0.02 all arms land at 115–128 and the comparison
   is about content. Sleep consolidates; it does not encode at waking rates.

Consolidation, once concepts existed to merge, compresses 78 → 69 cells at
purity 0.982 with `sound → vision` unchanged — modest, and no longer nothing.

### Where the eye runs out, and what it is not

Four candidate limits, each with a control (`benchmarks/vision_ceiling.py`),
1-NN on CIFAR-10 colour against 0.914 for the ear:

| lever | swept | result |
|---|---|---|
| width | 1024 → 2048 → 4096 cells | 0.323 → 0.315 → 0.315 — **flat** |
| aperture | rf 7 / 10 / 14 | 0.323 / 0.315 / 0.272 — bigger is worse |
| spike noise | spiking vs **noiseless** filter response | 0.322 vs 0.323 — **costs nothing** |
| integration | 50 ms vs 200 ms window | 0.322 vs 0.323 — nothing |

The third row is the one that matters for this project. The whole architecture
is committed to spiking neurons, and it would have been reasonable to suspect
that a windowed firing rate under membrane noise was throwing away what natural
images need. It is not: removing the neuron model entirely and reading the raw
filter response scores **the same number**, −0.002. Spiking is exonerated on
real photographs, and quadrupling the integration window buys nothing either, so
the rate estimate is not noise-limited. (It does cost ~20× the compute for that
identical result, which is a fair thing to know.)

Width is the striking negative. Single-layer unsupervised features on CIFAR are
known to scale with feature count, and going from 20 to 83 distinct filters per
location moves 1-NN by nothing at all.

So **nothing available within a single layer moves this**. That localises the
ceiling to depth — a second cortical stage — which is precisely the thing §1
measured as *harmful* (V1→V2→V3 at 39.2% against 63.4% for one wide layer). But
that was measured at narrow widths and on MNIST digits, where a single layer was
already near its ceiling and there was nothing for a second stage to add.
Photographs are the opposite case, so it was retried there.

### Depth fails a fifth time, on the case most favourable to it

`neurobrain/vision/secondstage.py` is a standard V2: complex-cell max-pooling
over V1 positions, a wider aperture across all V1 filters, and units learned by
competition with the instar rule — Hubel & Wiesel's simple/complex distinction
and HMAX's S/C alternation, no gradients anywhere. Built on the best
single-layer configuration (4096 cells, rf 7, stride 2, an 11×11 grid), 3 seeds:

Two learning rules were tried for it, not one. The second is Földiák's **trace
rule** — one winner chosen per *sequence of views of the same object* and then
updated on all of them, so invariance is picked up from the world changing more
slowly than the retina does (small shifts stand in for fixational drift). That
rule is the visual counterpart of what `PredictiveA1` does for hearing, and
hearing is the front end in this project that works, which made it the obvious
thing to try.

| code | CIFAR proto | CIFAR **1-NN** | MNIST 1-NN |
|---|---|---|---|
| V1 alone | 0.299 | **0.337** | **0.917** |
| V1 + V2, competition | 0.245 | 0.210 | 0.819 |
| V1 + V2, competition, concatenated | 0.312 | 0.323 | 0.912 |
| V1 + V2, trace rule | 0.255 | 0.202 | 0.845 |
| V1 + V2, trace rule, concatenated | 0.311 | 0.313 | 0.913 |

V2 alone loses a third of the accuracy on photographs and a tenth on digits.
Concatenating it with V1 — which lets a read-out take whichever it prefers, so
it can only fail by V2 adding nothing — lands at −0.014 and −0.004. The stage is
not adding information; it is discarding it.

The trace rule *did* do what it was designed to do: V2-alone on MNIST rises
0.819 → 0.845, which is the shift invariance it was supposed to buy. It simply
does not help on photographs, because shift is not what CIFAR is hard about.
A mechanism working exactly as intended and still not moving the number is a
more useful negative than a mechanism that failed.

This was the experiment most likely to overturn the project's central
architectural claim, run under the conditions chosen to favour depth: a first
layer far from its ceiling, wide rather than narrow, on photographs rather than
digits, with two different objectives. **Width instead of depth stands.**

One honest limit: these are two second stages, not all possible ones. They
settle that the eye's gap to the ear will not be closed by stacking another
competitive or slowness-driven layer on top — not that no hierarchy could.

### What did work was not a mechanism at all

After six mechanisms failed, the two remaining candidates were the ordinary
handicaps still sitting on the input. One of them is worth 5 points:

| input | proto | **1-NN** | 5-NN |
|---|---|---|---|
| 28×28 centre crop, opponent | 0.317 | 0.320 | 0.343 |
| **32×32 full frame**, opponent | 0.345 | **0.343** | 0.362 |
| 28×28, global ZCA whitening | 0.105 | 0.108 | 0.108 |
| 32×32, global ZCA whitening | 0.132 | 0.140 | 0.115 |

CIFAR had been centre-cropped to 28×28 for no better reason than that MNIST is
28×28 and the eye was built for digits. Handing it the whole frame is worth
+0.023 on 1-NN before anything downstream sees the code, and it carries through:
end-to-end on the audiovisual task, **`sound → vision` goes 0.581 → 0.775** and
the eye's own 1-NN goes 0.312 → 0.362.

The whitening rows in that table are **not a result** — 0.108 is chance, which
means the implementation was broken, not that whitening is harmful. Global ZCA
over 3072 dimensions from 2000 samples is badly conditioned, and re-normalising
each channel afterwards destroys the structure it just computed.

Redone correctly, per-patch. For stationary image statistics the ZCA matrix over
`p×p` patches *is* a convolution, so its centre row reshaped to `p×p` is the
retinal decorrelation filter (Atick & Redlich 1992) and can be applied in one
pass:

| input (32×32) | proto | **1-NN** | 5-NN |
|---|---|---|---|
| opponent channels | 0.345 | **0.343** | 0.362 |
| + per-patch ZCA, 5×5 | 0.222 | 0.188 | 0.192 |
| + per-patch ZCA, 9×9 | 0.217 | 0.197 | 0.238 |
| + per-patch ZCA, 13×13 | 0.225 | 0.212 | 0.207 |

**Whitening hurts, consistently, at every kernel width.** The explanation is
specific to this eye rather than general: whitening flattens the spectrum, which
removes low-frequency structure — and the low-frequency structure of a
*colour-opponent* channel is the broad chromatic signal that says sky, grass,
fur. Colour is measurably where this eye gets much of its class information
(+0.040 on the prototype read-out when opponency was added), so decorrelating it
away costs more than the edge enhancement returns. Whitening earns its keep in
pipelines that follow it with thousands of learned dictionary elements and a
discriminative read-out; it does not here.

That is the seventh mechanism to fail, and it makes the shape of §7.8 hard to
miss: **six mechanisms and one preprocessing step failed to move the eye, and a
cropping decision inherited from MNIST was worth more than all of them.**

### The neuron model is ~90% of the compute and none of the accuracy

`vision_ceiling.py` found that replacing the spiking rate code with the raw
filter response changes CIFAR 1-NN by −0.002. Profiling the real-world pipeline
shows what that costs, and the same test on the *auditory* side gives the same
answer:

| stage | spiking | without the neuron model | accuracy |
|---|---|---|---|
| `WideV1` on static images | 4.5 s | 0.2 s — **18×** | 0.322 → 0.323 |
| `AuditoryBelt` on 600 real clips | 12.6 s | 7.4 s — 1.7× | 0.527 → **0.527** |

The belt row is the striking one: identical to three decimal places on five real
ESC-50 categories. Across both senses, simulating Izhikevich dynamics over a
50 ms window contributes **nothing** to recognition accuracy on real data.

This is not an argument for removing it. §1 measured what the window is actually
for — 4-way motion direction at 72.0% against a 28.5% static control, and
left-versus-right at 75.5% on sequences containing *identical frames in reversed
order*, which no static code can pass. The window carries **movement and
temporal order**, and identity is simply not where it pays.

The practical consequence is that any experiment about *identity* — every
measurement in this section — can use the static path and run an order of
magnitude faster, and any experiment about motion or streams cannot. That
distinction was not previously stated anywhere, and stating it is worth more
than the speed: it says precisely what the project's central biological
commitment buys and where.

### Looking transfers to photographs; recognising does not

Every measurement above hands the mind a whole photograph at once, centred and
static. The eye — a fovea at full resolution, a heavily blurred periphery,
saliency-driven saccades, inhibition of return, fixational drift inside every
50 ms window — is not used at all, and had only ever been pointed at scenes made
of digits. So it was pointed at scenes made of CIFAR-10 photographs: twelve on a
256×256 canvas, no positions given, nothing segmented.

| | on-object (corrected) | ×chance | named |
|---|---|---|---|
| **CIFAR-10 photographs** | **0.883** | 6.2× | **0.134** |
| MNIST digits (reference) | 0.830 | 5.8× | 0.559 |

*chance of landing on an object by accident is 0.144; naming chance is 0.100.*

**The eye finds photographs slightly better than it finds digits.** A saliency
map built on local contrast, tuned entirely on high-contrast strokes against an
empty field, transfers to natural images without modification — which was not
obvious, since a photograph has no empty background and no single bright stroke.

And then naming collapses to 0.134 against a chance of 0.100.

This is the cleanest localisation in the section. The perception stack's failure
on natural images is **entirely in recognition, not in attention**: the mind
looks in the right place and cannot say what is there. Seven mechanisms inside
the recognition path have now failed to move it and one cropping decision moved
it more than all of them, and the faculty that every one of those experiments
skipped turns out to be the one part that already works.

### It does not imagine. It recalls an average — and the word matters

The section above is headed "a mind learns from a sight it never saw", and the
effect is real: 6 of 6 seeds, against a confabulation control at exactly zero.
But "imagines" was doing unearned work in that sentence, and asking what the
imagined content actually *is* settles it. The imagined sight is
`Wv[concept_from_sound(a)]` — a stored weight vector, and a concept cell's `Wv`
is the running average of everything bound to it. `benchmarks/imagination_shape.py`
measures the four things that would distinguish imagining from recalling, on
held-out sounds, 5 seeds:

| | imagined | a real unseen photograph |
|---|---|---|
| closeness to a **remembered** sight | **0.979** | 0.203 |
| closeness to its **class prototype** | **0.214** | 0.060 |
| residual outside the hull of experience | 0.706 | 0.940 |

> **Corrected.** This table first read 0.886 / 0.277 / 0.186 / 0.074 / 0.725 /
> 0.908. Those numbers compared imagined codes against the stored bank in **raw
> code space**, while `Wv` and everything `imagine_vision` returns live in
> `prep_v` space — the same photograph sits at cosine 0.899 to itself across the
> two, so every distance was read through a 0.1 fog. Found by
> `benchmarks/composition.py` and locked by `tests/test_imagination_space.py`.
> Every conclusion below survives the correction and several get sharper: at
> 0.979 the produced code is not "close to" a memory, it is *often a memory*.

**What it imagines is 4.8× closer to memory than opening its eyes is.** That is
the decisive number. If perception is more novel than imagination, imagination
is the wrong word: nothing is being invented, something stored is being replayed.

That first row is a **mean over two populations** and saying "almost verbatim"
about it would overstate what it covers. Split: **36.8% is bit-identical** to a
stored photograph (cosine > 0.999) and the rest sits at 0.967. Both halves are
far closer to memory than a real unseen observation (0.203); they are not the
same failure.

The second row says the produced code sits 3.6× closer to the class centre than
a real member of that class does. The defensible claim there is that it
**behaves like a learned class prototype** — a centroid, a manifold centre and a
learned attractor all produce that signature, and the measurement does not
separate them. It is specifically *not* "a class average": a cell that won once
holds one exemplar, not a mean of several, and that is a third of them. The
third row says it stays further *inside* the convex hull of experience than a
real photograph does, so it interpolates and never extrapolates.

And the inner world is not infinite. Over 144 held-out sounds it produces **59
effectively distinct imaginings — 5.88 bits** — from 112 concept cells. That is
the whole vocabulary: a finite set of attractors, one per concept, and the
"imagining" is a lookup into it.

So the honest reading of the +0.0231 gain is **self-distillation, not
imagination**: replaying a category's own average back into the layer sharpens
the categories, which helps, and is exactly what the consolidation literature
describes rather than what the generative-replay literature does. It also
explains the anchoring result below without any appeal to imagination — feeding
back an average works while feeding back *noise* does not, and feeding back both
halves of the average (`dreamt`, −0.0301) collapses the categories it is
averaging over.

The claim this project can defend is therefore narrower than the one it made:
**replaying a concept's own average into the layer that formed it improves real
cross-modal recall.**

#### Giving the concepts something to vary along

A mean has one output. So concept cells were given the *directions they vary in*
— a few principal components per cell, learned by Oja's rule with Sanger
deflation, which is local, Hebbian and gradient-free — and `imagine_vision`
samples `mean + Σ z·σ·direction`. At temperature 0 this reduces exactly to the
old behaviour, so the two sit on one axis.

It took one bug to get there, and the bug is worth recording because it produced
a *flat* result rather than an error: Oja's rule is multiplicative in the current
correlation, and in 12288 dimensions a randomly initialised direction has
essentially none. The modes never left their initialisation, variance stayed at
0.003, and sampling at temperature 16 moved fidelity by 0.002 — a mechanism
reporting "no effect" because it had never started. Seeding an unused mode from
the first residual it meets fixes it (variance 0.003 → 0.45).

With it working, 5 seeds:

| temperature | fidelity (↓ is more novel) | coherence |
|---|---|---|
| 0.0 (the mean) | 0.886 | 0.775 |
| 1.0 | 0.788 | 0.658 |
| 2.0 | 0.751 | 0.556 |
| 8.0 | 0.714 | 0.511 |
| 16.0 | 0.700 | 0.479 |

*a real unseen photograph scores 0.277.*

Sampling **does** produce genuine novelty — 0.886 → 0.700 is a real move, not
noise — and it trades against coherence monotonically, which is the shape a
temperature should have. But it saturates at 0.700, still 2.5× closer to memory
than a real photograph, and **more directions do not help**: 4, 16, 64 and 128
modes give 0.693 / 0.728 / 0.725 / 0.706. The dimensional explanation was the
obvious one and it is wrong.

The actual reason is structural, and it also indicts the test. The imagined code
is assembled from stored components — a mean of training codes plus directions
derived from their residuals — so it necessarily lies in the span of memory. A
real photograph does not, and this eye makes distinct photographs *nearly
orthogonal* (between-exemplar cosine ≈ 0), which is why a fresh one scores 0.277.
So "further from memory than a real photograph" may be unpassable by **any**
recombination-based process in this representation, and a bar that no
construction from experience could clear is a bar about the representation
rather than about imagining.

What survives, stated at the width it deserves: the mechanism now produces
**controllably novel** members of a concept rather than one average, the
novelty-coherence trade is measured, and the limit is that everything it makes
stays inside the span of what it has seen. Whether that counts as imagining is a
question the numbers can inform and not settle — but "it recalls one average per
concept", which is what was true before, is no longer the description.

### Accumulating glances is blocked by the same missing property

`natural_scene.py` left the sharpest split in the project: the eye finds
photographs (on-object 0.883) and cannot name them (0.134). Seven mechanisms
inside the recognition path had already failed, so the next attempt used **more
of the faculty that works** instead of a better code — the eye lands on the same
object several times during a free view, and every glance is currently
classified alone and thrown away.

It does not work, and the reason is worth more than the attempt.

**The glances are near-duplicates.** Median pixel offset between successive
fixations on the same object: **0.0**. Their codes sit at cosine 0.879 against
0.347 for different objects. The corrective saccade drives to the same saliency
peak every time, so there is no independent evidence to accumulate — pooling
gains +0.009 (d=0.22) and the reason is arithmetic, not biology.

Two things fell out of building the test, both corrections to my own design:

* Pooling *codes* and pooling *scores* are **mathematically identical** here —
  `unit(mean(C))` is proportional to `sum(C)`, so nearest-prototype cannot tell
  them apart. They agreed to three decimals on every row, which is the
  arithmetic confirming itself rather than two mechanisms agreeing.
* Two of my own benchmarks reported per-fixation naming on MNIST as 0.559 and
  0.733. Re-run on identical settings they agree (0.645 against 0.641), and
  `which_object` places **100%** of on-object fixations (195/195, 256/256). The
  gap was seed variance across a small number of scenes — which is itself a
  warning about how much weight the single-digit gains in this section can take.

**Then the fix, and the trap.** Letting the eye scan *parts* of an object rather
than re-centring (`SaccadicEye.free_view(explore=)`) removes the redundancy
exactly as designed — 0.856 → 0.168. It still does not pay:

| explore | redundancy | MNIST k=1 | MNIST k=4 | CIFAR k=1 | CIFAR k=3 |
|---|---|---|---|---|---|
| 0 | 0.842 | **0.733** | 0.753 | 0.114 | 0.128 |
| 4 | 0.235 | 0.257 | 0.349 | 0.128 | 0.139 |
| 8 | 0.224 | 0.101 | 0.090 | 0.106 | 0.147 |
| 12 | 0.293 | 0.094 | 0.117 | 0.085 | 0.150 |

The wide V1 code is retinotopic and has **no translation invariance at all** —
which this module already knew, since the corrective saccade exists because
free-viewing recognition collapsed to 17.2% against 78.3% for centred crops. So
an off-centre glance is a bad glance, and the two requirements are in direct
opposition: **diverse glances must be off-centre, and off-centre glances cannot
be read.** The correction that makes single glances work is exactly what makes
repeated glances redundant.

That closes the loop on this section. Accumulation, the second stage, the trace
rule, pooling, width — every one of them is blocked by the same missing property.
The eye's problem has one name, and it is not depth or capacity: it is that
nothing in this visual pathway is invariant to where the thing is.

### The invariant code was not invariant — a contract violated by the learning rule

The section above ended by naming translation invariance as the missing
property. Testing that named something more specific and more fixable.

`WideV1.pooling_index` promises a shift-invariant code by grouping cells that
"share a filter but sit at different locations", and it identifies those cells
**purely by index**, `arange(n_cells) // n_pos`. That is a filter identity only
if every hypercolumn holds the same bank. Measured, by the cosine between filter
*k* at different positions:

| | filter-index cosine across positions |
|---|---|
| untrained `WideV1` | **0.977** — tied |
| after `develop_v1` | **0.454** — against 0.319 for random pairs |
| `AuditoryBelt`'s layer (never developed) | **0.978** |

`develop_v1` lets each column drift independently, so after development "filter
k" means a different pattern at every position and pooling by filter index sums
**unrelated cells**. The code was not invariant, and the giveaway had been
sitting in the data: a fully pooled code fell from 0.382 to chance over a 10 px
offset, which a code that discards position cannot do unless the grouping is
wrong.

The other sense is the control that confirms it. `AuditoryBelt` builds a
`WideV1` and never develops it, so its bank stays tied at 0.978 — which is why
the identical pooling operation genuinely works there, and is part of why
hearing outperforms vision in this project.

**The fix is weight tying** (`develop_v1(..., tie=True)`), which is also the
standard assumption about V1 rather than a convenience: an orientation channel
is repeated across the retinotopic map, and that repetition is what makes the
map a map. It restores the cosine to 1.000, and with it the invariance:

| MNIST, accuracy vs offset | 0 px | 4 px | 8 px | 10 px | fall |
|---|---|---|---|---|---|
| `pooled both`, untied | 0.382 | 0.161 | 0.117 | 0.108 | **+0.274** |
| `pooled both`, **tied** | 0.233 | 0.241 | 0.208 | 0.191 | **+0.042** |
| `pooled cols`, tied | 0.628 | **0.333** | 0.199 | 0.196 | +0.432 |
| position-specific | 0.773 | 0.237 | 0.103 | 0.105 | +0.668 |

The tied invariant code is nearly **flat** across a 10 px shift where the untied
one collapses. And there is a crossover: past 4 px the tied `pooled cols` beats
position-specific (0.333 against 0.237, and 0.196 against 0.105 at 10 px), which
is exactly the regime free viewing lives in.

**And accumulation works there.** The question `multi_fixation.py` could not
answer, re-asked on a tied bank:

| free-viewing arm | k=1 | k=4 | accumulation |
|---|---|---|---|
| MNIST `pooled both`, untied, explore 8 | 0.093 | 0.069 | −0.024 |
| MNIST `pooled both`, **tied**, explore 8 | 0.130 | 0.183 | **+0.054** |
| CIFAR `pooled both`, **tied** | 0.065 | 0.178 | **+0.113** |

Pooling several glances helps only where the code is genuinely invariant, and
there it is the largest accumulation effect measured anywhere in this project.

Two things this does **not** claim. The invariant codes are weak in absolute
terms — 0.233 against 0.773 for position-specific on centred digits — so this
does not rescue overall accuracy, it repairs a mechanism that was silently
broken and shows the mechanism behaving as documented. And the first version of
this measurement was itself invalid: a 28×28 object shifted 10 px inside a 28×28
frame loses a third of itself, so every code including the invariant one fell to
chance and the conclusion would have been "invariance is not the missing
property". Giving the object a 48×48 frame to move in is what made it a
translation test rather than an occlusion test.

### The assembled mind, finally on photographs

Ten benchmarks in this project use real photographs and recordings. Eleven use
the assembled `UnifiedMind` — episodes, `watch`, `dream`, the transition model
`_T`, the acceptance gate. **The overlap was zero.** Every conclusion about
replay and the world model was drawn on handwriting; every conclusion about real
sensation was drawn with none of that machinery attached.

It was a wall rather than an oversight. `build_unified_mind` loaded MNIST inside
itself, seeded a counting curriculum 0→1→…→9, hard-coded ten digit cues, and
`perceive` reshaped its input to `(1, 28, 28)`. `build_mind_on` takes the data as
an argument and drops the two digit-specific parts: the counting prior, because
"after 3 comes 4" means nothing between a cat and an airplane, and the fixed ten
cues.

**Four hard-codings, each of which failed silently rather than raising.** They
are worth listing because every one produced an exact zero that reads like a
scientific result:

* `perceive` reshaping to `(1, 28, 28)` — `contrast_normalise` flattens anyway,
  so `(1, -1)` was always correct.
* `perceive` returning `str(int(label))` while the pallium, workspace and world
  model were keyed by *names*. A mind whose memory says "cat" and whose
  perception says "3" scores **exactly 0.000** on every cross-store probe.
* `_replay_into_perception` doing `int(label)` inside a `try` and returning on
  failure. With named concepts that is a silent no-op: detection was frozen at
  +0.0000 **at every learning rate**, including 20× the one §7.7 needed.
* `p.size != 784`, and a `(2000, 28, 28)` noise batch in the recogniser's
  don't-know calibration.

And one measurement bug of mine: I reported `_T` as "100 of 100 entries before
and after" and read it as saturation. `_grow_T` initialises every entry to 0.05,
so the nonzero count is n² before any experience and cannot show learning at all.
Mean normalised row-max does.

With those fixed, the mind runs on photographs:

| | value | reference |
|---|---|---|
| episodes laid down per day | 180 | — |
| world-model sharpness | **0.393** | 0.100 if uniform |
| distinct surprise values | 79 | 2 in the original defect |
| detection | 0.129 | chance 0.100 |
| recall (1-NN) | 0.196 | chance 0.100 |

**What replay does here, against what it did on digits:**

| metric | delta | d | wins |
|---|---|---|---|
| detection | +0.0087 | 0.25 | 2/4 |
| world model | **+0.0000** | 0.00 | 0/4 |
| recall 1-NN | +0.0030 | 0.63 | 3/4 |

Detection now *moves* — the wire that the third hard-coding had cut is connected,
and the exact zero is gone. It is small and unreliable, but it is no longer
structural.

The world model is the real result, and it is negative: **the +14.6 point gain
§7 measured on digits does not transfer.** The obvious explanation — that MNIST's
counting curriculum supplied a wrong prior for replay to correct, and this mind
has none — was tested and **refuted**: giving the photograph mind an equally
arbitrary prior makes replay *worse*, not better (−0.0101 at 10 repetitions,
−0.0065 at 40).

What is left is upstream. The world model is built out of percepts, and the eye
names photographs at 0.129 against 0.559 for digits. Replaying a sequence of
mostly-wrong percepts cannot sharpen anything, however good the replay
machinery is. §7's world-model result was real and **conditional on perception
working** — a condition digits met and photographs do not.

### Replaying transitions does not learn. It undoes a prior.

`real_mind.py` left the world-model result unexplained: §7's +14.6 points from
replaying the day's order does not transfer to photographs, and the reason
looked upstream — the eye names photographs at 0.129, and replaying
mostly-wrong percepts cannot sharpen anything.

That story makes a prediction, and hearing is the test of it: 1-NN 0.914 on six
real ESC-50 categories. `real_time.py` builds the same mind on **belt codes
instead of pixels** — `build_mind_on` only flattens, so sound codes work where a
digit mind takes images — and walks it down an imposed street of real urban
recordings (car horn → engine → train …; the sounds are real, the succession is
designed, because ESC-50 clips have no natural succession).

Two of my own faults had to be removed first, and both would have produced a
publishable-looking wrong answer:

* **The three arms lived three different days.** The day was generated inside
  the arm loop, so the generator advanced between arms and the comparison was
  across days, not across nights. That artefact alone produced a clean-looking
  "+0.0405, d=1.84, 4/4".
* **Perception collapsed to one class.** Belt codes are non-negative and share a
  large component across every clip; `contrast_normalise` removes each sample's
  *own* mean and does nothing to a component common to all of them. Same-class
  cosine 0.986, different-class 0.976 — a separation of 0.010 — and
  `GrowingCategoryMap` grew **one** category at every vigilance from 0.3 to 0.9.
  The mind named all 60 sounds "engine", so `world_self` read a perfect 1.000.
  `PopulationAdaptation`, the fix already measured for vision, takes the
  separation to **+0.390**, 31 categories, and accuracy 0.221 → 0.750. The same
  defect in the other sense.

With perception working — the ear names its own clips at **0.818** over 4.2
distinct percepts, against 0.129 for the eye — and the world model genuinely
learning the street (0.480 against a chance of 0.167):

**replay changes it by exactly +0.0000.**

So the upstream story is wrong. Good percepts are not what was missing. Sweeping
the strength of a competing prior says what is:

| prior repetitions | before | after | delta | wins |
|---|---|---|---|---|
| 0 | 0.480 | 0.480 | **+0.0000** | 0/4 |
| 5 | 0.472 | 0.476 | +0.0042 | 1/4 |
| 20 | 0.472 | 0.472 | +0.0000 | 0/4 |
| **60** | **0.126** | **0.472** | **+0.3464** | **4/4** |

A strong prior *destroys* the world model, 0.480 → 0.126, and replay restores it
to **0.472 — the value it has with no prior at all**. Replaying transitions is
not learning. It is **undoing self-inflicted damage**, and it cannot do more than
return the model to what the day already said.

That is also the retrospective explanation of §7's +14.6. `build_unified_mind`
seeds a counting curriculum, 0→1→…→9, twenty times. Counting is a wrong prior
about which digit follows which, replay diluted it back toward the lived order,
and the gain was repair rather than acquisition. The mechanism was never adding
knowledge; it was removing a curriculum.

Arithmetically it could not have been otherwise: `predict_next` is an argmax over
a row of `_T`, and replaying the day's transitions in proportion adds a scaled
copy of what is already there. A scaled copy cannot move an argmax. Only a
*competing* distribution can be tipped.

### The imagining has to be anchored to something real

The goal asks for an *infinite* inner world — new percepts and concepts made
from the mind's own imagination rather than from replayed episodes. There are
finitely many episodes and unboundedly many points between concepts, so three
arms were added that use no episode at all:

| arm | what is imagined | balanced | wins | purity |
|---|---|---|---|---|
| `imagined` | **the sight only**, real recording kept | **+0.0231** | **6/6** | 1.000 |
| `dreamt` | sight *and* sound, from one concept cell | −0.0301 | 0/6 | 0.923 |
| `blended` | both, between two same-label concepts | −0.0046 | 1/6 | 0.706 |
| `chimera` | both, between two different concepts | −0.0035 | 2/6 | 0.592 |

**Only the anchored one helps.** The moment the night stops containing a real
sensory signal and the mind invents both halves, the gain does not merely
vanish — `dreamt` is the *worst* arm in the table at −0.0301, worse than
confabulating a random sight. Purity says why: self-generated content drags
categories together (1.000 → 0.923 → 0.706 → 0.592) because nothing outside the
mind is holding them apart.

That is the same shape as this project's earlier finding in §7.6 — replay into
perception needs an external teaching signal — arriving independently in a
different mechanism, and it is the well-attested failure mode of training a
generative system on its own output. The blending is not useless in principle:
on the *sparse* day it is the only arm that helps at all (+0.0058, 4/6, and
+0.0140 on the starved categories for the chimera version), which is where
interpolation between concepts should pay. But those are effects of a fraction
of a point and none of them clears the bar.

So the honest form of the goal's claim, as measured: **a mind can learn from a
sight it never saw, provided something it really heard is holding that sight in
place.** An entirely self-generated inner world, in this architecture, degrades
the concepts it is made of.

### A third of what it "imagines" is a memory, byte for byte

`benchmarks/composition.py`, 5 seeds, 360 real CIFAR-10 / ESC-50 pairs over six
shared categories. It set out to test *combining* concepts and to replace a
metric I had got wrong, and on the way it found something worse than the thing
it was built to fix.

Measured in the space the concept cells actually live in, the imagined sight is
not 0.886 from the nearest memory. It is **0.979**, and the reason is
structural rather than a matter of degree:

> of **112** concept cells recruited from 216 pairs, **59 won exactly once**.
> Vigilance recruits by copying the pair into an uncommitted cell *verbatim*,
> and `_grow_subspace` is only reached on the non-recruiting path — so a
> once-winning cell's `Wv` row **is** a stored photograph and its `mode_var` is
> exactly zero.

For those cells `imagine_vision` returns the exemplar unchanged at **every
temperature**. **36.8% of held-out sounds wake one.** That is what the 0.70
saturation in the earlier temperature sweep was made of: it was never a weak
generative model, it was a third of the population that cannot move at all,
averaged against two thirds that can.

The two-thirds that can move do something almost as awkward. A cell that won
exactly twice learns one mode, and that mode is necessarily the line joining its
two members — so sampling along it interpolates between two memorised
photographs and at high temperature lands on one. On the cells that *do* have a
subspace, the verbatim rate climbs 0.000 → 0.044 as temperature goes 0 → 4.
**Turning up the noise on a single concept walks toward memory, not away.**

#### A metric that is about imagining, and its two failure modes

The old bar — "further from memory than a real photograph is" — was
unpassable by construction: this eye makes distinct photographs nearly
orthogonal, so anything assembled from stored parts is inside the span of memory
and can never clear it. The replacement is a *pair*, since neither half means
anything alone: **novelty** (1 − cos to the nearest stored thing) and
**coherence** (does it still read as its own category). The target is where real
held-out data sits, not an unreachable corner.

Two arms exist purely to attack that metric, and both landed:

- **`crossed`** — chimeras from concepts the mind grouped *differently* — reached
  a scalar gap of 0.423, the best in the table, by driving coherence to 0.275,
  *below* real data's 0.361. A plain distance is gameable from the low side, so
  ranking now requires an arm to be **at least as coherent as real data** and
  scores it on novelty. `crossed` is disqualified by its own result.
- **`real+imagined`** — the goal's own phrasing — reached gap 0.165 at
  `anchor=0.75`, and sits at cosine **0.949 to the very photograph it is
  supposed to be imagining about**. Its gap measures how much of the answer was
  copied from the question. It is reported with that cosine printed beside it
  and excluded from the ranking.

#### What composition actually buys

| arm | novelty | coherence | verbatim |
|---|---|---|---|
| a stored exemplar | 0.000 | 0.790 | 1.000 |
| **sampled T=0** (the mean — behaviour before any of this) | 0.021 | 0.831 | **0.368** |
| sampled T=8 (best single concept) | 0.208 | 0.557 | **0.397** |
| **composed T=1** (best ranked arm) | **0.389** | 0.843 | **0.001** |
| *control:* 3 stored photographs averaged, no concept layer | 0.362 | 0.915 | 0.001 |
| crossed T=4 *(disqualified — coherence below real)* | 0.383 | 0.275 | 0.000 |
| **a real unseen photograph** ← the target | **0.797** | **0.361** | 0.000 |
| gaussian noise | 0.975 | 0.178 | 0.000 |

**One real win: composition ends verbatim recall.** 0.397 → 0.001. A singleton
concept has no spread of its own — one observation carries no variation, which
is not a bug to patch — but the *category* it belongs to does, and composing
three siblings reaches it. That is worth having and it is the mechanism's own
result.

**And one honest negative, which is the finding.** The control decides it:
averaging three stored photographs *with no concept layer at all* reaches
novelty 0.362 against composition's 0.389. Paired over 5 seeds that margin is
+0.0276 (sd 0.0100, d = +2.76, 5/5) — consistent, and negligible. Averaging *k*
near-orthogonal unit codes sits 1/√k from each of them by arithmetic alone;
at k=3 that is 0.42 of novelty for free, which is essentially all of what the
composed arm scores. **The novelty of a composition is that arithmetic.** The
concept cells contribute the coherence (0.843 vs the control's 0.915 — they
contribute slightly *less*), and temperature spends it.

So the ceiling stands and is now quantified: **nothing self-generated exceeds
novelty 0.389 against real data's 0.797.** Everything this architecture can
assemble stays in the span of memory, and no amount of mixing or sampling
changes that — because mixing and sampling are both linear operations on stored
vectors. Getting past it requires a source of variation that is not a stored
code: composition over *parts* rather than whole codes, or a generative step
that is not a weighted sum. That is the next thing to build, and it is now a
specific requirement rather than a wish.

### A yellow bus: the first thing this project builds that memory cannot assemble

The ceiling above has an algebraic cause, and naming it supplies the way past it.
Every way this architecture had of imagining — reading a concept's mean, sampling
its subspace, mixing concepts, anchoring to a percept — is a **weighted sum of
stored vectors**, and sums of stored vectors lie in the span of stored vectors.
So the right instrument is not cosine to the nearest memory but the residual
after projecting onto the span of everything stored: **exactly zero means the
code is a linear combination of things already seen**, whatever the cosines say.

`benchmarks/factored.py`, 5 seeds, same 360 real pairs. The stored bank spans a
**215-dimensional** subspace of 12288.

| arm | span residual | in span | whole-code reads | **form block reads** | took donor's colour |
|---|---|---|---|---|---|
| a stored code | 0.0000 | 1.000 | 0.135 | 0.135 | — |
| the concept mean | **0.0000** | **1.000** | 0.831 | 0.660 | — |
| sampled T=4 | **0.0000** | **1.000** | 0.571 | 0.465 | — |
| composed T=4 | **0.0000** | **1.000** | 0.579 | 0.493 | — |
| **factored (form + donor colour)** | **0.5287** | **0.000** | 0.261 | **0.660** | 1.000 |
| *control:* both blocks, same cell | 0.0000 | 1.000 | 0.831 | 0.660 | 0.000 |
| a real unseen photograph | 0.8298 | 0.000 | 0.361 | 0.361 | — |
| gaussian noise | 0.9912 | 0.000 | 0.176 | 0.163 | — |

**Every mixing and sampling arm scores exactly 0.000 and is in the span 100% of
the time.** That is `composition.py`'s ceiling stated as what it actually is —
not "hard to escape" but *provably impossible*, confirmed numerically to six
decimal places. No parameter of those mechanisms was ever going to matter.

#### What crossing factors does instead

The eye's code was already factored and nobody had used it: `rate()` concatenates
the three retinal opponent channels, so `V = [luminance | red-green | blue-yellow]`
and **form and colour occupy disjoint blocks**. With `Wv[A] = [f_A, c_A]` and
`Wv[B] = [f_B, c_B]`, the recombination `[f_A, c_B]` is in `span{A, B}` only if
one scalar is both 1 and 0. It is not a sum, so the ceiling does not apply.

Measured: **residual 0.5287, outside the span in 100% of cases**, at 64% of a
real unseen photograph's residual. The control settles that this is the
*crossing* and not the slicing — rebuilding both blocks from the **same** cell
returns residual exactly 0.0000 and reproduces that cell bit for bit.

And it is not damage. The whole-code read-out falls 0.831 → 0.261, but that
read-out is looking at a code whose two chromatic blocks now belong to a
different object, so it charges the recombination for its own confusion. Asked
of the form block alone, the recombination reads **0.660 — identical to the
concept mean's 0.660**. The form is untouched; only the colour changed, which
is the entire intent. It clears the coherence floor `composition.py` imposed
(0.660 against real data's 0.361), which the chimera arm did not.

So: **a code the world never presented, that memory cannot assemble, that the
mind still reads as its own object.** A bus that is yellow. It is the first
construction in this project that is novel in the strong sense rather than the
cosine sense, and the mechanism is four lines of slicing — the representation
had been factored all along.

What this does **not** yet do, stated so the claim stays the size of the
evidence: the recombination is a *sight*, not a hypothesis. There is no causal
model saying which crossings are possible, no constraint on the result, and
nothing consumes it — a bus with a frog's colour and a bus with a bird's colour
are equally available and equally unjudged. Leaving the span was the blocking
problem, and it is solved; choosing *which* point outside the span is worth
imagining is the next one, and it is what a world model would be for.

### The inner world is real and nothing consumes it

A sight nothing uses is a curiosity. The goal is the *re-creation of the outside
world inside*, and the only evidence that anything has been re-created is that
the mind gets better at the world it has to live in. `benchmarks/inner_world.py`
runs `cross_modal_dream.py`'s protocol on content that benchmark could not
produce — every arm keeps the **real recording** and invents only the sight,
which is the goal's own phrasing, and the arms differ only in what is imagined.

| arm | what the night was made of | span residual | sound→vision | d | wins |
|---|---|---|---|---|---|
| `stored` | the real sight | 0.0000 (in span) | **+0.0336** | **2.49** | **6/6** |
| `imagined` | the concept's expected sight | 0.0000 (in span) | −0.0093 | −0.89 | 0/6 |
| **`factored`** | **form + a foreign concept's colour** | **0.4927 (outside)** | −0.0023 | −0.19 | 2/6 |
| `factored_own` | the same, colour from its own category | 0.4933 (outside) | +0.0000 | 0.00 | 2/6 |
| `confabulated` | a random sight | 0.9912 (outside) | −0.0069 | −0.42 | 2/6 |

**Only replaying the real sight helps.** Nothing self-generated does, in span or
out of it. And the accuracy table was never going to be the right question:
binding a yellow bus to the real sound of a bus asserts something **false**
about the world, so a night of recombinations should not raise overall accuracy,
and does not.

What a recombination *is* evidence for is that form survives a change of colour.
So the probe that matters exchanges R and B on held-out photographs — luminance
`(r+g+b)/3` is exactly invariant to that swap, so form is held fixed by
construction and only the thing the recombination varied is varied:

| arm | names it, colours swapped | fraction of naming kept | Δ | d |
|---|---|---|---|---|
| `no_dream` | 0.244 | 69.0% | — | — |
| `stored` | **0.267** | **77.0%** | **+0.0231** | +1.25 |
| `imagined` | 0.262 | 77.1% | +0.0174 | +1.03 |
| **`factored`** | 0.248 | **72.3%** | +0.0035 | +0.28 |
| `confabulated` | 0.250 | **72.5%** | +0.0058 | +1.11 |

**The factored night is indistinguishable from a random sight** — 72.3% against
72.5%. That is the same shape as `composition.py`'s verdict arrived at from the
other end: the mechanism's effect equals a structureless control's effect, so
whatever structure it has is not what is doing the work. And what *did* move
robustness is replaying the real sight (+0.0231): the gain came from
**sharpening** the concept, not from imagined variation.

#### Why, specifically — and it is not that the mechanism was ignored

The recombination recruits only ~4 new cells out of a pool with 145 free, so it
is not being rejected as novel and filed away. It **updates existing concepts**,
drifting each one's colour block toward the average of all of them. In this
world that is a loss rather than an invariance: colour here is *signal*, worth
**+0.171** to the read-out (`factored.py`: whole code 0.831, form block alone
0.660). Imagining a yellow bus is sound. Concluding "colour does not matter"
from it is not — and binding it as a fact is the only thing this architecture
knows how to do with an imagining.

That is the finding, and it relocates the gap precisely. Across three benchmarks:

- mixing stored codes **provably cannot** produce anything new (`composition.py`)
- crossing factors **provably can**, and does, coherently (`factored.py`)
- and the mind has **no operation that benefits from it** (`inner_world.py`)

The missing piece was never novelty, and it is no longer a generative model
either — that now exists. It is that every route from an imagining back into the
mind runs through `bind`, which treats what it is given as an observation. An
imagined thing needs to be *evaluated* rather than memorised: something that asks
whether a crossing is possible before the concepts absorb it. That is what a
causal model and a constraint on the result would be for, and it is now the
blocking problem rather than a wish — demonstrated by three arms that leave the
span and none that gains from having left it.

### The constraint, not the solver, is the weak part

The gap above says an imagined thing has to be *evaluated* rather than
memorised, so the next piece is the judgement: something that asks whether a
crossing is possible before the concepts absorb it. Built the project's way —
`FactorCompatibility`, a Hebbian outer product between the two factors
accumulated online from the pairs the world presented, no gradients, read as a
compatibility rather than a recall.

It barely works, and `benchmarks/constraint.py` measures why before the blame
lands on the rule. A compatibility over (form, colour) can only work if **form
constrains colour**, and the only route from one to the other is through the
category — so the most any such model could know is how much colour a category
determines. That is the ceiling, and reporting a score without it would make a
weak *world* look like a weak *model*.

| | AUC | sd | share of the available signal |
|---|---|---|---|
| **the ceiling** — how much colour a category determines | **0.545** | 0.027 | — |
| Hebbian, random projection rank 64 | 0.541 | 0.024 | **92%** |
| Hebbian, random projection rank 256 | 0.536 | 0.024 | 81% |
| Hebbian, projected on the mind's own concept cells | 0.528 | 0.045 | 62% |
| Hebbian, random projection rank 16 | 0.509 | 0.025 | 21% |

**The rule captures 92% of what is there to capture. There is almost nothing
there.** Form carries 0.045 of AUC above chance about colour, so a plausibility
model over these two factors has nothing to say — and it is right not to: in a
world of cars and buses of every colour, *a yellow bus is not implausible*.

(The concept-cell projection was the obvious improvement and it is **worse**
than random directions at rank 64 — 62% against 92%. The concept basis is tuned
for discriminating categories, not for spanning within-category variation, so
projecting onto it discards exactly the variation a compatibility needs.)

#### The finding, which is about the factorisation and not the rule

The eye handed form and colour over already separated, and they are close to
independent. That independence is not incidental to either result — **it is the
same property twice**:

- it is *why crossing them works*: the form block is untouched by a change of
  colour, so a recombination is coherent (form reads 0.660, identical to the
  concept mean)
- it is *why nothing objects to the crossing*: independent factors carry no
  information about each other, so no compatibility over them can say a
  crossing is wrong

**The same independence that makes factored recombination work makes it
unconstrainable.** A constraint solver needs factors that constrain each other,
and these were chosen — by the retina, for good reasons of its own — for
precisely the opposite property.

So the third component is not blocked on a rule that has yet to be invented; the
Hebbian one is adequate. It is blocked on a factorisation over which
plausibility is a real question. Form × colour is not one — but **sight × sound
is**, and it was already in the architecture, unexamined from this angle: `Wv`
and `Wa` are two factors of one concept, and whether a bark goes with a dog is a
question the world genuinely answers.

#### And the same rule, asked that question, gives the session's answer

| the same Hebbian outer product, over sight × sound | AUC |
|---|---|
| with the visual code the eye actually produces | 0.533 |
| **with a visual code that clusters by category** | **0.879** |

Nothing about the rule changes between those two rows. Only whether similar
things are similar in what it is handed. **The judgement machinery works — it
reaches 0.879 — and the eye's code is what fails it.**

That is the same upstream defect that has blocked every downstream result in
§7.8, where the eye names photographs at 0.134 against 0.559 for digits. It now
has a much wider consequence than "recognition is weak", because a
representation in which same-category things are not similar cannot support a
generative model, a constraint solver **or** a causal model, however each is
built. The three components are not three independent gaps. They are one gap,
seen three times.

This also settles what to do next, and it is not more imagination machinery.
Every mechanism built in this section works as specified: recombination leaves
the span, the compatibility captures ~92% of the constraint available to it,
replay sharpens what it replays. Each of them is then measured against a world
it cannot see clearly.

> **Corrected below.** This paragraph originally concluded that the eye is the
> whole project's critical path and the only thing whose resolution unblocks the
> rest. Running the same machinery in the modality where perception *works*
> shows that is not right — the payback failure survives a front end that
> clusters. The eye is a real limit and it is not this one.

**One candidate for that, tested and rejected.** The mind evidently *has* a
clustered representation somewhere — `sound → vision` works at 0.775 while raw
sight similarity sits at chance — and the obvious place to look is the
population code over its own concept cells, which is what cortex would read
rather than the 12288-d input. Measured (same-category vs different-category
similarity, 3 seeds): raw code 0.497 / 0.592 / 0.609, concept profile 0.561 /
0.547 / 0.612. A wash. Sharpening the profile with kWTA(8), which is the project's
own move everywhere else, is **worse than chance** — 0.362 / 0.382 / 0.433.

That last number is the informative one. Below chance means two photographs of
the same category activate *more disjoint* concept sets than two photographs of
different categories, which is what an exemplar-based layer does: the cells
partition a category's members, so a fresh member matches one sub-cluster and a
stored member of the same category usually belongs to another. It is the same
fact as the 59 singleton cells, seen from the read-out side. The clustering that
makes `sound → vision` work lives in the *averaging* inside `Wv`, not in any
code the layer can hand to something else.

### The same machinery in the ear, and what it corrects

The oracle above used class prototypes built from labels, so it shows the rule
is adequate rather than that a code good enough for it is reachable. The ear
settles that without supervision. Hearing works here — 1-NN 0.914 over six real
ESC-50 categories — and the belt already emits two factors,
`[spectral envelope | tonotopic profile]`, exactly as the eye emits form and
colour. `benchmarks/heard_world.py`: same rule, same code path, real field
recordings, nothing supervised.

| | the eye | **the ear** | labelled oracle |
|---|---|---|---|
| does the front end cluster? | 0.573 | **0.788** | — |
| can the two factors be judged? | 0.541 | **0.818** | 0.879 |

**AUC 0.818 unsupervised, essentially at the labelled oracle.** So the
judgement is buildable and the representation was genuinely what blocked it on
vision. That much of the previous section holds.

#### And then the positive control failed, which is the finding

With a front end that clusters, a judgement that discriminates, and
recombinations that leave the span (residual 0.4976 against 0.0000 for a stored
code), the night was run five ways — and `crossed_judged` versus `crossed_anti`
draws the *same* candidates and inverts only which is kept, so the gap between
them is selection and nothing else.

| arm | names it | Δ | d |
|---|---|---|---|
| `no_dream` | 0.620 | — | — |
| **`stored`** — replaying a **real** recording | 0.616 | **−0.0046** | −0.14 |
| `crossed` — unjudged | 0.597 | −0.0231 | −0.48 |
| `crossed_judged` — most plausible | 0.593 | −0.0278 | −0.44 |
| `crossed_anti` — least plausible | 0.602 | −0.0185 | −0.29 |
| `reverse` — **un**learn a random crossing | 0.569 | −0.0509 | −0.49 |
| `reverse_judged` — unlearn a judged-implausible one | 0.611 | −0.0093 | −0.11 |

The last two arms test the one mechanism the diagnosis actually calls for. Every
arm before them binds an imagining as **fact**, which is why they cannot help. An
implausible crossing is not a fact — it is a **negative example**, generated free
and in unlimited supply by `imagine_factored` and identified without labels by
the compatibility. `AssociationArea.unbind` is `bind`'s update with the sign
reversed, which is Crick & Mitchison's (1983) account of REM as reverse learning
and the negative phase that makes contrastive Hebbian learning and wake-sleep
work. It does not help either (−0.0093, d=−0.11) — and the rule has a geometric
no-op inside it that explains the shape of the failure. The anti-instar step
`w − lr(x − w)` is exactly zero when `w == x`: there is no direction in which to
push a unit vector away from itself. Vigilance recruits by copying, and a
crossing is built from one real code and one foreign one, so **the half taken
from the cell's own memory is frozen and only the foreign half is unlearned**
(measured on a crossing: the auditory half moved −0.284 → −0.397, the visual
half stayed at 1.0000). `reverse` is unlearning half of each crossing, which is
the likeliest reason it damages rather than helps, and it is a property of this
rule rather than of reverse learning as such.

It does show the judge doing something real, and this is the only place in the
run where an unsupervised judgement changes an outcome at all: unlearning a
**random** crossing costs −0.0509, unlearning a **judged-implausible** one costs
−0.0093. Paired, that is **+0.0417 (d=+0.51, 3/6)** — the right sign and the
right size, below the gate. A random crossing is often a perfectly valid pairing
and weakening it damages a real concept; the judge mostly avoids those. It
decides what is *safe* to unlearn without making unlearning pay.

**And the positive control fails.** Replaying a real recording does not help either,
so the route from a night into perception is inert here and the null for
selection is a null for the *channel*, not for judging. (This was caught by
running the 50-way task first, where every arm including `stored` lost; the
6-class world was chosen so replay would have something to sharpen, and it still
lost. A benchmark whose positive control fails cannot answer its own question,
and saying so is the result.)

#### What that corrects, and what it leaves

The previous section concluded the eye is the critical path and the only thing
whose resolution unblocks the rest. **That is too strong.** The payback failure
reproduces with a front end that clusters at 0.788 and a judgement at 0.818, so
it is not downstream of the eye. What all of it is downstream of is narrower and
more structural:

> the architecture has exactly **one** route from an imagining back into the
> mind — `bind` — and that route implements **consolidation**, not learning. It
> sharpens the average of what it is given. An imagining is therefore either
> absorbed as a false observation or it is noise, and in both cases the best it
> can do is leave the concepts where they were.

That is why `stored` helps in `inner_world.py` (+0.0336) and not here: there the
probe is cross-modal retrieval, which reads the averaged `Wv` weights that
consolidation sharpens; here it is direct recognition, which the waking day
already saturates. Replay only ever moved the numbers that averaging moves. It
was never learning, which §7's correction said in one place and this now says
generally.

So the standing conclusion, at the width the measurements support: **this project
can now imagine — provably, outside the span of memory, coherently — and can
judge what it imagines, unsupervised, at 0.818. What it cannot do is learn from
either.** A generative model and a constraint solver both exist and both are
wired into a consolidator. Seven arms across two modalities, with plasticity of
*both* signs, and none of them beats doing nothing.

The missing piece is not a better imagination, a better judge, or a better eye —
all three were built or bounded here. It is a plasticity rule driven by a
*prediction error* rather than by a co-occurrence. Note what reversing the sign
established: an error-driven rule is not merely "Hebbian with a minus", or
`reverse_judged` would have worked. What is absent is a **teaching signal
computed from the mind's own failure to predict**, which nothing in this
architecture currently produces — `bind` compares an input to a weight, never a
prediction to an outcome.

That is the same conclusion §7.6 reached from the other end (replay into
perception needs an external teaching signal), the same one step 15b asks for the
world model (a learning rule, not a replay rule), and now the same one the
imagination path arrives at. **Three independent routes, one missing mechanism.**

#### So it was built — and the reason it is inert is the session's real finding

`AssociationArea.bind_contrastive` is that rule: a positive phase on the real
pair, and a negative phase on **the layer's own completion of one modality from
the other**, sampled rather than averaged. When the completion matches what
arrived the two phases land on the same cell and cancel — a well-predicted pair
teaches nothing, which is the defining property instar lacks. That is
Contrastive Hebbian Learning / wake-sleep (Hinton et al. 1995), local and
gradient-free, and it is the version in which **imagination is the negative
phase**: the mind's own generative samples are what the rule learns against.
(It also explains why `unbind` on crossings was the wrong negative phase — a
random crossing is not what the model believes, so unlearning it teaches nothing
about the model.)

It does not help either: −0.0046, d=−0.14. And the diagnostic beside it says why
in one number. **The mean prediction error the rule ever saw was 0.054.** The
layer's completion is already 95% correct, so the negative phase almost never
fires and the rule degenerates into plain `bind` — 0.620 against 0.625 for plain
binding over two passes.

A layer that **memorises** always predicts its own experience correctly. Forcing
it to generalise is exactly what creates something to learn from:

| pairs per cell | prediction error | plain | contrastive |
|---|---|---|---|
| 1.03 — one cell per experience | **0.054** | 0.625 | 0.620 |
| 1.81 | 0.221 | 0.602 | 0.574 |
| 1.92 — forced to merge | **0.262** | 0.588 | 0.551 |

The error signal rises **5×** when the layer is made to generalise, so the
diagnosis is right. And accuracy falls when it does, for both rules — with 57
training clips over 6 classes an exemplar store is simply the better recogniser.

> **Corrected by `scale.py` below.** This paragraph originally concluded that
> the remaining limit is a *scale* problem rather than an architecture one.
> CIFAR-10 has 5,000 images per class where ESC-50 has 40, and at 30× the data
> the layer sits at **exactly 1.00 pairs per cell** — unchanged. The exemplar
> store is a property of the rule, not of how much was seen. And the error-driven
> rule does start paying at scale, which the auditory world was too small to
> show.

### One fact, seen from six angles

Every negative in this section is the same property of the concept layer:

| the finding | the fact underneath |
|---|---|
| 36.8% of imaginings are byte-identical memories | cells hold **single exemplars** |
| composition's novelty is arithmetic, not conceptual | it mixes memorised points |
| replay only ever sharpens, never learns | averaging is all there is to do |
| a night of recombinations = a night of random sights | `bind` can only treat them as observations |
| error-driven learning is inert (error 0.054) | a memoriser is never surprised |
| forcing generalisation creates error (0.262) but costs accuracy | too little data to generalise |

**The concept layer memorises instead of generalising, and that single fact
produces all six.** It is a better position than six separate defects, and it
names what to do rather than what to try: not a better imagination (built, and
it works), not a better judge (built, 0.818 unsupervised), not a better eye
(bounded, and it is not the blocker) — but a layer in which being wrong about
the next thing is possible at all. Surprise is the prerequisite for learning
from surprise.

### Imagination pays, once there is enough of a world to be wrong about

`benchmarks/scale.py` settles the last question and turns the section around.
Two candidate causes for the memoriser: **pressure** (vigilance recruits and
nothing forces a merge) and **scale** (57 clips over 6 classes is nothing to
generalise from). Only scale can be tested by adding data, and the ear cannot —
ESC-50 has 40 clips per class. CIFAR-10 has 5,000. Same layer, same rules, the
eye's own two factors as the two modalities, 100 → 3000 photographs, 5 seeds,
held out throughout.

| n_train | pairs per cell | prediction error | plain | contrastive | Δ |
|---|---|---|---|---|---|
| 100 | 1.00 | 0.079 | 0.185 | 0.184 | −0.0012 |
| 300 | 1.00 | 0.175 | 0.235 | 0.232 | −0.0024 |
| 1000 | 1.00 | 0.194 | 0.255 | 0.258 | +0.0024 |
| **3000** | **1.00** | 0.181 | 0.269 | **0.293** | **+0.0240** |

**Pairs per cell is 1.00 at every size.** Thirty times the data does not merge a
single extra experience, so the exemplar store is a property of the *rule* —
vigilance recruits whenever the match falls short, and nothing ever forces a
merge — not of how much has been seen. That is a fixable thing in a way that
"needs more data" is not, and it corrects the paragraph above.

**And the error-driven rule pays: +0.0240, d=+2.10, 5/5**, monotone in data
(−0.0012 → −0.0024 → +0.0024 → +0.0240). It is the first mechanism in this
section that beats its control, and the crossover is where prediction error
becomes real: at 100 photographs the layer's completion is right to within
0.079 and there is nothing to learn, so the negative phase is noise and the
rule loses. At 3000 the completions are wrong often enough to teach.

What that negative phase *is* matters for the goal. It is not a random sample
and not an arbitrary crossing — it is **the layer's own completion of one
modality from the other**, sampled from its concepts. So:

> what the mind imagines is what it learns against, and doing so improves what
> it does with real, held-out photographs.

That is the second half of the goal, at the size the evidence supports: not "the
inner world replaces the outer" but **the inner world is what the outer is
measured against, and measuring against it is what learning is.** The three
components asked for earlier are now built and each is measured — a generative
model that leaves the span of memory, a judge that scores crossings unsupervised
at 0.818, and a learning rule driven by the mind's own prediction failure that
gains 5/5 at d=2.10. What is still missing is the merge: a layer that generalises
would have more to be wrong about, and every number here says that is where the
next gain is.

### Phase 9.1 — forced consolidation, and what it costs

The single fact behind most of the negatives above is that the concept layer
**memorises**: 52% of its cells hold exactly one photograph, which is why 38.8%
of what it "imagines" is that photograph byte for byte at every temperature.
Vigilance says "if it is new, make a cell" and nothing ever says "these two were
the same thing". So: a merger that runs after the day and is *forced*.

**By rank, not by threshold — and that is not a detail.** Over 111 cells from
216 real pairs, the visual similarity between concept cells has median −0.009
and a 99th percentile of 0.206. A threshold of **0.85 merges zero pairs**; 0.75
merges zero; 0.1 admits 189 at once. No setting merges the right amount, because
the scale belongs to the code — distinct photographs are near-orthogonal in it —
not to how alike two concepts are. The *ordering* is fine: the top 100 pairs by
summed similarity are **82% same-category**. So the merger holds a rate, exactly
the correction the vigilance controller needed.

`benchmarks/merger.py`, 5 seeds, 360 real pairs:

| | cells | singletons | **verbatim** | novelty | sound→vision | purity |
|---|---|---|---|---|---|---|
| no merging | 111.6 | 53.2% | **0.388** | 0.196 | 0.831 | 1.000 |
| `keep=0.8` | 89.4 | 39.4% | 0.306 | 0.223 | 0.838 | 0.995 |
| **`keep=0.6`** | 67.2 | 31.6% | **0.175** | 0.290 | **0.839** | 0.963 |
| `keep=0.4` | 44.4 | 31.5% | 0.138 | 0.339 | 0.707 | 0.807 |
| `absorb` (every singleton) | 52.2 | **0.0%** | **0.015** | 0.320 | 0.771 | 0.927 |
| **`keep=0.6`+`absorb`** | 46.0 | **0.0%** | **0.010** | 0.346 | 0.799 | 0.917 |

**The prediction holds and then some**: singletons 53.2% → **0.0%**, verbatim
38.8% → **1.0%**, novelty 0.196 → 0.346. Panel 7 shows it — the same five
photographs, two verbatim memories before and none after.

**And it is not free, which the sweep is for.** Ranked merging alone buys
verbatim 0.175 at *no* cost (recall +0.008); driving verbatim to zero costs
0.032 of recall and 0.083 of purity. Compressing past `keep=0.4` collapses
recall (0.707 → 0.543 → 0.426). Which point is right depends on what the layer
is for, and that curve is the deliverable rather than any single setting.

> **A bug this found, worth stating because it is the class that keeps
> recurring.** The first run said merging *improved* recall (+0.026) while
> barely touching verbatim. Both were artefacts. `concept_from_vision` takes an
> `argmax` over **every** row, and my merge cleared a retired cell's `wins`
> while leaving its `Wv` intact — so the read-out kept returning merged-away
> cells still holding verbatim photographs, with `wins == 0`. The merger had
> worked and nothing downstream could see it. The pre-existing `consolidate`
> already handled this correctly (it re-randomises the retired row, with a
> comment saying why zeroing would be wrong); the two new methods skipped it.
> Fixed by a shared `_retire`, and the real numbers are above.

### Phase 9.2 — the prior ranks, and makes the imagining safer rather than more real

The proposal was a prior that censors meaningless crossings — "if form = bus,
yellow is likely". `benchmarks/prior.py` draws 8 candidate crossings per
imagining and keeps one by four rules, so *selection* is the only thing that
varies.

| selection rule | novelty | coherence | gap to real |
|---|---|---|---|
| unjudged (first draw) | 0.355 | 0.269 | 0.452 |
| **most plausible** | 0.330 | **0.321** | **0.469** |
| least plausible | 0.377 | 0.257 | 0.433 |
| random of the same draws | 0.356 | 0.261 | 0.453 |
| a real unseen photograph | 0.797 | 0.361 | 0.000 |

**The prior really does rank**: best-minus-worst on coherence is **+0.0639
(d=1.52, 5/5)**, and +0.0597 against keeping one of the *same* draws at random,
which is the control that separates choosing from merely drawing more.

**And censoring moves the output the wrong way.** The gap to real data gets
*worse*, 0.452 → 0.469 — a gap is a distance, so up is away. It buys coherence
by spending novelty (0.355 → 0.330), and real data needs both. Censoring makes
the imagining **safer, not more real**, which is what a prior over factors this
weakly coupled can do: AUC 0.524 over form × colour, 0.564 over sight × sound,
both near the 0.500 of ranking nothing. `constraint.py` already said that is the
world and not the rule.

### Phase 10.3 — seeking *disagreeing* evidence works, and the basis is why

*(Filed here beside Phase 9's arms because it shares their benchmark; it is part of the Phase 10 roadmap summarised in §9.5.)*

The rule: after each glance, keep the candidate whose response most
**challenges** what the mind currently believes rather than the one that most
agrees with it. Measured against an oracle that draws 6 candidates per glance
and keeps one **at random**, so the comparison is selection alone and not the
larger sample.

The first implementation scored disagreement over **class prototypes** and came
out negative everywhere. That was my substitution: the specification says *among
the concept cells*, and the difference is not cosmetic. A prototype is an
average over a whole category, so the drive across six of them is already
pooled, and a glance can move the concept-cell profile without moving that
summary at all — which is exactly the glances that distinguish members of one
category. Over the concept cells, 8 seeds, per-arm random streams:

| k | agree-greedy | **disagree-greedy** |
|---|---|---|
| 2 | −0.0008 (4/8) | −0.0046 (3/8) |
| 3 | −0.0021 (3/8) | **+0.0150**, d=+0.50 (6/8) |
| 5 | −0.0200, d=−1.12 (1/8) | **+0.0163, d=+0.89, 7/8** ✔ |

**The criterion is what matters, not whether one chooses at all.** At k=5,
keeping the glance that challenges the belief is worth +0.0163 while keeping the
one that confirms it is worth −0.0200 — a gap of **+0.036** between two rules
that draw the same candidates and differ only in which is kept.

And it **grows with the number of looks** (−0.0046, +0.0150, +0.0163 at
k = 2, 3, 5) while the agreement rule gets worse over the same range, which is
what a rule that resists settling on a fixed point should do. `oracle-diverse
k=5` at **0.1842** is the best arm measured anywhere in this benchmark — above
one glance (0.1450), random pooling (0.1571), and keeping one of the same six at
random (0.1679).

The standing caveat holds: every number here sits between 0.14 and 0.19 on a
task whose chance is 0.100, so these are real effects on a perceiver that is
barely working. What the result establishes is the *sign and the direction of
the trend*, not a usable eye.

> **Two errors of mine that hid this.** An earlier run reported agreement-greedy
> helping at k=2,3 and *reversing* at k=5, which I read as confirmation bias
> compounding — "the signature of the rule rather than noise". It was noise: all
> arms drew from one shared random stream, so adding an arm shifted every later
> arm's glance locations and the reversal moved from k=5 to k=3. Independent
> per-arm streams and 8 seeds instead of 4 make it absent. Then the verdict
> logic examined only the agreement arm, so it printed "no headroom for a cheap
> approximation" while the disagreement arm was passing its gate. A verdict that
> cannot see one of its own arms is not a verdict; it now gates both.



## 9.5 Real sensors, and Phase 10

> **Phase 10 in one place.** The sections below were written as the work
> happened and the later ones correct the earlier ones. The settled account:
>
> | sub-phase | verdict |
> |---|---|
> | **10.1** — concept feedback re-teaching V1 | **Refuted, not merely failed.** Its premise is false: V1 filters already respond to 271 of 360 photographs before any intervention, so the single-image binding it was designed to break does not exist here. Two rules were tried, one of them verified to carry its teacher; a *deliberately misleading* teacher scores as well as the correct one under both. |
> | **10.2** — the shift test | **Premise confirmed far harder than stated; numeric target not met, and the collapse repaired.** A 5-px shift takes concept-waking from 0.290 to **0.021**, below the 0.167 chance rate. The predicted 0.55–0.60 was never reached (best 0.264), but the collapse is repaired two independent ways: pooling chosen fixations (gap 0.267 → 0.017) and `relational_code` (gap → 0.005, single look). |
> | **10.3** — disagreement-seeking gaze | **Met**, once scored over the concept cells as specified rather than over class prototypes: +0.0163 at k=5, d=+0.89, 7/8, growing with the number of looks while agreement-seeking gets worse. |
>
> Two earlier sections below — "Phase 10.1 — concept feedback to V1 does not
> work, and the rule is why" and "Phase 10.1 finished, and the aperture
> hypothesis with it" — attribute the failure to the *rule's shape* (collinear
> updates in a retinotopic bank). That diagnosis is correct about the delta rule
> and **superseded as an explanation**: the reconstructive rule fixed the
> collinearity, provably carried its teacher, and still did nothing, because the
> target was never the problem. See "Phase 10.1's premise was false".
>
> The eye's ceiling has now survived **ten** controlled interventions — width,
> aperture, depth, whitening, resolution, capacity, two teaching rules, tying,
> fixation choice, second-order spatial statistics — with centred cluster AUC
> ranging over 0.540–0.605 against the ear's 0.788 (§9.9 shows 0.063 of that gap is the class set, not the sense). Everything that buys
> robustness costs discrimination; nothing raises discrimination.


### The mind on cameras that are showing something right now

Every measurement above ran on an archive. `neurobrain/sensing/live.py` points
the same machinery at live public traffic cameras, and reports honestly what
this host can reach:

| sensor | here | why |
|---|---|---|
| webcam | unavailable | no `/dev/video0` in this container |
| microphone | unavailable | no `/dev/snd` |
| **CCTV** | **LIVE** | 511ny.org, 24 of 25 ids returning real scenes |
| text | available | always |

The webcam and microphone paths are written and take the same route downstream —
on a laptop they work with no change to anything else — but they cannot be
*exercised* here and the report says so rather than substituting something.

**The placeholder guard is the part worth keeping.** Ten different camera ids
returned **byte-identical 15136-byte images** with identical statistics: HTTP
200, a valid image, a plausible mean and variance, no information. A status
check passes it and a size check passes it. Only hashing the content against
what *other* cameras returned catches it, and 1 of 25 ids is rejected that way.

Measured on 24 live cameras, no labels available:

| | cosine |
|---|---|
| the same pixels encoded twice (**the noise floor**) | 0.870 |
| the same camera, 20 s later | 0.867 |
| a different camera, same moment | 0.033 |
| **identity AUC** | **0.974** |

**The code separates places on live input it was never tuned for** — 0.867
against 0.033. But the temporal claim has to be stated with its floor: 0 of 24
cameras changed their pixels in 20 s (they refresh slower than the survey), and
same-camera-later sits *at* the encode-twice floor. So this shows the code is
stable and cameras are distinguishable — **not** that it tracks a place through
change. A longer gap is needed for that.

Forced consolidation groups 24 live scenes into 14 concepts, and the imagination
result reproduces on live data: **span residual 0.610 for a factored
recombination against 0.000 for sampling one concept.** The street it imagines —
the form of one camera's scene carrying another's colour — is not a linear
combination of anything it saw.

### Phase 10.1 — concept feedback to V1 does not work, and the rule is why

> **Superseded in part.** The rule-shape diagnosis below is right about the delta rule and is not why concept feedback fails — see *Phase 10.1's premise was false*.

`align_v1` moves each filter by `ΔWt[c] = lr · (target[c] − rate[c]) ·
patch[pos_c]` with the target supplied by the merged concept layer — no labels,
the mind's own opinion as teacher. `benchmarks/align.py`, 4 seeds:

| target | cluster AUC | Δ |
|---|---|---|
| no alignment | 0.577 | — |
| the correct concept mean | 0.565 | −0.0121 |
| the image's **own** rate | 0.562 | −0.0148 |
| a **deliberately wrong** concept | 0.566 | −0.0116 |

**A correct teacher, an uninformative teacher and a misleading teacher all give
the same answer.** No teaching signal is reaching the filters at all, and the
cause is the rule's shape rather than the teacher's quality:

> This bank is retinotopic — **8 cells share each of the 64 positions**, and
> every cell at a position is handed the same patch. So `ΔWt[c] = lr · e[c] ·
> patch[pos_c]` gives all 8 an update along **one shared direction**, differing
> only by a scalar. Verified directly: the update vectors for two cells at the
> same position are identical up to that scalar. A target code differs across
> those 8 cells; the update has one direction available and cannot.

What would be needed is an error that is a **vector over the patch** rather than
a scalar per cell — reconstruct the patch from the code and use the pixel-space
residual weighted by each cell's own activity, `ΔWt[c] = lr · r[c] · (patch −
Wtᵀr)`. That is the sparse-coding dictionary update, equally local and equally
gradient-free, and it gives each cell a different effective direction. Recorded
as the specific next thing to try.

### Phase 10.1 finished, and the aperture hypothesis with it

> **Superseded in part.** The rule-shape diagnosis below is right about the delta rule and is not why concept feedback fails — see *Phase 10.1's premise was false*.

The delta rule could not carry a teacher at all. The **reconstructive** rule can
-- verified before it was tested, by measuring that two different targets leave
the filters 0.341 apart rather than 0.030 -- and with it:

| rule | correct target | uninformative | misleading |
|---|---|---|---|
| delta (scalar error per cell) | −0.0121 | −0.0148 | −0.0116 |
| **reconstructive** (error is a vector over the patch) | **+0.0020** | **+0.0036** | **+0.0045** |

All three within 0.003, the *misleading* target nominally highest. **Concept
feedback does not teach this eye**, under a rule that demonstrably transmits its
teacher. Phase 10.1 is closed as a negative.

That is four independent attacks, and together they exhaust "adapt the filters
better": §7.8's eleven mechanisms, concept feedback under two rules, and gaze
selection in both directions. So I proposed the remaining easy explanation --
that the constraint is what the eye *receives*, since 32×32 has been its input
since it was built for MNIST and the live-camera panel shows a traffic scene
reduced to mush at that size. CIFAR cannot test it (natively 32×32, so more
resolution returns interpolation), but the live cameras are 240–576px native.

`benchmarks/aperture.py`, 24 live cameras, cell count **held fixed**:

| input | positions | separation | identity AUC |
|---|---|---|---|
| 32×32 | 169 | **0.818** | 0.972 |
| 48×48 | 441 | 0.831 | 0.970 |
| 64×64 | 841 | 0.825 | 0.970 |
| 96×96 | 2025 | **0.789** | 0.969 |
| *32×32 with 4× the cells* | 169 | 0.777 | 0.969 |

**The hypothesis is wrong.** Three times the linear resolution makes the code
slightly *worse* (−0.029), and capacity does not help either (−0.041 from four
times the cells). Neither adaptation, nor field of view, nor capacity moves this
representation.

> **What this test cannot show, stated with it.** Identity AUC is at ceiling —
> 0.003 of spread across a threefold change in resolution — so it could not have
> revealed an improvement had one existed; `separation` is the unsaturated part
> and it falls. And telling two *cameras* apart is an easier question than
> telling two *categories* apart, which is the one that matters downstream and
> needs labels the live world does not provide. This bounds the aperture
> hypothesis rather than closing it.

The convergent negative is now the finding: **five classes of intervention, none
of which moves the visual code.** I do not have a sixth worth proposing on
present evidence, and proposing one anyway is how the previous four got spent.
What none of them touched is that this architecture has no operation that makes
two different photographs of the same thing land near each other — `tie=True`
restores the pooling contract but the invariant codes it produces are weak
(0.233 against 0.773 for position-specific). That is a property of the
representation's construction rather than of any rule applied to it.

### The whole mind on live cameras — the payback reproduces

`payback.py`'s result was measured on archives. `benchmarks/live_mind.py` runs
the same protocol on **live traffic cameras** over a real time gap, with text as
the second modality — the camera's own name, honest metadata rather than a label
invented for the benchmark. Concepts bind on the early rounds; every number is
measured on the last, minutes later.

The first attempt used 25 camera ids and could not resolve the effect: one
camera *is* 0.042 of the score in a 24-place world, so the archived effect
(+0.0833) would have been two cameras. Scanning ids 380–519 found **104 distinct
live cameras** — the rest serve a no-signal card the placeholder guard rejects
by content — which puts the resolution at **0.0094** and makes the measurement
possible.

**106 cameras, 318 observations (3 per place), 450 s gap, all scenes changed**
(same camera later 0.712 against an encode-twice floor of 0.941). Chance 0.009.

| arm | name→place | place→name |
|---|---|---|
| `no_dream` | 0.755 | 0.915 |
| `stored` — replay the real sight | **+0.0189** d=+2.45 4/4 | **+0.0165** d=+3.50 4/4 |
| `imagined_as_fact` — believe it | +0.0024 d=+0.17 3/4 | **−0.0755 d=−4.90 0/4** |
| **`imagined_as_error`** — learn against it | **+0.0236 d=+4.33 4/4** | **+0.0165 d=+3.50 4/4** |
| `merged+error` | +0.0047 d=+0.50 3/4 | −0.0519 d=−1.77 0/4 |

> **A live re-run does not reproduce this.** Re-run with the corrected arm
> added (§9.7), `stored` and `imagined_as_error` came back **identical on every
> metric** — 0.631 / 0.9833 / 110.75 cells, the same deltas, the same win
> counts. The verdict printed "the archived result reproduces on live sensors",
> reading a tie with the positive control as a reproduction.
>
> My first reading of that identity was that the negative phase never fired.
> **It is wrong, and the counter says so**: the phase fires on **1.63%** of
> replays, about 5 of 300 per seed. It fires and changes nothing measurable —
> five perturbed cells out of ~111 never flip an argmax when one camera is
> 0.0095 of the score. The effect is real and smaller than this world can
> resolve, which is a different failure from the rule not running, and only the
> instrument distinguishes them.
>
> So the detector is not the fire rate. `live_mind.py` and `payback.py` now
> check whether the error arm differs from its own control **at all**, and void
> the payback question when it does not, whatever the counter says.
>
> The table above is **not** retracted: its two arms *differ*, which cannot
> happen unless the negative phase both fired and moved a read-out, so that run
> measured something this one did not. What is retracted is any claim that the
> live result has been reproduced.

**The archived result reproduces on live sensors**, and reproduces in the part
that matters most: *learning against what the mind imagines beats replaying what
it actually saw* — +0.0236 against +0.0189 — while **believing the identical
imaginings hurts** place recognition, 0 of 4 seeds. Same completions, same replay
order, same number of plasticity events; the only difference is whether the
fantasy is believed or learned against.

The effect is +2.5 cameras of 106, above the 1.5-camera floor the benchmark
imposes on itself, on both read-out directions and at 4 of 4 seeds each.

Forced consolidation does **not** add anything here (+0.0047), unlike on the
archive where it was superadditive. 106 places with 3 observations each has far
less within-category redundancy than 6 categories with ~36 members, and merging
is the operation that redundancy feeds — so this is the expected shape rather
than a contradiction.

### The bug this found, which was silently degrading every held-out visual result

`set_stats` floored the divisor at an absolute `std + 1e-6`. When a feature has
**no** variance across the training set — a V1 cell that never fired for any of
them — every held-out sample gets that feature multiplied by a million, and the
prepared code becomes nothing but dead dimensions.

Measured on 24 live cameras with 12288-d codes: the 1st percentile of `v_sd` sat
at exactly 1e-6, and **20 of 24 held-out frames woke an *uncommitted* concept
cell** — the read-out was answering with random rows.

| divisor floor | held-out frames waking an unbound cell | place recognition |
|---|---|---|
| `+1e-6` (as shipped) | **20 of 24** | 0.167 |
| 1% of the mean std | **4 of 24** | **0.833** |

Training samples are unaffected — their value in a zero-variance feature *is*
the mean, so the ratio is 0/floor either way — which is exactly why this never
appeared on a training-set probe and only surfaced when something genuinely new
was presented. It is fixed, and it is the reason the live figures above are
0.802 rather than 0.000.

### Phase 10.2 as specified — the premise is confirmed, the remedy is not

The test asked for is panel 1's: show a held-out photograph, and ask whether the
concept it wakes is one whose training majority is its own category. Baseline
35%, predicted 55–60% after alignment, with the object shifted 5 px.
`benchmarks/phase10_2.py` measures exactly that, with a 32 px object in a 48 px
frame so the shift keeps the object whole, applied at test time only, 4 seeds:

| arm | centred | +5 px | −5 px |
|---|---|---|---|
| no align | **0.290** | **0.021** | **0.023** |
| align to concept (delta rule) | 0.311 | 0.019 | 0.016 |
| reconstruct concept | 0.295 | 0.019 | 0.019 |
| reconstruct own *(uninformative)* | 0.286 | 0.026 | 0.024 |
| reconstruct wrong *(misleading)* | 0.309 | 0.026 | 0.026 |

**The premise is confirmed far more strongly than stated.** A 5-pixel shift does
not degrade the code, it *annihilates* it: 0.290 → 0.021, which is not merely
worse than the 35% baseline but well **below the 0.167 chance rate** — the
shifted code wakes concepts of systematically wrong categories. Five pixels.

**The remedy does not work.** Best arm across all three shifts is 0.120 against
the 0.55–0.60 predicted, and it is `reconstruct wrong` — the **deliberately
misleading** teacher. The correct concept gives 0.111. Whatever moved, the
concept layer did not move it.

> **Two setup errors of mine, one of which was hiding the real number.** The
> first version used a 40 px frame, which leaves only 4 px of margin, so both
> ±5 requests silently **clipped to ±4** — verified by measuring which columns
> the object occupies. That version reported 0.269 at +5 and 0.085 at −5, a 3×
> asymmetry I could not explain. With a proper 48 px frame the asymmetry
> vanishes (0.021 vs 0.023) and the true magnitude appears: the asymmetry *was*
> the clipping geometry, and ±4 was flattering ±5 by an order of magnitude. The
> benchmark now asserts that the frame can hold the requested shift rather than
> quietly running a different experiment.

That collapse is the most useful number Phase 10 produced. It says the eye's
problem is not a shortage of teaching signal and not a shortage of resolution —
both tested, both negative — but that **the code carries almost no information
that survives a five-pixel translation**, and everything built on top inherits
that. `translation.py` measured the same property indirectly (`tie=True` restores
the pooling contract but its invariant codes score 0.233 against 0.773 for
position-specific ones); this is the same fact at the level the concept layer
actually consumes.

### Phase 10.2 retried with 10.3's perception — the translation collapse is repaired

`phase10_2.py` gave the eye exactly one look, always at the frame centre, so a
shifted object was simply somewhere else. Phase 10.3 had meanwhile produced a
translation mechanism nobody had pointed at the translation failure.
`benchmarks/phase10_2b.py` measures 10.2's quantity with 10.3's perception —
*k* glances pooled, chosen at random or by disagreement over the concept cells.

| arm | centred | +5 px | −5 px | **shifted mean** |
|---|---|---|---|---|
| one centred look | 0.288 | 0.021 | 0.021 | **0.021** |
| 3 random glances | 0.254 | 0.248 | 0.241 | 0.245 |
| 3 disagreeing glances | 0.259 | 0.264 | 0.240 | 0.252 |
| **5 random glances** | 0.281 | 0.262 | 0.266 | **0.264** |
| 5 disagreeing glances | 0.264 | 0.245 | 0.252 | 0.248 |

**The collapse is repaired.** 0.021 → 0.264, a factor of 12.7, d = +22.1, 4 of 4
seeds — and back above the 0.167 chance rate that a single look falls *below*
when the object moves.

The sharper way to read it is the gap between centred and shifted, because that
is the translation sensitivity itself:

    one centred look     0.288 centred, 0.021 shifted   gap 0.267
    5 pooled glances     0.281 centred, 0.264 shifted   gap 0.017

**94% of the eye's position sensitivity is gone**, and the centred performance
is essentially unchanged (0.288 → 0.281) — so this is not the invariance-for-
selectivity trade that `translation.py` measured for pooled codes, where
invariance cost almost all the accuracy. Nothing was learned to achieve it: the
same V1, the same concept layer, only several fixations instead of one.

**And the choosing is not what does it here.** Best chosen 0.252 against best
random 0.264 — the extra looks carry the effect, not their selection. That does
not contradict `uncertain_look.py`, where disagreement-choosing passed its gate
at k=5; the two measure different things on different material (1-NN over 1200
training photographs with ±8 jitter, against concept-waking over 216 with ±5).
The honest statement is that **choosing helps where it was measured to help and
does not here**, and pooling helps in both.

**The numeric prediction is still not met**: 0.264 against the 0.55–0.60 asked
for. That target was about raising the eye's *ceiling*, and the ceiling has not
moved — centred accuracy is 0.288 whatever is done to it, which is the same wall
five other interventions hit. What has moved is that the eye no longer falls off
a cliff when the world shifts by five pixels, which was the failure Phase 10.2
was designed to expose.

### Phase 10.1's premise was false — V1 was never single-image bound

Every earlier test of Phase 10.1 measured a downstream quantity and came back
negative. But the phase's own step 5 is a claim about V1 itself — *this should
break V1's single-image binding* — and that had never been measured. A mechanism
can do exactly what it was designed to do and still not produce the predicted
benefit, and those are different failures.

`benchmarks/phase10_1b.py` measures it three ways, 4 seeds, 360 photographs:

| arm | top1 share | **participation** | lifetime kurtosis |
|---|---|---|---|
| **no align** | 0.026 | **271.0** | 10.5 |
| align to concept | 0.074 | 250.6 | 25.6 |
| reconstruct concept | 0.034 | 240.5 | 9.8 |
| reconstruct own *(uninformative)* | 0.047 | **180.4** | 12.2 |
| reconstruct wrong *(misleading)* | 0.025 | 242.5 | 7.2 |

**The eye does not have the problem Phase 10.1 was built to fix.** Before any
alignment, a filter already responds to **271 of 360 photographs** — 75% of
everything it is shown — and its single best photograph accounts for 2.6% of its
total response. That is a broadly-tuned filter bank, not a set of
single-photograph detectors.

And alignment pushes the wrong way. Participation *falls* under every arm — to
240 with the concept teacher, to 180 with the uninformative one — so the rule
makes filters **more** selective, not less. Lifetime kurtosis rises under the
delta rule (10.5 → 25.6), the same direction. Whatever alignment does, it is the
opposite of what step 5 asked for.

This retires Phase 10.1 completely and explains why every variant of it failed,
including the one that provably carried its teacher: **the remedy was aimed at a
diagnosis that does not hold.** `Wv` is not "stuck to raw pixels" in the sense of
being tied to individual images.

#### What the eye's problem actually is, now that this one is excluded

Phase 10.2 found it, and found it harder than it was stated: a **5-pixel shift**
takes concept-waking from 0.290 to **0.021**, below the 0.167 chance rate. The
filters are broadly tuned across *images* and catastrophically narrow across
*positions*. Those are different axes, and every intervention that worked on the
first axis was working on an axis that was never broken.

The fix follows from the correct diagnosis rather than from more learning:
pooling several fixations closes 94% of the position sensitivity (gap 0.267 →
0.017) with no change to V1 or the concept layer at all. **The eye needed to
move, not to be taught.**

### The last structural difference between the eye and the ear, measured

The eye's wall is cluster AUC ≈ 0.578 and centred concept-waking ≈ 0.29,
unmoved by eight classes of intervention. The ear reaches **0.788** on the same
kind of task, and `selforganize.py` had already recorded a structural difference
between them without anyone testing it as a *cause*: `AuditoryBelt` builds a
`WideV1` and **never develops it**, so its bank stays tied at 0.978, while the
eye's drifts to 0.454 per column after `develop_v1` — which is why
`pooling_index`, that groups cells by index on the assumption that index means
filter identity, works there and not here.

`benchmarks/ear_config.py` gives the eye each configuration on identical
photographs, 4 seeds:

| configuration | tie | cluster AUC | wakes +0 px | +5 px | −5 px |
|---|---|---|---|---|---|
| developed, untied *(the eye now)* | 0.661 | 0.578 | 0.286 | 0.029 | 0.031 |
| **developed, tied** | **1.000** | **0.594** | 0.288 | 0.052 | 0.049 |
| undeveloped, tied *(the ear's exact setup)* | 0.983 | 0.583 | 0.299 | 0.038 | 0.038 |

Tying passes the significance gate — +0.0164, d = +1.37, 4 of 4 — and **explains
7.8% of the gap**. The distance to the ear is 0.210; tying the bank recovers
0.016 of it. A reliable effect and a small one, which are different things.

And the ear's *exact* configuration (undeveloped, tied) scores 0.583, **below**
the developed-tied arm — so it is not the absence of development that helps
either. The benchmark's verdict now requires an effect to account for at least a
quarter of the gap it was proposed to explain, because "4 of 4 seeds" was about
to be reported as though it had explained the wall.

That closes the last structural difference this architecture makes available
between its working sense and its broken one. The eye's ceiling is not per-column
drift, not development, not width, aperture, depth, whitening, resolution,
capacity, concept feedback, or fixation choice. **Nine classes of intervention,
and the wall has moved by 0.016.**

What that leaves is not another parameter. Hearing and vision differ in
something this codebase does not expose as a setting: a cochleagram is already a
*time × frequency* map in which the same sound at a different moment occupies
different columns of the same rows, so the belt's pooling operates over a
genuine nuisance dimension. The eye's retinotopic grid has no equivalent — a
photograph's category is not a translation of anything — and the pooling that
works for the ear is being asked to do a different job. Testing that means
building a visual representation with an explicit nuisance dimension, which is a
change of architecture rather than of configuration, and it is where the
evidence points rather than where a benchmark could take it.

### The ear's trick done properly: invariance that keeps *relative* structure

`ear_config.py` closed the last configuration difference between the senses,
leaving a difference of kind. A cochleagram is a **time × frequency** map, so
pooling time discards a real nuisance dimension and keeps what identifies the
sound. The retinotopic grid has no such split — pooling position discards
everything spatial and returns a bag of features, which is why `translation.py`
measured fully pooled codes at 0.233 against 0.773.

`WideV1.relational_code` is the operation that actually corresponds:

    C[k, f, g] = Σ_p  A[f, p] · A[g, p + offset_k]

Summing over *p* is what makes it translation-invariant — shift the image and
every *p* shifts, leaving the sum alone — while the offset index keeps **how the
parts are arranged relative to each other**. Second-order statistics over
relative position; nothing learned, nothing differentiated.

| code | dim | cluster AUC | centred | shifted | **gap** |
|---|---|---|---|---|---|
| rate *(the eye now)* | 12288 | **0.605** | 0.290 | 0.036 | 0.254 |
| pooled *(`pooling_index`)* | 30 | 0.540 | 0.280 | 0.251 | 0.029 |
| **relational** | 2187 | 0.560 | 0.285 | **0.280** | **0.005** |

**What it delivers.** The best translation invariance in the project: a
centred-to-shifted gap of **0.005** against 0.254 for the standard code, and
shifted concept-waking **0.036 → 0.280, d = +7.0, 4 of 4**. It does that with a
*single look*, matching what pooling five chosen fixations achieved (0.280
against 0.264) at a fifth of the compute — and confirming from a second
direction that the 10.2 collapse is a property of the representation rather than
of the eye's sampling.

**What it does not.** Cluster AUC 0.560 sits below the position-specific code's
0.605. It recovers about **30%** of the structure plain pooling throws away
(0.540 → 0.560) and no more. So relative arrangement is *partly* recoverable
structure and not a replacement for absolute position — the invariance is real
and it is still bought at a price, a smaller one than before.

That is the tenth intervention, and the pattern across all of them is now hard
to miss: **everything that makes this eye robust costs it discrimination, and
nothing raises discrimination at all.** Centred cluster AUC has ranged over
0.540–0.605 across width, aperture, depth, whitening, resolution, capacity, two
teaching rules, tying, fixation choice and now second-order spatial statistics.
The ear sits at 0.788 on its own material. Whatever separates them is not on the
list of things this architecture lets one vary.

## 9.6 A second visual pathway, rejected — and what its ablations found

`LocalSpatialEye` is a new pathway rather than an eleventh repair: 8×8
overlapping patches, a feature bank per patch, and every local feature vector
placed by **where it sits in the object** rather than in the frame — the patch
grid itself laid out from the object's centroid, so translation cancels in the
*sampling* and not merely in the labelling.

Hypotheses and the rejection bar were fixed in `localspatial.py` before any
measurement: **H2 — cluster AUC must beat the best existing arm by ≥0.02 with
d ≥ 0.8 and ≥75% of seeds.** Invariance alone was declared not to count, because
pooling and `relational_code` both already achieve it and cluster at 0.540/0.560.

`benchmarks/pathways.py`, 4 seeds, 240 photographs, luminance for every arm,
`drive` for every arm (the project's own measurement: 1-NN 0.322 vs 0.323 for
`rate`, at 18× the cost — spiking pays on motion, and these are still images).

| pathway | dim | cluster AUC | Δ vs rate | self-sim | proto |
|---|---|---|---|---|---|
| rate *(current)* | 4096 | 0.571 | — | 0.293 | 0.297 |
| pooled | 10 | 0.557 | −0.014 (2/4) | 0.707 | 0.339 |
| relational | 729 | 0.575 | +0.004 (1/4) | 0.754 | **0.362** |
| **local-spatial** *(as specified)* | 1152 | 0.587 | +0.015 d=0.35 (3/4) | **0.931** | 0.305 |
| **ablate: no object frame** | 1152 | **0.629** | **+0.058 d=2.15 (4/4)** ✔ | 0.666 | 0.323 |
| **ablate: no resampling** | 1152 | **0.630** | **+0.058 d=2.18 (4/4)** ✔ | 0.701 | 0.333 |
| ablate: no position | 128 | 0.581 | +0.009 (3/4) | 0.912 | 0.279 |
| ablate: global bank | 1152 | 0.597 | +0.026 d=0.98 (4/4) ✔ | 0.385 | 0.318 |
| ablate: independent banks | 1152 | 0.532 | −0.040 (1/4) | **0.955** | 0.302 |

### The architecture as specified is rejected

**H1 not met**: self-similarity under ±5 px is 0.931 against a 0.95 bar. On a
synthetic object it reached 0.993 — real photographs fill the frame, so there is
no figure for the centroid to lock onto and the object-centred frame is
estimated from whole-image contrast.

**H2 not met**: +0.015 over `rate` (d=0.35, 3/4) against a +0.02 bar, and
+0.012 over the best rival. Requirement 11 says reject, and it is rejected.

**H3 is refuted, in the opposite direction to my prediction.** I expected the
object-centred frame to be where the gain came from. Removing it *raises*
clustering by 0.043 (0.587 → 0.630) while costing invariance (0.931 → 0.701).
The centroid is estimated per image from contrast, and that estimate is stable
when the *same* photograph moves and inconsistent *between* photographs — so it
cancels translation and adds between-image jitter, which is precisely the wrong
trade for clustering.

> **§9.8 measures this paragraph and corrects it.** Both halves of the mechanism
> are real (within-image error 0.04 px, between-image spread 1.48 px), but
> "jitter" is the wrong diagnosis: an error of *the same size* dealt to the
> wrong images costs **nothing** (0.619 against a perfect origin's 0.617). The
> cost comes from the error being a **function of image content**, not from its
> magnitude. Note also that every object here sits at the frame centre, so the
> frame has no translation to cancel and its error is pure cost; once position
> genuinely varies the same comparison reverses.

### What the ablations did find, which is the useful part

**H4 is confirmed, and it is the largest movement of this metric in the
project.** Local receptive fields with **frame-absolute** position score
**0.630 against `rate`'s 0.571 — +0.058, d = 2.18, 4 of 4 seeds**, and beat the
same binning over a *global* bank (0.597), which is the control that separates
locality from the position code. Ten previous interventions moved this number
over 0.540–0.605 in total; patching alone moves it 0.058 in one step.

That configuration is **not the architecture that was specified** — it is the
specification with its defining feature removed. Reporting it as a success of
the proposal would be exactly the substitution this project keeps catching, so
it is reported as what it is: a rejected hypothesis whose control found
something.

**Two dissociations worth keeping.** `relational` has the best downstream
prototype accuracy (0.362) while clustering worse than local-spatial (0.575 vs
0.587), and `independent banks` has the best invariance in the whole table
(0.955) with the worst clustering (0.532). Both are H5's concern made concrete:
downstream accuracy, invariance and representation quality move independently,
and an architecture judged on any one of them alone can be sold on a number that
means nothing about the other two.

---

## 9.7 What the pathways do once a mind is built on them

`pathways.py` judged the four visual pathways as *representations*. Requirements
4, 10d and 10e ask what happens when a **concept layer**, a **night** and an
**imagination** sit on top of each. `benchmarks/pathway_downstream.py`, 4 seeds,
240 CIFAR photographs paired with real ESC-50 recordings, hypotheses H6–H9 fixed
in the docstring before the run.

| pathway | dim | cluster AUC | 1-NN floor | cells | per pair | see→name | sound→vision |
|---|---|---|---|---|---|---|---|
| rate *(current)* | 4096 | 0.571 | 0.292 | 87.0 | 0.60 | 0.242 | **0.570** |
| relational | 729 | 0.575 | 0.292 | 86.5 | 0.60 | 0.273 | 0.359 |
| local-spatial *(as specified)* | 1152 | 0.587 | 0.255 | 89.2 | 0.62 | 0.297 | 0.448 |
| **local-absolute** | 1152 | **0.629** | **0.318** | 87.8 | 0.61 | **0.344** | 0.510 |

### H6 confirmed: cluster AUC is what the layer consumes

Naming accuracy orders the pathways **exactly** as cluster AUC does —
local-absolute > local-spatial > relational > rate — and *not* as invariance
does, nor as the 1-NN floor does (which puts `rate` second and `local-spatial`
last). So `pathways.py` rejected the architecture on the right quantity. Had the
order tracked invariance instead, the rejection would have been decided on a
number the rest of the mind does not read.

**But the layer is not a pass-through, and one mean nearly hid that.** Averaged
over the four pathways, concept-layer naming minus the 1-NN floor is −0.0000,
which reads as "the layer adds nothing". Split by pathway it is not a null at
all:

| pathway | layer − 1-NN | d | seeds |
|---|---|---|---|
| rate | **−0.0495** | −1.81 | 0/4 |
| relational | −0.0182 | −0.51 | 1/4 |
| local-spatial | **+0.0417** | +1.02 | 3/4 |
| local-absolute | +0.0260 | +0.68 | 2/4 |

The layer holds ~0.61 cells per training pair, so it merges roughly two pairs in
five. **Merging helps a representation that clusters and hurts one that does
not** — the same operation, opposite sign, decided entirely by the code beneath
it. Reported as one average it cancels to zero; that average would have been a
fabricated null, and the per-pathway split is the result.

### H7 partly holds — and the arm it is measured against had a bug in it

Building this benchmark turned up a confound in the project's headline replay
result. `imagine_from_sound` returns a vector in **prep_v** space.
`bind_contrastive` consumes it there and is correct. But `bind` preps whatever
it is handed, so `imagined_as_fact` — in `payback.py:112` and `live_mind.py:120`
as well as here — was binding a **twice-normalised** fantasy: cos **0.407** to
the one the model actually generated, waking a *different* concept cell 21 times
out of 36. The two arms differed in the learning rule **and** in whether the
imagining survived intact.

`imagined_as_fact_fixed` un-preps before binding, an exact inverse
(`prep_v(v_hat·sd + mu) == v_hat`, measured cos 1.000). Sound→vision deltas
against `no_dream`:

| pathway | `stored` *(control)* | believe | believe *(undistorted)* | learn against | against − undistorted |
|---|---|---|---|---|---|
| rate | +0.0052 | −0.0104 | −0.0391 | +0.0026 | **+0.0417** d=0.87, 2/4 |
| relational | **−0.0052** | −0.0130 | +0.0000 | −0.0026 | −0.0026 d=−0.05, 2/4 |
| local-spatial | +0.0208 | −0.0156 | −0.0104 | +0.0234 | **+0.0339** d=2.17, **4/4** |
| local-absolute | +0.0182 | +0.0000 | −0.0339 | +0.0156 | **+0.0495** d=0.83, 3/4 |

**The correction does not overturn the claim — it widens it.** On three of the
four pathways, believing the *undistorted* fantasy is **worse** than believing
the mangled one, so the bug had been flattering the arm the claim is made
against. Learning-against beats believing-undistorted by +0.0417, +0.0339 and
+0.0495 where it beat the buggy arm by +0.0130, +0.0391 and +0.0156.

**`payback.py` re-run with the corrected arm** (6 seeds, 360 archived pairs, the
run the original number came from) says the same thing and settles it for the
published table:

| arm | sound→vision | see→name | swapped |
|---|---|---|---|
| `stored` *(control)* | +0.0255 d=+0.91 5/6 | −0.0081 | +0.0243 d=+1.44 5/6 |
| `imagined_as_fact` *(as published)* | −0.0162 d=−0.66 2/6 | −0.0116 | −0.0012 |
| `imagined_as_fact_fixed` *(undistorted)* | **−0.0174 d=−1.11 1/6** | −0.0162 d=−0.77 0/6 | +0.0093 |
| `imagined_as_error` | **+0.0208** d=+0.87 5/6 | −0.0081 | +0.0185 d=+1.23 4/6 |
| `merged+error` | **+0.0833 d=+3.65 6/6** | −0.0139 | +0.0058 |

Believing the undistorted fantasy scores −0.0174 where believing the mangled one
scored −0.0162: the same conclusion, marginally stronger, and **+0.0833 for
`merged+error` is untouched**. The bug was real, it made two arms differ in more
than the rule they were meant to isolate, and fixing it changed nothing about
what the benchmark concluded. That is the outcome worth recording — a found bug
whose correction leaves the result standing is evidence *for* the result, and it
is only visible because the corrected arm was run rather than reasoned about.

#### How rarely the error rule actually fires, which nothing had measured

Instrumenting the arms turned up something larger than the space bug.
`bind_contrastive` unlearns **only** where the completion wakes a different cell
from the real pair — `if win_n != win_p` — and that is deliberate: a
well-predicted pair should teach nothing, which is the defining property of an
error-driven rule. Nothing had ever counted how often the condition is met.

On the archive it is **about 5 events of 400 per seed**, and counted separately
per arm it says something the aggregate did not:

| arm | negative phase fires on | events per seed |
|---|---|---|
| `imagined_as_error` | **1.21%** | ~5 of 400 |
| `merged+error` | **1.50%** | ~6 of 400 |

The layer holds roughly one cell per experience, so hearing a sound wakes the
cell that stored that pair and the completion is nearly always *right*. The
mechanism that produces the project's largest effect therefore fires about five
times a night. That does not invalidate the effect — every arm replays the same
400 pairs in the same order, and the controls are unaffected — but it means
**+0.0833 is the product of five plasticity events, not four hundred**, and the
rule is *rare* rather than weak.

**The per-arm split is the superadditivity's mechanism, measured directly.**
Forcing the layer to generalise raises the fire rate by a quarter, 1.21% →
1.50%: a merged layer has more completions to get *wrong*, so the error rule has
more occasions to run. That was the stated explanation for `merged` +0.0255 and
error +0.0208 alone summing to +0.0463 but reaching +0.0833 together, and it had
been an argument rather than an observation until the counter existed. The
effect is modest and in the predicted direction; it is offered as consistency,
not as a demonstration that the mechanism is *sufficient* — a 24% change in
occasions does not by itself account for an 80% change in outcome.

The counter is now permanent in `payback.py` and `live_mind.py`, and an **exact
zero is treated as void**: at zero the error arm performs the positive phase and
nothing else, so it *is* `stored` and any gap between them is zero by
construction. A first version of that guard voided at <2% and threw away this
finding as a defect; the threshold is now zero, with a loud warning below 5%,
because a small non-zero rate is the measurement rather than a fault.

The one pathway that flips is `relational`, and its **positive control is
negative**: replaying *real stored pairs* scores −0.0052 there, so the payback
channel is not open on that pathway and neither of its imagination arms means
anything. A first pass at the verdict gated on all four pathways regardless,
printed "the claim survives only against the arm with the space bug in it", and
was wrong — letting a pathway with a closed channel veto the result is
cherry-picking with the sign reversed. The gate now requires `stored > 0`.

So: **H7 holds on every pathway where the payback channel is open (3/3), and
holds by a larger margin once the fantasy is undistorted.** Learning against an
imagining is a property of the rule, not of the representation. What was
genuinely wrong is the *size* attributed to it, and the direction of that error
was the flattering one — the effect had been understated.

### H8 confirmed on every pathway, and the better code has less room to escape

Mixing whole codes cannot leave the span of what is stored; crossing factors
can. The argument is algebraic and never mentions the eye, so it should not care
which pathway it runs on. It does not:

| pathway | mixing | crossing |
|---|---|---|
| rate | **0.0000** | 0.5112 |
| relational | **0.0000** | 0.4396 |
| local-spatial | **0.0000** | 0.3827 |
| local-absolute | **0.0000** | 0.3362 |

Exactly zero for mixing on all four, to every digit numpy prints. On the local
pathways the crossed factor is **spatial and explicit** — the top row of
position bins from one concept, the middle and bottom rows from another, so the
imagining is the top of one object carried on the rest of another. That is a
part recombination rather than the colour swap `factored.py` had to use.

The crossing residual **falls monotonically as clustering rises** (0.511 → 0.336
from worst-clustering to best). A representation whose categories are better
separated leaves less of an imagined crossing outside the span it already
covers, which is the same trade this project keeps finding rather than a new
one: novelty and structure are bought from the same budget.

### H9 confirmed: the rejected architecture is not rescued downstream

Requirement 11's failure mode is an architecture that changes downstream
learning without improving the representation. `local-spatial` — the
specification, rejected on cluster AUC — names at **0.297** against
`local-absolute`'s **0.344** (+0.0469, d=1.79, **4/4 seeds**), the same
direction as the representation and the same direction as the rejection. It is
not rescued at the concept layer, not by any of the four night arms, and not by
the imagination. **The rejection stands on the representation and on everything
built above it.**

---

## 9.8 The limiting factor, measured — and it is not what §9.6 said

§9.6 rejected `LocalSpatialEye` and explained the rejection by argument:

> The centroid is estimated per image from contrast, and that estimate is
> stable when the *same* photograph moves and inconsistent *between*
> photographs — so it cancels translation and adds between-image jitter.

No number in the project tested that. `benchmarks/limiting_factor.py` does, with
an **oracle**: the object's true position, which the benchmark knows because it
placed the object itself. The oracle is an *instrument, not a proposal* —
nothing in the mind can supply it, and no arm using it is an architecture.

4 seeds, 240 photographs, a 64 px frame, each object displaced ±10 px on both
axes, one developed bank shared by every arm so they differ only in the origin.

| arm | origin | cluster AUC | self +5 px |
|---|---|---|---|
| `local-absolute` | none (frame) | 0.541 | 0.788 |
| `local-spatial` *(as built)* | estimated from contrast | 0.573 | 0.981 |
| `oracle+shuffled-error` | true, displaced 1.48 px at random | **0.619** | **0.996** |
| `oracle-centroid` | true | **0.617** | **0.996** |

### H10 holds: the origin estimate is the limiting factor

A perfect origin — same filters, same bank, same binning, same patches — is
worth **+0.0435 (d=1.50, 4/4)** over the estimated one, and takes invariance
from 0.981 to 0.996. (H13 below shows this is a **lower bound**: the oracle
knows where the *tile* was placed, not where the object sits inside it, and an
origin better than the oracle's exists.) With a perfect origin the object frame beats having no
frame by **+0.0759 (d=1.64, 4/4)**; as built it manages +0.0325 (d=0.64, 3/4).
**The object-centred idea is not what failed. Locating the object without a
label is.**

This does not contradict §9.6's measurement, and the two together are the
finding. The object frame *cost* 0.043 in `pathways.py` and a perfect origin
*gains* 0.043 here because those are the same estimator error in two
conditions: when every object sits at the frame centre there is nothing for the
frame to cancel and its jitter is pure cost; once position genuinely varies the
frame earns its keep and the estimator caps it.

### H11 holds as stated

| | |
|---|---|
| within-image error (one photo, shifted 5 px) | **0.04 px** |
| between-image spread (across one category) | **1.48 px** |
| bias against the true centre | 1.60 px |

The estimator tracks a moving photograph almost exactly and scatters ~1.5 px
between photographs. Both halves of §9.6's asserted mechanism, now measured.

### H12 refutes the obvious reading of H10, and this is the real conclusion

"The estimator is the limiting factor" invites one repair — make it *more
accurate*. H12 tests whether accuracy is what matters: take the estimator's own
error vectors and deal them to the **wrong images**. Magnitude and distribution
are preserved exactly; the link between an error and the picture that caused it
is destroyed.

**A 1.48 px error costs nothing when it is unrelated to the picture.**
`oracle+shuffled-error` scores 0.619 against the perfect oracle's 0.617 —
+0.0020, d=0.06, 2/4, indistinguishable — while the estimator's *own* 1.48 px
costs 0.044. Same magnitude, opposite consequence.

> The damage is not that the centroid is **imprecise**. It is that its error is
> a **function of image content**, so two photographs of one category are given
> systematically different origins *because they look different* — which is
> exactly the difference the category is supposed to survive.

That separates two requirements this pathway conflated:

* **invariance needs within-image consistency.** The shuffled arm keeps 0.996
  despite a wrong origin, because the same wrong origin is used for both views
  of one photograph, so translation still cancels.
* **clustering needs content-independence.** Only the true origin and
  content-independent noise reach ~0.618; a content-correlated error of the
  same size loses 0.044.

A contrast-weighted centroid cannot satisfy the second, because contrast *is*
image content — the statistic and the nuisance are the same quantity. **That is
the limiting factor, and it is a property of the estimator's definition rather
than of its tuning.** No amount of sharpening a content-derived origin fixes it;
the fix would have to be an origin that does not read the content, which is a
different architecture and is deliberately not proposed here.

### H13 — the dose–response, which survives the test but qualifies the oracle

H12 rested on a **single contrast**, and a single contrast is consistent with
causes it does not test. H13 is the experiment that could overturn it: scale the
estimator's error by α, with a **matched shuffled control at every dose**. If
content correlation is the cause, the correlated curve falls with α while the
uncorrelated one stays flat. Both falling together would put the cause back on
magnitude and withdraw the section above.

α=0 *is* `oracle-centroid`, α=1 correlated *is* `local-spatial`, α=1 shuffled
*is* `oracle+shuffled-error` — the endpoints are the arms already reported, not
recomputations of them.

| α | correlated | shuffled | correlated − shuffled |
|---|---|---|---|
| 0.00 | 0.617 | 0.617 | 0.0000 |
| 0.25 | **0.627** | 0.607 | **+0.0198** d=+2.31, 4/4 |
| 0.50 | 0.611 | 0.612 | −0.0008 |
| 0.75 | 0.584 | 0.612 | −0.0283 d=−0.54, 2/4 |
| 1.00 | 0.573 | 0.619 | −0.0454 d=−0.89, 1/4 |

**The control is flat at every dose** — 0.607 to 0.619 across the whole range,
rank correlation with dose +0.40. Uncorrelated displacement up to the full size
of the estimator's own error costs nothing at any magnitude. That is H12's core
claim and it is now measured at five points rather than one: **imprecision as
such is not the limiting factor.**

**The correlated curve falls, but not monotonically**, and the exception is the
interesting part. Rank correlation −0.90, dropping 0.0434 from α=0 to α=1, and
cleanly monotone above α=0.5 (0.611 → 0.584 → 0.573) against a control that
does not move (0.612 → 0.612 → 0.619). But at **α=0.25 the correlated error
beats a perfect origin** by +0.0106 (d=+0.82, 3/4) and beats its own matched
control by +0.0198 (d=+2.31, **4/4**). A quarter of the estimator's error is
better than none.

#### What that says about the oracle, which is a limitation of the instrument

The oracle returns the centre of the **32 px tile** the benchmark placed — not
the centre of the *object inside it*. A CIFAR subject is not centred in its own
frame, and the measured 1.60 px bias between the contrast centroid and the tile
centre is that offset. So a small dose of contrast-derived correction moves the
origin *toward the object* and away from the tile, which is why α=0.25 wins.

Two consequences, and both are corrections to how §9.8's earlier numbers should
be read:

* **"A perfect origin" means perfect knowledge of tile position, not of the
  object.** The oracle is not the ceiling — α=0.25 exceeds it — so the +0.0435
  attributed to the origin estimate is a **lower bound** on what a true
  object-locator would be worth, not an estimate of it.
* **The limiting factor is unchanged and better supported.** Origin quality
  dominates this pathway, uncorrelated error of the same size is free at every
  dose tested, and the damage tracks the correlated component specifically. What
  the α=0.25 point adds is that the relationship is not "less content in the
  origin is better" — a little content helps, and a lot destroys.

The conclusion therefore stands as stated: a contrast-weighted centroid cannot
serve as an object frame's origin, because **contrast is image content and the
statistic is the nuisance**. What H13 adds is that this is a matter of degree —
the failure is not that content enters the origin at all, but that at full
strength it dominates it. Per the loop's failure branch, no replacement
estimator is proposed here.

### What was wrong with this benchmark three times before it was right

Recorded because all three failures produced a confident verdict line, and none
was caught by reading the code:

1. **A vacuous oracle.** Every photograph placed at the frame centre makes the
   object's true centre *be* the frame centre, so the oracle grid and the
   frame-absolute grid are the same computation. It reported the oracle
   recovering **100.0%** of the gap, with all four seeds identical to three
   decimals in both arms. Caught on the exact-identity signature.
2. **A sign-flipping denominator.** The verdict divided by "what removing the
   object frame recovers" — positive when objects are centred, **negative** once
   they move. It printed −292% and "H10 is FALSIFIED" while the direct
   contrasts all pointed the other way.
3. **A frame too small for its own shift.** 32 px of object in a 48 px frame
   leaves 16 px of travel; ±8 px of jitter consumed it, so at dx=+8 the 5 px
   shift moved the object **0 px**. Within-image error read 1.66 px instead of
   0.04 px and H11 came back False. Second occurrence in this project —
   `phase10_2.py` was the first.

Every false verdict here came from a **gate**, not from the data. The benchmark
now asserts that the oracle origin varies, asserts that the shift is not
clipped, warns if two arms score identically, and reports direct contrasts
instead of ratios.

---

## 9.9 Time was not it either — and the target it was aimed at was mismeasured

§9.8 ended with the eye's limiting factor localised to an origin estimate whose
error is correlated with image content. The literature says that whole line of
attack was aimed at the wrong thing.
[Halvagal & Zenke, *Nat Neuro* 2023](https://www.nature.com/articles/s41593-023-01460-y)
report that Hebbian plasticity **alone fails to produce invariant object
representations** — this project's own measured result, reached independently —
and their remedy is not a better estimator but a **predictive term over time**.
Every image this project had ever trained a filter bank on was static and
independent.

So the machinery was built: objects that translate, **rotate and loom**
(`streams.warp_tile`, tested in `tests/test_warp.py`), sequences that follow one
object with no label read (`sensing/sequences.py`), and `develop_v1_temporal` —
the three LPL terms, all local, no gradients.

### The rule had to be repaired before it could be tested

The first version **collapsed completely**: effective dimension 0.00, all 36
filter pairs identical. Instrumenting the three terms said why rather than
leaving it to guesswork — the variance term ran at magnitude 1–3 against the
decorrelation term's 0.001, and its `1/(var+eps)` gain is positive feedback,
since shrinking variance raises the gain that shrinks it further. Putting every
term on a common scale (z-scored activity, **correlation** rather than
covariance) and estimating the statistics in a frozen warm-up pass is what makes
the rule run at all.

**H15 confirmed:** the decorrelation term is load-bearing. Ablate it and the
bank still collapses to effective dimension **0.00**.

### H14 is falsified — and it survives a power check

240 photographs, 4 seeds, the bar registered in `temporal_invariance.py` before
the run.

| arm | cluster AUC | self ±5 px | code PR |
|---|---|---|---|
| **static** *(the current rule)* | **0.591** | **0.350** | 23.2 |
| temporal | 0.585 | 0.253 | 33.6 |
| shuffled-time *(control)* | 0.580 | 0.279 | 25.9 |
| temporal: no-predictive | 0.581 | 0.253 | 35.4 |
| temporal: no-decorrelation | 0.577 | 0.088 | 67.7 |

* **temporal − static −0.0062** (d=−0.19, 1/4) — no gain
* **temporal − shuffled-time +0.0048** (d=+0.19, 2/4) — ordering buys nothing
* the registered bar was missed by −0.0454
* **not a collapse artifact** — the temporal code's participation ratio (33.6)
  *exceeds* the static rule's (23.2)
* the effect it most directly predicted **reversed**: shift-invariance fell from
  0.350 to 0.253. The term meant to buy invariance reduced it.

A null from an instrument too blunt to see the effect would be worthless, and
this project has produced three such verdicts. So the power was measured: the
predictive term receives a **2.4× different signal** on ordered against shuffled
sequences (mean `|z_t − z_{t−1}|` = 0.0333 against 0.0796). The distinction goes
into the rule and does not come out of the representation. **That is a
falsification, not a null.**

### H16: capacity is not the constraint either, and the gap reverses

`pathways.py`'s configuration gives a *tied* bank `4096 // 441 = 9` distinct
filters, where LPL in the paper is a deep network with thousands. Only
`n_cells` was varied; rule, data and evaluation held identical.

| filters | static | temporal | shuffled | temporal − shuffled |
|---|---|---|---|---|
| 9 | 0.586 | 0.583 | 0.573 | +0.0099 (d=+0.36, 2/3) |
| 27 | 0.583 | 0.582 | 0.590 | −0.0081 (d=−1.15, 0/3) |
| 55 | 0.579 | 0.580 | 0.593 | **−0.0135** (d=−2.87, 0/3) |

Six times the filters moves the representation **not at all** (static 0.586 →
0.579), and the temporal rule's gap against its own control goes the *wrong*
way, until at 55 filters the shuffled control beats it 0/3. Capacity was never
the binding constraint.

### The control that should have been run first

Thirteen interventions have now landed in 0.540–0.630. Two numbers that were
never measured put that band in its place:

| | cluster AUC |
|---|---|
| **raw pixels**, no V1 at all | **0.524** |
| the eye's best arm ever (`local-absolute`) | 0.630 |
| **the ear, on the 6 audiovisual classes** | **0.716** |
| the ear, on `heard_world.py`'s 6 easiest classes | 0.779 |

#### And the class set moves the ear more than the whole gap does

The correction above is the weak form. Drawing **eight random 6-class subsets**
of ESC-50 and measuring the ear on each, holding metric, split and code path
fixed:

| | cluster AUC |
|---|---|
| 8 random 6-class draws | **0.695 – 0.836**, mean 0.757 |
| `heard_world.py`'s chosen 6 | 0.779 |
| the 6 audiovisual classes | 0.716 |
| the eye's best arm | 0.630 |

**The spread caused by class choice alone is 0.141 — larger than the entire
matched eye-ear gap of 0.086.** Which six categories you happen to test on moves
the ear more than the difference between the two senses does. The comparison
that motivated thirteen interventions is dominated by task selection, and the
audiovisual set the eye is judged on sits *below* the random-draw mean.

This does not mean the eye is fine. It means "the eye is far behind the ear" was
never a measurement of the eye: it was a measurement of two different tasks, and
the honest statement is that on the one task where both are measured identically
the ear leads by 0.086, which is inside the range that class choice alone spans.

**The eye is not doing nothing:** it adds **+0.106** over the raw pixels. And
the project's headline framing — *"the eye at 0.63 against the ear's 0.788"*,
which motivated eleven interventions — is **partly a cross-condition
comparison**. `heard_world.py` selects the six ESC-50 classes *with the most
clips*; the eye is judged on the six *audiovisual* classes, three of which are
animal vocalisations. Holding the metric, the split and the code path fixed and
changing only the class set moves the ear by **0.063**. The matched gap is
**0.086, not 0.158** — the ear still leads, but by a little over half of what
was claimed.

> A correction to this section's own first draft, which is the reason it is
> stated at all: the pixel control initially put the ear at **0.623** and I very
> nearly reported that the eye now *exceeds* it. That run omitted
> `PopulationAdaptation`, which `heard_world.py` documents as not optional —
> without it, belt codes sit at same-class cosine 0.986 against different-class
> 0.976. The corrected figure is 0.716. A measurement that flatters your current
> hypothesis is exactly the one to re-run before publishing it.

### What is now excluded, and what is not

Three classes of explanation for the eye's ceiling have been tested and
excluded: **geometry** (§9.8 — the origin estimate, content-correlated),
**time** (H14, with power), and **capacity** (H16, six-fold). The one variable
never changed across all thirteen interventions is **the input itself** — 32-px
luminance tiles. The ear receives a time × frequency map in which pooling over
time discards a genuine nuisance dimension and keeps identity; 32 × 32 pixels
offer no such factorisation, which is the argument `WideV1.relational_code`
already makes and which nothing since has resolved.

That is where the evidence points, and it is named rather than acted on: per
this project's own discipline, a falsified hypothesis is not replaced with the
next architectural idea in the same breath.

---

## 9.10 The instrument, not the architecture — the project's largest win does not replicate

§9.9 found that choosing which six ESC-50 classes to test on moves the ear by
**0.141**, more than the entire matched eye-ear gap of 0.086. That raised an
obvious question about the *eye*, and the answer invalidates this project's
headline architectural result.

`pathways.py` reported **local-absolute over rate: +0.058, d = 2.18, 4 of 4
seeds**, written up in §9.6 as "the largest movement of this metric in the
project". Re-run across **six random 6-class CIFAR subsets** (2 seeds each,
everything else identical):

| classes | rate | local-absolute | delta |
|---|---|---|---|
| 0,1,3,5,8,9 | 0.555 | 0.532 | **−0.0226** |
| 0,1,4,5,8,9 | 0.540 | 0.530 | **−0.0104** |
| 0,2,3,6,8,9 | 0.519 | 0.540 | +0.0208 |
| 0,1,2,4,6,8 | 0.561 | 0.604 | +0.0429 |
| 1,3,5,6,7,9 | 0.542 | 0.500 | **−0.0417** |
| 1,2,3,5,8,9 | 0.535 | 0.561 | +0.0259 |

**Mean +0.0025, range −0.042 to +0.043, positive on 3 of 6.** The effect is
approximately zero and its *sign* depends on which categories are tested. The
published +0.058 was one draw from a distribution centred on nothing.

### Why the original looked so solid, and why that is a design error

The four seeds varied the train/test split and the network initialisation. They
did **not** vary the categories. So "4 of 4 seeds, d = 2.18" measured how
*reproducible* the result was within one task, and reproducibility within a task
says nothing about whether the effect exists across tasks. Four correlated
samples from a single condition were reported as four independent confirmations.
That is pseudo-replication, and it is not specific to this one result:

> **Every one of the thirteen interventions in this project used the same six
> audiovisual categories and varied only seeds.** The measured spread from class
> choice alone is 0.104 for the eye and 0.141 for the ear, while the effects
> being accepted and rejected were 0.015–0.058.

The instrument's resolution was never established, and it is coarser than most
of what was measured with it. This does not automatically reverse the thirteen
conclusions — a *null* result is only reinforced by extra noise, so "intervention
X changed nothing" mostly survives. What does not survive is the positive
claims, and the largest of them is now measured at zero.

### What this means for the eye

The eye's real position, stated with what is actually known:

* it adds **+0.106** over raw pixels (0.524 → 0.630), which is a genuine effect
* the matched gap to the ear is **0.086**, inside the range class choice spans
* **no architectural variant tested has a demonstrated advantage over any other**
  once class set is treated as a random factor rather than a constant

Thirteen interventions were run against a baseline whose measurement error was
never quantified. The finding is not that the eye is fine, nor that the
architectures were bad — it is that this project could not have told the
difference, and did not know it.

---

## 9.11 What is actually wrong with the eye: it cannot turn seeing into learning

Every visual number in this project was compared to raw pixels, to the ear, or
to another variant of itself. `benchmarks/vision_reference.py` supplies the
reference it never had — a gradient-trained CNN on the *same* 6 categories, the
same 32-px grayscale frames, the same `cluster_auc`, the same split.

| arm | cluster AUC | 6-way accuracy |
|---|---|---|
| raw pixels | 0.521 | 0.271 *(probe)* |
| **the eye, this project** | **0.589** | 0.316 *(probe)* |
| CNN, backprop, **the same 240 images** | 0.575 | 0.312 |
| CNN, backprop, **12 000 images** | **0.698** | **0.585** |

**On the same data, gradients buy nothing.** −0.0134 for the CNN, comfortably
inside the noise §9.10 established.

> **§9.14 corrects the framing of this section.** The CNN figures here are its
> *final* layer, three blocks deep, against a single-stage V1. At matched depth
> the front ends are comparable; the gap is the missing hierarchy, not the rule. A loss function, labels and backpropagation
produce no better a representation than a local Hebbian rule when both see 240
photographs. The learning rule was never the eye's problem.

**The entire CNN advantage is data**: +0.110 cluster AUC and +0.273 accuracy
from 50× the images. So the obvious question, which thirteen interventions never
asked — does *V1's* rule improve with data?

| `develop_v1` trained on | cluster AUC |
|---|---|
| 240 images | **0.595** |
| 2 400 images | 0.561 |
| 12 000 images | 0.582 |

**Flat.** Fifty times the data moves the eye's representation not at all, while
moving the CNN's by +0.123 over the same range.

### The diagnosis, and it is one thing rather than thirteen

`scale.py` already measured this for the concept layer: pairs-per-cell stays at
**1.00** at every data size from 100 to 3000, so the layer stores one cell per
experience no matter how much it sees. The same is now measured one level down.

> **Neither layer of this visual stack can convert additional experience into a
> better representation.** The eye does not see badly — it sees about as well as
> a CNN given the same tiny budget. What it cannot do is *learn more from seeing
> more*.

That is a single coherent account of why thirteen interventions all landed in
0.540–0.630: every one of them varied the *structure* of a rule family that is
insensitive to data volume, and evaluated it at a budget where the insensitivity
does not show. Geometry (§9.8), time (§9.9), capacity (§9.9) and now the
learning rule itself have all been excluded as the differentiator, because in
this regime **nothing differentiates** — the band is what a 240-image budget
buys, whatever rule is applied to it.

### And the read-out is not hiding anything

A supervised linear probe on the eye's own code reaches **0.316**, against the
unsupervised read-out's 0.271 and raw pixels' 0.271. The gap is +0.045. So the
category information is *not* sitting in the code unreachable by the concept
layer — the code genuinely carries about that much and no more. The CNN at 12k
carries 0.585.

The honest summary is neither "the eye is fine" nor "the architecture is wrong":
it is that **this project has been measuring a data-starved regime and
attributing the result to architecture.**

---

## 9.12 What the literature says about data-efficiency, and a pilot that refutes the obvious fix

§9.11 measured the eye's real defect: **it cannot convert more experience into a
better representation** (240 → 12 000 images moves cluster AUC 0.595 → 0.582,
while a CNN over the same range gains +0.123). The literature has a direct
answer to why, and a demonstration that local rules need not have this property.

**Local rules can scale — the ceiling is not Hebbian learning.**
[SoftHebb (Journé et al., ICLR 2023)](https://arxiv.org/abs/2209.11883) reaches
**80.3% on CIFAR-10** with five hidden layers, no backpropagation, no feedback,
no target and no error signal. That is far above anything this project has
produced and it is achieved under the same constraint this project holds.

**The theory of why.**
[Moraitis et al., *Neuromorph. Comput. Eng.* 2022](https://arxiv.org/abs/2107.05747)
prove that a **soft** winner-take-all network under Hebbian plasticity maintains
a *Bayesian generative model* of its input and minimises cross-entropy to the
data distribution **without any loss function**. A density estimator keeps
refining as data grows. A hard winner-take-all rule is **vector quantization**:
k centroids that converge and then stop. `develop_v1` is the latter —
`argmax` plus a Kohonen neighbourhood — which is a textbook self-organising map.

That made a clean hypothesis: the flatness is caused by hard WTA.

### The pilot refutes it

Replacing the winner rule alone — softmax over the column at temperature 0.25 in
place of `argmax` + neighbourhood, everything else identical
(`benchmarks/soft_wta_pilot.py`):

| rule | 240 img | 2 400 img | 12 000 img | slope |
|---|---|---|---|---|
| hard-WTA *(current)* | 0.5951 | 0.5614 | 0.5817 | −0.0134 |
| soft-WTA | 0.5851 | 0.5661 | 0.5145 | **−0.0706** |

Soft-WTA is not merely flat, it is **worse** with more data. So the single
component that the theory identifies as the difference does not, on its own,
produce the property it is credited with.

**The lesson is about method, not about SoftHebb.** SoftHebb is a *system*: soft
WTA plus an anti-Hebbian term for non-winners, plus a specific weight
normalisation, plus adaptive per-neuron learning rates, plus depth, plus a
particular convolutional widening. Extracting one mechanism and expecting its
headline property to come with it is the same error as citing a number measured
under different conditions — and this project has now made both. Any adoption of
this line has to be **faithful and whole, or not attempted**.

---

## 9.13 Is the benchmark wrong? Partly — and it does not rescue the system

A fair challenge: the papers are established results, so if they do not transfer
here, perhaps the measurement is at fault and the system is fine. Both halves
were tested.

### The metric does have a real defect, measured

`cluster_auc` draws **one** same-class and **one** different-class neighbour per
query and then takes the cross product, so query *k*'s same-score is compared
against query *j*'s different-score. Against a proper estimator — every query
scored against all same-class and all different-class items, per-query AUCs
averaged, no sampling and no cross-query comparison — over 6 seeds:

| code | current (sampled) | correct (full pairs) |
|---|---|---|
| raw pixels | 0.5377 ± 0.0238 | **0.5315 ± 0.0042** |
| the eye | 0.5939 ± 0.0222 | **0.5633 ± 0.0069** |

* **3–6× too noisy.** Every null result measured with it had less power than
  claimed, so "eleven interventions moved nothing" is weaker evidence than it
  reads.
* **Biased, and unevenly.** It reads +0.031 high on the eye's structured code
  and only +0.006 on raw pixels, so it flatters structure and **inflates the gap
  between a good code and a bad one**. The eye's advantage over raw pixels is
  +0.032, not the +0.056 previously reported.

`cluster_auc_full` is now in `pathways.py` and is the estimator for new work;
the old one is kept because every figure before this section was measured with
it.

### But the system is not vindicated by it

The defect inflates differences — it does not manufacture the *ordering*, and
the central conclusion of §9.11 never depended on this metric at all. Linear
probe accuracy is an independent measurement with no cosine ranking in it:

| | probe accuracy (6-way, chance 0.167) |
|---|---|
| raw pixels | 0.271 |
| the eye | 0.316 |
| CNN, same 240 images | 0.312 |
| CNN, 12 000 images | **0.585** |

The eye and a CNN tie on equal data, and the CNN nearly doubles with more data
while the eye stays flat. That is measured without `cluster_auc` and it is the
finding.

### Why the papers do not transfer, concretely

Three reasons, and none of them is "the papers are wrong":

1. **The comparison was never like-for-like.** SoftHebb's 80.3% is unsupervised
   features **plus a supervised linear classifier**. This project's headline
   metric has no learning in it at all. The number that compares to SoftHebb is
   our *probe* — 0.316 on 6 classes — not our cluster AUC.
2. **Scale.** SoftHebb uses five hidden layers on 50 000 images. This project
   ran one layer on 240–12 000, and §9.11 shows one layer is flat in data
   regardless of rule.
3. **Systems do not decompose.** §9.12's pilot took the single mechanism the
   theory names — soft WTA — and it made data-scaling *worse* (−0.071 against
   −0.013). The published property belongs to the whole system, not to the part
   that carries the explanation.

So: the benchmark was partly wrong, the system is not fine, and the reason the
literature does not reproduce here is that neither the measurement nor the
implementation was ever matched to it.

---

## 9.14 V1 was never the weak part — the missing stages were

A reader's objection, and it is correct: the visual system is V1 → V2 → V4 →
IT, object identity lives at the far end of that, and V1 codes oriented edges.
Every visual number in this project is `cluster_auc` on **V1's output**, asked
to separate *categories* — which is not the question V1 answers.

Worse, §9.11 compared that V1 code against a CNN's **final** layer, three
convolutional blocks deep, and attributed the resulting 0.11 gap to the
learning rule. Those are not the same depth. `benchmarks/depth_matched.py`
reads the **same trained network** at three taps — not three networks, which
would confound depth with capacity and training budget:

| stage | cluster AUC | probe accuracy |
|---|---|---|
| **this project's V1** | **0.565** | 0.316 |
| CNN conv1 *(the honest V1 analogue)* | 0.549 | 0.399 |
| CNN conv2 | 0.613 | 0.514 |
| CNN conv3 *(what §9.11 compared against)* | **0.706** | **0.580** |

**At matched depth the two front ends are comparable**, and the two metrics
disagree about which is ahead: this project's V1 is **+0.016 on cluster AUC**
and **−0.083 on the probe**. The disagreement is worth keeping rather than
resolving — V1's code has the better cosine-neighbourhood structure, conv1's is
the more linearly separable, and neither difference is large.

**Depth inside the CNN is worth +0.157 cluster AUC and +0.181 probe accuracy** —
roughly ten times the matched-depth difference on one metric and twice it on the
other.

> §9.11's conclusion that "the learning rule was never the eye's problem" stands
> and is strengthened. What does not stand is the implication that the 0.11 gap
> measured the rule at all. It measured **three stages of hierarchy this project
> does not have.**

### Which makes one old negative result the central open question

`second_stage.py` (§7) built a V2 and it *lost* a third of the accuracy, with a
concatenated arm — which can only fail by V2 adding nothing — landing at −0.014.
That was read as "depth does not help here". It cannot bear that reading:

* the V2 used the **same competitive rule** §9.11 later measured to be
  insensitive to data;
* it was tested at **one small data budget**;
* and depth in a gradient-trained network of the same shape is now measured to
  be worth **+0.157**, the largest single effect in any visual measurement this
  project has made.

So the question is not *does depth help* — in a comparable architecture it
plainly does — but **why this project's V2 destroyed information where the
CNN's second block adds it.** That has never been separated from the rule's
data-insensitivity, and the two are the same suspect.

The project's eye has been V1 → concepts throughout: one cortical stage standing
in for four, with the association layer asked to make up the difference. Thirteen
interventions varied that one stage.

---

## 9.15 The four-stage eye existed all along — measured, it stops helping after V2

§9.14 concluded the missing hierarchy was the eye's real gap. Looking for where
to add stages turned up something worse: **`neurobrain/vision/ventral.py`
already contains a complete spiking ventral stream** — retina → V1 (edges) → V1
complex (phase-invariant) → V2 (corners, junctions) → pool → V4
(curvature/shape) — every area convolutional and retinotopic, every filter
learned by competitive Hebbian plasticity, plus a `TraceInvarianceLayer` that
learns transformation invariance from objects moving in time.

It appears in **no benchmark** and **zero times in EVALUATION.md**. Sixty
kilobytes of four-stage visual cortex, built and never measured, while thirteen
interventions were run on the one-stage `WideV1`.

`benchmarks/ventral_audit.py` measures every area on the same photographs every
other visual number used.

| stage | dim | cluster AUC | probe | participation | live units | spikes |
|---|---|---|---|---|---|---|
| `WideV1` *(what every number used)* | 4096 | 0.565 | 0.316 | 27.2 | 0.64 | — |
| V1 complex | 968 | 0.562 | 0.354 | 20.8 | 0.75 | 254 |
| **V2** | 1764 | **0.576** | **0.358** | 66.8 | 0.71 | 588 |
| pool | 324 | 0.570 | 0.288 | 35.0 | 0.89 | 214 |
| **V4** | 44 | **0.526** | **0.246** | **7.9** | 0.66 | **12** |

> ### CORRECTION — this section's diagnosis of V4 was wrong
>
> The "12 spikes" below is not near-silence. `SpikingConvLayer.forward` runs 12
> timesteps and lets **exactly one map win per location**, so a layer's spike
> ceiling is `locations × 12`. And the canvas was not what this section claims:
> the upscaling was `np.kron(im, np.ones((56 // 32, 56 // 32)))`, and
> **`56 // 32 == 1`** — a no-op. Every run here was at **32 px**, not 56.
>
> At 32 px this hierarchy runs out of space: V1 (k=11) → 22, pool → 11, V2
> (k=5) → 7, pool → 3, V4 (k=3) → **1**. V4 is `(44, 1, 1)`, a **single
> location**, ceiling `1 × 12 = 12`. So V4 was at **100% saturation**, not
> nearly silent, and the "tuned to synthetic curvature, photographs don't
> contain it" explanation below is unsupported.
>
> Measured ceilings by canvas: 32 px → 12, 56 px → 588, 96 px → 3 468, 128 px →
> 7 500. V2 and V4 sit at **100% of ceiling at every size**, which is its own
> problem and a different one.
>
> `upscale()` and `stage_extents()` in `neurobrain/vision/ventral.py` now do
> the resize correctly and report each area's ceiling beside its count. The
> tables in §9.15 and §9.16 are 32-px results and are kept as such; the real
> lesson is the one the goal already named — **this architecture needs real
> dimensions to function at all**, and at 32 px its last stage has one pixel.

**No area is dead** — 0.66 to 0.89 of units fire somewhere across the set, so
every stage does carry signal and a loop built on any of them would have
something to work with. That was the thing worth checking before trusting any
of it.

**The four-stage eye does beat the one-stage eye, modestly.** V2's probe is
0.358 against `WideV1`'s 0.316, **+0.042** — with 1 764 units against 4 096.

**But the hierarchy is not monotone: only 1 of 3 steps improves.** It rises to
V2 and then falls, and V4 — the stage nearest to object identity, the one the
whole architecture is pointed at — is the worst thing in the table. Its cluster
AUC of 0.526 is **below raw pixels' 0.532**, its participation ratio is 7.9, and
real photographs raise an average of **12 spikes** in it.

### The stream is not broken; it is out of domain

Its own self-test, printed while building, is excellent:

> learned view-invariance 34%, discriminability +0.34 · **shape recognition 99%**
> (5 classes, chance 20%) · **complex-object recognition 92%** (12 classes,
> chance 8%) · **IT cell purity 87%**

So the architecture works on what it was built for. The catch is what it was
built *on*: each area is developed on its own hand-made stimulus set —
`oriented_edges` for V1, `corner_images` + `curve_images` for V2,
`curve_images` + `shape_images` for V4 — line drawings, not photographs. V4's
filters are tuned to synthetic curvature, and a CIFAR photograph does not
contain it. Twelve spikes is what that looks like from the outside.

> The eye's stages are not too few and not badly designed. **The later ones were
> never shown the data the whole project is evaluated on.**

That is a different defect from any of the thirteen, and it is the first one
that is cheap to act on: develop the higher areas on photographs rather than on
line drawings, and re-run this table.

### What it means for the loop

The closed circuit should be built on **V2**, not V4 — V2 is where this stream's
information actually peaks on photographs. Building a predictive loop on a stage
raising 12 spikes would be measuring the loop's ability to amplify nothing.

---

## 9.16 The loop, on the four-stage eye — and a verdict of mine that was wrong

The drawn proposal was a circuit instead of backpropagation: a stage sends its
guess down, the stage below returns only what the guess missed, and the two
settle until they agree. `benchmarks/loop_on_ventral.py` closes it on the
ventral stream, at **V2** — where §9.15 measured this stream's information to
peak on photographs — with **V4** as the control, since V4 raises 12 spikes and
a loop that "improves" it is manufacturing structure from nothing.

| arm (front end V2) | cluster AUC | probe | participation |
|---|---|---|---|
| feedforward *(1764-d)* | 0.576 | **0.358** | 66.8 |
| loop-1 *(256-d)* | 0.548 | 0.292 | 46.8 |
| loop-20 *(256-d)* | 0.571 | 0.278 | **16.2** |
| loop-20-noteach *(256-d)* | 0.564 | 0.285 | 55.6 |

**The control is clean**: the same loop on V4 moves the probe by exactly
**+0.0000**. It does not invent structure where there is none, so the rest of
the table can be read.

### The benchmark's own verdict was wrong, for the third time in this project

It printed *"the loop does not beat the feedforward stage it was built on"* off
the −0.080 probe gap. That gap compares a **1764-dimensional** code against a
**256-dimensional** one. It is the same error as §9.14 (comparing across depths)
and §9.9 (comparing across class sets): a difference read as an effect when it
is a difference in conditions.

Matched at 256 dimensions:

| 256-d code | probe | participation |
|---|---|---|
| random projection of V2 | 0.257 | 67.1 |
| **PCA of V2** | **0.278** | 66.8 |
| **loop-20** | **0.278** | **16.2** |
| **loop-1** | **0.292** | 46.8 |

**The loop's apparent loss was entirely the bottleneck.** At equal
dimensionality the settled loop is exactly level with PCA (0.278 against 0.278)
and clearly above a random projection (0.257). Predictive coding here is a
competent dimensionality reduction — it is not destroying information.

### But the drawn claim specifically — *continue till best result* — is refuted

One feedback pass is the best 256-d code in the table (**0.292**, above PCA's
0.278). Settling to a fixed point gives that back (**0.278**, exactly PCA), and
the mechanism is visible in the third column: **participation collapses from
46.8 to 16.2** as the loop iterates. Each round explains away more, and what it
explains away includes the category structure.

Learning the loop's generative weights is worth **−0.007** — the untrained loop
is marginally *better* than the trained one.

> The feedback is mildly useful. The **iteration** is what costs, and the
> iteration is the part the circuit was drawn around.

Stated with the caution it needs: three seeds, and the differences between 0.278
and 0.292 are small. The participation collapse from 46.8 to 16.2 is not small,
and it is the reason to believe the direction rather than the size.

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
14. ~~**Make the concept layer form concepts.**~~ — **done.** Reliability-weighted
    combination plus a homeostatic vigilance took it from 1.00 cells per
    experience to 0.32–0.53, purity ≥0.98, with cross-modal retrieval rising
    0.581 → 0.671 and the synthetic bank unchanged. Consolidation now merges
    (1.13× at no cross-modal cost) where it previously merged nothing.
15b. **The world model needs a learning rule, not a replay rule.** §7.8 shows
    transition replay can only undo a prior -- with none it is arithmetically a
    no-op (+0.0000 on real sound with perception at 0.818), with a strong wrong
    one it restores exactly the no-prior value. Anything that genuinely improves
    the model has to add information the day did not already contain: predicting
    *n* steps ahead, learning transitions between concept cells rather than
    between names, or replaying counterfactual orders rather than lived ones.

15. ~~**Re-run the dream now that concepts exist.**~~ — **done**, and it is the
    section's headline: imagining a sight the mind never saw improves real
    cross-modal recall in 6 of 6 seeds (d=1.79), worth 74% of replaying the real
    sights, against a confabulation control at exactly +0.0000. Novelty-
    prioritised replay came out *below* uniform (+0.016 vs +0.023) — the
    opposite of the prediction, and cheap to re-test once the sparse case works.
16. **Build something that actually imagines.** The bar has now been fixed and
    the mechanism measured against it, and the answer is sharper than
    "half-done".

    Concept cells learn the directions they vary in (Oja + Sanger, local and
    gradient-free) and sample members rather than returning one mean. But
    measured in the space those weights actually live in, a third of what the
    mind produces is a **stored photograph byte for byte** — 59 of 112 cells win
    exactly once, hold their training pair verbatim, and have no subspace at
    all, so no temperature moves them. Composing several concepts is what ends
    that: verbatim 0.397 → 0.001.

    The replacement bar is the pair (novelty, coherence) against where real
    held-out data sits, with a coherence floor — added because the chimera arm
    won the naive scalar version by *degrading* below real data, and the anchored
    arm won it by copying 0.949 of the answer from the question. Both are now
    reported with the artefact printed beside them.

    Against that bar the honest result is a negative with a number on it:
    averaging three stored photographs **with no concept layer at all** reaches
    novelty 0.362 where composing three concepts reaches 0.389 — a paired
    +0.0276 (d = 2.76, 5/5), consistent and negligible. Mixing *k* near-orthogonal
    codes is 1/√k novel by arithmetic; that arithmetic is the whole effect.
    **Nothing self-generated exceeds novelty 0.389 against real data's 0.797**,
    because mixing and sampling are both linear operations on stored vectors.

    That requirement — a source of variation that is **not a weighted sum of
    stored codes** — has since been met. Recombining *factors* rather than whole
    codes (the eye already emits form and colour in disjoint blocks) produces a
    span residual of 0.5287 where every mixing arm produces exactly 0.000, at
    64% of a real photograph's residual, with the form block reading identically
    to the concept mean. See "A yellow bus" above. What remains open is not
    novelty any more but *selection*: nothing yet says which of the available
    crossings is worth imagining. Sampling the concept instead of the episode — the
    "infinite inner world" version — was tried and *degrades* the concepts
    (`dreamt` −0.0301, the worst arm measured), because nothing real is holding
    the categories apart. The direction that survives is a night that is
    **partly** anchored: keep a real recording, imagine the sight, and let the
    *variation* rather than the content be self-generated. Interpolating between
    two same-label concepts is the only self-generated arm that helps on a
    sparse day (+0.0058), which is a signal worth chasing rather than a result.
17. **The eye's problem has one name: no translation invariance.** Eight
    mechanisms have now failed against it, and the eighth named it. Multi-glance
    accumulation is blocked because the corrective saccade -- which exists
    precisely because off-centre objects cannot be read -- lands on the same
    pixel every time (median offset 0.0, glance-to-glance cosine 0.879), so
    there is no independent evidence to pool. Letting the eye scan an object's
    parts removes the redundancy (0.856 -> 0.168) and destroys single-glance
    accuracy (MNIST 0.733 -> 0.257), because diverse glances must be off-centre
    and off-centre glances cannot be read.

    Tested, and it found a real bug rather than a missing feature.
    `pooling_index` groups cells by index on the assumption that every
    hypercolumn holds the same filter bank -- true untrained (cosine 0.977) and
    false after `develop_v1`, which lets each column drift (0.454, against 0.319
    for random pairs). The "invariant" code was summing unrelated cells.
    `develop_v1(tie=True)` restores it, invariance appears (a fully pooled code
    falls +0.042 over 10px instead of +0.274), and glance accumulation starts
    working (+0.054 on MNIST, +0.113 on CIFAR) where it previously did not.

    What remains open is that the invariant codes are *weak* -- 0.233 against
    0.773 for position-specific on centred digits -- so the repair does not
    lift overall accuracy. The selectivity has to come from somewhere else, and
    that is now the question rather than invariance.

18. **(superseded) six mechanisms failed and one non-mechanism worked.**
    1-NN 0.337 on photographs against 0.914 for the ear. Width is flat from
    1024 to 4096 cells; a larger aperture is worse; a longer integration window
    does nothing; **spiking costs nothing at all** against the noiseless filter
    response (−0.002); and a second stage discards information under *both* a
    competitive objective and Földiák's trace rule (−0.014 and −0.024
    concatenated). Each was measured with a control and none of them is the
    limit.

    Giving the eye the **whole 32×32 frame** instead of a 28×28 centre crop —
    a handicap inherited from MNIST and never revisited — is worth +0.023 on
    1-NN and carries end-to-end: `sound → vision` 0.581 → 0.775. That six
    mechanisms failed and one cropping decision was worth more than all of them
    is the most useful thing in §7.8.

    Per-patch whitening was then tested correctly — as the convolution the
    patch covariance implies — and **hurts** at every kernel width (0.343 →
    0.188/0.197/0.212). It flattens the spectrum, and the low-frequency
    structure of a colour-opponent channel is exactly the chromatic signal this
    eye depends on. Seven mechanisms have now failed and one cropping decision
    was worth more than all of them.

    That experiment has now been run, and it *removes* a candidate rather than
    supplying one: on 256×256 scenes of scattered photographs the eye reaches an
    on-object rate of **0.883, better than the 0.830 it manages on digits**, and
    then names what it found at 0.134 against a chance of 0.100. Looking
    transfers to the real world completely; recognising does not transfer at
    all. Whatever fixes this is inside the recognition path, and it is not any
    of the seven things tried.

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
