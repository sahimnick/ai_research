# benchmarks/

Runnable measurements of the perception stack, the assembled mind, and the cost
of both. Every script writes a JSON file and prints as it goes, so a partial run
is still a usable result. Findings from a full pass are written up in
[`../EVALUATION.md`](../EVALUATION.md).

## Running

```bash
pip install numpy
PYTHONPATH=. python3 benchmarks/vision.py            out_vision.json
PYTHONPATH=. python3 benchmarks/streaming.py         out_streaming.json
PYTHONPATH=. python3 benchmarks/unified.py           out_unified.json
PYTHONPATH=. python3 benchmarks/efficiency.py        out_efficiency.json
PYTHONPATH=. python3 benchmarks/binding_diagnostic.py out_binding.json
```

numpy is the only requirement. Torch is optional and untouched by these runs.
(Until AUDIT.md A1 was fixed these all needed a stub-torch shim on `PYTHONPATH`
just to get `import neurobrain` to succeed; that shim is gone. `PYTHONPATH=.`
is still needed because the package is not installed -- run from the repo root.)

`vision.py`, `streaming.py` and `efficiency.py` download MNIST and
Fashion-MNIST on first run and cache them; they need network access once.

## The real-world pass

These use **CIFAR-10 photographs and ESC-50 field recordings** instead of digits
and synthetic tones, and they are what EVALUATION.md section 7.8 is written from.
Both datasets download and cache on first run (CIFAR is 170 MB; the fetch is
resumable and streaming, and `load_cifar10` will read a partial download).

```bash
PYTHONPATH=. python3 benchmarks/real_audio.py        out_real_audio.json
PYTHONPATH=. python3 benchmarks/belt_gain.py         out_belt_gain.json
PYTHONPATH=. python3 benchmarks/real_binding.py      out_real_binding.json
PYTHONPATH=. python3 benchmarks/concept_formation.py out_concept_formation.json
PYTHONPATH=. python3 benchmarks/cross_modal_dream.py out_dream.json
PYTHONPATH=. python3 benchmarks/natural_v1.py        out_natural_v1.json
PYTHONPATH=. python3 benchmarks/vision_ceiling.py    out_vision_ceiling.json
PYTHONPATH=. python3 benchmarks/second_stage.py      out_second_stage.json
PYTHONPATH=. python3 benchmarks/natural_scene.py     out_natural_scene.json
PYTHONPATH=. python3 benchmarks/imagination_shape.py out_imagination_shape.json
PYTHONPATH=. python3 benchmarks/multi_fixation.py    out_multi_fixation.json
PYTHONPATH=. python3 benchmarks/translation.py       out_translation.json
PYTHONPATH=. python3 benchmarks/real_mind.py         out_real_mind.json
PYTHONPATH=. python3 benchmarks/real_time.py         out_real_time.json
PYTHONPATH=. python3 benchmarks/composition.py       out_composition.json
PYTHONPATH=. python3 benchmarks/factored.py          out_factored.json
PYTHONPATH=. python3 benchmarks/inner_world.py       out_inner_world.json
PYTHONPATH=. python3 benchmarks/constraint.py        out_constraint.json
PYTHONPATH=. python3 benchmarks/heard_world.py       out_heard_world.json
PYTHONPATH=. python3 benchmarks/scale.py             out_scale.json
PYTHONPATH=. python3 benchmarks/pathways.py          out_pathways.json
PYTHONPATH=. python3 benchmarks/pathway_downstream.py out_pathway_downstream.json
PYTHONPATH=. python3 benchmarks/limiting_factor.py   out_limiting_factor.json
PYTHONPATH=. python3 benchmarks/temporal_invariance.py out_temporal.json
PYTHONPATH=. python3 benchmarks/temporal_capacity.py  out_temporal_capacity.json
```

| script | question | key control |
|---|---|---|
| `real_audio.py` | Does the auditory front end hear real environmental sound? | five front ends on identical clips; prototype, 5-NN **and** linear probe, because a probe-minus-prototype gap means the geometry is wrong rather than the code empty |
| `belt_gain.py` | How much tonotopy should the belt keep? | swept on real audio **and** the synthetic bank it was originally tuned on, 8 paired splits |
| `real_binding.py` | Do concepts form from two corpora that share only meaning? | `shuffled` for the label paths, `mismatched` for the cross-modal ones, and a 1-NN unimodal floor |
| `concept_formation.py` | Does the layer form concepts or keep one cell per experience? | cells-per-pair below 1.0 with purity and recall held |
| `cross_modal_dream.py` | Can the mind learn from a sight it never saw? | a **confabulated** arm that performs identical binds on a random imagined sight |
| `natural_v1.py` | What does the eye need for photographs? | colour, adaptation and discovered fields added cumulatively, with MNIST carried through as a regression check |
| `vision_ceiling.py` | Which part of the eye is the limit? | width, aperture, spike noise and integration window, each against a control |
| `second_stage.py` | Does depth help where width did not? | two learning rules for V2, and a concatenated arm that can only fail by V2 adding nothing |
| `natural_scene.py` | Does the saccadic eye work on scenes of photographs? | on-object rate against the chance of landing on one, with digit scenes as reference |
| `imagination_shape.py` | Does it imagine, or recall an average? | the same distances asked of a **real unseen photograph**, so "novel" has a scale; measured in `prep_v` space, which the first version got wrong |
| `multi_fixation.py` | Does aggregating glances beat one look? | pooling codes and pooling scores are the *same* operation (`unit(mean(C)) ∝ sum(C)`) — counted once, not twice |
| `translation.py` | Is the code translation-invariant? | a 48-px frame, because a 28-px object in a 28-px frame tests occlusion, not translation |
| `real_mind.py` | Does the assembled mind behave the same on photographs? | the same arms and probes as `acceptance.py`, so the two tables can be read against each other |
| `real_time.py` | Is perception what gates the world model? | one **shared** day across all three arms; distinct-percept count beside every self-consistency score, since a collapsed perceiver predicts itself at 1.000 |
| `composition.py` | Does combining concepts imagine? | *k* stored photographs averaged with **no concept layer** — mixing *k* near-orthogonal codes is 1/√k novel by arithmetic alone; plus a chimera arm that exists to catch the metric being gamed from below |
| `factored.py` | Can it produce anything memory cannot assemble? | the **span residual**, not a cosine — exactly 0 means a linear combination of things already seen; plus a same-cell control proving the escape is the *crossing* and not the slicing |
| `inner_world.py` | Does out-of-span imagining pay back into the waking world? | a night of **random sights** at the same span residual — if the recombination matches it, the structure is not what is working; plus an R↔B swap that leaves luminance exactly unchanged |
| `scale.py` | Is the exemplar store caused by the rule or by too little data? | 30x the data with everything else held fixed — pairs-per-cell answers it directly; the two rules see identical data in identical order and differ only in whether the negative phase runs |
| `heard_world.py` | Does judging an imagining pay back, where perception works? | `crossed_judged` and `crossed_anti` draw the **same** candidates and invert only which is kept, so their gap is selection alone — and `stored` is the positive control that decides whether the payback channel is open at all |
| `merger.py` | Does forcing the layer to generalise stop it returning memories? | a **sweep** over compression, not one setting — recall and purity reported at every level, because merging 82%-pure pairs still merges 18% wrong ones |
| `limiting_factor.py` | What actually limits the object-centred frame? | an **oracle** origin — the object's true position, which the benchmark knows because it placed it — as an instrument rather than a proposal; and the arm that decides the answer is the oracle **displaced by the estimator's own errors, dealt to the wrong images**, which holds the error's size fixed and destroys only its link to content. A 1.48 px error costs nothing that way and 0.044 when it is real, so the fault is content-correlation, not imprecision. Asserts that the oracle origin varies and that the shift is not clipped, because both were silently false in earlier versions |
| `temporal_invariance.py` | Does invariance come from TIME rather than geometry? | a **shuffled-time** control that pools frames globally rather than within a sequence — shuffling within would leave the predictive term most of its signal, since an object moves slowly and frames 1 and 5 are still the same object; plus a **power check**, because a null from a blunt instrument is worthless: the predictive term provably receives a 2.4x different signal on ordered vs shuffled input and still yields the same code |
| `temporal_capacity.py` | Is the tied bank's 9 filters what binds it? | only `n_cells` varies — rule, data, sequences and evaluation identical — so it localises the fault instead of proposing a second architecture |
| `pathway_downstream.py` | What do the pathways do once a concept layer, a night and an imagination sit on them? | a **1-NN floor** on raw codes beside every naming score, because the layer holds ~0.6 cells per pair and may be reading the representation straight back — reported per pathway, since averaging the four cancelled a +0.042 against a −0.050 into a fabricated null; and `stored` gates every replay claim, because a pathway where replaying *real* pairs does not help cannot decide anything about imagined ones |
| `pathways.py` | Does a local-receptive-field pathway improve the representation? | the bar is **pre-registered** in `localspatial.py` and cluster AUC is computed from raw codes with no learning, so a gain cannot come from the read-out adapting; five ablations, each disabling one component, share an identically-developed bank so the ablation varies one thing |
| `relational.py` | Can a code be translation-invariant without becoming a bag of features? | the **already-invariant** code (`pooling_index`) as the arm to beat, not the position-specific one — a new invariant code that ties plain pooling has bought nothing |
| `ear_config.py` | Is the eye's ceiling caused by the one structural difference from the working sense? | the verdict requires an effect to explain a **quarter of the gap it was proposed to explain**, not merely to pass a significance gate — tying the bank is 4/4 seeds and 7.8% of the distance to the ear |
| `phase10_1b.py` | Are V1's filters actually bound to single images? | three independent sparsity measures (top-1 share, participation ratio, lifetime kurtosis) so the answer does not rest on one statistic's threshold — they agree that filters respond to 75% of all photographs, so the premise being fixed was never true |
| `phase10_2b.py` | Does pooling glances survive the shift that breaks a single look? | a **random-glance** arm beside the chosen-glance one, which separates "more looks" from "better-chosen looks" — and it is the random arm that wins, so the claim is about pooling |
| `phase10_2.py` | Does a 5-pixel shift break the eye, and does concept feedback fix it? | asserts the frame can hold the requested shift, because a 40px frame silently clipped ±5 to ±4 and flattered the result by an order of magnitude; plus a **deliberately misleading** teacher alongside the correct one |
| `live_mind.py` | Does the whole mind work on sensors that are live, and does the payback reproduce? | binds on early rounds and measures on a later one, and **refuses to report** unless the pixels actually changed; effects are expressed in *cameras* because one camera is 0.042 and anything smaller is below what a 24-place world can resolve |
| `aperture.py` | Is the eye limited by what it receives rather than how it learns? | cell count **held fixed** across input sizes, plus a separate 4x-cells arm — otherwise field of view and capacity cannot be told apart; run on live cameras because CIFAR is natively 32x32 and more resolution there is interpolation |
| `align.py` | Can the concept layer teach V1, with no labels? | the same rule and budget with an **uninformative** target (the image's own rate) and a **misleading** one (a wrong concept) — all three agreeing is what proves no signal is arriving |
| `live_world.py` | Does the machinery behave on sensors that are live right now? | the **same pixels encoded twice**, as the noise floor beneath "the same camera later"; plus content hashing across cameras, which is the only thing that catches a no-signal card served with HTTP 200 |
| `payback.py` | Does the inner world improve the waking one? | `imagined_as_fact` and `imagined_as_error` replay the **same completions in the same order** with the same number of plasticity events — the only difference is meant to be whether the fantasy is believed, and for a while it was not: `bind` re-prepped an already-prepped imagining (cos **0.407** to what the model generated), so `imagined_as_fact_fixed` is the arm the comparison should always have used. Both are kept, because the uncorrected number is published; plus `merged` alone, which separates generalising from dreaming |
| `prior.py` | Does censoring implausible crossings help? | keep the **least** plausible of the same draws, which inverts only the selection; plus keep one at random, which separates choosing from drawing more |
| `uncertain_look.py` | Is it worth choosing where to look next? | an **oracle** that tries 6 candidates per glance and keeps the best — the ceiling any cheap rule could reach; against drawing the same 6 and keeping one at random |
| `constraint.py` | Can a plausibility model judge an imagined crossing? | the **ceiling** — how much colour a category determines at all — reported beside every model, so a weak world is not read as a weak rule; plus the same rule handed a visual code that clusters, which separates "weak rule" from "weak representation" |
| `eye_live_world.py` | Do all four eye stages hold a place at native resolution, in colour, while it moves? | **raw downsampled pixels through the identical pipeline** — same pairing, same fit-once-then-freeze adaptation — which is what revealed the identity task to be saturated: 24 cameras point at 24 different streets and pixels score it perfectly too, so the eye's 1.000 was reporting geography. Plus a floor that re-encodes the *same* frames through the *same* frozen baseline, and a refusal if any code comes out as the zero vector — the first version had neither and reported a structural 0.000 as an eye result |
| `eye_shift.py` | Does the eye survive translation better than the pixels it is computed from? | pixels rebuilt on **each stage's own grid** (87×87, 41×41), because a coarser code is automatically more shift tolerant and that confound would otherwise explain the whole result; the window is cropped from a larger native frame so the scene really moves, and the run **refuses to report** if the crop did not — the trap a 64-px frame fell into once already |
| `eye_track.py` | Does a stored tag find its object, and follow it when it moves? | the object is composited into a real frame at **known** positions, because live cameras give realism but no ground truth; a **shuffled tag** (a different object's) for chance, and an instrument check that refuses to report unless an arm can find the tag in the frame the tag came from — which is what caught the mechanism scoring 0.000 there against the wrong tag's 0.188 |
| `dpc_multistep.py` | Does training on several horizons fix §9.22's negative rollout? | **persistence** and **constant velocity** both, because on smooth motion "keep going the way you were" is arithmetic that a world model must beat to be one — and it is the arm the direct predictions clear and the free rollout does not; the run **refuses to report** if the targets barely move, since persistence would then be unbeatable by construction. States are displacements, not coordinates, after the first version diverged to `nan` |
| `eye_unified.py` | Is this one eye, or one stage with scaffolding? | four read-outs from a **single forward pass** — where, which, relation, how-big — so the stages are compared doing different jobs on the same code; the verdict is gated on the winners' **margins**, not merely on there being several of them, because an argmax over three near-equal numbers is decided by noise (§9.10) |
| `eye_properties.py` | Does the code carry distance and shadow — the goal's "ویژگیهای خاص"? | distance uses a cue **the world produced**, not one the benchmark invented: a tracked object's annotated box grows as it approaches, so VOT ground truth already holds the label. **Two splits**, because §9.27 showed this code carries ~nothing across unseen scenes about anything, so a by-sequence null could not have said distance specifically was missing — and within-sequence it is emphatically present (0.585 vs pixels' 0.360). Shadow is normalised by the pixel change it caused, so the statistic is code-change *per unit pixel change*, and raw pixels give a ratio of 1 by construction — the null the eye lands exactly on |
| `eye_track_drift.py` | Does tracking fail because the tracker drifts, or because the code cannot localise precisely? | re-anchors the search window on the previous **ground truth** instead of the previous prediction, which removes error accumulation and changes nothing else — an **instrument, not a tracker**, since it consumes ground truth every frame. Answer: drift costs only +0.05, so precision is the limit. It is also where §9.24's "pixels beat the eye two to one" was found not to replicate: the ordering reverses on a different twelve sequences and the arms are not separated on 32. Writes **per-sequence** results so a paired test is possible |
| `eye_attention_video.py` | Does top-down selection survive an object whose appearance really changes? | the two competitors differ in **exactly one way** — a real VOT target that translates, rotates, scales and relights, against a composited distractor that is pixel-identical — both cued through the same machinery in the same frames, which is what locates the decay rather than merely measuring it; the field is fixed per sequence so "harder to find" cannot mean "left the picture". Written to test a prediction §9.28 had asserted without running, and it refuted it |
| `eye_attention.py` | Does a cue change what the eye picks out of a scene, or only where it looks? | the **image is identical** across the two trials and only the cue changes, so a feed-forward saliency arm has a switch rate of 0.000 *by construction* — that is the arm to beat, not a number to gloss; each object is tagged on a **different background** than any test scene, or a tag could win by matching wallpaper; and which object sits in which slot is randomised, so a cue cannot be answered by a positional bias. A third object never present in the image gives chance |
| `eye_breadth.py` | Is the eye's non-generalisation architectural, or just too few scenes? | the **frame budget is held constant** across breadths, so "more scenes" is not also "more data"; three splits varying *which sequences* are held out, not merely the seed (§9.10); and the run reports itself **underpowered** when the pixel control does not respond either — a manipulation that cannot move a code built to benefit from it cannot license a negative about one that does not move. It is also where a control I reinterpreted favourably after seeing it (+0.071) collapsed to +0.020 ± 0.033 under replication |
| `eye_information.py` | How much of the image survives to each area, and do the four act as one eye? | everything cut to the **same D** before decoding, so a wider code cannot win on width (§9.16's error); the pixel arm is declared a **ceiling** rather than a rival, because the decoding target is a linear function of its own input and it must win; **two splits** reported, since a by-video split alone made every eye arm negative and looked like an architectural claim when it was a distribution shift; and each area is rescaled before concatenation, without which "all four" is really "whichever area has the most variance" — that fix alone reversed the unified-eye verdict |
| `eye_vot.py` | How does the eye do on a benchmark this project did not choose? | **VOT2019**, real 30fps video with human ground truth, standard one-pass evaluation — and the tracker is deliberately declared weak up front (fixed template, no model update, no scale search), so the absolute numbers are **not** comparable to published trackers and only the arms' relative standing is claimed. `shuffled-tag` runs a *different* sequence's tag through the same search dynamics for chance. It is where §9.23's 0.834 on self-chosen CCTV fell to 0.094, against raw pixels at 0.172 |
| `eye_track_real.py` | Does a tag still find its place once the pixels there have really changed? | a **static** camera at t0 and t0+105s gives real appearance change with exact ground truth — the camera did not move, so the place is known while its content is not; scored with the metrics trackers are scored with (precision@20px, success AUC over IoU) rather than one hit rate, and it **refuses to report** unless the world actually changed. The arms fail differently: pixel NCC is exact or nothing (median 0.0px, precision 0.875), the eye is reliably close and never exact (1.5px, 0.938) because a V2 cell is 2.17px |
| `dpc_video.py` | Can a learned dynamics predict the next state better than "nothing moved"? | **persistence** is the arm to beat and skill is reported relative to it, because `continuous.py` records this project losing to that baseline once already; plus `shuffled-time` re-dealt *across* sequences (§9.9's lesson), `K=1` to separate "dynamics helped" from "several dynamics helped", and a rollout with no new input, since one-step accuracy is not what a world model owes a planner. Sequences come from panning real frames because NY511 was **measured** at a 97 s update period — it is stills, not video |
| `adaptation_row0.py` | What did `PopulationAdaptation`'s dead first row cost every number in this suite? | three arms on **identical codes** — as-published, row 0 repaired through the final baseline, and baseline fitted then frozen — so the defect's cost is isolated from everything else; the answer is that every arm of every benchmark carries the same one dead row in 240, so *differences* between arms are untouched |

## What each script measures

| script | question | key control |
|---|---|---|
| `vision.py` | Does the wide spiking V1 classify real images, and do *discovered* receptive fields beat designed ones? | designed vs developed filters, on MNIST **and** Fashion-MNIST |
| `streaming.py` | Do the eye and ear work on unsegmented streams? | saliency-driven looking vs chance; foveation correction on vs off; pre-segmented vs streaming |
| `binding_diagnostic.py` | Is `StreamingBrain.bind` actually binding vision to sound? | exact-code vs held-out query; visual code replaced by noise and by zeros |
| `unified.py` | Does the assembled mind perceive, imagine, and consolidate? | imagination chain diversity; imagine → re-perceive round trip; dream before/after |
| `efficiency.py` | Where do time and memory go, and are the known defects still live? | per-stage ms, V1 and population scaling, direct probes for A2–A5 |

## Rendered reports

These write JPGs into `--outdir` (see [`../reports/`](../reports/)):

| script | panels |
|---|---|
| `report_vision.py --dataset {mnist,fashion}` | learned prototypes · multi-object detection with tags and labels · scene-to-scene spread · attention and inhibition-of-return · smooth pursuit · confusion |
| `report_audio.py` | learned sounds · multi-event detection in an unsegmented soundscape · continuous tracking · detection-threshold sweep · confusion |
| `report_binding.py` | the bound pairs · **the vision ablation** · concept cells in both senses · sound → visual-code recall |

```bash
python3 benchmarks/report_vision.py --dataset mnist --outdir reports
python3 benchmarks/report_audio.py --outdir reports
python3 benchmarks/report_binding.py --outdir reports
```

They need `matplotlib` and `pillow` on top of numpy.

## Reading the controls

A number without its control is not a result. `binding_diagnostic.py` is the
clearest example: the same mechanism scores **100%** under an exact-code query
and **6.25%** under a shuffled-label control, and the gap between those two is
the entire finding.

The sharpest control in the suite is the **visual ablation** — rebind with
every visual code replaced by noise, then by zeros. If the answer does not
move, the binding is not cross-modal no matter what its accuracy says.
