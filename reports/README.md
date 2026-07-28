# reports/

Rendered JPGs from `benchmarks/report_*.py`. Regenerate with:

```bash
python3 benchmarks/report_vision.py --dataset mnist   --outdir reports
python3 benchmarks/report_vision.py --dataset fashion --outdir reports
python3 benchmarks/report_audio.py                    --outdir reports
python3 benchmarks/report_binding.py                  --outdir reports
```

Colour convention throughout: **green = correct**, **red = wrong**,
**grey = background / false alarm**, **dotted blue = ground truth**.

---

## Vision (`mnist_*`, `fashion_*`)

| panel | shows |
|---|---|
| `01_learned_prototypes` | what was learned: one example and the class prototype it matches against, plus the V1 receptive fields |
| `02_multi_detection` | one cluttered scene, every saccade tagged / classified / labelled, with the scan path |
| `02b_scene_spread` | the same detector on three scenes — the honest view |
| `03_attention_ior` | the saliency map attention is drawn to, and what inhibition-of-return does to coverage |
| `04_pursuit` | a moving object tracked across 8 frames, with the tag it carries |
| `05_confusion` | which classes it confuses, and per-class recall |

**Numbers (MNIST, scene seed 0):** 45 saccades → 24 landed on an object (53%)
→ 15 named correctly (62%). Pursuit held the same tag on 8/8 frames.

**Detection rate swings hugely with the scene** — 53% / 82% / 100% across
three seeds at 45 saccades, and 57% / 100% / 100% at 60. The multi-seed mean
of 85.6% in [`../EVALUATION.md`](../EVALUATION.md) is the number to quote; a
single scene is an illustration, not a result. `02b` exists to make that
visible rather than to hide it behind an average.

## Audio (`audio_*`)

| panel | shows |
|---|---|
| `01_learned_sounds` | each class through the full front end: waveform → cochleagram → belt code |
| `02_multi_detection` | one 18 s soundscape, every onset found / tagged / labelled against the real events |
| `03_tracking` | a 400 ms window slid every 50 ms, labelled continuously — the audio analogue of pursuit |
| `04_threshold` | the attention knob: detection threshold vs precision / recall / F1 |
| `05_confusion` | which sounds it confuses |

**Numbers:** 18 onsets found for 24 real events (precision 100%, recall 75%),
16/18 named correctly (89%). The six misses are visible in `02` as blue bands
with no green line.

`04_threshold` is the actionable one: the shipped default `threshold=2.5`
scores F1 85.7%, while `1.5` scores **95.8%** — the default is trading a
quarter of all events for precision it does not need. Above 3.0 the ear goes
nearly deaf (recall 17%).

`harmonic` is at **0% recall** — a complete blind spot inside an otherwise
good average.

## Binding (`binding_*`)

| panel | shows |
|---|---|
| `01_pairs` | the audio-visual pairs that were bound |
| `02_ablation` | **the test that matters** — vision real vs noise vs zeros |
| `03_concept_cells` | what each concept cell holds in *both* senses |
| `04_recall` | hear a sound → the visual code it recalls → which digit that is |

`02_ablation` is the whole argument, before and after the rewrite:

| condition | before (stored-label NN) | after (concept cells) |
|---|---|---|
| vision real | 78% | 70% |
| vision **noise** | 78% | **34%** |
| vision **zeros** | 78% | 59% |
| labels shuffled | 6% | 11% |
| **ablation spread** | **0%** — vision ignored | **36%** — vision participates |

Before the rewrite, replacing every visual code with Gaussian noise changed
the answer by *nothing*. After it, the same substitution costs 36 points.
That gap is the evidence that binding now reads vision at all.

Note the asymmetry between noise and zeros, which is the informative part:
**zeros** (vision silent) falls back to audio alone and still works at 59%,
while **noise** (vision actively lying) drags the concept cells apart and
collapses to 34%. Vision is *read*; it is not yet *necessary*.

`04_recall` is the capability that did not exist before: a held-out sound
returns a **visual code** — not a stored integer — matched against digit
prototypes at **46.9%** against 12.5% chance (3.8×). Modest, and limited by a
tiny sound dataset (8 classes × 8 clips), but it is real cross-modal recall
and the old code could not express it at all.

`03_concept_cells`: 13 of 32 cells were ever used, 10 of them pure (one class
only).
