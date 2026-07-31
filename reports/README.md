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

**Numbers (with the new `threshold=1.5` default):** 26 onsets found for 24 real
events (precision 92%, **recall 100%**), 21 named correctly. Every real event is
now found — compare the old default, which reported 18/24 and left six blue
bands in `02` with no green line at all.

`04_threshold` is the panel that drove the change. The default was 2.5 and is
now 1.5, chosen over a 182-run sweep on *named yield* rather than F1 — see
`benchmarks/ear_threshold.py` and the box in `../EVALUATION.md` §2. Above 3.0
the ear goes nearly deaf (recall 17%).

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

## The imagination panels

`python3 benchmarks/report_imagination.py --outdir reports` — six panels on real
CIFAR-10 photographs and ESC-50 recordings, written to be checkable rather than
flattering.

| file | what it shows |
|---|---|
| `imag_01_recall.jpg` | a held-out photograph and the stored photographs the concept it wakes is made of. 52% of concepts hold exactly **one** photograph, and a held-out photograph wakes the right category only **35%** of the time — the eye's failure, visible |
| `imag_02_temperature.jpg` | imagining one concept at T = 0, 1, 4, 16, each beside the nearest thing ever stored. Where the caption reads **← A MEMORY** the cosine is 1.000 and the "imagining" *is* that photograph — at every temperature, for the singleton concepts |
| `imag_03_yellow_bus.jpg` | factored recombination in pixels: the luminance of one photograph carrying the chroma of another. Span residual **0.5287** against **0.000000** for every mixing operation. The picture is an *illustration* — cos to the code-space output is 0.351, and the caption says why |
| `imag_04_plane.jpg` | every arm on the (novelty, coherence) plane. The target is where real held-out data sits, not a corner |
| `imag_05_ear.jpg` | the sense that works: cluster AUC 0.788 against the eye's 0.573 |
| `imag_06_scale.jpg` | the error-driven rule overtaking plain binding as the world grows (+0.0240, d=2.10, 5/5), and the prediction error that makes it possible |

The V1 decoder used in panels 2, 3 and 5 is a transpose decode with a
correlation of **0.339** to its input. Every panel that uses it says so, and
panels 1 and 3 work in image space where they can.
