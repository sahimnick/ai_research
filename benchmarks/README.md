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
