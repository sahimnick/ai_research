# benchmarks/

Runnable measurements of the perception stack, the assembled mind, and the cost
of both. Every script writes a JSON file and prints as it goes, so a partial run
is still a usable result. Findings from a full pass are written up in
[`../EVALUATION.md`](../EVALUATION.md).

## Running

```bash
pip install numpy
python3 benchmarks/vision.py            out_vision.json
python3 benchmarks/streaming.py         out_streaming.json
python3 benchmarks/unified.py           out_unified.json
python3 benchmarks/efficiency.py        out_efficiency.json
python3 benchmarks/binding_diagnostic.py out_binding.json
```

numpy is the only requirement. Torch is optional and untouched by these runs.
(Until AUDIT.md A1 was fixed these all needed a stub-torch shim on `PYTHONPATH`
just to get `import neurobrain` to succeed; that shim is gone.)

`vision.py`, `streaming.py` and `efficiency.py` download MNIST and
Fashion-MNIST on first run and cache them; they need network access once.

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
