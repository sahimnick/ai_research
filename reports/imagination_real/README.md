# The imagination system on the real brain — visual report

Six figures from `benchmarks/imagination_real.py`. Full write-up in
`EVALUATION.md` §9.39; raw numbers in `out_imagination_real.json`.

Reproduce:

```bash
python3 benchmarks/imagination_real.py out_imagination_real.json vot
```

5 seeds, 416 crops from 13 VOT2019 sequences, bound to real ESC-50 recordings
through the H23 adapter (`neurobrain/minds/eyebrain.py`). `minds/mind.py` is
unmodified.

| figure | what it shows | how to read it |
|---|---|---|
| `01_imagery_plane.png` | **Test 1 — mental imagery.** Every arm on §9.19's novelty × coherence plane, with the decisive region enlarged beside it | The target is the **red star**, not the top-right corner. Novelty only counts above the green line, because collapsing coherence buys novelty for free. The purple diamond is the control with **no concept layer** — it sits on the target |
| `02_verbatim_temperature.png` | **Test 1b.** Verbatim recall and novelty against sampling temperature | Verbatim is flat at 0.000, which is a **cell-count artefact** (no singleton cells in this configuration), not a repair of §9.19's 0.368 |
| `03_transitions.png` | **Test 2 — sequential imagination.** Transition matrices: the world, the imagined walk, and a shuffled-time control | The control has no diagonal — that is what destroying temporal structure looks like. The imagined diagonal is **brighter than the world's** while its off-diagonal is dimmer: the mind imagines a world too static |
| `04_rollout.png` | **Test 3 — imagining forward.** Rollout accuracy against persistence, horizons 1–8 | The **dashed line is the arm to beat**. The gap is real at all four horizons (CIs clear zero, paired over 13 videos) and small — about two frames in a hundred |
| `05_surprise.png` | **Test 4 — surprise.** Distributions for within-sequence transitions and ones spliced from another video | If the two histograms coincided, surprise would be a constant rather than a prediction error. They do not: d = +1.55 |
| `06_cross_modal.png` | **Test 5 — cross-modal imagery.** Hear a held-out recording, imagine the sight, match it to visual categories | **The diagonal is the claim.** 0.427 against chance 0.200. Only the recordings are held out — the visual prototypes come from all crops |

## The one-line result

The imagination system has a working model of **how its world behaves** —
sequences, forward prediction, surprise, cross-modal recall — and no working
generator of **what its world contains**: mental imagery fails its own test, and
averaging three stored crops with no concept layer does as well as the concept
layer does.

## The bound on all of it

H23 (§9.37) measured cross-scene concept naming at chance. Every number here
rests on concepts reliable **within** a scene. None has been shown to survive a
scene the eye did not develop on.
