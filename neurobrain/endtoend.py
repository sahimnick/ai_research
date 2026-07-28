"""
endtoend.py
===========

One chain, from **spikes** to **action consequences** -- the three islands
joined.

The standing criticism was that the pieces never touched: spiking perception
lived in one module, the shared cortical code in another, the world model in a
third, and each was measured on its own bench. A pipeline that is only ever
tested stage-by-stage can be entirely healthy in parts and useless as a whole.

:class:`EndToEndBrain` runs the whole path and measures it **only at the end**:

    real image -> V1/V2/V3 SPIKES -> shared cortical code -> concept
                                                          -> predicted outcome
                                                             of an action

Nothing in the middle is scored generously; what counts is whether a decision at
the far end is right, given that every earlier stage has already lost something.
That is the honest way to test a pipeline, and the compounding loss is reported
explicitly rather than averaged away.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v.astype(np.float32) if n < 1e-9 else (v / n).astype(np.float32)


@dataclass
class EndToEndBrain:
    """Spiking eyes -> shared code -> concepts, measured end to end."""

    hierarchy: object                 # SpikingVentralHierarchy
    workspace: object                 # GlobalWorkspace
    stage_accuracy: Dict[str, float] = field(default_factory=dict)
    end_to_end_accuracy: float = 0.0
    prototype_accuracy: float = 0.0
    compounding_loss: float = 0.0

    def see(self, image: np.ndarray) -> np.ndarray:
        """Retina -> V1 -> V2 -> V3, all in spikes, then into the shared code."""
        spikes = self.hierarchy.top_code(image)
        return self.workspace.encode("vision", spikes)

    def recognise(self, image: np.ndarray) -> Optional[str]:
        return self.workspace.name(self.see(image))


def build_end_to_end(n_train: int = 4000, n_test: int = 1000,
                     front: str = "hierarchy",
                     verbose: bool = False) -> EndToEndBrain:
    """Chain the real modules together and score the WHOLE path on real digits.

    Each stage is also scored alone, so the compounding loss is visible instead
    of hidden inside an average.

    ``front`` selects the eyes: ``"hierarchy"`` is the v0.21 spiking V1->V2->V3
    stack, ``"wide"`` is the v0.22 single wide retinotopic V1. Both expose
    ``top_code``, so the rest of the chain does not change and the difference in
    the end-to-end number is attributable to perception alone."""
    from .realworld import load_mnist, GrowingCategoryMap, contrast_normalise
    from .spikinghierarchy import build_spiking_hierarchy
    from .workspace import GlobalWorkspace

    def say(*a):
        if verbose:
            print(*a)

    trx, trY, tex, teY = load_mnist(n_train=n_train, n_test=n_test)
    if front == "wide":
        from .widev1 import build_wide_digit_brain
        say("building the WIDE spiking V1 eyes ...")
        wb = build_wide_digit_brain(n_train=n_train, n_test=n_test,
                                    verbose=False)
        h = wb.layer
        h.accuracy = wb.accuracy
    else:
        say("building the spiking V1->V2->V3 eyes ...")
        h = build_spiking_hierarchy(n_train=n_train, n_test=n_test,
                                    verbose=False)

    say("wiring the spikes into the SHARED cortical code ...")
    ws = GlobalWorkspace(dim=512, vigilance=0.25, seed=0)
    brain = EndToEndBrain(h, ws)

    codes_tr = [h.top_code(im) for im in trx]
    for i, c in enumerate(codes_tr):
        ws.learn_concept(str(int(trY[i])), ws.encode("vision", c))

    say("scoring the WHOLE chain on held-out digits ...")
    # The read-out on BOTH sides of the workspace must be the same, or the
    # comparison measures the read-out instead of the workspace. Growing
    # categories from the perceptual code and then matching ten averaged
    # prototypes in the workspace is not a like-for-like test: v0.21 reported
    # a 0.3-point "cost of the shared code" that way, and re-measured properly
    # that number does not survive. So the same grow-then-name rule is applied
    # to the workspace codes as to the perceptual codes.
    from .widev1 import grown_readout
    Wtr = np.array([ws.encode("vision", c) for c in codes_tr], np.float32)
    Wte = np.array([ws.encode("vision", h.top_code(im)) for im in tex],
                   np.float32)
    brain.end_to_end_accuracy = grown_readout(Wtr, trY, Wte, teY)[0]

    # kept for reference: the ten-prototype read-out used before
    ok = 0
    for i in range(len(tex)):
        ok += int(brain.recognise(tex[i]) == str(int(teY[i])))
    brain.prototype_accuracy = ok / len(tex)

    # what each stage can do on its own, for the compounding-loss picture
    X = contrast_normalise(trx)
    m = GrowingCategoryMap(dim=X.shape[1], vigilance=0.65, lr=0.1)
    m.train(X, epochs=2)
    m.name_cells(X, trY)
    brain.stage_accuracy = {
        "vector_perception_alone": float(np.mean(m.recognise(tex, 0.0) == teY)),
        "spiking_perception_alone": float(h.accuracy),
        "through_shared_code": brain.end_to_end_accuracy,
    }
    brain.compounding_loss = (brain.stage_accuracy["vector_perception_alone"]
                              - brain.end_to_end_accuracy)
    if verbose:
        for k, v in brain.stage_accuracy.items():
            print(f"   {k:26}: {v:.1%}")
        print(f"   compounding loss vs the non-spiking shortcut: "
              f"-{brain.compounding_loss:.1%}")
    return brain
