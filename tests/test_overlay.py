"""The drawing must put the box where the percept says it is.

Every number in EVALUATION.md is computed from `Percept.where`, but what a
reader judges the system by is the *picture*. Those are two different pieces of
code, and a coordinate slip in the second would make a correct system look
broken -- or, worse, an incorrect one look fine.

This checks the rendering itself, in pixels, against the percept it was given.
It is a test of the drawing, not of the eye.

It exists because the drawing was wrong once already: with a restricted search
window the attention map was returned sub-window-sized and stretched across the
whole frame, so the heat landed where the eye had never looked and disagreed
with the box beside it.
"""
import sys

import numpy as np

sys.path.insert(0, ".")
from neurobrain.viz.overlay import draw_percept                  # noqa: E402
from neurobrain.vision.unified_eye import Percept                # noqa: E402


def _box_centre(img, colour, tol=40):
    """Find the drawn rectangle of a given colour, return its centre."""
    d = np.abs(img.astype(np.int16) - np.asarray(colour, np.int16)).sum(2)
    ys, xs = np.nonzero(d <= tol)
    if len(ys) == 0:
        return None
    return (float(ys.mean()), float(xs.mean()))


def test_box_lands_where_the_percept_says():
    frame = np.full((224, 224), 0.5, np.float32)
    for (y, x) in ((60.0, 80.0), (150.0, 40.0), (110.0, 190.0)):
        p = Percept(where={"t": (y, x)}, confidence={"t": 0.9})
        p._boxes = {"t": (44.0, 44.0)}
        img = draw_percept(frame, p, show=["box"], scale=1)
        got = _box_centre(img, (255, 92, 92))
        assert got is not None, "no box was drawn at all"
        dy, dx = abs(got[0] - y), abs(got[1] - x)
        assert dy <= 3 and dx <= 3, (
            f"percept said ({y}, {x}), the drawing put it at "
            f"({got[0]:.1f}, {got[1]:.1f})")


def test_scale_does_not_move_the_box():
    """A 2x render must put the box at 2x the coordinates, not somewhere else."""
    frame = np.full((224, 224), 0.5, np.float32)
    p = Percept(where={"t": (70.0, 100.0)}, confidence={"t": 0.9})
    p._boxes = {"t": (44.0, 44.0)}
    got = _box_centre(draw_percept(frame, p, show=["box"], scale=2),
                      (255, 92, 92))
    assert got is not None
    assert abs(got[0] - 140.0) <= 6 and abs(got[1] - 200.0) <= 6, got


def test_lost_draws_no_box():
    """An abstention must not put a rectangle anywhere -- that is the whole
    point of abstaining. §9.35: answering on every frame scores 0.779."""
    frame = np.full((224, 224), 0.5, np.float32)
    p = Percept(lost={"t": 0.11})
    img = draw_percept(frame, p, show=["box", "label", "lost"], scale=1)
    assert _box_centre(img, (255, 92, 92)) is None, (
        "a box was drawn for a tag the eye reported as LOST")


def test_attention_peak_agrees_with_the_box():
    """The heat and the box must point at the same place."""
    frame = np.full((224, 224), 0.5, np.float32)
    att = np.zeros((28, 28), np.float32)
    att[7, 12] = 1.0                       # peak at map cell (7, 12)
    p = Percept(where={"t": ((7 + 0.5) * 224 / 28, (12 + 0.5) * 224 / 28)},
                confidence={"t": 0.9}, attention={"t": att})
    p._boxes = {"t": (44.0, 44.0)}
    img = draw_percept(frame, p, show=["attention"], scale=1)
    grey = np.full((224, 224), 0.5, np.float32)
    diff = np.abs(img.astype(np.float32)
                  - (grey * 255)[..., None]).sum(2)
    ys, xs = np.nonzero(diff > 30)
    assert len(ys), "the attention layer drew nothing"
    cy, cx = ys.mean(), xs.mean()
    assert abs(cy - p.where["t"][0]) <= 12 and abs(cx - p.where["t"][1]) <= 12, (
        f"heat centred at ({cy:.0f}, {cx:.0f}), box at {p.where['t']}")


if __name__ == "__main__":
    test_box_lands_where_the_percept_says()
    test_scale_does_not_move_the_box()
    test_lost_draws_no_box()
    test_attention_peak_agrees_with_the_box()
    print("all overlay tests passed")
