"""
overlay.py
==========

Draw what the eye is doing **on top of the frame it is doing it to**.

`UnifiedEye.look` returns a :class:`~neurobrain.vision.unified_eye.Percept` full
of numbers. This turns one into a picture: where each tag was found, how sure it
was, the path it has taken, where it is predicted to go next, the attention map
that drove the decision, and the relations between tags.

Every layer is independent and switchable, so a viewer can turn one on at a time
and see what each contributes:

    attention   the correlation map that decided *where* -- the closest thing
                this eye has to a segmentation. It is a similarity field over
                the area's own grid, NOT a learned object mask.
    box         the located region, at the tag's own size
    label       tag name + match confidence
    path        where the tag has been, over previous frames
    prediction  where it is expected next, by constant velocity over the path
    relation    the spatial relation between two tags, drawn between them
    grid        the area map's cell boundaries -- what resolution the eye
                actually sees at, which is coarser than the frame
    lost        tags the eye looked for and was not confident enough to
                report. Drawn as a banner, NOT as a box: §9.35 measured that
                answering on every frame scores 0.848 on held-out sequences
                and abstaining below 0.35 scores 0.910, so a weak frame is an
                absence of evidence and drawing a box on it would be
                inventing a detection

Honest about what is drawn
--------------------------
The attention layer is a **template-correlation field**, not a segmentation
mask. Nothing in this project segments objects; §9.21 measures localisation and
the map shown is exactly the quantity that measurement scored. Drawing it in a
way that looked like a mask would misrepresent what the eye computes.

The prediction layer is **constant velocity over the drawn path**, so it is a
display convention rather than the DPC model of §9.32 -- which measured +0.085
at one step but −0.030 on free rollout, and is not what draws this arrow.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

#: Every layer this module can draw. Passing a subset to :func:`draw_percept`
#: switches the rest off.
LAYERS = ("attention", "box", "label", "path", "prediction", "relation",
          "grid", "lost")

_PALETTE = ((255, 92, 92), (92, 200, 255), (140, 240, 140), (255, 205, 80),
            (220, 140, 255), (255, 160, 90))


def _to_rgb(frame: np.ndarray) -> np.ndarray:
    a = np.asarray(frame, np.float32)
    if a.ndim == 3:
        a = a.mean(0) if a.shape[0] in (1, 3) else a.mean(2)
    if a.max() > 1.5:
        a = a / 255.0
    a = np.clip(a, 0.0, 1.0)
    return np.repeat((a * 255).astype(np.uint8)[..., None], 3, axis=2)


def _upsample(m: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    yi = np.clip((np.arange(shape[0]) * m.shape[0]) // max(shape[0], 1),
                 0, m.shape[0] - 1)
    xi = np.clip((np.arange(shape[1]) * m.shape[1]) // max(shape[1], 1),
                 0, m.shape[1] - 1)
    return m[np.ix_(yi, xi)]


def _heat(v: np.ndarray) -> np.ndarray:
    """A blue->yellow->red ramp. Kept monotone in luminance so it survives a
    greyscale print, which a rainbow ramp does not."""
    v = np.clip(v, 0.0, 1.0)
    r = np.clip(2.2 * v - 0.6, 0, 1)
    g = np.clip(1.8 * v - 0.25, 0, 1) * np.clip(2.0 - 1.6 * v, 0, 1)
    b = np.clip(1.1 - 2.2 * v, 0, 1)
    return np.stack([r, g, b], -1)


def draw_percept(frame: np.ndarray, percept, history=None,
                 show: Optional[Iterable[str]] = None,
                 colours: Optional[Dict[str, Tuple[int, int, int]]] = None,
                 alpha: float = 0.45, scale: int = 1,
                 truth: Optional[Dict[str, Tuple[float, float, float, float]]]
                 = None) -> np.ndarray:
    """Render ``percept`` over ``frame``. Returns an RGB uint8 image.

    ``history`` is an optional list of earlier percepts, oldest first, used by
    the ``path`` and ``prediction`` layers.
    ``show`` selects layers; omit it for everything.

    ``truth`` draws ground-truth boxes ``{name: (y, x, h, w)}`` in white beside
    the eye's own, so the fit can be judged from the picture rather than taken
    on trust. Whether the box lands on the object is the whole question, and a
    render with only the prediction in it cannot answer it.
    """
    show = set(LAYERS if show is None else show)
    img = _to_rgb(frame)
    H, W = img.shape[:2]
    names = sorted(percept.where)
    colours = dict(colours or {})
    for i, n in enumerate(names):
        colours.setdefault(n, _PALETTE[i % len(_PALETTE)])

    # --- attention: the correlation field that decided where ----------------
    if "attention" in show and percept.attention:
        acc = np.zeros((H, W, 3), np.float32)
        wt = np.zeros((H, W), np.float32)
        for n in names:
            a = percept.attention.get(n)
            if a is None:
                continue
            u = _upsample(np.asarray(a, np.float32), (H, W))
            m = u.max()
            u = u / m if m > 1e-9 else u
            acc += _heat(u) * u[..., None]
            wt += u
        m = wt.max()
        if m > 1e-9:
            wt = wt / m
            acc = acc / np.maximum(wt[..., None], 1e-6)
            img = (img * (1 - alpha * wt[..., None])
                   + acc * 255 * alpha * wt[..., None]).astype(np.uint8)

    # --- grid: the resolution the eye actually sees at -----------------------
    if "grid" in show and getattr(percept, "maps", None):
        m = next(iter(percept.maps.values()))
        gh, gw = m.shape[-2:]
        # Draw at most ~16 lines each way. A 103x103 map drawn cell by cell is
        # a solid mesh that hides the frame underneath -- it shows the grid
        # exists without showing anything about the picture.
        sr, sc = max(1, gh // 16), max(1, gw // 16)
        img = img.copy()
        for r in range(0, gh, sr):
            y = int(r * H / gh)
            if 0 <= y < H:
                img[y, :, :] = (img[y, :, :] * 0.6 + 90).astype(np.uint8)
        for c in range(0, gw, sc):
            x = int(c * W / gw)
            if 0 <= x < W:
                img[:, x, :] = (img[:, x, :] * 0.6 + 90).astype(np.uint8)

    from PIL import Image, ImageDraw, ImageFont
    pil = Image.fromarray(img)
    if scale > 1:
        pil = pil.resize((W * scale, H * scale), Image.NEAREST)
    d = ImageDraw.Draw(pil, "RGBA")
    S = scale
    fs = max(11, 7 * S)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", fs)
    except OSError:
        font = ImageFont.load_default()

    def _text(xy, txt, fg, bg):
        """Text on its own plate, clamped inside the frame so a label near an
        edge is still readable rather than half-drawn off it."""
        x, y = xy
        try:
            l, t, r, b = d.textbbox((0, 0), txt, font=font)
            tw, th = r - l, b - t
        except AttributeError:
            tw, th = len(txt) * fs // 2, fs
        x = min(max(0, x), pil.size[0] - tw - 4)
        y = min(max(0, y), pil.size[1] - th - 4)
        d.rectangle([x, y, x + tw + 4, y + th + 4], fill=bg)
        d.text((x + 2, y + 2), txt, fill=fg, font=font)
        return th + 4

    def P(pt):
        return (pt[1] * S, pt[0] * S)

    # --- path and prediction -------------------------------------------------
    hist: List = list(history or [])
    for n in names:
        col = colours[n]
        track = [h.where[n] for h in hist if n in getattr(h, "where", {})]
        track.append(percept.where[n])
        if "path" in show and len(track) > 1:
            d.line([P(p) for p in track], fill=col + (200,), width=max(1, S))
            for p in track[:-1]:
                x, y = P(p)
                d.ellipse([x - 2 * S, y - 2 * S, x + 2 * S, y + 2 * S],
                          fill=col + (140,))
        if "prediction" in show and len(track) >= 3:
            v = np.asarray(track[-1]) - np.asarray(track[-3])
            nxt = np.asarray(track[-1]) + v / 2.0
            d.line([P(track[-1]), P(nxt)], fill=col + (255,),
                   width=max(1, 2 * S))
            x, y = P(nxt)
            r = 5 * S
            d.ellipse([x - r, y - r, x + r, y + r], outline=col + (255,),
                      width=max(1, S))
            d.line([(x - r, y), (x + r, y)], fill=col + (255,))
            d.line([(x, y - r), (x, y + r)], fill=col + (255,))

    # --- boxes and labels ----------------------------------------------------
    for n in names:
        col = colours[n]
        y, x = percept.where[n]
        bh = bw = 44.0
        tag = getattr(percept, "_boxes", {}).get(n)
        if tag:
            bh, bw = tag
        x0, y0 = (x - bw / 2) * S, (y - bh / 2) * S
        x1, y1 = (x + bw / 2) * S, (y + bh / 2) * S
        if "box" in show:
            d.rectangle([x0, y0, x1, y1], outline=col + (255,),
                        width=max(1, S))
        if "label" in show:
            conf = percept.confidence.get(n, 0.0)
            _text((x0, y0 - fs - 6), f"{n} {conf:.2f}", (10, 12, 16, 255),
                  col + (225,))

    # --- ground truth, for judging the fit by eye ---------------------------
    if truth:
        for n, (ty, tx, th_, tw_) in truth.items():
            x0, y0 = (tx - tw_ / 2) * S, (ty - th_ / 2) * S
            x1, y1 = (tx + tw_ / 2) * S, (ty + th_ / 2) * S
            for k in range(0, int(max(x1 - x0, y1 - y0)), 8 * S):
                d.line([(x0 + k, y0), (min(x0 + k + 4 * S, x1), y0)],
                       fill=(255, 255, 255, 230), width=max(1, S))
                d.line([(x0 + k, y1), (min(x0 + k + 4 * S, x1), y1)],
                       fill=(255, 255, 255, 230), width=max(1, S))
                d.line([(x0, y0 + k), (x0, min(y0 + k + 4 * S, y1))],
                       fill=(255, 255, 255, 230), width=max(1, S))
                d.line([(x1, y0 + k), (x1, min(y0 + k + 4 * S, y1))],
                       fill=(255, 255, 255, 230), width=max(1, S))
            _text((x1 + 2, y0), f"truth:{n}", (20, 22, 28, 255),
                  (255, 255, 255, 220))

    # --- relations -----------------------------------------------------------
    if "relation" in show:
        for (a, b), r in sorted(getattr(percept, "relations", {}).items()):
            if a not in percept.where or b not in percept.where:
                continue
            pa, pb = P(percept.where[a]), P(percept.where[b])
            d.line([pa, pb], fill=(235, 235, 245, 150), width=max(1, S))
            mx, my = (pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2
            _text((mx, my - fs), f"{a} {r} {b}", (225, 230, 240, 255),
                  (18, 20, 26, 205))

    # --- lost: say so, do not draw a box ------------------------------------
    if "lost" in show and getattr(percept, "lost", None):
        y = 4
        for n, c in sorted(percept.lost.items()):
            y += _text((4, y), f"{n}: LOST ({c:.2f} < threshold)",
                       (255, 235, 235, 255), (150, 40, 40, 210)) + 2

    if getattr(percept, "frame_size", None) is not None:
        _text((4, pil.size[1] - 22), f"frame log-size {percept.frame_size:.2f}",
              (230, 235, 245, 235), (18, 20, 26, 190))
    return np.asarray(pil)


def annotate_boxes(percept, tags) -> None:
    """Attach each tag's own (h, w) so :func:`draw_percept` boxes match it."""
    percept._boxes = {n: (t.box[2], t.box[3]) for n, t in tags.items()}


def strip(frames: Sequence[np.ndarray], pad: int = 4,
          bg: int = 18) -> np.ndarray:
    """Lay images side by side into one wide image."""
    ims = [np.asarray(f, np.uint8) for f in frames]
    h = max(i.shape[0] for i in ims)
    w = sum(i.shape[1] for i in ims) + pad * (len(ims) + 1)
    out = np.full((h + 2 * pad, w, 3), bg, np.uint8)
    x = pad
    for i in ims:
        out[pad:pad + i.shape[0], x:x + i.shape[1]] = i
        x += i.shape[1] + pad
    return out
