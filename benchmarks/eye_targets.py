"""The same eye on many kinds of target: human, ball, car, bird, animal, object.

One eye, one mechanism, ten different things to follow. Each sequence gets its
own developed eye (§9.27: this code does not leave the footage it grew on), one
tag placed on the ground-truth box of frame 0, and then consecutive frames --
the regime §9.35 measured at 0.814 against 0.194 for large jumps.

Produces, per target:

    a strip of tracked frames, ground truth drawn in white dashes beside the
    eye's own box, so the fit is judged from the picture;
    coverage (how often it answered rather than saying LOST),
    precision (of the answers, how many were on the object),
    median IoU between the drawn box and the true one.

And one legend image that says what every mark on those pictures means, because
a reader should not have to infer it.

What this cannot show, and no run of it will
--------------------------------------------
There is **no detector here**. The tracked box comes from a tag placed by hand
on frame 0, and nothing in this project can name what it is following: no object
classes, no labels.

`propose` and `segment` now mark untagged regions too, and they are neither
detection nor YOLO:

* **proposals** are *objectness* -- "this region is unlike its surroundings" --
  with no class attached. Measured against ground truth they contain the object
  in **0.31** of frames at top-10, against **0.11** for the same number of
  boxes placed at random: real signal, about 3x chance, and far from a detector.
* **segments** group map cells by what they respond to. That is closer to
  superpixels than to YOLO's instance masks, which come from a network trained
  on labelled instances; one object here is routinely split across labels.

Usage:  python3 benchmarks/eye_targets.py [outdir] [vot_dir]
"""
import json
import os
import sys

import numpy as np

from neurobrain.viz.overlay import LAYERS, annotate_boxes, draw_percept, strip
from neurobrain.vision.unified_eye import Percept, UnifiedEye

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eye_vot import load_gt, resize                           # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "reports/eye_targets"
VOT_DIR = sys.argv[2] if len(sys.argv) > 2 else "vot"
SIZE = 224
N_FRAME = 40

#: what each sequence actually contains, so the panels can be read
TARGETS = [
    ("bolt1", "human -- sprinter"),
    ("pedestrian1", "human -- walking"),
    ("ball2", "ball -- football"),
    ("soccer1", "ball -- in a crowd"),
    ("car1", "car"),
    ("birds1", "bird"),
    ("rabbit", "animal -- rabbit"),
    ("tiger", "animal -- tiger"),
    ("book", "object -- book"),
    ("frisbee", "object -- frisbee"),
]


def load_seq(d, name, n=N_FRAME):
    from PIL import Image
    p = os.path.join(d, name)
    if not os.path.isdir(p):
        return []
    jpgs = sorted(f for f in os.listdir(p) if f.endswith(".jpg"))
    gtp = os.path.join(p, "groundtruth.txt")
    if not jpgs or not os.path.exists(gtp):
        return []
    gt = load_gt(gtp)
    out = []
    for i in range(min(len(jpgs), len(gt))):
        if gt[i] is None:
            continue
        y, x, h, w = gt[i]
        if h < 3 or w < 3:
            continue
        a = np.asarray(Image.open(os.path.join(p, jpgs[i])).convert("L"),
                       np.float32) / 255.0
        sy, sx = SIZE / a.shape[0], SIZE / a.shape[1]
        f = resize(a, sy, sx)
        if f.shape != (SIZE, SIZE):
            continue
        out.append((f, ((y + h / 2) * sy, (x + w / 2) * sx,
                        max(12.0, h * sy), max(12.0, w * sx))))
        if len(out) >= n:
            break
    return out


def save(img, path, title=None):
    from PIL import Image, ImageDraw, ImageFont
    a = np.asarray(img, np.uint8)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    if title:
        pil = Image.new("RGB", (a.shape[1], a.shape[0] + 26), (18, 20, 26))
        pil.paste(Image.fromarray(a), (0, 26))
        ImageDraw.Draw(pil).text((8, 6), title, fill=(228, 233, 242),
                                 font=font)
    else:
        pil = Image.fromarray(a)
    pil.save(path, quality=92)
    print(f"  wrote {path}", flush=True)


def legend(path):
    """One picture that says what every mark means."""
    from PIL import Image, ImageDraw, ImageFont
    W, H = 1180, 300
    pil = Image.new("RGB", (W, H), (18, 20, 26))
    d = ImageDraw.Draw(pil, "RGBA")
    try:
        big = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 17)
        f = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except OSError:
        big = f = ImageFont.load_default()
    d.text((16, 12), "WHAT EVERY MARK MEANS", fill=(235, 240, 250), font=big)
    rows = [
        ((255, 92, 92), "solid coloured box",
         "where the eye reports the tagged thing is. One box per tag -- "
         "nothing else is marked."),
        ((255, 255, 255), "white dashed box",
         "ground truth, from the dataset's human annotation. The two should "
         "coincide."),
        ((255, 205, 80), "text  name 0.42",
         "the tag's name and its match confidence (the correlation peak, 0-1)."),
        ((150, 40, 40), "red banner  LOST",
         "the eye looked and was NOT confident enough to answer. No box is "
         "drawn. §9.35: this is what lifts precision 0.848 -> 0.910."),
        ((255, 140, 60), "orange/yellow heat",
         "the template-correlation field -- how well the tag matches at each "
         "place. NOT a segmentation mask; nothing here segments."),
        ((92, 200, 255), "thin line + dots",
         "the path the tag has taken over previous frames."),
        ((140, 240, 140), "circle with a cross",
         "predicted next position, by CONSTANT VELOCITY over that path -- a "
         "display convention, not the learned model (§9.32: -0.030 rollout)."),
        ((160, 168, 182), "faint grid",
         "the cortical map's cell size -- the resolution the eye truly sees "
         "at, coarser than the frame."),
    ]
    y = 44
    for col, name, txt in rows:
        d.rectangle([18, y + 2, 40, y + 16], outline=col + (255,), width=2)
        d.text((52, y), name, fill=col + (255,), font=big)
        d.text((250, y + 1), txt, fill=(176, 186, 202), font=f)
        y += 31
    d.text((16, H - 26),
           "No detector: every box comes from a tag placed by hand on frame 0. "
           "Other people/cars/objects are never marked.",
           fill=(232, 176, 176), font=f)
    pil.save(path, quality=94)
    print(f"  wrote {path}", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    if not os.path.isdir(VOT_DIR):
        print(f"no video at {VOT_DIR!r} -- nothing to report.")
        return
    legend(os.path.join(OUT, "00_legend.jpg"))

    rows, panels = [], []
    for name, what in TARGETS:
        fr = load_seq(VOT_DIR, name)
        if len(fr) < 12:
            print(f"  {name}: only {len(fr)} usable frames -- skipped")
            continue
        eye = UnifiedEye(size=SIZE)
        eye.develop([f for f, _ in fr])
        y0, x0, h0, w0 = fr[0][1]
        eye.add_tag(name, fr[0][0], (y0 - h0 / 2, x0 - w0 / 2, h0, w0))

        percepts, errs, ious, ans = [], [], [], []
        for f, (ty, tx, th, tw) in fr:
            p = eye.look(f, relations=False)
            annotate_boxes(p, eye.tags)
            percepts.append(p)
            if name in p.where:
                py, px = p.where[name]
                e = float(np.hypot(py - ty, px - tx))
                inter = (max(0.0, (h0 + th) / 2 - abs(py - ty))
                         * max(0.0, (w0 + tw) / 2 - abs(px - tx)))
                u = h0 * w0 + th * tw - inter
                errs.append(e)
                ious.append(inter / u if u > 0 else 0.0)
                ans.append(1.0)
            else:
                ans.append(0.0)
        travel = float(np.hypot(fr[-1][1][0] - fr[0][1][0],
                                fr[-1][1][1] - fr[0][1][1]))
        step = float(np.mean([np.hypot(fr[i + 1][1][0] - fr[i][1][0],
                                       fr[i + 1][1][1] - fr[i][1][1])
                              for i in range(len(fr) - 1)]))
        cov = float(np.mean(ans))
        prec = float(np.mean(np.asarray(errs) <= h0 / 2)) if errs else 0.0
        miou = float(np.median(ious)) if ious else 0.0
        rows.append({"sequence": name, "target": what, "frames": len(fr),
                     "travel_px": round(travel, 1),
                     "px_per_frame": round(step, 1),
                     "coverage": round(cov, 3), "precision": round(prec, 3),
                     "median_iou": round(miou, 3)})
        print(f"  {name:<12} {what:<22} cover {cov:.2f}  prec {prec:.2f}  "
              f"IoU {miou:.2f}  ({step:.1f}px/frame, {travel:.0f}px total)",
              flush=True)

        idx = np.linspace(0, len(fr) - 1, 6).astype(int)
        tiles = []
        for j in idx:
            ty, tx, th, tw = fr[j][1]
            tiles.append(draw_percept(fr[j][0], percepts[j],
                                      history=percepts[max(0, j - 6):j],
                                      scale=2, truth={name: (ty, tx, th, tw)}))
        panels.append((name, what, strip(tiles), cov, prec, miou, step))

    for name, what, img, cov, prec, miou, step in panels:
        save(img, os.path.join(OUT, f"target_{name}.jpg"),
             f"{what.upper()}  ({name})  --  answered {cov:.0%}, right "
             f"{prec:.0%} of those, median IoU {miou:.2f}, "
             f"{step:.0f}px/frame motion")

    # --- the three new read-outs, on one frame each ------------------------
    fr = load_seq(VOT_DIR, TARGETS[4][0], 14) or load_seq(VOT_DIR,
                                                          TARGETS[0][0], 14)
    if fr:
        eye = UnifiedEye(size=SIZE)
        eye.develop([f for f, _ in fr])
        y0, x0, h0, w0 = fr[0][1]
        eye.add_tag("tagged", fr[0][0], (y0 - h0 / 2, x0 - w0 / 2, h0, w0))
        j = len(fr) // 2
        p6 = eye.look(fr[j][0], relations=False, n_proposals=6, n_segments=5)
        annotate_boxes(p6, eye.tags)
        tiles = [
            draw_percept(fr[j][0], p6, show=["box", "label"], scale=2),
            draw_percept(fr[j][0], p6, show=["proposals"], scale=2),
            draw_percept(fr[j][0], p6, show=["segments"], scale=2),
            draw_percept(fr[j][0], p6,
                         show=["segments", "proposals", "box", "label"],
                         scale=2)]
        save(strip(tiles), os.path.join(OUT, "01_new_layers.jpg"),
             "TAGGED TARGET  |  PROPOSALS (objectness, no labels -- finds the "
             "object 0.31 of frames in top-10 vs 0.11 random)  |  SEGMENTS "
             "(feature grouping, NOT YOLO instance masks)  |  ALL")
        rep = eye.features(fr[j][0])
        json.dump(rep, open(os.path.join(OUT, "features.json"), "w"), indent=1)
        print("\nfeature extraction, probed with gratings:")
        for a, v in rep.items():
            print(f"  {a:<12} {v['channels']:>4} channels, "
                  f"{v['oriented_fraction']:.0%} orientation-selective, "
                  f"mean selectivity {v['mean_selectivity']:.2f}")
            for c in v["top"][:3]:
                print(f"      ch{c['channel']:<4} {c['kind']:<14} "
                      f"prefers {c['prefers_deg']:>5.1f}deg  "
                      f"sel {c['selectivity']:.2f}")

    if rows:
        print(f"\n{'sequence':<13}{'target':<24}{'px/f':>6}{'cover':>8}"
              f"{'prec':>7}{'IoU':>7}")
        for r in sorted(rows, key=lambda r: -r["precision"]):
            print(f"{r['sequence']:<13}{r['target']:<24}"
                  f"{r['px_per_frame']:>6.1f}{r['coverage']:>8.2f}"
                  f"{r['precision']:>7.2f}{r['median_iou']:>7.2f}")
        p = np.array([r["precision"] for r in rows])
        c = np.array([r["coverage"] for r in rows])
        print(f"\n{len(rows)} targets: precision {p.mean():.2f} "
              f"(worst {p.min():.2f}, best {p.max():.2f}), "
              f"coverage {c.mean():.2f}")
        json.dump({"targets": rows,
                   "mean_precision": round(float(p.mean()), 4),
                   "mean_coverage": round(float(c.mean()), 4)},
                  open(os.path.join(OUT, "targets.json"), "w"), indent=1)
    print(f"\nreport in {OUT}/")


if __name__ == "__main__":
    main()
