"""A picture of what this project does: input video in, overlays and numbers out.

Produces JPG panels showing the whole path through the system on **real video**:

    01_input          the frames as they arrive
    02_layers         each overlay alone, so it is clear what each contributes
    03_sequence       the loop running: overlays on consecutive frames
    04_areas          what each cortical area actually sees, per frame
    05_measurements   the numbers, beside the pictures they came from

Nothing here is a new measurement. Every number plotted is produced by the same
`UnifiedEye` the benchmarks measure, on the same footage, so the pictures and
EVALUATION.md cannot drift apart.

Usage:  python3 benchmarks/eye_visual_report.py [outdir] [vot_dir] [seq]
"""
import os
import sys

import numpy as np

from neurobrain.viz.overlay import LAYERS, annotate_boxes, draw_percept, strip
from neurobrain.vision.unified_eye import UnifiedEye

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eye_vot import load_gt, resize                           # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "reports/eye_loop"
VOT_DIR = sys.argv[2] if len(sys.argv) > 2 else "vot"
SIZE = 224
OBJ = 44


def load_seq(d, name, n=14):
    from PIL import Image
    p = os.path.join(d, name)
    jpgs = sorted(f for f in os.listdir(p) if f.endswith(".jpg"))
    gt = load_gt(os.path.join(p, "groundtruth.txt"))
    step = max(1, len(jpgs) // n)
    out = []
    for i in range(0, min(len(jpgs), len(gt)), step):
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
        ty, tx = (y + h / 2) * sy, (x + w / 2) * sx
        if not (OBJ < ty < SIZE - OBJ and OBJ < tx < SIZE - OBJ):
            continue
        out.append((f, (ty, tx), float(np.log(h * sy * w * sx))))
        if len(out) >= n:
            break
    return out


def save(img, path, title=None):
    from PIL import Image, ImageDraw
    a = np.asarray(img, np.uint8)
    if title:
        pil = Image.new("RGB", (a.shape[1], a.shape[0] + 22), (18, 20, 26))
        pil.paste(Image.fromarray(a), (0, 22))
        ImageDraw.Draw(pil).text((6, 6), title, fill=(228, 233, 242))
    else:
        pil = Image.fromarray(a)
    pil.save(path, quality=92)
    print(f"  wrote {path}  ({pil.size[0]}x{pil.size[1]})", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    if not os.path.isdir(VOT_DIR):
        print(f"no video at {VOT_DIR!r} -- nothing to report.")
        return
    names = sorted(d for d in os.listdir(VOT_DIR)
                   if os.path.isdir(os.path.join(VOT_DIR, d)))
    seq = sys.argv[3] if len(sys.argv) > 3 else None
    picked, frames = None, None
    for n in ([seq] if seq else names):
        s = load_seq(VOT_DIR, n)
        if s and len(s) >= 8:
            picked, frames = n, s
            break
    if not frames:
        print("no usable sequence -- nothing to report.")
        return
    print(f"sequence '{picked}', {len(frames)} frames at {SIZE}px\n", flush=True)

    eye = UnifiedEye(size=SIZE)
    eye.develop([f for f, _, _ in frames])
    eye.fit_size([f for f, _, _ in frames], [a for _, _, a in frames])
    f0, (ty0, tx0), _ = frames[0]
    eye.add_tag("target", f0, (ty0 - OBJ / 2, tx0 - OBJ / 2, OBJ, OBJ))
    # a second tag somewhere else, so relations have something to relate
    eye.add_tag("corner", f0, (12, 12, OBJ, OBJ))

    percepts, hist = [], []
    for f, _, _ in frames:
        p = eye.look(f)
        annotate_boxes(p, eye.tags)
        percepts.append(p)

    # 01 -- the input
    save(strip([np.repeat((np.clip(f, 0, 1) * 255).astype(np.uint8)[..., None],
                          3, 2) for f, _, _ in frames[:8]]),
         os.path.join(OUT, "01_input.jpg"),
         f"INPUT  --  {picked}, {SIZE}px, every frame as it arrives")

    # 02 -- each layer alone
    mid = len(frames) // 2
    tiles = []
    for layer in LAYERS:
        h = percepts[max(0, mid - 4):mid]
        tiles.append(draw_percept(frames[mid][0], percepts[mid], history=h,
                                  show=[layer], scale=2))
    save(strip(tiles[:4]), os.path.join(OUT, "02_layers_a.jpg"),
         "LAYERS, one at a time  --  " + " | ".join(LAYERS[:4]))
    save(strip(tiles[4:]), os.path.join(OUT, "02_layers_b.jpg"),
         "LAYERS, one at a time  --  " + " | ".join(LAYERS[4:]))

    # 03 -- the loop running, everything on
    run = []
    for i in range(min(6, len(frames))):
        j = i + max(0, len(frames) - 6)
        run.append(draw_percept(frames[j][0], percepts[j],
                                history=percepts[max(0, j - 5):j], scale=2))
    save(strip(run), os.path.join(OUT, "03_sequence.jpg"),
         "THE LOOP RUNNING  --  every overlay, consecutive frames, "
         "one forward pass each")

    # 04 -- what each area sees
    P = percepts[mid]
    tiles = []
    from PIL import Image
    for a, m in P.maps.items():
        e = np.linalg.norm(np.asarray(m, np.float32), axis=0)
        e = e / (e.max() + 1e-9)
        im = Image.fromarray((e * 255).astype(np.uint8)).resize(
            (SIZE * 2, SIZE * 2), Image.NEAREST).convert("RGB")
        tiles.append(np.asarray(im))
    save(strip(tiles), os.path.join(OUT, "04_areas.jpg"),
         "WHAT EACH AREA SEES  --  " + " | ".join(
             f"{a} {tuple(m.shape)}" for a, m in P.maps.items()))

    # 05 -- the numbers, beside the pictures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    truth = np.array([[t[0], t[1]] for _, t, _ in frames])
    got = np.array([p.where.get("target", (np.nan, np.nan))
                    for p in percepts])
    err = np.hypot(got[:, 0] - truth[:, 0], got[:, 1] - truth[:, 1])
    conf = np.array([p.confidence.get("target", np.nan) for p in percepts])
    fsize = np.array([p.frame_size if p.frame_size is not None else np.nan
                      for p in percepts])
    true_a = np.array([a for _, _, a in frames])

    fig, ax = plt.subplots(1, 4, figsize=(19, 3.9), facecolor="#12141a")
    for a in ax:
        a.set_facecolor("#171a21")
        a.tick_params(colors="#9aa4b5")
        for s in a.spines.values():
            s.set_color("#2a2f3a")
    ax[0].plot(truth[:, 1], truth[:, 0], "-o", color="#5cc8ff", label="truth",
               ms=3)
    ax[0].plot(got[:, 1], got[:, 0], "-o", color="#ff5c5c", label="eye", ms=3)
    ax[0].invert_yaxis()
    ax[0].set_title("path: truth vs eye", color="#e4e9f2")
    ax[0].legend(facecolor="#171a21", labelcolor="#cfd6e4", edgecolor="#2a2f3a")
    ax[1].plot(err, color="#ffcd50")
    ax[1].axhline(OBJ / 2, color="#8cf08c", ls="--", label=f"hit ≤{OBJ//2}px")
    ax[1].set_title("centre error (px)", color="#e4e9f2")
    ax[1].legend(facecolor="#171a21", labelcolor="#cfd6e4", edgecolor="#2a2f3a")
    ax[2].plot(conf, color="#dc8cff")
    ax[2].set_title("match confidence", color="#e4e9f2")
    ax[2].set_ylim(0, 1)
    ax[3].plot(true_a, color="#5cc8ff", label="true log-size")
    ax[3].plot(fsize, color="#ff5c5c", label="read out")
    ax[3].set_title("angular size", color="#e4e9f2")
    ax[3].legend(facecolor="#171a21", labelcolor="#cfd6e4", edgecolor="#2a2f3a")
    for a in ax:
        a.set_xlabel("frame", color="#9aa4b5")
    fig.suptitle(f"MEASUREMENTS  --  {picked}: hit rate "
                 f"{np.mean(err <= OBJ/2):.2f}, median error "
                 f"{np.nanmedian(err):.0f}px, mean confidence "
                 f"{np.nanmean(conf):.2f}", color="#e4e9f2")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "05_measurements.jpg"), dpi=110,
                facecolor="#12141a")
    print(f"  wrote {os.path.join(OUT, '05_measurements.jpg')}")
    plt.close(fig)

    print(f"\nhit rate {np.mean(err <= OBJ/2):.3f}   median error "
          f"{np.nanmedian(err):.1f}px   mean confidence {np.nanmean(conf):.3f}")
    print(f"report in {OUT}/")


if __name__ == "__main__":
    main()
