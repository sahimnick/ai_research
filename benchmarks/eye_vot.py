"""The eye on VOT2019: a real tracking benchmark, run the way it is meant to be.

§9.23 scored tag-driven localisation with the *metrics* a tracking benchmark
uses, on live CCTV. It could not be compared to any published number, because no
benchmark **dataset** was involved. This closes that: VOT2019 sequences, real
30 fps video, human-annotated ground truth, one-pass evaluation.

What this is, and is emphatically not
-------------------------------------
The tracker here is a **fixed template**: tag the target in frame 0, then find
it again in every later frame by normalised cross-correlation over the area's
map. No model update, no scale search, no motion model, no re-detection. That is
a deliberately weak tracker by VOT standards -- roughly the classical NCC
baseline -- and its absolute scores will be far below a modern tracker's.

So this measures **whether the eye's representation supports localisation on
real benchmark video**, with the eye, raw pixels and a shuffled control all run
through the identical protocol. It does not measure how this project compares to
the state of the art in tracking, and no such claim should be read out of it.

Protocol
--------
One-pass evaluation (OPE), the standard: initialise from ground truth on the
first frame, then track using only the tracker's own predictions. Error
accumulates, which is the point.

Every sequence is rescaled so its target is ~``TARGET`` px, so a 1280x720 frame
with a small object and an 800x336 frame with a large one present the eye with
the same geometry; the search window is then a constant ``SEARCH`` px around the
previous prediction. Without that, "which sequence" and "how big the object is"
would be the same variable.

Reported as the two standard summaries -- precision at 20 px and success AUC
over IoU thresholds 0..1 -- per sequence and overall, against:

    pixels-NCC     the same template matching on raw luminance
    shuffled-tag   a DIFFERENT sequence's tag, tracked through this sequence:
                   keeps the protocol and the search dynamics, destroys only
                   the correspondence

Usage:  python3 benchmarks/eye_vot.py out_eye_vot.json [vot_dir] [n_seq]
"""
import json
import os
import sys

import numpy as np

from neurobrain.vision.ventral import build_ventral_stream_on, locate_template

def _arg(i, cast, default):
    """argv read defensively: this module is also IMPORTED (by
    eye_track_drift), and then argv belongs to the importing benchmark."""
    try:
        return cast(sys.argv[i])
    except (IndexError, ValueError, TypeError):
        return default


VOT_DIR = _arg(2, str, "vot")
N_SEQ = _arg(3, int, 12)
TARGET = 40          # every target rescaled to about this many pixels
SEARCH = 160         # the window the eye sees, centred on the last prediction
STRIDE = 2           # every STRIDE-th frame, to keep the run affordable
STAGES = ("V2", "V4")


def load_gt(path):
    """VOT ground truth: 8-number rotated polygons or 4-number boxes."""
    out = []
    for line in open(path):
        v = [float(x) for x in line.replace(",", " ").split()]
        if len(v) == 8:
            xs, ys = v[0::2], v[1::2]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        elif len(v) == 4:
            x0, y0, x1, y1 = v[0], v[1], v[0] + v[2], v[1] + v[3]
        else:
            out.append(None)
            continue
        out.append((y0, x0, y1 - y0, x1 - x0))     # (y, x, h, w)
    return out


def resize(a, sy, sx):
    """Bilinear resize -- no scipy in this project, and nearest would alias."""
    h, w = a.shape
    nh, nw = max(1, int(round(h * sy))), max(1, int(round(w * sx)))
    yi = np.linspace(0, h - 1, nh)
    xi = np.linspace(0, w - 1, nw)
    y0 = np.floor(yi).astype(int)
    x0 = np.floor(xi).astype(int)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (yi - y0)[:, None]
    wx = (xi - x0)[None, :]
    return ((a[np.ix_(y0, x0)] * (1 - wy) * (1 - wx)
             + a[np.ix_(y1, x0)] * wy * (1 - wx)
             + a[np.ix_(y0, x1)] * (1 - wy) * wx
             + a[np.ix_(y1, x1)] * wy * wx).astype(np.float32))


def load_seq(d, stride):
    from PIL import Image
    jpgs = sorted(f for f in os.listdir(d) if f.endswith(".jpg"))
    gt = load_gt(os.path.join(d, "groundtruth.txt"))
    if not jpgs or len(gt) < len(jpgs):
        return None
    g0 = gt[0]
    if g0 is None or min(g0[2], g0[3]) < 4:
        return None
    s = TARGET / max(g0[2], g0[3])
    if not (0.02 < s < 4.0):
        return None
    frames, boxes = [], []
    for i in range(0, len(jpgs), stride):
        if gt[i] is None:
            continue
        a = np.asarray(Image.open(os.path.join(d, jpgs[i])).convert("L"),
                       np.float32) / 255.0
        frames.append(resize(a, s, s))
        y, x, h, w = gt[i]
        boxes.append((y * s, x * s, h * s, w * s))
    return (frames, boxes) if len(frames) >= 8 else None


def window(f, cy, cx, size):
    """A ``size`` crop centred on (cy, cx), clipped into the frame."""
    h, w = f.shape
    y0 = int(np.clip(round(cy - size / 2), 0, max(0, h - size)))
    x0 = int(np.clip(round(cx - size / 2), 0, max(0, w - size)))
    return f[y0:y0 + size, x0:x0 + size], y0, x0


def area_map(stream, view, area):
    h = stream.hierarchy
    names = [getattr(l, "name", type(l).__name__) for l in h.layers]
    h.forward(view[None])
    return np.asarray(h.layers[names.index(area)].log["output"], np.float32)


def patch_of(m, y, x, bh, bw, img):
    _, gh, gw = m.shape
    r0, c0 = int(np.clip(y * gh / img, 0, gh - 1)), int(np.clip(x * gw / img,
                                                                0, gw - 1))
    r1 = int(np.clip(np.ceil((y + bh) * gh / img), r0 + 1, gh))
    c1 = int(np.clip(np.ceil((x + bw) * gw / img), c0 + 1, gw))
    return m[:, r0:r1, c0:c1]


def ncc(view, tmpl):
    ph, pw = tmpl.shape
    if view.shape[0] < ph or view.shape[1] < pw:
        return None
    p = tmpl - tmpl.mean()
    pn = np.linalg.norm(p) + 1e-9
    W = np.lib.stride_tricks.sliding_window_view(view, (ph, pw))
    Wc = W - W.mean((2, 3), keepdims=True)
    num = np.einsum("ijkl,kl->ij", Wc, p)
    den = np.sqrt(np.einsum("ijkl,ijkl->ij", Wc, Wc)) * pn + 1e-9
    r, c = np.unravel_index(int(np.argmax(num / den)), num.shape)
    return r + ph / 2.0, c + pw / 2.0


def iou(a, b):
    ay, ax, ah, aw = a
    by, bx, bh, bw = b
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    inter = iy * ix
    u = ah * aw + bh * bw - inter
    return inter / u if u > 0 else 0.0


def track(stream, frames, boxes, mode, tag_src=None):
    """One-pass evaluation. Returns per-frame centre error and IoU."""
    src = tag_src if tag_src is not None else (frames, boxes)
    f0, b0 = src[0][0], src[1][0]
    y0, x0, bh, bw = b0
    v0, wy, wx = window(f0, y0 + bh / 2, x0 + bw / 2, SEARCH)
    if mode == "pixels":
        yy = int(np.clip(round(y0 - wy), 0, max(0, v0.shape[0] - 2)))
        xx = int(np.clip(round(x0 - wx), 0, max(0, v0.shape[1] - 2)))
        tmpl = v0[yy:yy + max(2, int(round(bh))),
                  xx:xx + max(2, int(round(bw)))].copy()
    else:
        m0 = area_map(stream, v0, mode)
        tmpl = patch_of(m0, y0 - wy, x0 - wx, bh, bw, SEARCH)

    th, tw = boxes[0][2], boxes[0][3]
    cy, cx = boxes[0][0] + th / 2, boxes[0][1] + tw / 2
    errs, ious = [], []
    for t in range(1, len(frames)):
        view, wy, wx = window(frames[t], cy, cx, SEARCH)
        if min(view.shape) < 8:
            break
        if mode == "pixels":
            r = ncc(view, tmpl)
            if r is None:
                break
            py, px = r
        else:
            m = area_map(stream, view, mode)
            if m.shape[1] < tmpl.shape[1] or m.shape[2] < tmpl.shape[2]:
                break
            (rr, cc), _ = locate_template(m, tmpl)
            gh, gw = m.shape[1], m.shape[2]
            py = (rr + (tmpl.shape[1] - 1) / 2.0) * view.shape[0] / gh
            px = (cc + (tmpl.shape[2] - 1) / 2.0) * view.shape[1] / gw
        cy, cx = wy + py, wx + px
        gy, gx, gh_, gw_ = boxes[t]
        errs.append(float(np.hypot(cy - (gy + gh_ / 2), cx - (gx + gw_ / 2))))
        ious.append(iou((cy - th / 2, cx - tw / 2, th, tw), boxes[t]))
    return errs, ious


def summarise(errs, ious):
    e, i = np.asarray(errs), np.asarray(ious)
    thr = np.linspace(0.0, 1.0, 101)
    return (float(np.mean(e <= 20.0)),
            float(np.mean([(i >= t).mean() for t in thr])),
            float(np.median(e)))


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "out_eye_vot.json"
    if not os.path.isdir(VOT_DIR):
        print(f"no VOT directory at {VOT_DIR!r} -- nothing reported.")
        json.dump({"error": "no vot dir", "path": VOT_DIR},
                  open(out_path, "w"), indent=1)
        return
    names = sorted(d for d in os.listdir(VOT_DIR)
                   if os.path.isdir(os.path.join(VOT_DIR, d)))[:N_SEQ]
    print(f"loading {len(names)} VOT2019 sequences from {VOT_DIR} "
          f"(every {STRIDE}nd frame) ...", flush=True)
    seqs = {}
    for n in names:
        s = load_seq(os.path.join(VOT_DIR, n), STRIDE)
        if s is not None:
            seqs[n] = s
            print(f"  {n:<12} {len(s[0]):>4} frames, target "
                  f"{s[1][0][2]:.0f}x{s[1][0][3]:.0f}px after rescale",
                  flush=True)
    if len(seqs) < 4:
        print(f"only {len(seqs)} usable sequences -- nothing reported.")
        json.dump({"error": "too few sequences", "n": len(seqs)},
                  open(out_path, "w"), indent=1)
        return

    print(f"\ngrowing the eye on first frames ({SEARCH}px windows) ...",
          flush=True)
    firsts = []
    for fr, bx in seqs.values():
        v, _, _ = window(fr[0], bx[0][0] + bx[0][2] / 2,
                         bx[0][1] + bx[0][3] / 2, SEARCH)
        if v.shape == (SEARCH, SEARCH):
            firsts.append(v)
    stream = build_ventral_stream_on(np.stack(firsts), size=SEARCH,
                                     verbose=False)

    arms = list(STAGES) + ["pixels", "shuffled"]
    per = {a: {} for a in arms}
    keys = list(seqs)
    for si, n in enumerate(keys):
        fr, bx = seqs[n]
        for a in arms:
            if a == "shuffled":
                other = seqs[keys[(si + 1) % len(keys)]]
                e, i = track(stream, fr, bx, "V2", tag_src=other)
            else:
                e, i = track(stream, fr, bx, a)
            per[a][n] = summarise(e, i) if e else (0.0, 0.0, float("nan"))
        print(f"  {n:<12} " + "  ".join(
            f"{a} {per[a][n][1]:.3f}" for a in arms), flush=True)

    res = {"n_sequences": len(seqs), "target_px": TARGET, "search_px": SEARCH,
           "frame_stride": STRIDE, "protocol": "OPE, fixed template, no update",
           "per_sequence": {a: {n: [round(v, 4) for v in per[a][n]]
                                for n in per[a]} for a in arms},
           "overall": {}}

    print(f"\n{'arm':<12}{'prec@20px':>11}{'success AUC':>13}"
          f"{'median err':>12}")
    for a in arms:
        p = float(np.mean([per[a][n][0] for n in per[a]]))
        u = float(np.mean([per[a][n][1] for n in per[a]]))
        m = float(np.nanmedian([per[a][n][2] for n in per[a]]))
        res["overall"][a] = {"precision_20px": round(p, 4),
                             "success_auc": round(u, 4),
                             "median_err_px": round(m, 2)}
        print(f"{a:<12}{p:>11.3f}{u:>13.3f}{m:>12.1f}")

    best = max(STAGES, key=lambda s: res["overall"][s]["success_auc"])
    ba = res["overall"][best]["success_auc"]
    sa = res["overall"]["shuffled"]["success_auc"]
    pa = res["overall"]["pixels"]["success_auc"]
    res["best_stage"] = best
    res["beats_shuffled"] = bool(ba > sa + 0.05)
    res["beats_pixels"] = bool(ba > pa + 0.02)

    print(f"\n--- the eye on a real tracking benchmark ---")
    print(f"  best stage {best}: success AUC {ba:.3f}   "
          f"shuffled {sa:.3f}   pixels-NCC {pa:.3f}")
    if res["beats_shuffled"]:
        print(f"  The eye's map localises on real benchmark video, above a "
              f"shuffled tag. The representation\n  carries what tracking "
              f"needs; whether the TRACKER is any good is a different "
              f"question, and\n  this one has no model update, no scale search "
              f"and no re-detection.")
    else:
        print(f"  The eye does NOT clear the shuffled control on real "
              f"benchmark video. §9.23's CCTV result\n  does not transfer: a "
              f"static camera with a 105s gap is a far easier problem than "
              f"30fps video\n  with a moving object.")
    print(f"  Absolute numbers here are NOT comparable to published VOT "
          f"trackers -- this is a fixed\n  template, the weakest tracker "
          f"there is, and the point is the arms' relative standing.")

    json.dump(res, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
