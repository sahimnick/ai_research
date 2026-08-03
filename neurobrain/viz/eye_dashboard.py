"""
eye_dashboard.py
================

Watch the eye work, live, with every overlay switchable while it runs.

    from neurobrain.viz.eye_dashboard import serve_eye
    serve_eye("webcam")            # your laptop camera
    serve_eye("cctv")              # live traffic cameras
    serve_eye("/path/to/frames")   # a directory of images

Then open http://127.0.0.1:8137. The page shows the processed view and a
checkbox per overlay -- attention, box, label, path, prediction, relation,
grid -- so any one or all of them can be turned on **while looking at the same
frame**, which is the only way to see what each contributes.

Click the picture to tag whatever is under the cursor. The eye then looks for
that thing in every later frame, which is the loop of §9.34 driven by hand
instead of by a benchmark.

Honest about the picture
------------------------
The attention layer is a **template-correlation field**, not a segmentation
mask -- nothing in this project segments. The prediction cross is constant
velocity over the drawn path, a display convention, not the DPC model of §9.32
(+0.085 one step, −0.030 on free rollout).

And the measured limits apply to what you will see: this eye does not leave the
distribution it developed on (§9.27), and it localises coarsely rather than
precisely (§9.30). Point it at your desk, let it develop there, and it will
work far better on your desk than on the street outside.
"""
from __future__ import annotations

import io
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import List, Optional

import numpy as np

from .overlay import LAYERS, annotate_boxes, draw_percept
from ..vision.unified_eye import UnifiedEye

DEFAULT_PORT = 8137


class _Source:
    """Frames from a webcam, live CCTV, or a directory -- one interface."""

    def __init__(self, spec: str = "webcam"):
        self.spec, self.kind, self._i, self._files = spec, None, 0, []
        if spec == "webcam":
            from ..sensing.live import Webcam
            self.cam = Webcam()
            ok, why = self.cam.available()
            if not ok:
                raise RuntimeError(f"webcam unavailable: {why}")
            self.kind = "webcam"
        elif spec == "cctv":
            from ..sensing.live import CctvCamera, NY511_LIVE_IDS
            self.cam = CctvCamera(ids=list(NY511_LIVE_IDS))
            self.kind = "cctv"
        elif os.path.isdir(spec):
            self._files = sorted(
                os.path.join(spec, f) for f in os.listdir(spec)
                if f.lower().endswith((".jpg", ".jpeg", ".png")))
            if not self._files:
                raise RuntimeError(f"no images in {spec}")
            self.kind = "dir"
        else:
            raise RuntimeError(f"unknown source {spec!r}")

    def read(self) -> np.ndarray:
        if self.kind == "webcam":
            a = np.asarray(self.cam.read().data, np.float32)
        elif self.kind == "cctv":
            for i in range(len(self.cam.urls)):
                r = self.cam.read((self._i + i) % len(self.cam.urls))
                if getattr(r, "live", False):
                    self._i = (self._i + i + 1) % len(self.cam.urls)
                    a = np.asarray(r.data, np.float32)
                    break
            else:
                raise RuntimeError("no live camera")
        else:
            from PIL import Image
            p = self._files[self._i % len(self._files)]
            self._i += 1
            a = np.asarray(Image.open(p).convert("RGB"), np.float32)
        if a.ndim == 3:
            a = a.mean(2) if a.shape[2] in (3, 4) else a.mean(0)
        return a / 255.0 if a.max() > 1.5 else a


class EyeService:
    """One eye, one frame source, and the switches the page drives."""

    def __init__(self, source: str = "webcam", size: int = 224,
                 develop_frames: int = 12):
        self.src = _Source(source)
        self.size = size
        self.eye = UnifiedEye(size=size)
        self.layers = set(LAYERS) - {"grid", "segments"}
        self.cue = None            # None = report every tag
        self.n_proposals = 6
        self.n_segments = 5
        self.history: List = []
        self.frame = None
        self.percept = None
        self.lock = threading.Lock()
        self.status = "developing"
        self._develop(develop_frames)
        self.status = "running"

    def _develop(self, n: int):
        frames = []
        for _ in range(n):
            try:
                frames.append(self.src.read())
            except Exception:
                time.sleep(0.2)
        if not frames:
            raise RuntimeError("no frames from the source to develop on")
        self.eye.develop(frames)

    def step(self):
        f = self.src.read()
        p = self.eye.look(
            f, cue=self.cue,
            n_proposals=self.n_proposals if "proposals" in self.layers else 0,
            n_segments=self.n_segments if "segments" in self.layers else 0)
        annotate_boxes(p, self.eye.tags)
        with self.lock:
            self.frame, self.percept = f, p
            self.history.append(p)
            self.history = self.history[-24:]

    def render(self, scale: int = 2) -> bytes:
        with self.lock:
            f, p, h = self.frame, self.percept, list(self.history[:-1])
        if f is None:
            self.step()
            with self.lock:
                f, p, h = self.frame, self.percept, []
        img = draw_percept(f, p, history=h, show=self.layers, scale=scale)
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, "JPEG", quality=82)
        return buf.getvalue()

    def add_tag(self, name: str, y: float, x: float, side: float = 44.0):
        with self.lock:
            f = self.frame
        if f is None:
            return {"error": "no frame yet"}
        y = float(np.clip(y - side / 2, 0, self.size - side))
        x = float(np.clip(x - side / 2, 0, self.size - side))
        self.eye.add_tag(name, f, (y, x, side, side))
        return {"ok": True, "tags": sorted(self.eye.tags)}

    def state(self) -> dict:
        with self.lock:
            p = self.percept
        if p is None:
            return {"status": self.status, "tags": sorted(self.eye.tags)}
        return {"status": self.status,
                "min_confidence": self.eye.min_confidence,
                "cue": self.cue,
                "proposals": [{k: round(v, 1) if isinstance(v, float) else v
                               for k, v in b.items()} for b in p.proposals],
                "lost": {k: round(v, 3) for k, v in p.lost.items()},
                "tags": sorted(self.eye.tags),
                "layers": sorted(self.layers),
                "where": {k: [round(v[0], 1), round(v[1], 1)]
                          for k, v in p.where.items()},
                "confidence": {k: round(v, 3)
                               for k, v in p.confidence.items()},
                "relations": {f"{a}|{b}": r
                              for (a, b), r in p.relations.items()},
                "frame_size": (round(p.frame_size, 3)
                               if p.frame_size is not None else None),
                "describe": p.describe()}


def _handler(svc: EyeService):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            b = body if isinstance(body, bytes) else str(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            p = self.path.split("?")[0]
            if p == "/":
                return self._send(200, PAGE, "text/html; charset=utf-8")
            if p == "/frame.jpg":
                try:
                    svc.step()
                    return self._send(200, svc.render(), "image/jpeg")
                except Exception as e:
                    return self._send(503, json.dumps({"error": str(e)}))
            if p == "/state":
                return self._send(200, json.dumps(svc.state()))
            if p == "/features":
                try:
                    with svc.lock:
                        fr = svc.frame
                    return self._send(200, json.dumps(svc.eye.features(fr)))
                except Exception as e:
                    return self._send(503, json.dumps({"error": str(e)}))
            self._send(404, json.dumps({"error": "not found"}))

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                body = {}
            p = self.path.split("?")[0]
            if p == "/layers":
                svc.layers = {l for l in body.get("layers", [])
                              if l in LAYERS}
                return self._send(200, json.dumps(
                    {"layers": sorted(svc.layers)}))
            if p == "/tag":
                return self._send(200, json.dumps(svc.add_tag(
                    str(body.get("name", f"tag{len(svc.eye.tags)+1}")),
                    float(body.get("y", 0)), float(body.get("x", 0)),
                    float(body.get("side", 44)))))
            if p == "/threshold":
                svc.eye.min_confidence = float(body.get("value", 0.35))
                return self._send(200, json.dumps(
                    {"min_confidence": svc.eye.min_confidence}))
            if p == "/cue":
                v = body.get("value") or None
                svc.cue = v if (v in svc.eye.tags) else None
                return self._send(200, json.dumps({"cue": svc.cue}))
            if p == "/clear":
                svc.eye.tags.clear()
                svc.history.clear()
                return self._send(200, json.dumps({"ok": True}))
            self._send(404, json.dumps({"error": "not found"}))
    return H


def serve_eye(source: str = "webcam", port: int = DEFAULT_PORT,
              size: int = 224, develop_frames: int = 12,
              open_browser: bool = False):
    """Develop an eye on ``source``, then serve the live view on ``port``."""
    print(f"developing the eye on {develop_frames} frames from {source!r} ...",
          flush=True)
    svc = EyeService(source, size=size, develop_frames=develop_frames)
    httpd = HTTPServer(("127.0.0.1", port), _handler(svc))
    url = f"http://127.0.0.1:{port}"
    print(f"eye dashboard on {url}\n  click the picture to tag something; "
          f"toggle overlays on the right", flush=True)
    if open_browser:
        import webbrowser
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    return svc


PAGE = ("""<!doctype html><html><head><meta charset="utf-8">
<title>NeuroBrain -- the eye, live</title><style>
:root{--bg:#0e1117;--pan:#161a22;--line:#242a36;--fg:#e4e9f2;--dim:#93a0b4;
--acc:#5cc8ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:13px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{padding:10px 16px;border-bottom:1px solid var(--line);
display:flex;gap:14px;align-items:baseline}
h1{font-size:15px;margin:0;font-weight:600}
.sub{color:var(--dim);font-size:12px}
.wrap{display:grid;grid-template-columns:1fr 320px;gap:14px;padding:14px}
#view{width:100%;background:#000;border:1px solid var(--line);border-radius:8px;
cursor:crosshair;display:block}
.card{background:var(--pan);border:1px solid var(--line);border-radius:8px;
padding:12px;margin-bottom:12px}
.card h2{font-size:12px;margin:0 0 8px;color:var(--dim);font-weight:600;
letter-spacing:.06em;text-transform:uppercase}
label.row{display:flex;align-items:center;gap:8px;padding:3px 0;cursor:pointer}
input[type=checkbox]{accent-color:var(--acc);width:15px;height:15px}
.hint{color:var(--dim);font-size:11.5px}
button{background:#1d2430;color:var(--fg);border:1px solid var(--line);
border-radius:6px;padding:6px 10px;cursor:pointer;font-size:12px}
button:hover{border-color:var(--acc)}
pre{margin:0;white-space:pre-wrap;word-break:break-word;color:var(--dim);
font:11.5px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace}
kbd{background:#1d2430;border:1px solid var(--line);border-radius:4px;
padding:1px 5px;font:11px ui-monospace,monospace}
@media(max-width:900px){.wrap{grid-template-columns:1fr}}
</style></head><body>
<header><h1>the eye, live</h1>
<span class="sub">one forward pass per frame &mdash; every overlay below comes
out of it</span></header>
<div class="wrap">
  <div><img id="view" src="/frame.jpg" alt="processed view"></div>
  <div>
    <div class="card"><h2>overlays</h2><div id="lay"></div>
      <div class="hint" style="margin-top:8px">
      <b>attention</b> is a template-correlation field, not a segmentation
      mask. <b>prediction</b> is constant velocity over the drawn path.</div>
    </div>
    <div class="card"><h2>confidence gate</h2>
      <div class="hint">Below this the eye reports <b>LOST</b> instead of
      drawing a box. Measured (&sect;9.35): always answering = 0.779 correct;
      gate at 0.35 = <b>0.910</b> correct on the half it answers.</div>
      <input id="thr" type="range" min="0" max="0.9" step="0.05" value="0.35"
        style="width:100%;accent-color:var(--acc);margin-top:8px"
        oninput="document.getElementById('tv').textContent=this.value;
                 fetch('/threshold',{method:'POST',
                 body:JSON.stringify({value:parseFloat(this.value)})})">
      <div class="hint">threshold = <span id="tv">0.35</span></div></div>
    <div class="card"><h2>target</h2>
      <div class="hint">Which tag to follow. <b>all</b> reports every tag at
      once.</div>
      <select id="cue" onchange="fetch('/cue',{method:'POST',
        body:JSON.stringify({value:this.value})})"
        style="width:100%;margin-top:8px;background:#1d2430;color:var(--fg);
        border:1px solid var(--line);border-radius:6px;padding:6px">
        <option value="">all</option></select></div>
    <div class="card"><h2>features extracted</h2>
      <div class="hint">What each area is tuned to, <b>probed</b> with oriented
      gratings rather than assumed.</div>
      <button style="margin-top:8px" onclick="feats()">measure now</button>
      <pre id="ft" style="margin-top:8px"></pre></div>
    <div class="card"><h2>tags</h2>
      <div class="hint">Click the picture to tag what is under the cursor.
      The eye looks for it in every later frame.</div>
      <div style="margin-top:8px;display:flex;gap:6px">
        <button onclick="clr()">clear tags</button>
        <button onclick="tick()">step now</button></div>
      <pre id="tags" style="margin-top:8px"></pre></div>
    <div class="card"><h2>read-out</h2><pre id="st"></pre></div>
    <div class="card"><h2>measured limits</h2>
      <pre>does not leave the distribution it developed on (9.27)
localises coarsely, not precisely (9.30)
colour hurts it (9.18)
a shadow looks like an object to it (9.31)
stages not shown to specialise (9.33)</pre></div>
  </div>
</div>
<script>
const LAYERS=""" + json.dumps(list(LAYERS)) + """;
const OFF=new Set(["grid"]);
const box=document.getElementById('lay');
LAYERS.forEach(l=>{const w=document.createElement('label');w.className='row';
 const c=document.createElement('input');c.type='checkbox';c.value=l;
 c.checked=!OFF.has(l);c.onchange=push;
 w.appendChild(c);w.appendChild(document.createTextNode(l));box.appendChild(w);});
function push(){const on=[...box.querySelectorAll('input:checked')].map(c=>c.value);
 fetch('/layers',{method:'POST',body:JSON.stringify({layers:on})});}
const view=document.getElementById('view');
view.addEventListener('click',e=>{const r=view.getBoundingClientRect();
 const y=(e.clientY-r.top)/r.height*""" + "224" + """;
 const x=(e.clientX-r.left)/r.width*""" + "224" + """;
 const name=prompt('name this tag','tag'+(Date.now()%1000));
 if(!name)return;
 fetch('/tag',{method:'POST',body:JSON.stringify({name:name,y:y,x:x})})
  .then(r=>r.json()).then(refresh);});
function clr(){fetch('/clear',{method:'POST'}).then(refresh);}
function tick(){view.src='/frame.jpg?t='+Date.now();}
function refresh(){fetch('/state').then(r=>r.json()).then(s=>{
 document.getElementById('st').textContent=JSON.stringify(s,null,1);
 document.getElementById('tags').textContent=(s.tags||[]).join('\\n')||'(none)';
});}
setInterval(tick,700); setInterval(refresh,900); refresh();
</script></body></html>""").encode()


if __name__ == "__main__":                                   # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser(
        description="Watch the eye work, live, with switchable overlays.")
    ap.add_argument("source", nargs="?", default="webcam",
                    help="'webcam', 'cctv', or a directory of images")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--develop", type=int, default=12,
                    help="frames to develop the eye on before serving")
    a = ap.parse_args()
    serve_eye(a.source, port=a.port, size=a.size, develop_frames=a.develop)
