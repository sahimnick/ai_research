"""
console.py
==========

**Mission control for the whole brain** -- a live graph of the running system
with a management dashboard beside it.

:mod:`dashboard` already served the low-level spiking :class:`~neurobrain.brain.Brain`.
This is a different thing: it drives the **integrated** brain -- saccadic eye,
wide spiking V1, cochlea, auditory belt, the shared cortical workspace, memory
and the predictive world model -- while it is actually running on sensory
streams, and shows what each part is doing as it happens.

What it shows
-------------
* **The graph.** Every module is a node, every information path an edge. Nodes
  light up with their current activity and carry their live read-outs (cells
  firing, code sparsity, categories grown). The edges show what is actually
  flowing, so a module that has quietly stopped contributing is visible rather
  than assumed healthy.
* **The senses.** The scene with the eye's scan-path drawn on it, the current
  foveal image, the peripheral saliency map, and the rolling cochleagram with
  detected onsets marked.
* **The percept.** The shared workspace code as a raster, the name read out of
  it (or ``None``, which is the honest answer when nothing matches), and the
  running prediction error.
* **Counters.** Spikes per tick, saccades, onsets found, bindings formed,
  wall-clock rate -- the numbers you need to tell "thinking" from "hung".

Running it
----------
    from neurobrain.console import serve_console
    serve_console(port=8080)               # then open http://localhost:8080

or from the shell::

    python -m neurobrain.console --port 8080

The server is the standard library's ``http.server`` and the page is one
self-contained file with no external assets, so it works offline.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _u8(a: np.ndarray, lo: Optional[float] = None,
        hi: Optional[float] = None) -> List[int]:
    """Quantise an array to 0..255 for cheap transport to the browser."""
    a = np.asarray(a, np.float32)
    lo = float(a.min()) if lo is None else lo
    hi = float(a.max()) if hi is None else hi
    if hi - lo < 1e-9:
        return [0] * a.size
    return np.clip((a - lo) * 255.0 / (hi - lo), 0, 255
                   ).astype(np.uint8).reshape(-1).tolist()


# ---------------------------------------------------------------------------
# The live brain behind the dashboard
# ---------------------------------------------------------------------------
@dataclass
class NodeState:
    """One module in the graph, and how alive it is right now."""

    key: str
    label: str
    group: str
    x: float
    y: float
    activity: float = 0.0
    detail: str = ""


class BrainConsole:
    """Drives the integrated brain on live streams and reports its state.

    One :meth:`tick` is one perceptual cycle: the eye makes a saccade, foveates,
    and integrates a 50 ms window over its fixational drift; the ear advances
    along the soundscape and reports any onset it finds; both codes enter the
    shared workspace; the percept is named if anything matches, and the world
    model's prediction for the next step is scored against what arrives."""

    def __init__(self, scene_size: int = 256, n_objects: int = 12,
                 v1_cells: int = 1024, n_events: int = 24, seed: int = 0,
                 n_train_images: int = 600):
        from .streams import (build_scene, SaccadicEye, build_soundscape,
                              StreamingBrain)
        from .realworld import load_mnist
        from .widev1 import _unit

        self.seed = seed
        trx, trY, _, _ = load_mnist(n_train=n_train_images, n_test=10)
        self.scene = build_scene(trx, trY, size=scene_size,
                                 n_objects=n_objects, seed=seed)
        self.eye = SaccadicEye(self.scene, seed=seed)
        self.brain = StreamingBrain(v1_cells=v1_cells, seed=seed)
        self.scape = build_soundscape(n_events=n_events, seed=seed)
        self.onsets = self.brain.ear.detect_onsets(self.scape.wave)

        # A named memory for the workspace to match against. It is grown the
        # way the rest of the project grows categories -- ART vigilance on the
        # code, labels applied afterwards purely to NAME what grew. Ten averaged
        # prototypes (one per digit) were tried first and read out at 25%; the
        # grown map is the same read-out every other measurement in the project
        # uses, so the console shows a number comparable to them.
        from .spikingvision import SpikingCategoryMap
        from .widev1 import VIGILANCE_LADDER

        self._unit = _unit
        n_mem = min(500, len(trx))
        # The memory MUST be built through the same path the percept takes.
        # Built from static rate() codes while the percept came from
        # rate_over() on a drifting fixation, the two distributions did not
        # match and the console read out one label for everything (16.7%).
        # So the training images are drifted here exactly as the eye drifts.
        def _as_seen(im):
            fr = [np.roll(np.roll(np.asarray(im, np.float32),
                                  int(self.eye.rng.integers(-1, 2)), axis=0),
                          int(self.eye.rng.integers(-1, 2)), axis=1)
                  for _ in range(self.eye.n_drift_frames)]
            return self._unit(self.brain.v1.rate_over(fr))

        codes = np.array([self.brain.ws.encode("vision", _as_seen(im))
                          for im in trx[:n_mem]], np.float32)
        # Two read-outs are kept. The console DISPLAYS the class-prototype one
        # because it is simply the better decoder for this code -- measured on
        # freely-viewed fixations, prototypes 66.7% against 38.3% for the grown
        # map -- but it is supervised, so it is labelled as such on the page.
        # The project's unsupervised numbers live in
        # :func:`~neurobrain.streams.streaming_experiment`, not here.
        self.prototypes = np.stack([
            _unit(codes[trY[:n_mem] == d].mean(0)) if (trY[:n_mem] == d).any()
            else np.zeros(codes.shape[1], np.float32) for d in range(10)])
        self.cortex = SpikingCategoryMap(dim=codes.shape[1], vigilance=0.10,
                                         max_cells=4000)
        for v in VIGILANCE_LADDER:
            m = SpikingCategoryMap(dim=codes.shape[1], vigilance=v,
                                   max_cells=4000)
            for x in codes:
                m.learn(x)
            if m.n_categories <= n_mem / 3.0:
                m.name_cells(codes, trY[:n_mem])
                self.cortex = m
        if not self.cortex.n_categories:         # never leave it mute
            for x in codes:
                self.cortex.learn(x)
            self.cortex.name_cells(codes, trY[:n_mem])
        for i in range(min(50, n_mem)):
            self.brain.ws.learn_concept(str(int(trY[i])), codes[i])

        self.running = False
        self.tick_count = 0
        self.saccades = 0
        self.onsets_heard = 0
        self.bindings = 0
        self.correct = 0
        self.scored = 0
        self.spikes_last = 0.0
        self.rate_hz = 0.0
        self.log: List[str] = []
        self.path: List[Tuple[int, int]] = []
        self.last_fovea = np.zeros((28, 28), np.float32)
        self.last_code = np.zeros(self.brain.ws.dim, np.float32)
        self.last_name: Optional[str] = None
        self.last_truth: Optional[int] = None
        self.last_grown: Optional[str] = None
        self.last_coch = np.zeros((self.brain.ear.coch.n_freq, 40), np.float32)
        self._lock = threading.Lock()
        self._t0 = time.time()

    # -- one perceptual cycle ----------------------------------------------
    def tick(self) -> None:
        with self._lock:
            self._tick_locked()

    def _tick_locked(self) -> None:
        t0 = time.time()
        self.tick_count += 1

        # --- the eye ------------------------------------------------------
        r, c = self.eye.next_target()
        r, c = self.eye.foveate(r, c)
        fix = self.eye.fixate(r, c)
        self.saccades += 1
        self.path.append((int(r), int(c)))
        self.path = self.path[-40:]
        self.last_fovea = np.asarray(fix.frames[len(fix.frames) // 2],
                                     np.float32)

        rate = self.brain.v1.rate_over(fix.frames)
        self.spikes_last = float(rate.sum() * self.brain.v1.window_ms / 1000.0)
        v_code = self.brain.ws.encode("vision", self._unit(rate))
        self.brain.ws.broadcast(v_code, source="vision")
        self.last_code = v_code
        sim = self.prototypes @ v_code
        self.last_name = str(int(sim.argmax())) if float(sim.max()) > 0 else None
        self.last_grown = (str(int(self.cortex.recognise(v_code[None])[0]))
                           if self.cortex.n_categories else None)
        self.last_truth = fix.true_label
        if fix.true_label >= 0:
            self.scored += 1
            self.correct += int(self.last_name == str(fix.true_label))

        # --- the ear ------------------------------------------------------
        if self.onsets:
            at = self.onsets[self.tick_count % len(self.onsets)]
            coch = self.brain.ear.listen(self.scape.wave, at)
            self.last_coch = coch
            a_code = self.brain.ws.encode("sound", self.brain.belt.code(coch))
            self.onsets_heard += 1
            if fix.true_label >= 0:
                self.brain.bind(v_code, a_code, fix.true_label)
                self.bindings += 1

        what = self.last_name if self.last_name is not None else "-"
        truth = fix.true_label if fix.true_label >= 0 else "bg"
        self._say(f"saccade -> ({r},{c})  sees '{what}'  (really {truth})")
        dt = max(time.time() - t0, 1e-6)
        self.rate_hz = 1.0 / dt

    def _say(self, msg: str) -> None:
        self.log.append(f"[{self.tick_count:05d}] {msg}")
        self.log = self.log[-14:]

    # -- background thread --------------------------------------------------
    def start(self, hz: float = 4.0) -> None:
        if self.running:
            return
        self.running = True

        def loop():
            while self.running:
                self.tick()
                time.sleep(max(0.0, 1.0 / hz))

        threading.Thread(target=loop, daemon=True).start()

    def stop(self) -> None:
        self.running = False

    # -- the graph ----------------------------------------------------------
    def graph(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Modules as nodes, information paths as edges, with live activity."""
        v1 = self.brain.v1
        ws = self.brain.ws
        code_on = float((self.last_code > 0).mean())
        specs = [
            NodeState("scene", "Scene / world", "input", 0.05, 0.50,
                      1.0, f"{self.scene.shape[0]}x{self.scene.shape[1]} px, "
                           f"{len(self.scene.positions)} objects"),
            NodeState("retina", "Retina (fovea+periphery)", "sensor",
                      0.22, 0.28, min(1.0, self.last_fovea.mean() / 60.0),
                      f"fovea {self.eye.fovea}px, drift "
                      f"{self.eye.n_drift_frames} frames"),
            NodeState("colliculus", "Superior colliculus (saccades)", "sensor",
                      0.22, 0.60, min(1.0, self.saccades / 50.0),
                      f"{self.saccades} saccades, IOR on"),
            NodeState("cochlea", "Cochlea", "sensor", 0.22, 0.86,
                      float(self.last_coch.mean()),
                      f"{self.brain.ear.coch.n_freq} bands"),
            NodeState("v1", "Wide spiking V1", "cortex", 0.44, 0.28,
                      min(1.0, self.spikes_last / 2000.0),
                      f"{v1.n_cells} cells, {v1.per_column}/column, "
                      f"{v1.rf}x{v1.rf} RF, {v1.window_ms}ms"),
            NodeState("belt", "Auditory belt", "cortex", 0.44, 0.86,
                      float(self.last_coch.mean()),
                      f"{self.brain.belt.n_groups} shift-invariant units"),
            NodeState("workspace", "Global workspace (shared code)", "core",
                      0.64, 0.55, code_on * 6.0,
                      f"dim {ws.dim}, {self.cortex.n_categories} categories, "
                      f"{code_on:.0%} active"),
            NodeState("memory", "Cross-modal bindings", "memory", 0.84, 0.80,
                      min(1.0, self.bindings / 30.0),
                      f"{self.bindings} bindings"),
            NodeState("naming", "Name read-out (class prototypes)", "output",
                      0.84, 0.32,
                      1.0 if self.last_name else 0.15,
                      f"'{self.last_name}'" if self.last_name
                      else "None (no match -- honest)"),
            NodeState("worldmodel", "Predictive world model", "core",
                      0.64, 0.15, min(1.0, self.tick_count / 60.0),
                      f"accuracy {self.accuracy():.0%} over {self.scored}"),
        ]
        edges = [("scene", "retina"), ("scene", "colliculus"),
                 ("scene", "cochlea"), ("retina", "v1"),
                 ("colliculus", "retina"), ("cochlea", "belt"),
                 ("v1", "workspace"), ("belt", "workspace"),
                 ("workspace", "naming"), ("workspace", "memory"),
                 ("workspace", "worldmodel"), ("worldmodel", "colliculus"),
                 ("memory", "workspace")]
        nodes = [dict(id=n.key, label=n.label, group=n.group, x=n.x, y=n.y,
                      activity=round(min(max(n.activity, 0.0), 1.0), 3),
                      detail=n.detail) for n in specs]
        return nodes, [dict(source=a, target=b) for a, b in edges]

    def accuracy(self) -> float:
        return self.correct / self.scored if self.scored else 0.0

    # -- everything the page needs -----------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            nodes, edges = self.graph()
            sal = self.eye._saliency
            return {
                "running": self.running,
                "tick": self.tick_count,
                "nodes": nodes,
                "edges": edges,
                "metrics": {
                    "saccades": self.saccades,
                    "onsets": self.onsets_heard,
                    "bindings": self.bindings,
                    "spikes_last_tick": round(self.spikes_last),
                    "recognition": round(self.accuracy(), 3),
                    "scored": self.scored,
                    "tick_rate_hz": round(self.rate_hz, 1),
                    "uptime_s": round(time.time() - self._t0, 1),
                    "v1_cells": self.brain.v1.n_cells,
                    "workspace_dim": self.brain.ws.dim,
                    "concepts": self.cortex.n_categories,
                },
                "fovea": {"w": self.last_fovea.shape[1],
                          "h": self.last_fovea.shape[0],
                          "px": _u8(self.last_fovea, 0.0, 255.0)},
                "saliency": {"w": sal.shape[1], "h": sal.shape[0],
                             "px": _u8(sal)},
                "cochlea": {"w": self.last_coch.shape[1],
                            "h": self.last_coch.shape[0],
                            "px": _u8(self.last_coch)},
                "code": _u8(self.last_code),
                "path": [{"r": r, "c": c} for r, c in self.path],
                "scene_size": int(self.scene.shape[0]),
                "objects": [{"r": int(r), "c": int(c), "label": int(l)}
                            for (r, c), l in zip(self.scene.positions,
                                                 self.scene.labels)],
                "name": self.last_name,
                "truth": (self.last_truth if self.last_truth is not None
                          and self.last_truth >= 0 else None),
                "log": list(reversed(self.log)),
            }


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------
class _NpJSON(json.JSONEncoder):
    """NumPy scalars are not JSON-serializable; unwrap them on the way out.

    Every metric here comes from a float32 array at some point, so without this
    the /state endpoint dies on the first request."""

    def default(self, o):                        # noqa: D102
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return super().default(o)


def _dump(obj: Any) -> bytes:
    return json.dumps(obj, cls=_NpJSON).encode()


def _handler(console: "BrainConsole"):
    from http.server import BaseHTTPRequestHandler
    from urllib.parse import urlparse, parse_qs

    class H(BaseHTTPRequestHandler):
        def _send(self, body: bytes, ctype: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):                       # noqa: N802
            u = urlparse(self.path)
            if u.path in ("/", "/index.html"):
                self._send(PAGE.encode(), "text/html; charset=utf-8")
            elif u.path == "/state":
                self._send(_dump(console.snapshot()), "application/json")
            elif u.path == "/cmd":
                q = parse_qs(u.query)
                a = (q.get("do") or [""])[0]
                if a == "start":
                    console.start(hz=float((q.get("hz") or ["4"])[0]))
                elif a == "stop":
                    console.stop()
                elif a == "step":
                    console.tick()
                self._send(_dump({"ok": True, "running": console.running}),
                           "application/json")
            else:
                self.send_error(404)

        def log_message(self, *a):              # keep the terminal readable
            pass

    return H


def serve_console(port: int = 8080, host: str = "127.0.0.1",
                  autostart: bool = True, hz: float = 4.0,
                  console: Optional[BrainConsole] = None,
                  blocking: bool = True) -> BrainConsole:
    """Build the integrated brain, start it, and serve the dashboard."""
    from http.server import HTTPServer

    c = console or BrainConsole()
    if autostart:
        c.start(hz=hz)
    srv = HTTPServer((host, port), _handler(c))
    print(f"NeuroBrain console on http://{host}:{port}  "
          f"({c.brain.v1.n_cells} V1 cells, workspace dim {c.brain.ws.dim})")
    if blocking:
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            c.stop()
            print("\nstopped.")
    else:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    return c


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>NeuroBrain console</title>
<style>
:root{--bg:#0b1020;--fg:#dce6ff;--dim:#7c8bb5;--line:#1d2949;--card:#111936;
      --ok:#4ade80;--warn:#fbbf24}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);
  font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
header{padding:10px 16px;border-bottom:1px solid var(--line);display:flex;
  gap:16px;align-items:center;flex-wrap:wrap}
h1{font-size:15px;margin:0;font-weight:600;letter-spacing:.3px}
button{background:var(--card);color:var(--fg);border:1px solid var(--line);
  border-radius:6px;padding:5px 12px;cursor:pointer;font:inherit}
button:hover{border-color:var(--dim)}
.wrap{display:grid;grid-template-columns:1fr 340px;gap:12px;padding:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
  padding:10px 12px}
.card h2{font-size:11px;margin:0 0 8px;color:var(--dim);font-weight:600;
  text-transform:uppercase;letter-spacing:.8px}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(96px,1fr));
  gap:8px}
.metric b{display:block;font-size:17px;font-weight:600}
.metric span{color:var(--dim);font-size:10px;text-transform:uppercase}
canvas{width:100%;image-rendering:pixelated;border-radius:6px;
  background:#060a16;display:block}
.row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
#log{height:150px;overflow:auto;font-size:11px;color:var(--dim);
  white-space:pre-wrap}
.pill{padding:2px 8px;border-radius:99px;border:1px solid var(--line);
  font-size:11px}
.on{color:var(--ok);border-color:var(--ok)} .off{color:var(--warn)}
@media(max-width:900px){.wrap{grid-template-columns:1fr}}
</style></head><body>
<header>
  <h1>NeuroBrain &mdash; live console</h1>
  <span id="state" class="pill off">idle</span>
  <button onclick="cmd('start')">run</button>
  <button onclick="cmd('stop')">pause</button>
  <button onclick="cmd('step')">step</button>
  <span id="up" style="color:var(--dim)"></span>
</header>
<div class="wrap">
  <div>
    <div class="card"><h2>brain graph &mdash; modules and information flow</h2>
      <canvas id="g" height="420"></canvas></div>
    <div class="card" style="margin-top:12px"><h2>scene &amp; scan path</h2>
      <canvas id="scene" height="300"></canvas></div>
  </div>
  <div>
    <div class="card"><h2>vitals</h2><div class="metrics" id="m"></div></div>
    <div class="card" style="margin-top:12px"><h2>percept</h2>
      <div style="font-size:22px;font-weight:600" id="pc">&mdash;</div>
      <div style="color:var(--dim)" id="tr"></div>
      <canvas id="code" height="46" style="margin-top:8px"></canvas></div>
    <div class="card" style="margin-top:12px"><h2>sensors</h2>
      <div class="row">
        <div><div style="color:var(--dim);font-size:10px">fovea</div>
          <canvas id="fov" height="120"></canvas></div>
        <div><div style="color:var(--dim);font-size:10px">peripheral saliency</div>
          <canvas id="sal" height="120"></canvas></div>
      </div>
      <div style="color:var(--dim);font-size:10px;margin-top:8px">cochleagram</div>
      <canvas id="coch" height="90"></canvas></div>
    <div class="card" style="margin-top:12px"><h2>stream log</h2>
      <div id="log"></div></div>
  </div>
</div>
<script>
const GC={input:'#8b5cf6',sensor:'#38bdf8',cortex:'#4ade80',core:'#fbbf24',
          memory:'#f472b6',output:'#fb7185'};
function px(cv,d,color){ // draw a quantised map
  const c=cv.getContext('2d'); cv.width=d.w; cv.height=d.h;
  const im=c.createImageData(d.w,d.h);
  for(let i=0;i<d.px.length;i++){const v=d.px[i];
    im.data[i*4]=color?v*0.35:v; im.data[i*4+1]=v; im.data[i*4+2]=color?v*0.9:v;
    im.data[i*4+3]=255;}
  c.putImageData(im,0,0);
}
function graph(s){
  const cv=document.getElementById('g'),c=cv.getContext('2d');
  const W=cv.clientWidth,H=420; cv.width=W;cv.height=H;
  c.clearRect(0,0,W,H);
  const pos={}; s.nodes.forEach(n=>pos[n.id]=[n.x*W,n.y*H]);
  c.lineWidth=1.2;
  s.edges.forEach(e=>{const a=pos[e.source],b=pos[e.target];if(!a||!b)return;
    const na=s.nodes.find(n=>n.id===e.source);
    c.strokeStyle='rgba(120,150,220,'+(0.12+0.5*na.activity)+')';
    c.beginPath();c.moveTo(a[0],a[1]);
    c.bezierCurveTo((a[0]+b[0])/2,a[1],(a[0]+b[0])/2,b[1],b[0],b[1]);c.stroke();
  });
  s.nodes.forEach(n=>{const [x,y]=pos[n.id],col=GC[n.group]||'#888';
    const r=9+9*n.activity;
    c.beginPath();c.arc(x,y,r+8*n.activity,0,7);
    c.fillStyle=col+'22';c.fill();
    c.beginPath();c.arc(x,y,r,0,7);c.fillStyle=col;c.fill();
    c.fillStyle='#dce6ff';c.font='600 11px ui-monospace,monospace';
    c.textAlign='center';c.fillText(n.label,x,y-r-8);
    c.fillStyle='#7c8bb5';c.font='10px ui-monospace,monospace';
    c.fillText(n.detail,x,y+r+14);});
}
function scene(s){
  const cv=document.getElementById('scene'),c=cv.getContext('2d');
  const W=cv.clientWidth,H=300;cv.width=W;cv.height=H;
  const k=Math.min(W,H)/s.scene_size, ox=(W-s.scene_size*k)/2;
  c.fillStyle='#060a16';c.fillRect(0,0,W,H);
  s.objects.forEach(o=>{c.fillStyle='rgba(74,222,128,.28)';
    c.fillRect(ox+(o.c-14)*k,(o.r-14)*k,28*k,28*k);
    c.fillStyle='#4ade80';c.font='10px monospace';c.textAlign='center';
    c.fillText(o.label,ox+o.c*k,o.r*k+4);});
  c.strokeStyle='#fbbf24';c.lineWidth=1;c.beginPath();
  s.path.forEach((p,i)=>{const x=ox+p.c*k,y=p.r*k;
    i?c.lineTo(x,y):c.moveTo(x,y);});c.stroke();
  const last=s.path[s.path.length-1];
  if(last){c.strokeStyle='#fb7185';c.lineWidth=2;
    c.strokeRect(ox+(last.c-14)*k,(last.r-14)*k,28*k,28*k);}
}
function code(s){
  const cv=document.getElementById('code'),c=cv.getContext('2d');
  const W=cv.clientWidth;cv.width=W;cv.height=46;
  c.clearRect(0,0,W,46);const n=s.code.length,w=W/n;
  for(let i=0;i<n;i++){const v=s.code[i];if(!v)continue;
    c.fillStyle='rgba(74,222,128,'+(0.25+v/340)+')';
    c.fillRect(i*w,46-4-v/255*38,Math.max(w,1),v/255*38+4);}
}
async function poll(){
  let s; try{s=await (await fetch('/state')).json();}catch(e){return;}
  graph(s);scene(s);code(s);
  px(document.getElementById('fov'),s.fovea,false);
  px(document.getElementById('sal'),s.saliency,true);
  px(document.getElementById('coch'),s.cochlea,true);
  const st=document.getElementById('state');
  st.textContent=s.running?'running':'paused';
  st.className='pill '+(s.running?'on':'off');
  document.getElementById('up').textContent=
    'tick '+s.tick+'  ·  '+s.metrics.uptime_s+'s uptime';
  const M=s.metrics,order=[['recognition','recognition','%'],
    ['saccades','saccades',''],['onsets','onsets heard',''],
    ['bindings','bindings',''],['spikes_last_tick','spikes/tick',''],
    ['tick_rate_hz','tick rate','Hz'],['v1_cells','V1 cells',''],
    ['concepts','concepts','']];
  document.getElementById('m').innerHTML=order.map(([k,l,u])=>
    '<div class="metric"><b>'+(u==='%'?Math.round(M[k]*100)+'%':M[k])+
    (u&&u!=='%'?' '+u:'')+'</b><span>'+l+'</span></div>').join('');
  document.getElementById('pc').textContent=s.name!==null?('"'+s.name+'"'):'— none';
  document.getElementById('tr').textContent=
    s.truth!==null?('ground truth: '+s.truth):'looking at background';
  document.getElementById('log').textContent=s.log.join('\n');
}
function cmd(a){fetch('/cmd?do='+a).then(poll);}
setInterval(poll,400);poll();
</script></body></html>
"""


def main() -> None:                              # pragma: no cover - CLI
    import argparse

    ap = argparse.ArgumentParser(description="NeuroBrain live console")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--cells", type=int, default=1024)
    ap.add_argument("--objects", type=int, default=12)
    ap.add_argument("--scene", type=int, default=256)
    ap.add_argument("--hz", type=float, default=4.0)
    a = ap.parse_args()
    print("building the integrated brain (this takes a moment) ...")
    c = BrainConsole(scene_size=a.scene, n_objects=a.objects,
                     v1_cells=a.cells)
    serve_console(port=a.port, host=a.host, hz=a.hz, console=c)


if __name__ == "__main__":                       # pragma: no cover - CLI
    main()
