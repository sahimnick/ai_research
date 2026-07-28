"""
dashboard.py
============

An **interactive web dashboard** for the brain.

Run it and open the page in a browser: type a word (or pick an image/sound
stimulus), press *Send*, and watch the activation travel from the input sense
through the brain to the motor output -- live, in the browser -- while a panel
shows what concept the brain recognised and what it now expects next.

It is a single self-contained Python file using only the standard library
(``http.server``) plus the brain itself, so there is nothing extra to install.
The HTML/JS front-end is embedded below and drawn on a ``<canvas>``.

    from neurobrain import build_default_brain, teach_starter_curriculum
    from neurobrain.dashboard import serve

    brain = build_default_brain()
    teach_starter_curriculum(brain, verbose=False)
    serve(brain)                      # open http://127.0.0.1:8080
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, Optional

import numpy as np

from ..core.brain import Brain
from ..learning.teacher import Teacher
from ..tools.runtime import BrainRuntime
from .visualizer import NetworkVisualizer


class BrainService:
    """A window onto a **continuously running** brain (a :class:`BrainRuntime`).

    The runtime loops forever in a background thread; the dashboard just reads
    its latest state (for the live graph) and posts inputs / lessons into it.
    """

    def __init__(self, brain: Brain,
                 encoders: Optional[Dict[str, object]] = None,
                 teacher: Optional[Teacher] = None,
                 runtime: Optional[BrainRuntime] = None,
                 max_neurons_per_region: int = 34, max_edges: int = 520):
        self.runtime = runtime or BrainRuntime(
            brain, teacher=teacher, encoders=encoders, fps=60.0)
        self.brain = brain
        self._viz_params = (max_neurons_per_region, max_edges)
        self._rebuild_graph()
        if not self.runtime.running:
            self.runtime.start()

    def _rebuild_graph(self) -> None:
        mpr, me = self._viz_params
        self.viz = NetworkVisualizer(self.brain, max_neurons_per_region=mpr,
                                     max_edges=me).build()
        self.node_gids = np.array(self.viz.node_gids, dtype=np.int64)

    # -- graph for the front-end -----------------------------------------
    def graph(self) -> dict:
        nodes = []
        for i, gid in enumerate(self.viz.node_gids):
            x, y = self.viz.node_pos[i]
            nodes.append({
                "x": round(float(x), 4), "y": round(float(y), 4),
                "region": self.viz.node_region[i],
                "inhib": self.viz.node_color[i] == "#E8734C",
            })
        regions = [{"name": n, "role": r.role,
                    "x": r.position[0], "y": r.position[1]}
                   for n, r in self.brain.regions.items()]
        return {"nodes": nodes, "edges": self.viz.edges, "regions": regions}

    # -- the live frame (polled continuously by the browser) -------------
    def state(self) -> dict:
        s = self.runtime.state()
        thought = s.get("thought")
        prediction = None
        if thought:
            concept = self.brain.world.concept_for_word(thought)
            name = concept.name if concept else thought
            nxt = self.brain.world.predict_next(name)
            if nxt:
                c = self.brain.world.concepts.get(nxt)
                prediction = (c.word or c.name) if c else nxt
        s["prediction"] = prediction
        s["activation"] = self.runtime.node_activation(self.node_gids)
        s["region_rates"] = self.runtime.region_rates()
        return s

    # -- feeding / teaching / growth into the live loop ------------------
    def feed(self, modality: str, value: str) -> dict:
        self.runtime.feed_modality(modality, str(value))
        return {"ok": True}

    def teach(self, mode: str, value: str) -> dict:
        res = self.runtime.teach(mode, value)
        self._rebuild_graph()
        res["graph_changed"] = True
        return res

    def grow(self, kind: str) -> dict:
        res = self.runtime.grow_now(kind)
        self._rebuild_graph()
        res["graph_changed"] = True
        return res

    def attend(self, modality=None, concepts=None) -> dict:
        return self.runtime.attend(modality=modality, concepts=concepts)

    def imagine(self, enabled=None, strength=None, temperature=None) -> dict:
        return self.runtime.set_imagination(enabled=enabled, strength=strength,
                                            temperature=temperature)

    def stats(self) -> dict:
        return self.runtime.stats()


def _handler_factory(service: BrainService):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):        # keep the console quiet
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, _PAGE, "text/html; charset=utf-8")
            elif self.path == "/api/graph":
                self._send(200, json.dumps(service.graph()))
            elif self.path == "/api/stats":
                self._send(200, json.dumps(service.stats()))
            elif self.path == "/api/state":
                self._send(200, json.dumps(service.state()))
            else:
                self._send(404, json.dumps({"error": "not found"}))

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            try:
                req = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/api/feed":
                    result = service.feed(req.get("modality", "text"),
                                          str(req.get("value", "")))
                elif self.path == "/api/teach":
                    result = service.teach(req.get("mode", "word"),
                                           str(req.get("value", "")))
                elif self.path == "/api/grow":
                    result = service.grow(req.get("kind", "synaptogenesis"))
                elif self.path == "/api/attend":
                    result = service.attend(modality=req.get("modality"),
                                            concepts=req.get("concepts"))
                elif self.path == "/api/imagine":
                    result = service.imagine(enabled=req.get("enabled"),
                                             strength=req.get("strength"),
                                             temperature=req.get("temperature"))
                else:
                    self._send(404, json.dumps({"error": "not found"}))
                    return
                self._send(200, json.dumps(result))
            except Exception as exc:  # pragma: no cover - defensive
                self._send(400, json.dumps({"error": str(exc)}))

    return Handler


def serve(brain: Brain, encoders: Optional[Dict[str, object]] = None,
          teacher=None, runtime: Optional[BrainRuntime] = None,
          host: str = "127.0.0.1", port: int = 8080) -> None:
    """Start the dashboard server (blocking). Open http://host:port .

    The brain runs continuously in a background loop; the page shows it live.
    """
    service = BrainService(brain, encoders, teacher=teacher, runtime=runtime)
    httpd = ThreadingHTTPServer((host, port), _handler_factory(service))
    print(f"NeuroBrain dashboard running at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard.")
        httpd.server_close()


# The single-page front-end (HTML + CSS + vanilla JS, drawn on a canvas).
_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>NeuroBrain — live dashboard</title>
<style>
  :root { --bg:#0e1117; --panel:#161b22; --line:#232a34; --txt:#e6e6e6;
          --muted:#9aa4b2; --hot:#ffd24a; --accent:#4C9BE8; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--txt);
         font-family:system-ui,Segoe UI,Roboto,sans-serif; }
  header { padding:14px 20px; border-bottom:1px solid var(--line); }
  header h1 { margin:0; font-size:18px; }
  header p { margin:4px 0 0; color:var(--muted); font-size:13px; }
  .wrap { display:flex; gap:14px; padding:14px; flex-wrap:wrap; }
  .stage { flex:1 1 620px; min-width:320px; }
  canvas { width:100%; background:#0b0e13; border:1px solid var(--line);
           border-radius:10px; display:block; }
  .side { flex:0 0 300px; display:flex; flex-direction:column; gap:12px; }
  .card { background:var(--panel); border:1px solid var(--line);
          border-radius:10px; padding:14px; }
  .card h2 { margin:0 0 10px; font-size:13px; color:var(--muted);
             text-transform:uppercase; letter-spacing:.05em; }
  input,select,button { font:inherit; border-radius:8px; border:1px solid var(--line);
          background:#0b0e13; color:var(--txt); padding:9px 10px; }
  button { background:var(--accent); color:#04121f; border:none; cursor:pointer;
           font-weight:600; }
  button:hover { filter:brightness(1.08); }
  .row { display:flex; gap:8px; }
  .row > * { min-width:0; }
  .row input { flex:1; }
  .big { font-size:26px; font-weight:700; }
  .big.hot { color:var(--hot); }
  .muted { color:var(--muted); font-size:13px; }
  .bars { display:flex; flex-direction:column; gap:6px; }
  .bar { display:flex; align-items:center; gap:8px; font-size:12px; }
  .bar .name { width:88px; color:var(--muted); }
  .bar .track { flex:1; height:9px; background:#0b0e13; border-radius:5px; overflow:hidden;}
  .bar .fill { height:100%; width:0; background:linear-gradient(90deg,#4C9BE8,#ffd24a); }
  .chips { display:flex; flex-wrap:wrap; gap:6px; }
  .chip { background:#0b0e13; border:1px solid var(--line); border-radius:20px;
          padding:3px 10px; font-size:12px; color:var(--muted); }
  .examples button { background:#0b0e13; color:var(--txt); border:1px solid var(--line);
          font-weight:400; font-size:12px; padding:6px 9px; }
</style>
</head>
<body>
<header>
  <h1>🧠 NeuroBrain — live dashboard</h1>
  <p>The brain runs in a never-ending loop — always active, always learning.
     Feed it an input any time; watch the ongoing activity and read its
     current thought. Teach it new words and watch it grow.</p>
</header>
<div class="wrap">
  <div class="stage">
    <canvas id="cv" width="1000" height="600"></canvas>
    <div class="muted" id="clock" style="margin-top:6px;">loop starting…</div>
  </div>
  <div class="side">
    <div class="card">
      <h2>Input</h2>
      <div class="row" style="margin-bottom:8px;">
        <select id="mod">
          <option value="text">Text (word/phrase)</option>
          <option value="image">Image</option>
          <option value="sound">Sound</option>
        </select>
      </div>
      <div class="row">
        <input id="val" value="light" placeholder="type a word…"/>
        <button id="go">Feed ▶</button>
      </div>
      <div class="examples" style="margin-top:10px; display:flex; flex-wrap:wrap; gap:6px;">
        <button data-m="text" data-v="light">"light"</button>
        <button data-m="text" data-v="dark">"dark"</button>
        <button data-m="text" data-v="one">"one"</button>
        <button data-m="image" data-v="bright">bright img</button>
        <button data-m="image" data-v="dark">dark img</button>
        <button data-m="sound" data-v="high">high tone</button>
      </div>
    </div>
    <div class="card">
      <h2>Current thought</h2>
      <div class="muted">the brain is thinking about</div>
      <div class="big hot" id="recognised">—</div>
      <div class="muted" style="margin-top:8px;">expects next</div>
      <div class="big" id="prediction">—</div>
      <div class="muted" style="margin-top:10px;">💭 imagining
        <span id="imagining" style="color:#b06bcf;font-weight:600">—</span></div>
      <div class="muted" id="imagStream" style="margin-top:4px;font-size:11px;">&nbsp;</div>
    </div>
    <div class="card">
      <h2>Multi-faceted attention</h2>
      <div class="muted">attend to a sense (gain)</div>
      <div class="bar"><div class="name">vision</div>
        <input type="range" min="0" max="2" step="0.1" value="1" id="gVision" style="flex:1"></div>
      <div class="bar"><div class="name">sound</div>
        <input type="range" min="0" max="2" step="0.1" value="1" id="gSound" style="flex:1"></div>
      <div class="bar"><div class="name">text</div>
        <input type="range" min="0" max="2" step="0.1" value="1" id="gText" style="flex:1"></div>
      <div class="muted" style="margin-top:8px;">attend to concepts</div>
      <div class="row">
        <input id="focusVal" value="one two three" placeholder="concepts to hold in mind…"/>
        <button id="focusBtn">Focus</button>
      </div>
      <button id="clearAtt" style="margin-top:8px;background:#0b0e13;border:1px solid var(--line);color:var(--txt);font-weight:400;">clear attention</button>
    </div>
    <div class="card">
      <h2>Imagination (always on)</h2>
      <div class="bar"><div class="name">on/off</div>
        <input type="checkbox" id="imagOn" checked></div>
      <div class="bar"><div class="name">strength</div>
        <input type="range" min="0" max="16" step="0.5" value="9" id="imagStr" style="flex:1"></div>
      <div class="bar"><div class="name">creativity</div>
        <input type="range" min="0.2" max="1.6" step="0.05" value="0.75" id="imagTemp" style="flex:1"></div>
    </div>
    <div class="card">
      <h2>Teach the brain</h2>
      <div class="row" style="margin-bottom:8px;">
        <input id="teachVal" value="apple" placeholder="new word(s)…"/>
        <button id="teachWord">Learn</button>
      </div>
      <div class="row">
        <input id="seqVal" value="red green blue" placeholder="a b c (order)…"/>
        <button id="teachSeq">Teach order</button>
      </div>
      <div class="muted" id="teachMsg" style="margin-top:8px;">&nbsp;</div>
    </div>
    <div class="card">
      <h2>Brain size (it grows)</h2>
      <div class="bars">
        <div class="bar"><div class="name">neurons</div><div class="big" id="nNeurons" style="font-size:18px">—</div></div>
        <div class="bar"><div class="name">synapses</div><div class="big" id="nSyn" style="font-size:18px">—</div></div>
      </div>
      <div class="examples" style="margin-top:10px; display:flex; flex-wrap:wrap; gap:6px;">
        <button data-g="synaptogenesis">＋ synapses</button>
        <button data-g="neurogenesis">＋ neurons</button>
        <button data-g="prune">✂ prune</button>
      </div>
    </div>
    <div class="card">
      <h2>Region activity</h2>
      <div class="bars" id="bars"></div>
    </div>
    <div class="card">
      <h2>Vocabulary</h2>
      <div class="chips" id="vocab"></div>
    </div>
  </div>
</div>
<script>
// The brain runs forever on the server; this page polls its live state every
// frame and redraws — like watching a game engine's screen.
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
let G = null, act = null, busy = false;

function fit(){ const r = cv.getBoundingClientRect(); cv.width = r.width*2; cv.height = r.height*2; ctx.setTransform(2,0,0,2,0,0); }
window.addEventListener('resize', ()=>{ fit(); draw(); });
function px(x){ return 60 + x*(cv.width/2-120); }
function py(y){ return 74 + (1-y)*(cv.height/2-150); }

async function loadGraph(){ G = await (await fetch('/api/graph')).json(); fit(); draw(); }
function nodeColor(a, inhib){
  const base = inhib ? [232,115,76] : [76,155,232], hot=[255,210,74];
  return `rgb(${Math.round(base[0]*(1-a)+hot[0]*a)},${Math.round(base[1]*(1-a)+hot[1]*a)},${Math.round(base[2]*(1-a)+hot[2]*a)})`;
}
function draw(){
  if(!G) return;
  ctx.clearRect(0,0,cv.width,cv.height);
  for(const [i,j] of G.edges){
    const a = act ? (act[i]||0) : 0;
    ctx.strokeStyle = a>0.05 ? `rgba(255,210,74,${0.12+0.7*a})` : 'rgba(40,48,60,0.45)';
    ctx.lineWidth = a>0.05 ? 0.6+2*a : 0.5;
    ctx.beginPath();
    ctx.moveTo(px(G.nodes[i].x), py(G.nodes[i].y));
    ctx.lineTo(px(G.nodes[j].x), py(G.nodes[j].y));
    ctx.stroke();
  }
  ctx.textAlign='center'; ctx.font='bold 12px system-ui';
  for(const r of G.regions){
    ctx.fillStyle='#9aa4b2'; ctx.fillText(r.name, px(r.x), py(r.y)-58);
    ctx.fillStyle='#5b6570'; ctx.font='10px system-ui'; ctx.fillText('['+r.role+']', px(r.x), py(r.y)+66);
    ctx.font='bold 12px system-ui';
  }
  for(let i=0;i<G.nodes.length;i++){
    const a = act ? (act[i]||0) : 0;
    ctx.beginPath(); ctx.fillStyle = nodeColor(a, G.nodes[i].inhib);
    ctx.arc(px(G.nodes[i].x), py(G.nodes[i].y), 3+7*a, 0, 7); ctx.fill();
  }
}
function renderBars(rates){
  const el = document.getElementById('bars'); el.innerHTML='';
  const order = ['vision','sound','text','association','memory','motor'];
  for(const name of order){
    if(!rates || rates[name]==null) continue;
    const w = Math.min(100, (rates[name]||0)*100*6);
    el.insertAdjacentHTML('beforeend',
      `<div class="bar"><div class="name">${name}</div>
       <div class="track"><div class="fill" style="width:${w}%"></div></div></div>`);
  }
}
function showStats(s){
  if(s.neurons!=null) document.getElementById('nNeurons').textContent = s.neurons.toLocaleString();
  if(s.synapses!=null) document.getElementById('nSyn').textContent = s.synapses.toLocaleString();
  if(s.vocabulary){ const v=document.getElementById('vocab'); v.innerHTML='';
    s.vocabulary.forEach(w=> v.insertAdjacentHTML('beforeend',`<span class="chip">${w}</span>`)); }
}

// --- the continuous render loop: poll the live brain, redraw --------------
async function pollState(){
  if(busy){ return; }
  try{
    const s = await (await fetch('/api/state')).json();
    act = s.activation;
    draw(); renderBars(s.region_rates); showStats(s);
    document.getElementById('recognised').textContent = s.thought || '…';
    document.getElementById('prediction').textContent = s.prediction || '—';
    document.getElementById('imagining').textContent = s.imagining || '—';
    document.getElementById('imagStream').textContent =
      (s.imagination_stream||[]).join(' → ');
    const cs = (s.active_concepts||[]).join(', ') || '—';
    document.getElementById('clock').textContent =
      `frame ${s.frame}   ·   firing ${(s.rate*100).toFixed(1)}%   ·   current(⚡) ${s.tonic}   ·   active: ${cs}`;
  }catch(e){}
}
setInterval(pollState, 90);   // ~11 fps view of the loop

// --- feeding inputs into the live loop -----------------------------------
async function feed(mod, val){
  await fetch('/api/feed',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({modality:mod, value:val})});
}
document.getElementById('go').onclick = ()=> feed(document.getElementById('mod').value, document.getElementById('val').value);
document.getElementById('val').addEventListener('keydown', e=>{ if(e.key==='Enter') document.getElementById('go').click(); });
document.querySelectorAll('.examples button[data-m]').forEach(b=>{
  b.onclick = ()=>{ document.getElementById('mod').value=b.dataset.m;
    document.getElementById('val').value=b.dataset.v; feed(b.dataset.m, b.dataset.v); };
});

// --- teaching & growth (pauses the loop briefly on the server) -----------
async function reloadGraph(){ G = await (await fetch('/api/graph')).json(); }
async function teach(mode, val){
  const msg=document.getElementById('teachMsg'); msg.textContent='teaching… (may grow neurons)';
  busy = true;
  try{
    const res = await (await fetch('/api/teach',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({mode, value:val})})).json();
    if(res.error){ msg.textContent='error: '+res.error; return; }
    msg.textContent = res.message + (res.grew_neurons? `  (+${res.grew_neurons} neurons, +${res.grew_synapses} synapses)`:'');
    showStats(res); if(res.graph_changed) await reloadGraph();
  } finally { busy = false; }
}
async function grow(kind){
  const msg=document.getElementById('teachMsg'); msg.textContent='growing…';
  busy = true;
  try{
    const res = await (await fetch('/api/grow',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({kind})})).json();
    if(res.error){ msg.textContent='error: '+res.error; return; }
    msg.textContent = res.message; showStats(res); if(res.graph_changed) await reloadGraph();
  } finally { busy = false; }
}
document.getElementById('teachWord').onclick = ()=> teach('word', document.getElementById('teachVal').value);
document.getElementById('teachSeq').onclick = ()=> teach('sequence', document.getElementById('seqVal').value);
document.querySelectorAll('.examples button[data-g]').forEach(b=> b.onclick = ()=> grow(b.dataset.g));

// --- multi-faceted attention ---------------------------------------------
async function postAttend(body){
  await fetch('/api/attend',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
}
function sendModality(){
  postAttend({modality:{
    vision:+document.getElementById('gVision').value,
    sound:+document.getElementById('gSound').value,
    text:+document.getElementById('gText').value}});
}
['gVision','gSound','gText'].forEach(id=> document.getElementById(id).addEventListener('input', sendModality));
document.getElementById('focusBtn').onclick = ()=>
  postAttend({concepts: document.getElementById('focusVal').value.split(/\s+/).filter(Boolean)});
document.getElementById('clearAtt').onclick = ()=>{
  ['gVision','gSound','gText'].forEach(id=> document.getElementById(id).value=1);
  postAttend({modality:{vision:1,sound:1,text:1}, concepts:[]});
};

// --- imagination controls ------------------------------------------------
async function postImagine(body){
  await fetch('/api/imagine',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
}
document.getElementById('imagOn').addEventListener('change', e=> postImagine({enabled:e.target.checked}));
document.getElementById('imagStr').addEventListener('input', e=> postImagine({strength:+e.target.value}));
document.getElementById('imagTemp').addEventListener('input', e=> postImagine({temperature:+e.target.value}));

loadGraph();
</script>
</body>
</html>
"""
