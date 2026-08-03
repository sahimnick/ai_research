"""Class UML for the live vision subsystem, as SVG."""
import html, inspect, sys
sys.path.insert(0, ".")

W, H = 1560, 1080
BG, TXT, DIM = "#0e1117", "#e4e9f2", "#93a0b4"
LIVE, DATA, DORM, EDGE = "#5cc8ff", "#8cf08c", "#8a6a3a", "#3a4354"

o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
     f'viewBox="0 0 {W} {H}" font-family="DejaVu Sans, sans-serif">',
     f'<rect width="{W}" height="{H}" fill="{BG}"/>',
     '<defs>'
     '<marker id="inh" viewBox="0 0 12 12" refX="11" refY="6" markerWidth="12"'
     f' markerHeight="12" orient="auto"><path d="M0,0 L12,6 L0,12 z" '
     f'fill="none" stroke="{EDGE}" stroke-width="1.4"/></marker>'
     '<marker id="agg" viewBox="0 0 14 10" refX="13" refY="5" markerWidth="12"'
     f' markerHeight="10" orient="auto"><path d="M0,5 L7,0 L14,5 L7,10 z" '
     f'fill="{BG}" stroke="{EDGE}" stroke-width="1.4"/></marker>'
     '<marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" '
     f'markerHeight="8" orient="auto"><path d="M0,0 L10,5 L0,10 z" '
     f'fill="{EDGE}"/></marker></defs>']

def cls(x, y, w, name, stereo, attrs, ops, col=LIVE, dash=False):
    ah, oh = len(attrs) * 15, len(ops) * 15
    h = 30 + (14 if stereo else 0) + ah + 8 + oh + 12
    d = ' stroke-dasharray="6 4"' if dash else ''
    o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" '
             f'fill="#161a22" stroke="{col}" stroke-width="1.6"{d}/>')
    o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{28 + (12 if stereo else 0)}"'
             f' rx="6" fill="{col}" opacity=".14"/>')
    o.append(f'<text x="{x+w/2}" y="{y+19}" fill="{col}" font-size="14" '
             f'font-weight="bold" text-anchor="middle">{html.escape(name)}</text>')
    yy = y + 28
    if stereo:
        o.append(f'<text x="{x+w/2}" y="{yy+8}" fill="{DIM}" font-size="10" '
                 f'text-anchor="middle">{html.escape(stereo)}</text>')
        yy += 12
    o.append(f'<line x1="{x}" y1="{yy}" x2="{x+w}" y2="{yy}" stroke="{col}" '
             f'opacity=".45"/>')
    for i, a in enumerate(attrs):
        o.append(f'<text x="{x+9}" y="{yy+15+i*15}" fill="{DIM}" '
                 f'font-size="11">{html.escape(a)}</text>')
    yy += ah + 6
    o.append(f'<line x1="{x}" y1="{yy}" x2="{x+w}" y2="{yy}" stroke="{col}" '
             f'opacity=".45"/>')
    for i, m in enumerate(ops):
        o.append(f'<text x="{x+9}" y="{yy+15+i*15}" fill="{TXT}" '
                 f'font-size="11">{html.escape(m)}</text>')
    return (x, y, w, h)

def link(a, b, kind="assoc", label="", side="rl"):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if side == "rl":
        p1, p2 = (ax + aw, ay + ah / 2), (bx, by + bh / 2)
    elif side == "bt":
        p1, p2 = (ax + aw / 2, ay + ah), (bx + bw / 2, by)
    else:
        p1, p2 = (ax + aw / 2, ay), (bx + bw / 2, by + bh)
    m = {"inh": "url(#inh)", "agg": "url(#agg)"}.get(kind, "url(#ar)")
    o.append(f'<line x1="{p1[0]}" y1="{p1[1]}" x2="{p2[0]}" y2="{p2[1]}" '
             f'stroke="{EDGE}" stroke-width="1.4" marker-end="{m}"/>')
    if label:
        o.append(f'<text x="{(p1[0]+p2[0])/2}" y="{(p1[1]+p2[1])/2-5}" '
                 f'fill="{DIM}" font-size="10" text-anchor="middle">'
                 f'{html.escape(label)}</text>')

o.append(f'<text x="28" y="34" fill="{TXT}" font-size="20" font-weight="bold">'
         'NeuroBrain — class model of the vision subsystem</text>')
o.append(f'<text x="28" y="55" fill="{DIM}" font-size="12">solid = used by '
         'benchmarks · dashed = defined and exported but constructed nowhere · '
         'green = data-carrying</text>')

ue = cls(30, 80, 380, "UnifiedEye", "vision/unified_eye.py",
         ["size: int = 224", "areas: tuple = (V2, pool, V4)",
          "where_area = 'pool'   size_area = 'V4'",
          "min_confidence = 0.35   § gate, 0.910",
          "search_radius = 40.0   § near last seen",
          "exclusion_px = 30.0    § ids never share",
          "max_lost = 12          § id expiry",
          "stream: VentralStream", "tags: Dict[str, Tag]",
          "last: Dict[str, (y,x)]   _next_id: int"],
         ["develop(frames)", "add_tag(name, frame, box) -> Tag",
          "look(frame, cue, n_proposals, n_segments) -> Percept",
          "fit_size(frames, log_areas)", "features(frame) -> dict",
          "run(frames) -> List[Percept]"])

tag = cls(470, 80, 300, "Tag", "@dataclass",
          ["name: str", "templates: Dict[area, ndarray]",
           "box: (y, x, h, w)", "ident: int   § stable identity",
           "centre: Dict[area, (dy,dx)]"],
          ["§ centre fixes a half-cell bias",
           "  that put every box up-left"], col=DATA)

per = cls(830, 80, 340, "Percept", "@dataclass — one forward pass",
          ["where / confidence: Dict[str, …]",
           "lost: Dict[str, float]", "ident / unseen: Dict[str, int]",
           "proposals: List[dict]   § objectness",
           "segments: ndarray       § feature groups",
           "relations: Dict[(a,b), str]",
           "frame_size: float       § angular size",
           "attention / maps: Dict[str, ndarray]"],
          ["describe() -> str"], col=DATA)

vs = cls(30, 470, 380, "VentralStream", "vision/ventral.py",
         ["hierarchy: VisionHierarchy", "V4, IT, it_names",
          "it_class_proto        § averaged, 0.000",
          "it_class_v4tmpl       § spatial, 0.943"],
         ["descriptor(image)", "classify_object(image)",
          "attend(image, cue)          § averaged",
          "attend_spatial(image, cue)  § spatial",
          "area_map(image, area)", "tag(image, box, area) / locate(image, tag)"])

vh = cls(30, 700, 380, "VisionHierarchy", "",
         ["layers: List[Layer]"], ["forward(x) -> maps"])

lay = cls(30, 810, 380, "SpikingConvLayer · ComplexCellLayer ·\nSpikingPool · ITLayer",
          "the four stages", ["kernels, in_channels, name",
                              "log['output'] : (C, H, W)"],
          ["forward(x)"])

obj = cls(470, 300, 300, "objectness.py", "module-level functions",
          ["— no state —"],
          ["distinctiveness(map)", "propose(map, size, k) -> boxes",
           "segment(map, k) -> labels", "feature_report(stream, areas)"])

ov = cls(830, 340, 340, "viz/overlay.py", "module-level functions",
         ["LAYERS: 11 switchable names"],
         ["draw_percept(frame, percept, show, truth)",
          "annotate_boxes(percept, tags)", "strip(images)"], col=DATA)

svc = cls(830, 500, 340, "EyeService", "viz/eye_dashboard.py",
          ["eye: UnifiedEye", "src: _Source (webcam|cctv|dir)",
           "layers: set   cue: str|None", "history: List[Percept]"],
          ["step()", "render(scale) -> jpeg bytes",
           "add_tag(name, y, x)", "state() -> dict"], col=DATA)

ps = cls(470, 520, 300, "PredictiveStack", "world/topdown.py",
         ["W: ndarray  (one generative matrix)", "iters, gain, decay, sparsity"],
         ["settle(v1)", "learn(v1, lr)", "train(states)"], col=DORM, dash=True)

dpc = cls(470, 680, 300, "MultiDPC", "benchmarks/dpc_multistep.py",
          ["A: (k, 3, d) one matrix per horizon"],
          ["predict(s, h, cur)", "learn(s, cur, targets)"], col=DORM, dash=True)

dead = cls(470, 830, 700, "Dormant subsystems — 82 of 166 classes",
           "exported from neurobrain/__init__.py, constructed nowhere",
           ["cognition/  13 of 21   dopamine.py 629 lines, 6 classes",
            "minds/       8 of 10   perceptloop.py 400 lines",
            "memory/      9 of 15   rhythm, relational, psyche, development",
            "world/       9 of 18   objects, continuous, loop",
            "audition/    7 of 12   four A1/A2 variants in auditorycortex",
            "core/        5 of 13   Neuron, NeuronType never used by the package"],
           [], col=DORM, dash=True)

link(ue, tag, "agg", "tags", "rl")
link(tag, per, "", "ident", "rl")
link(ue, vs, "agg", "stream", "bt")
link(vs, vh, "agg", "hierarchy", "bt")
link(vh, lay, "agg", "layers", "bt")
link(ue, obj, "", "propose/segment", "rl")
link(per, ov, "", "drawn by", "bt")
link(ov, svc, "", "render()", "bt")
o.append(f'<line x1="410" y1="200" x2="830" y2="200" stroke="{EDGE}" '
         f'stroke-width="1.4" marker-end="url(#ar)"/>'
         f'<text x="620" y="194" fill="{DIM}" font-size="10" '
         f'text-anchor="middle">look() returns</text>')

o.append(f'<text x="28" y="{H-16}" fill="{DIM}" font-size="11">'
         'Generated by docs/diagrams/make_diagrams.py. § marks a value or '
         'design choice a measurement in EVALUATION.md §9 forced.</text>')
o.append("</svg>")
open("docs/diagrams/class_uml.svg", "w").write("\n".join(o))
print("wrote docs/diagrams/class_uml.svg")
