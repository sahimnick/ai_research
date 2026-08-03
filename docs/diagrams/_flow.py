"""Emit the project flowchart as SVG -- laid out by hand, no graphviz here."""
import html

W, H = 1500, 980
BG, LINE, TXT, DIM = "#0e1117", "#2a3140", "#e4e9f2", "#93a0b4"
LIVE, DORM, DATA = "#5cc8ff", "#8a6a3a", "#8cf08c"

o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
     f'viewBox="0 0 {W} {H}" font-family="DejaVu Sans, sans-serif">',
     f'<rect width="{W}" height="{H}" fill="{BG}"/>',
     '<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" '
     'markerWidth="7" markerHeight="7" orient="auto">'
     f'<path d="M0,0 L10,5 L0,10 z" fill="{LINE}"/></marker>'
     '<marker id="al" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
     f'markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" '
     f'fill="{LIVE}"/></marker></defs>']

def box(x, y, w, h, title, lines, col=LIVE, dash=False):
    d = ' stroke-dasharray="6 4"' if dash else ''
    o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" '
             f'fill="#161a22" stroke="{col}" stroke-width="1.6"{d}/>')
    o.append(f'<text x="{x+12}" y="{y+22}" fill="{col}" font-size="14" '
             f'font-weight="bold">{html.escape(title)}</text>')
    for i, ln in enumerate(lines):
        o.append(f'<text x="{x+12}" y="{y+42+i*16}" fill="{DIM}" '
                 f'font-size="11.5">{html.escape(ln)}</text>')

def arrow(x1, y1, x2, y2, label="", live=True, curve=0):
    m = "url(#al)" if live else "url(#a)"
    c = LIVE if live else LINE
    if curve:
        mx, my = (x1+x2)/2, (y1+y2)/2 + curve
        o.append(f'<path d="M{x1},{y1} Q{mx},{my} {x2},{y2}" fill="none" '
                 f'stroke="{c}" stroke-width="1.5" marker-end="{m}" opacity=".85"/>')
    else:
        o.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c}" '
                 f'stroke-width="1.5" marker-end="{m}" opacity=".85"/>')
    if label:
        o.append(f'<text x="{(x1+x2)/2}" y="{(y1+y2)/2+curve/2-6}" fill="{DIM}" '
                 f'font-size="10.5" text-anchor="middle">{html.escape(label)}</text>')

def head(x, y, t, s=""):
    o.append(f'<text x="{x}" y="{y}" fill="{TXT}" font-size="19" '
             f'font-weight="bold">{html.escape(t)}</text>')
    if s:
        o.append(f'<text x="{x}" y="{y+19}" fill="{DIM}" font-size="12">'
                 f'{html.escape(s)}</text>')

head(28, 34, "NeuroBrain — how a frame becomes a percept",
     "solid = measured and in use   ·   dashed = built but not exercised by any benchmark")

# --- row 1: input
box(28, 78, 250, 96, "SENSORS  sensing/live.py",
    ["CctvCamera   NY511, 97s period", "Webcam   v4l2 / avfoundation / dshow",
     "Microphone, TextSense"])
box(28, 196, 250, 76, "DATASETS  sensing/natural.py",
    ["CIFAR-10, ESC-50, MNIST", "VOT2019 (60 sequences, benchmarks/)"])

# --- row 2: the eye
box(320, 78, 300, 194, "THE EYE  vision/ventral.py",
    ["build_ventral_stream_on(frames)", "",
     "V1  SpikingConvLayer   107x107", "V1_complex  ComplexCellLayer",
     "V2  corners/junctions  103x103", "pool  SpikingPool      51x51",
     "V4  curvature/shape    49x49", "IT  ITLayer  (whole-object)"])
arrow(278, 126, 320, 126, "frames")
arrow(278, 232, 318, 200, "frames", curve=-14)

box(660, 78, 330, 194, "UnifiedEye  vision/unified_eye.py",
    ["develop(frames)   grow on YOUR footage", "add_tag(name, frame, box)",
     "look(frame) -> Percept    ONE pass", "",
     "min_confidence 0.35  -> LOST if below",
     "search_radius 40px   near last seen",
     "exclusion_px 30      ids never share",
     "max_lost 12          id expires"])
arrow(620, 176, 660, 176, "area maps")

box(1030, 78, 300, 194, "Percept  (one forward pass)",
    ["where / confidence / ident", "lost / unseen",
     "attention   correlation field", "proposals   objectness, no labels",
     "segments    feature grouping", "relations   left/right/above",
     "frame_size  angular size", "maps        raw area maps"], col=DATA)
arrow(990, 176, 1030, 176, "read-outs")

# --- row 3: mechanisms
box(320, 300, 300, 118, "objectness.py",
    ["propose()   0.31 recall @10 (0.11 rand)",
     "segment()   k-means on cell vectors",
     "feature_report()  probed w/ gratings",
     "distinctiveness()"])
box(660, 300, 330, 118, "locate_template()  ventral.py",
    ["normalised cross-correlation", "0.943 static · 0.910 gated held-out",
     "the averaged prototype scored 0.000", "(§9.21) — layout is what matters"])
arrow(470, 300, 470, 274, live=True)
arrow(825, 300, 825, 274, live=True)

# --- row 4: output
box(1030, 300, 300, 118, "viz/overlay.py",
    ["draw_percept(frame, percept, show=…)", "11 switchable layers",
     "tests/test_overlay.py asserts the box", "lands where the percept says"], col=DATA)
arrow(1180, 274, 1180, 300, live=True)
box(1030, 442, 300, 96, "viz/eye_dashboard.py",
    ["serve_eye('webcam'|'cctv'|dir)", "per-layer toggles, target switch",
     "click to tag  ·  127.0.0.1:8137"], col=DATA)
arrow(1180, 418, 1180, 442, live=True)
box(660, 442, 330, 96, "benchmarks/  (88 scripts)",
    ["eye_vot · eye_confidence · eye_targets", "eye_information · eye_breadth · dpc_*",
     "every claim in EVALUATION.md §9"], col=DATA)
arrow(1030, 490, 990, 490, live=True)

# --- dormant column
head(28, 330, "Built, not exercised", "no benchmark constructs these")
box(28, 352, 250, 100, "world/", ["PredictiveStack  (§9.16 measured)",
    "ObjectConcept, Relation, LoopTrace", "continuous.py, objects.py, loop.py"],
    col=DORM, dash=True)
box(28, 468, 250, 88, "cognition/  13 of 21 unused",
    ["dopamine.py  629 lines, 6 classes", "pallium, selfmodel, analogy,",
     "reasoning, discovery"], col=DORM, dash=True)
box(28, 572, 250, 74, "memory/  9 of 15 unused",
    ["rhythm, relational, psyche,", "development (SemanticCortex)"],
    col=DORM, dash=True)
box(28, 662, 250, 74, "minds/  8 of 10 unused",
    ["perceptloop (400 lines)", "integrated, ObjectFile"], col=DORM, dash=True)
box(28, 752, 250, 74, "audition/  7 of 12 unused",
    ["auditorycortex: 4 A1/A2 variants", "TemporalPool"], col=DORM, dash=True)
box(28, 842, 250, 62, "core/neuron.py",
    ["Neuron, NeuronType — exported,", "used nowhere in the package"],
    col=DORM, dash=True)

# --- the measured loop
head(320, 470, "The loop, measured")
box(320, 492, 300, 200, "what it does and does not",
    ["+ locate a tagged thing          0.910*",
     "+ select under competition       0.863",
     "+ spatial relation               0.902",
     "+ angular size (within scene)    0.585",
     "+ identity: track collapse       0.000",
     "- detect anything untagged       none",
     "- transfer to unseen scenes      R2 0.00",
     "- imagine forward (rollout)     -0.030",
     "- shadow vs object              at null",
     "- colour                        -0.135",
     "* held-out, gated, consecutive frames"], col=DATA)

o.append(f'<text x="28" y="{H-18}" fill="{DIM}" font-size="11">'
         'Generated by docs/diagrams/make_diagrams.py — regenerate after '
         'structural changes. Numbers are from EVALUATION.md §9.18–§9.36.</text>')
o.append("</svg>")
open("docs/diagrams/flowchart.svg", "w").write("\n".join(o))
print("wrote docs/diagrams/flowchart.svg")
