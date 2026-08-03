"""Map the project: classes, where they are used, and what is never used."""
import ast, json, os, re
from collections import defaultdict

PKG = "neurobrain"
SEARCH_DIRS = ["neurobrain", "benchmarks", "tests", "examples", "docs"]
#: this generator names a great many classes in order to draw them, and would
#: otherwise count itself as their user -- reporting dormant code as alive
#: precisely because a diagram mentions it
SKIP = os.path.join("docs", "diagrams")

mods, classes, funcs = {}, {}, defaultdict(list)
for root, _, files in os.walk(PKG):
    if "__pycache__" in root: continue
    for fn in files:
        if not fn.endswith(".py"): continue
        p = os.path.join(root, fn)
        try: t = ast.parse(open(p, encoding="utf-8").read())
        except SyntaxError: continue
        m = p[:-3].replace("/", ".")
        cls = [n.name for n in ast.walk(t) if isinstance(n, ast.ClassDef)]
        fns = [n.name for n in t.body
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and not n.name.startswith("_")]
        mods[m] = {"path": p, "classes": cls, "functions": fns,
                   "lines": len(open(p, encoding="utf-8").read().splitlines()),
                   "doc": (ast.get_docstring(t) or "").strip().split("\n")[0][:120]}
        for c in cls: classes[c] = m

# where is each name mentioned, outside the file that defines it?
text = {}
for d in SEARCH_DIRS:
    if not os.path.isdir(d): continue
    for root, _, files in os.walk(d):
        if "__pycache__" in root or SKIP in root: continue
        for fn in files:
            if fn.endswith((".py", ".md")):
                p = os.path.join(root, fn)
                try: text[p] = open(p, encoding="utf-8").read()
                except Exception: pass

def uses(name, home_path):
    hits = []
    pat = re.compile(r"\b" + re.escape(name) + r"\b")
    for p, s in text.items():
        if p == home_path: continue
        if pat.search(s): hits.append(p)
    return hits

rows = []
for c, m in sorted(classes.items()):
    home = mods[m]["path"]
    h = uses(c, home)
    code = [p for p in h if p.endswith(".py")]
    exported = any(p.endswith("__init__.py") for p in code)
    real = [p for p in code if not p.endswith("__init__.py")]
    rows.append({"cls": c, "module": m, "exported": exported,
                 "used_in": len(real), "where": real[:4],
                 "docs_only": (not real) and bool([p for p in h if p.endswith(".md")])})
json.dump({"modules": mods, "classes": rows},
          open("/tmp/claude-0/-home-user-ai-research/adef3224-f5ac-552b-8f3c-fd871605f1ac/scratchpad/projmap.json","w"), indent=1)

never = [r for r in rows if r["used_in"] == 0]
print(f"{len(mods)} modules, {len(classes)} classes")
print(f"classes never referenced outside their own file: {len(never)}")
for r in never:
    print(f"  {r['cls']:<28} {r['module']:<42} exported={r['exported']} docs_only={r['docs_only']}")

print("\n=== by subpackage: classes total / never used elsewhere ===")
sub = defaultdict(lambda: [0, 0, []])
for r in rows:
    s = r["module"].split(".")[1] if r["module"].count(".") >= 1 else "root"
    sub[s][0] += 1
    if r["used_in"] == 0:
        sub[s][1] += 1
        sub[s][2].append(r["cls"])
for s in sorted(sub, key=lambda k: -sub[k][1]):
    t, u, names = sub[s]
    print(f"  {s:<12} {t:>3} classes, {u:>3} unused  ({100*u/max(t,1):.0f}%)")

alive = [r for r in rows if r["used_in"] > 0]
print(f"\nalive (used outside their own file): {len(alive)}")
print("=== biggest modules with zero used classes ===")
for m, info in sorted(mods.items(), key=lambda kv: -kv[1]["lines"])[:40]:
    cs = [r for r in rows if r["module"] == m]
    if cs and all(r["used_in"] == 0 for r in cs):
        print(f"  {info['lines']:>5} lines  {m:<44} {len(cs)} class(es)")


# --- the audit, written out so it can be read without running this ----------
L = ["# Dormant code -- generated, do not edit by hand", "",
     "Regenerate with `python3 docs/diagrams/make_diagrams.py`.", "",
     "A class counts as **used** if its name appears in any .py or .md outside",
     "the file that defines it. Being exported from `neurobrain/__init__.py`",
     "does **not** count -- exporting a class is not using it, and this project",
     "exports almost everything.", "",
     "**%d modules, %d classes, %d never used outside their own file.**"
     % (len(mods), len(classes), len(never)), ""]
L += ["| subpackage | classes | unused | share |", "|---|---|---|---|"]
for k in sorted(sub, key=lambda z: -sub[z][1]):
    t_, u_, _n = sub[k]
    L.append("| `%s` | %d | **%d** | %d%% |"
             % (k, t_, u_, round(100 * u_ / max(t_, 1))))
L += ["", "## Every class with no user outside its own file", "",
      "| class | module | exported |", "|---|---|---|"]
for r in sorted(never, key=lambda z: (z["module"], z["cls"])):
    L.append("| `%s` | `%s` | %s |"
             % (r["cls"], r["module"], "yes" if r["exported"] else "no"))
open(os.path.join("docs", "diagrams", "DORMANT.md"), "w").write("\n".join(L) + "\n")
print("wrote docs/diagrams/DORMANT.md")
