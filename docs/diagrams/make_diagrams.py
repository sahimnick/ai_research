"""Regenerate the project diagrams and the dormant-code audit.

    python3 docs/diagrams/make_diagrams.py     # from the repo root

Writes flowchart.svg, class_uml.svg and DORMANT.md beside this file. The audit
is generated rather than written by hand so it cannot drift from the code: it
parses every module with `ast`, then checks whether each class name appears
anywhere outside the file that defines it (package __init__ exports do not
count as use -- exporting a class is not using it).
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

if __name__ == "__main__":
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    for part in ("_map.py", "_flow.py", "_uml.py"):
        print(f"--- {part} ---")
        runpy.run_path(os.path.join(HERE, part), run_name="__main__")
