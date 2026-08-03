"""
viz -- Watch it think: matplotlib animation and the live HTTP console.

Modules
-------
console     Live HTTP dashboard of the integrated brain
dashboard   ORPHAN -- superseded by console.py, imported by nothing
visualizer  Matplotlib animation of the activation path
"""

from .overlay import LAYERS, draw_percept, annotate_boxes, strip
from .eye_dashboard import EyeService, serve_eye
