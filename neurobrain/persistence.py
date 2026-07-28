"""
persistence.py
==============

Give the brain a memory that survives being switched off. Everything the brain
becomes -- the filters V1/V2/V4 learned, the IT object cells, the auditory
receptive fields, the association area's concept synapses, the world-model
transitions, and any structure that *grew* (extra maps, extra concept cells) --
lives in plain numpy arrays on ordinary Python objects. So a whole trained brain
can be written to one file and read back identical, ready to keep learning where
it left off.

    from neurobrain.persistence import save_brain, load_brain
    save_brain(mind, "brain.nbz")
    mind = load_brain("brain.nbz")     # same brain, same behaviour

This is the prerequisite for lifelong learning: without it, every run started
from nothing.
"""

from __future__ import annotations

import gzip
import pickle
from typing import Any

FORMAT_VERSION = 1


def save_brain(obj: Any, path: str) -> str:
    """Write a trained brain (a Mind, a stream, a circuit -- anything holding the
    numpy state) to ``path``, gzip-compressed. Returns the path."""
    blob = {"format": FORMAT_VERSION, "object": obj}
    with gzip.open(path, "wb") as f:
        pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_brain(path: str) -> Any:
    """Read a brain saved with :func:`save_brain`, restoring it exactly."""
    with gzip.open(path, "rb") as f:
        blob = pickle.load(f)
    if not isinstance(blob, dict) or "object" not in blob:
        raise ValueError(f"{path} is not a NeuroBrain save file")
    if blob.get("format") != FORMAT_VERSION:
        raise ValueError(f"unsupported save format {blob.get('format')} "
                         f"(this build reads {FORMAT_VERSION})")
    return blob["object"]
