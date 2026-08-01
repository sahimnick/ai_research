"""Real sensors: cameras on the network, a webcam, a microphone, typed text.

Every measurement in this project so far ran on a corpus -- CIFAR photographs
and ESC-50 recordings, real but archived. This is the same mind pointed at the
world as it is now: traffic cameras that will show something different in five
minutes, a laptop's own camera and microphone, and whatever someone types.

Three commitments, because a live sensor is much easier to fake than a data set
and there is nothing to check the answer against:

**Every source says whether it is real.** :class:`Reading` carries ``live``, and
nothing here returns a plausible-looking array without it. A source that cannot
reach its device raises or reports ``live=False``; it never quietly substitutes
noise, and no benchmark built on this should accept a reading without checking.

**Placeholders are detected, not trusted.** A camera endpoint that is down
usually still returns *an image* -- a "no signal" card, served with HTTP 200.
Measured on 511ny.org: ten different camera IDs returned byte-identical
15136-byte images with identical statistics, which a size check or a status
check would both have passed. :meth:`CctvCamera.read` hashes what it gets and
refuses anything it has seen from a different camera, which is the only test
that catches this.

**No new hard dependencies.** The project is NumPy-only and stays that way.
Cameras are fetched with `urllib`; the webcam and microphone shell out to
whatever the host already has (``ffmpeg``, ``fswebcam``, ``arecord``) and say so
plainly when it has none. Pillow is used if present and a minimal JPEG/PNG path
is not attempted without it.

Where this can and cannot run
-----------------------------
Measured in the container this was developed in: **no ``/dev/video*``, no
``/dev/snd``, and none of ``ffmpeg``, ``fswebcam``, ``arecord``, ``cv2`` or
``sounddevice``**. So the device *call* cannot be exercised here -- but that is
not the same as the path being untested, and the difference is worth insisting
on. What can be exercised is exercised: :func:`decode_wav` is checked against a
synthesised 440 Hz tone (the recovered peak has to come back at 440 Hz, not
merely "some array"), against a stereo file, and against an 8-bit file it must
reject; :func:`_decode_image` against real encoded bytes; and the command each
tool would be handed against what it should be. What remains unverified is one
subprocess invocation per sensor, and the report says which.

The network path works and is what the live benchmarks use. On a laptop the same
code takes the webcam instead, with no change to anything downstream.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: A public traffic-camera endpoint that serves a JPEG per numeric id.
#: Ids near 400 were verified live; ids far outside that band return the
#: placeholder this module refuses.
NY511 = "https://511ny.org/map/Cctv/{id}"
NY511_LIVE_IDS = tuple(range(395, 420))

#: Length of the "no signal" card 511ny serves for a dead camera, kept as a
#: fast pre-filter. The hash check below is the real test -- a placeholder of a
#: different length would still be caught by it.
KNOWN_PLACEHOLDER_BYTES = 15136


@dataclass
class Reading:
    """One observation from one sensor, and whether it is genuinely one.

    ``live`` is the field that matters. Anything built on top of this must
    check it: a mind fed silently-synthetic input produces results that look
    exactly like results.
    """

    kind: str                      # "image" | "audio" | "text"
    data: np.ndarray
    live: bool
    source: str
    at: float = field(default_factory=time.time)
    note: str = ""

    def __repr__(self) -> str:
        return (f"Reading({self.kind}, {self.data.shape}, "
                f"live={self.live}, {self.source}"
                + (f", {self.note}" if self.note else "") + ")")


class SensorUnavailable(RuntimeError):
    """Raised when a sensor cannot be reached. Never swallowed into fake data."""


def _decode_image(raw: bytes) -> np.ndarray:
    from PIL import Image                                  # noqa: WPS433
    import io
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"),
                      dtype=np.uint8)


def decode_wav(path: str) -> np.ndarray:
    """16-bit PCM WAV -> float32 in [-1, 1], mono.

    Split out of :meth:`Microphone.read` so it can be *tested*. The device call
    cannot be exercised on a host with no sound card -- this container has
    neither ``/dev/snd`` nor ``arecord`` nor ``ffmpeg`` -- but the parsing is
    ordinary code, and "written but never run" is a worse position than
    "verified except for the driver call". A stereo file is averaged to mono so
    the shape a caller receives does not depend on the hardware.
    """
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            raise SensorUnavailable(
                f"{path}: expected 16-bit PCM, got "
                f"{8 * w.getsampwidth()}-bit")
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16)
        ch = w.getnchannels()
    if ch > 1:
        a = a.reshape(-1, ch).mean(1)
    return (a.astype(np.float32) / 32768.0)


class CctvCamera:
    """Public cameras, fetched over HTTP. The world, as it is right now.

    ``ids`` are endpoint identifiers for :data:`NY511`, or pass ``urls`` for any
    other source that serves a still image. The placeholder guard is the point:
    an endpoint that is down returns a "no signal" card with HTTP 200, and both
    a status check and a size check pass on it.
    """

    def __init__(self, ids: Sequence[int] = NY511_LIVE_IDS,
                 urls: Optional[Sequence[str]] = None,
                 template: str = NY511, timeout: float = 15.0):
        self.urls = (list(urls) if urls is not None
                     else [template.format(id=int(i)) for i in ids])
        self.timeout = float(timeout)
        self._hashes: Dict[str, str] = {}       # url -> content hash
        self._seen: Dict[str, str] = {}         # hash -> url that owns it

    def read(self, which: int = 0) -> Reading:
        url = self.urls[which % len(self.urls)]
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as r:
                raw = r.read()
        except (urllib.error.URLError, OSError) as e:
            raise SensorUnavailable(f"{url}: {e}") from e
        h = hashlib.md5(raw).hexdigest()

        # A camera serving the same bytes as a *different* camera is serving a
        # placeholder, not a scene. This is the check that caught ten "working"
        # cameras returning one image.
        owner = self._seen.get(h)
        if owner is not None and owner != url:
            return Reading("image", np.zeros((1, 1, 3), np.uint8), False, url,
                           note=f"placeholder (same bytes as {owner})")
        if len(raw) == KNOWN_PLACEHOLDER_BYTES:
            return Reading("image", np.zeros((1, 1, 3), np.uint8), False, url,
                           note="placeholder (known no-signal card)")
        self._seen[h] = url
        changed = self._hashes.get(url) not in (None, h)
        self._hashes[url] = h
        return Reading("image", _decode_image(raw), True, url,
                       note=("changed since last read" if changed else ""))

    def survey(self) -> List[Reading]:
        """One frame from every camera, placeholders included and marked."""
        out = []
        for k in range(len(self.urls)):
            try:
                out.append(self.read(k))
            except SensorUnavailable as e:
                out.append(Reading("image", np.zeros((1, 1, 3), np.uint8),
                                   False, self.urls[k], note=str(e)))
        return out


class Webcam:
    """The host's own camera, through whatever tool the host already has.

    No ``cv2``: the project is NumPy-only and a camera is not a reason to
    change that. ``ffmpeg`` and ``fswebcam`` are both common and both write a
    still to a file, which is all this needs.
    """

    TOOLS = ("ffmpeg", "fswebcam")

    def __init__(self, device: str = "/dev/video0"):
        self.device = device
        self.tool = next((t for t in self.TOOLS if shutil.which(t)), None)

    def available(self) -> Tuple[bool, str]:
        if not os.path.exists(self.device):
            return False, f"no {self.device} (no camera on this host)"
        if self.tool is None:
            return False, f"none of {self.TOOLS} installed"
        return True, f"{self.tool} -> {self.device}"

    def read(self) -> Reading:
        ok, why = self.available()
        if not ok:
            raise SensorUnavailable(why)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "frame.jpg")
            cmd = ([self.tool, "-y", "-f", "v4l2", "-i", self.device,
                    "-frames:v", "1", p] if self.tool == "ffmpeg"
                   else [self.tool, "-d", self.device, "--no-banner", p])
            r = subprocess.run(cmd, capture_output=True, timeout=30)
            if not os.path.exists(p) or os.path.getsize(p) == 0:
                raise SensorUnavailable(
                    f"{self.tool} produced nothing: "
                    f"{r.stderr.decode(errors='replace')[-200:]}")
            return Reading("image", _decode_image(open(p, "rb").read()), True,
                           f"webcam {self.device}")


class Microphone:
    """The host's own microphone, via ``arecord`` or ``ffmpeg``."""

    def __init__(self, seconds: float = 2.0, sr: int = 8000):
        self.seconds, self.sr = float(seconds), int(sr)
        self.tool = next((t for t in ("arecord", "ffmpeg")
                          if shutil.which(t)), None)

    def available(self) -> Tuple[bool, str]:
        if not os.path.exists("/dev/snd"):
            return False, "no /dev/snd (no sound card on this host)"
        if self.tool is None:
            return False, "neither arecord nor ffmpeg installed"
        return True, f"{self.tool}"

    def read(self) -> Reading:
        ok, why = self.available()
        if not ok:
            raise SensorUnavailable(why)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "clip.wav")
            cmd = ([self.tool, "-f", "S16_LE", "-r", str(self.sr), "-c", "1",
                    "-d", str(int(self.seconds)), p] if self.tool == "arecord"
                   else [self.tool, "-y", "-f", "alsa", "-i", "default",
                         "-t", str(self.seconds), "-ar", str(self.sr),
                         "-ac", "1", p])
            subprocess.run(cmd, capture_output=True, timeout=60)
            if not os.path.exists(p) or os.path.getsize(p) < 64:
                raise SensorUnavailable(f"{self.tool} recorded nothing")
            return Reading("audio", decode_wav(p), True,
                           f"microphone {self.tool}")


class TextSense:
    """Typed text as a sensory channel.

    Not an embedding model and not pretending to be one: a character-trigram
    hash into a fixed sparse code, which is a projection with no learning in
    it. It gives the association area something with the right *shape* to bind
    against sight and sound, and its limits should be assumed rather than
    discovered -- it knows nothing about meaning, only about spelling.
    """

    def __init__(self, dim: int = 512):
        self.dim = int(dim)

    def read(self, text: str) -> Reading:
        v = np.zeros(self.dim, np.float32)
        s = f"  {text.lower().strip()}  "
        for i in range(len(s) - 2):
            h = hashlib.md5(s[i:i + 3].encode()).digest()
            v[int.from_bytes(h[:4], "big") % self.dim] += 1.0
        n = np.linalg.norm(v)
        return Reading("text", v / n if n > 1e-9 else v, True, "typed text",
                       note=text[:40])


def probe_all(verbose: bool = True) -> Dict[str, Tuple[bool, str]]:
    """What can this host actually sense? Report it rather than assume it."""
    out: Dict[str, Tuple[bool, str]] = {}
    w, m = Webcam(), Microphone()
    out["webcam"] = w.available()
    out["microphone"] = m.available()
    try:
        r = CctvCamera().read(0)
        out["cctv"] = (r.live, r.source if r.live else r.note)
    except SensorUnavailable as e:
        out["cctv"] = (False, str(e)[:80])
    out["text"] = (True, "always available")
    if verbose:
        for k, (ok, why) in out.items():
            print(f"  {k:<12}{'LIVE' if ok else 'unavailable':<14}{why}")
    return out
