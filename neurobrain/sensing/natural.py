"""
natural.py
==========

**The real world, not a dataset that was cut up to be easy.**

Everything measured in this package so far arrived pre-segmented and
low-variance: MNIST digits centred in 28x28, Fashion-MNIST the same, and
synthetic tones and chirps for hearing. Those are gifts. A digit is one
high-contrast object on a black field; a pure tone is one frequency with no
reverberation, no source, and no scene.

This module supplies the other thing:

:func:`load_cifar10`
    **Natural photographs** -- animals and vehicles, 32x32 colour, photographed
    rather than drawn. Cluttered backgrounds, real lighting, occlusion, and a
    within-class variability that hand-written digits do not have. Converted to
    the 28x28 grayscale the existing V1 expects, so the *same* eye can be
    pointed at it and the difference is the world, not the plumbing.

:func:`load_esc50`
    **Environmental sound** -- 50 classes of real field recording, grouped by
    ESC-50's own five categories:

        animals            dog, rooster, pig, cow, frog, cat, hen, insects, ...
        nature / water     rain, sea waves, crackling fire, crickets, birds,
                           water drops, wind, pouring water, thunderstorm, ...
        human non-speech   crying baby, sneezing, clapping, footsteps, ...
        interior           door knock, mouse click, keyboard, washing machine,
                           vacuum, clock alarm, glass breaking, ...
        **urban / exterior**   helicopter, chainsaw, **siren**, **car horn**,
                           **engine**, **train**, church bells, airplane,
                           fireworks, hand saw

    These are recordings of *sources in places*: a siren has a doppler shift, an
    engine has harmonics that drift, rain has no onset at all. Nothing about
    them is a 400 ms tone.

Both cache under the same directory as :mod:`realworld` and both raise on no
network, so callers can skip rather than silently score on nothing.
"""

from __future__ import annotations

import io
import json
import os
import ssl
import tarfile
import time
import urllib.request
import wave
from typing import Dict, List, Optional, Tuple

import numpy as np

from .realworld import _cache_dir

_CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
CIFAR_CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck")

# ESC-50 in WebDataset shards -- 44 MB each, so a usable subset is one download
# rather than the 600 MB the original archive costs.
_ESC_REC = "https://zenodo.org/api/records/14614287/files/{}/content"
_ESC_SHARDS = ["wds-audio-fold-{f}-{i:06d}.tar".format(f=f, i=i)
               for f in (1, 2, 3, 4, 5) for i in range(4)]

ESC50_CLASSES = (
    "dog", "rooster", "pig", "cow", "frog", "cat", "hen", "insects", "sheep",
    "crow", "rain", "sea_waves", "crackling_fire", "crickets", "chirping_birds",
    "water_drops", "wind", "pouring_water", "toilet_flush", "thunderstorm",
    "crying_baby", "sneezing", "clapping", "breathing", "coughing", "footsteps",
    "laughing", "brushing_teeth", "snoring", "drinking_sipping",
    "door_wood_knock", "mouse_click", "keyboard_typing", "door_wood_creaks",
    "can_opening", "washing_machine", "vacuum_cleaner", "clock_alarm",
    "clock_tick", "glass_breaking", "helicopter", "chainsaw", "siren",
    "car_horn", "engine", "train", "church_bells", "airplane", "fireworks",
    "hand_saw")

#: ESC-50's own five groupings, by class index.
ESC50_CATEGORIES: Dict[str, Tuple[int, ...]] = {
    "animals": tuple(range(0, 10)),
    "nature": tuple(range(10, 20)),
    "human": tuple(range(20, 30)),
    "interior": tuple(range(30, 40)),
    "urban": tuple(range(40, 50)),
}


def _ctx() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ca = os.environ.get("NEUROBRAIN_CA_BUNDLE", "/root/.ccr/ca-bundle.crt")
    if os.path.exists(ca):
        try:
            ctx.load_verify_locations(ca)
        except Exception:
            pass
    return ctx


def _fetch(url: str, path: str, chunk: int = 1 << 20, verbose: bool = True) -> str:
    """Download to ``path`` atomically, streaming, and **resumable**.

    Three things, all of which CIFAR-10 forced. It is 170 MB, and on a metered
    or proxied link that is half an hour:

    * **Streamed**, not ``r.read()`` into memory, so the partial file is visible
      on disk and a 170 MB download does not become a 170 MB allocation.
    * **Resumable** via an HTTP range request against the existing ``.part``,
      so a connection that dies at 90% costs the last 10% rather than all of it.
      Servers that ignore the range header are handled by starting over, which
      is what would have happened anyway.
    * **Atomic**: only a complete file is ever renamed into place, so a
      truncated archive can never become the cache and fail confusingly later.
    """
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"

    # One writer at a time. Resuming and appending are safe alone and ruinous
    # together: two processes that both see a 17 MB `.part`, both ask for
    # `bytes=17M-`, and both append produce a 43 MB file of interleaved
    # garbage that still looks like a plausible download. The lock is an
    # exclusive create, so it also survives the process being killed only in
    # the sense that it must then be cleared -- which is what the staleness
    # check below does.
    lock = path + ".lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        age = time.time() - os.path.getmtime(lock)
        if age < 900:
            raise RuntimeError(
                f"another process is already downloading {os.path.basename(path)} "
                f"(lock held {age:.0f}s). Wait for it, or delete {lock}.")
        os.unlink(lock)                      # stale: the holder died
        return _fetch(url, path, chunk=chunk, verbose=verbose)

    try:
        return _download(url, path, tmp, chunk, verbose)
    finally:
        try:
            os.unlink(lock)
        except OSError:
            pass


def _download(url: str, path: str, tmp: str, chunk: int, verbose: bool) -> str:
    have = os.path.getsize(tmp) if os.path.exists(tmp) else 0

    headers = {"User-Agent": "neurobrain"}
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=300, context=_ctx()) as r:
        resumed = r.status == 206
        if have and not resumed:            # server ignored Range; start over
            have = 0
        total = r.headers.get("Content-Length")
        total = (int(total) + have) if total else None
        with open(tmp, "ab" if resumed else "wb") as f:
            got = have
            while True:
                block = r.read(chunk)
                if not block:
                    break
                f.write(block)
                got += len(block)
                if verbose and total:
                    print(f"\r  {os.path.basename(path)}  "
                          f"{got / 1e6:.1f}/{total / 1e6:.1f} MB "
                          f"({100 * got / total:.0f}%)", end="", flush=True)
    if verbose and total:
        print(flush=True)
    # An empty read means the socket closed, which is not the same as the file
    # being finished. Renaming on that alone put a 137.5 MB archive into the
    # cache under the name of a 170.1 MB one, where it looked complete and
    # loaded as a short dataset -- the atomicity this function advertises,
    # defeated by never checking the length. The partial file is *kept* so the
    # next call resumes from it rather than starting over.
    if total is not None and got < total:
        raise IOError(f"{os.path.basename(path)}: got {got} of {total} bytes "
                      f"({100 * got / total:.1f}%); {tmp} kept for resume")
    os.replace(tmp, path)          # only a complete file becomes the cache
    return path


# ---------------------------------------------------------------- vision ---
def _cifar_members(path: str):
    """Yield ``(name, records)`` for every *complete* batch in the archive.

    Streaming (``r|gz``), which matters for more than memory. The archive is a
    single 170 MB gzip and on a slow link that is hours, but its members are
    laid out sequentially -- ``data_batch_1.bin`` is complete after about 30 MB
    of it. Reading sequentially and stopping at the truncation point means a
    partially-downloaded file is already a usable dataset, rather than nothing
    at all until the last byte lands.
    """
    import gzip
    with gzip.open(path, "rb") as gz:
        with tarfile.open(fileobj=gz, mode="r|") as tar:
            while True:
                try:
                    m = tar.next()
                except (tarfile.ReadError, EOFError, OSError):
                    return                       # truncated: stop cleanly
                if m is None:
                    return
                if not m.name.endswith(".bin"):
                    continue
                try:
                    f = tar.extractfile(m)
                    raw = f.read() if f is not None else b""
                except (tarfile.ReadError, EOFError, OSError):
                    return
                if len(raw) < 3073 or len(raw) % 3073:
                    return                       # a half-written batch
                yield m.name, np.frombuffer(raw, np.uint8).reshape(-1, 3073)


def load_cifar10(n_train: int = 5000, n_test: int = 1000, seed: int = 0,
                 grayscale: bool = True, size: int = 28,
                 allow_partial: bool = True
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Real photographs, returned in the shape the existing eye already reads.

    Colour is averaged to luminance and 32x32 is centre-cropped to ``size`` so
    the same :class:`~neurobrain.vision.widev1.WideV1` can be pointed at a cat
    and at a handwritten 3 with nothing else changed. That is the point: any
    difference in the numbers is the world getting harder, not the pipeline
    changing underneath.

    ``allow_partial`` lets the loader read a download still in flight, using
    whichever batches have arrived complete. The train/test split is then carved
    out of what is present -- always **disjoint**, since a leak here would
    silently inflate every number downstream -- and ``test_batch.bin`` is used
    as the test set when it has arrived. Set it to ``False`` to require the
    whole archive.
    """
    full = os.path.join(_cache_dir(), "cifar-10-binary.tar.gz")
    train, test = [], []

    def read(p):
        for nm, rec in _cifar_members(p):
            (test if "test_batch" in nm else train).append(rec)

    if os.path.exists(full) and os.path.getsize(full) > 0:
        read(full)
    else:
        # Try whatever has landed. A byte threshold would be a guess -- the
        # question is not how big the file is but whether a *batch* in it is
        # complete, and the reader already answers that by stopping at the
        # truncation point.
        part = full + ".part"
        if allow_partial and os.path.exists(part):
            read(part)
        if not train and not test:
            read(_fetch(_CIFAR_URL, full))
    if not train and not test:
        raise RuntimeError("no complete CIFAR-10 batch available")
    if not train:                            # only the test batch arrived
        train, test = test, []

    def unpack(recs):
        r = np.concatenate(recs)
        return (r[:, 1:].reshape(-1, 3, 32, 32).astype(np.float32),
                r[:, 0].astype(np.uint8))

    train_x, train_y = unpack(train)
    if test:
        test_x, test_y = unpack(test)
    else:
        # no test batch yet -- carve one off the front, disjoint by construction
        cut = min(max(n_test, 1), len(train_x) // 5)
        test_x, test_y = train_x[:cut], train_y[:cut]
        train_x, train_y = train_x[cut:], train_y[cut:]

    def prep(a: np.ndarray) -> np.ndarray:
        v = a.mean(1) if grayscale else a          # luminance, or (N, 3, H, W)
        if size != v.shape[-1]:
            # crop the last two axes. Slicing positionally from axis 1 works
            # only for grayscale; in colour it cuts the CHANNEL axis instead,
            # which silently returned a 1-channel image of the wrong width.
            o = (v.shape[-1] - size) // 2
            v = v[..., o:o + size, o:o + size]
        return np.clip(v, 0, 255).astype(np.uint8)

    rng = np.random.default_rng(seed)
    i = rng.permutation(len(train_x))[:n_train]
    j = rng.permutation(len(test_x))[:n_test]
    return (prep(train_x[i]), train_y[i].astype(np.uint8),
            prep(test_x[j]), test_y[j].astype(np.uint8))


# -------------------------------------------------------------- audition ---
def _read_wav(blob: bytes, sr_out: int = 8000) -> Optional[np.ndarray]:
    """WAV bytes -> mono float32 at ``sr_out``. Decimation, no scipy."""
    try:
        with wave.open(io.BytesIO(blob)) as w:
            n, ch, sw, sr = (w.getnframes(), w.getnchannels(),
                             w.getsampwidth(), w.getframerate())
            raw = w.readframes(n)
    except Exception:
        return None
    dt = {1: np.uint8, 2: np.int16, 4: np.int32}.get(sw)
    if dt is None:
        return None
    x = np.frombuffer(raw, dt).astype(np.float32)
    if dt is np.uint8:
        x = (x - 128.0) / 128.0
    else:
        x = x / float(np.iinfo(dt).max)
    if ch > 1:
        x = x.reshape(-1, ch).mean(1)
    if sr != sr_out:                       # anti-alias then decimate
        k = max(1, int(round(sr / sr_out)))
        if k > 1:
            pad = (-len(x)) % k
            if pad:
                x = np.concatenate([x, np.zeros(pad, np.float32)])
            x = x.reshape(-1, k).mean(1)
    return x.astype(np.float32)


def load_esc50(n_shards: int = 2, sr: int = 8000, dur_s: float = 2.0,
               categories: Optional[Tuple[str, ...]] = None, seed: int = 0
               ) -> Tuple[List[np.ndarray], np.ndarray, Tuple[str, ...]]:
    """Real environmental recordings: ``(waves, labels, class_names)``.

    ``categories`` selects ESC-50's own groupings -- e.g. ``("nature","urban")``
    for the birds/rain/wind against sirens/engines/trains contrast the goal
    actually cares about. Each clip is a 5 s field recording; ``dur_s`` takes
    the loudest window of it, because a real recording is mostly the room
    around the event rather than the event.
    """
    want: Optional[set] = None
    if categories:
        want = set()
        for c in categories:
            want.update(ESC50_CATEGORIES[c])

    waves: List[np.ndarray] = []
    labels: List[int] = []
    win = int(dur_s * sr)
    for shard in _ESC_SHARDS[:max(1, n_shards)]:
        p = os.path.join(_cache_dir(), "esc50", shard)
        try:
            _fetch(_ESC_REC.format(shard), p)
        except Exception:
            if not os.path.exists(p):
                continue
        with tarfile.open(p) as tar:
            members = {m.name: m for m in tar.getmembers()}
            for name, m in members.items():
                if not name.endswith(".wav"):
                    continue
                jm = members.get(name[:-4] + ".json")
                lab = None
                if jm is not None:
                    try:
                        lab = json.load(tar.extractfile(jm)).get("label")
                    except Exception:
                        lab = None
                if lab is None:                    # fold-clip-take-LABEL.wav
                    try:
                        lab = int(os.path.basename(name)[:-4].split("-")[-1])
                    except Exception:
                        continue
                lab = int(lab)
                if want is not None and lab not in want:
                    continue
                f = tar.extractfile(m)
                if f is None:
                    continue
                x = _read_wav(f.read(), sr_out=sr)
                if x is None or len(x) < win // 2:
                    continue
                if len(x) > win:                   # the loudest window, not the first
                    e = np.convolve(x * x, np.ones(win, np.float32) / win, "valid")
                    s = int(np.argmax(e))
                    x = x[s:s + win]
                if len(x) < win:
                    x = np.concatenate([x, np.zeros(win - len(x), np.float32)])
                n = float(np.max(np.abs(x)))
                waves.append((x / n if n > 1e-6 else x).astype(np.float32))
                labels.append(lab)

    if not waves:
        raise RuntimeError("no ESC-50 clips could be read (no network and no cache)")
    y = np.asarray(labels, np.int64)
    rng = np.random.default_rng(seed)
    p = rng.permutation(len(waves))
    return [waves[i] for i in p], y[p], ESC50_CLASSES


def esc50_category_of(label: int) -> str:
    """Which of ESC-50's five groupings a class index belongs to."""
    for name, idx in ESC50_CATEGORIES.items():
        if int(label) in idx:
            return name
    return "unknown"


# ------------------------------------------------------- the two together ---
#: CIFAR-10 and ESC-50 were built by different people for different tasks, and
#: they happen to **share real categories**. A cat is in both. So is a dog, a
#: frog, a bird, an airplane, and a car. That coincidence is the most valuable
#: thing in this module, because it gives a genuine audiovisual world without
#: anyone constructing one: the photographs and the recordings have no shared
#: provenance, no shared session, no shared noise floor, nothing whatsoever in
#: common except *what they are of*.
#:
#: The consequence matters. In a purpose-built multimodal set the two channels
#: are recorded together, so a binding can be learned from incidental
#: correlations -- the same room, the same microphone, the same lighting. Here
#: there is nothing like that to learn. Any association the mind forms between
#: the sight and the sound has to be semantic, because that is the only thing
#: the two streams share.
#:
#: ``name -> (cifar_class, (esc50_classes...))``
AV_MAP: Dict[str, Tuple[int, Tuple[int, ...]]] = {
    "airplane":   (0, (47,)),                  # airplane
    "automobile": (1, (44, 43)),               # engine, car_horn
    "bird":       (2, (14, 9, 1, 6)),          # chirping_birds, crow, rooster, hen
    "cat":        (3, (5,)),                   # cat
    "dog":        (5, (0,)),                   # dog
    "frog":       (6, (4,)),                   # frog
}

#: CIFAR classes with no sound in ESC-50. Not a gap to be filled -- a control.
#: A mind that hears nothing for a deer should end up with a *visual* concept
#: for it and no cross-modal one, and that difference is measurable.
AV_SILENT: Tuple[str, ...] = ("deer", "horse", "ship", "truck")


def load_audiovisual(n_per_class: int = 60, n_shards: int = 6, seed: int = 0,
                     sr: int = 8000, dur_s: float = 2.0, size: int = 28,
                     classes: Optional[Tuple[str, ...]] = None,
                     grayscale: bool = True
                     ) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray,
                                Tuple[str, ...]]:
    """A real audiovisual world: ``(images, waves, labels, names)``.

    Each sample is a real photograph of a thing and a real field recording of
    that same kind of thing, drawn independently. Labels are ``0..k-1`` over
    ``names``, which are the shared categories of :data:`AV_MAP`.

    Sounds are sampled *with* replacement when a class has fewer recordings than
    images -- ESC-50 gives 40 clips per class against CIFAR's 6,000 images, so
    the alternative would be throwing away almost all the vision. Each image
    still meets a sound drawn at random, so no image-sound pair repeats
    systematically.
    """
    names = tuple(classes) if classes else tuple(AV_MAP)
    for nm in names:
        if nm not in AV_MAP:
            raise KeyError(f"{nm!r} is not an audiovisual class; have {tuple(AV_MAP)}")

    trX, trY, teX, teY = load_cifar10(n_train=50000, n_test=10000, seed=seed,
                                      size=size, grayscale=grayscale)
    X_all = np.concatenate([trX, teX])
    Y_all = np.concatenate([trY, teY])

    esc_want = tuple(sorted({c for nm in names for c in AV_MAP[nm][1]}))
    waves, y_esc, _ = load_esc50(n_shards=n_shards, sr=sr, dur_s=dur_s, seed=seed)
    by_esc: Dict[int, List[int]] = {c: [] for c in esc_want}
    for i, v in enumerate(y_esc):
        if int(v) in by_esc:
            by_esc[int(v)].append(i)

    rng = np.random.default_rng(seed)
    imgs: List[np.ndarray] = []
    snds: List[np.ndarray] = []
    labs: List[int] = []
    for k, nm in enumerate(names):
        cif, escs = AV_MAP[nm]
        pool = np.flatnonzero(Y_all == cif)
        pool = rng.permutation(pool)[:n_per_class]
        heard = [i for c in escs for i in by_esc.get(c, [])]
        if not len(heard):
            raise RuntimeError(f"no ESC-50 clips cached for {nm!r} "
                               f"(classes {escs}); raise n_shards")
        for i in pool:
            imgs.append(X_all[i])
            snds.append(waves[int(rng.choice(heard))])
            labs.append(k)

    order = rng.permutation(len(imgs))
    return (np.stack([imgs[i] for i in order]),
            [snds[i] for i in order],
            np.asarray([labs[i] for i in order], np.int64),
            names)
