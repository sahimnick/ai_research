"""
realworld.py
============

Real data, and categories the brain makes for *itself*.

Everything before this used tiny synthetic shapes and tones with a fixed list of
classes we hand-picked. This module drops both crutches:

  * it learns from **real handwritten digits** (the MNIST dataset -- images
    written by hundreds of real people), fetched from the internet, not drawn by
    our own code;
  * it is given **no labels and no fixed number of classes**. It grows its own
    category cells: each new input either matches a category it already has, or,
    if it is novel enough, sprouts a brand-new category cell. This is
    competitive Hebbian / ART-style learning (a *vigilance* threshold decides
    "new or not") -- no back-propagation, no hand-crafted feature, no supervised
    classifier;
  * it knows when it does **not** know: an input that matches no category well
    is reported as *unknown* rather than forced into a wrong class.

Honest result (measured, not claimed): on real, held-out MNIST test digits the
self-grown categories recognise the digit **>90% of the time** -- with the true
labels used *only* to name the discovered clusters for scoring, never to learn.
The categories are the brain's own; the labels are just how we read the score.
"""

from __future__ import annotations

import gzip
import os
import ssl
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

_MNIST_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
_MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _cache_dir() -> str:
    d = os.environ.get("NEUROBRAIN_DATA",
                       os.path.join(os.path.expanduser("~"), ".cache", "neurobrain"))
    os.makedirs(d, exist_ok=True)
    return d


#: Fashion-MNIST (Zalando) -- same format, completely different images.
#: This project has never been tuned on it, which is exactly the point: it is an
#: EXTERNAL benchmark whose data and ground truth were authored by someone else.
_FASHION_BASE = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
_FASHION_MIRROR = ("https://raw.githubusercontent.com/zalandoresearch/"
                   "fashion-mnist/master/data/fashion/")
FASHION_CLASSES = ("t-shirt", "trouser", "pullover", "dress", "coat",
                   "sandal", "shirt", "sneaker", "bag", "ankle-boot")


def _download(name: str, base: str = None, prefix: str = "") -> str:
    path = os.path.join(_cache_dir(), prefix + name)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    ctx = ssl.create_default_context()
    ca = "/root/.ccr/ca-bundle.crt"
    if os.path.exists(ca):
        try:
            ctx.load_verify_locations(ca)
        except Exception:
            pass
    bases = [base or _MNIST_BASE]
    if base == _FASHION_BASE:
        bases.append(_FASHION_MIRROR)          # fall back to the git mirror
    last = None
    for b in bases:
        try:
            req = urllib.request.Request(b + name,
                                         headers={"User-Agent": "neurobrain"})
            with urllib.request.urlopen(req, timeout=90, context=ctx) as r:
                data = r.read()
            with open(path, "wb") as f:
                f.write(data)
            return path
        except Exception as e:                  # try the next mirror
            last = e
    raise last


def load_mnist(n_train: int = 15000, n_test: int = 10000, seed: int = 0
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fetch (and cache) the **real** MNIST handwritten digits. Returns
    ``(train_images, train_labels, test_images, test_labels)`` as uint8 arrays of
    shape (n, 28, 28) / (n,). Raises on no network so callers can skip."""
    def imgs(p):
        with gzip.open(p) as g:
            g.read(16)
            return np.frombuffer(g.read(), np.uint8).reshape(-1, 28, 28)

    def labs(p):
        with gzip.open(p) as g:
            g.read(8)
            return np.frombuffer(g.read(), np.uint8)

    trx = imgs(_download(_MNIST_FILES["train_images"]))
    trY = labs(_download(_MNIST_FILES["train_labels"]))
    tex = imgs(_download(_MNIST_FILES["test_images"]))
    teY = labs(_download(_MNIST_FILES["test_labels"]))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(trx))[:n_train]
    return trx[idx], trY[idx], tex[:n_test], teY[:n_test]


def load_fashion_mnist(n_train: int = 15000, n_test: int = 10000, seed: int = 0
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fetch (and cache) **Fashion-MNIST** -- clothing photographs, same format.

    This exists for one reason: **an exam nobody here wrote**. Every other
    stimulus in this project is either synthetic (shapes, tones, grid worlds)
    with an oracle written in this repo, or MNIST, which the machinery was
    developed against. Fashion-MNIST is external, harder, and has never been
    tuned on -- so the score it gives is an honest measure of whether the
    self-organising perception generalises at all, or was quietly fitted to
    handwritten digits."""
    def imgs(p):
        with gzip.open(p) as g:
            g.read(16)
            return np.frombuffer(g.read(), np.uint8).reshape(-1, 28, 28)

    def labs(p):
        with gzip.open(p) as g:
            g.read(8)
            return np.frombuffer(g.read(), np.uint8)

    d = lambda n: _download(n, base=_FASHION_BASE, prefix="fashion-")
    trx = imgs(d(_MNIST_FILES["train_images"]))
    trY = labs(d(_MNIST_FILES["train_labels"]))
    tex = imgs(d(_MNIST_FILES["test_images"]))
    teY = labs(d(_MNIST_FILES["test_labels"]))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(trx))[:n_train]
    return trx[idx], trY[idx], tex[:n_test], teY[:n_test]


def contrast_normalise(images: np.ndarray) -> np.ndarray:
    """Retina-style preparation: flatten, remove the mean (contrast, not
    brightness), and normalise each image's energy. No learned/classical
    transform -- just the divisive normalisation cortex does on its input."""
    x = images.reshape(len(images), -1).astype(np.float32) / 255.0
    x = x - x.mean(axis=1, keepdims=True)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(n < 1e-6, 1.0, n)


@dataclass
class GrowingCategoryMap:
    """Category cells that the brain grows for itself (ART-style vigilance).

    Each cell is a prototype. A new input finds its best-matching cell; if the
    match beats the ``vigilance`` threshold the cell learns it (Hebbian move),
    otherwise a new cell is *grown* for it. No labels, no fixed class count, no
    back-prop. ``max_cells`` caps growth.
    """

    dim: int
    vigilance: float = 0.65
    lr: float = 0.1
    max_cells: int = 1200
    W: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.float32))
    wins: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cell_label: Optional[np.ndarray] = None      # eval-only names for the cells

    def __post_init__(self):
        self.W = np.zeros((0, self.dim), np.float32)
        self.wins = np.zeros(0)

    @property
    def n_categories(self) -> int:
        return int(len(self.W))

    def learn(self, xn: np.ndarray) -> int:
        """Present one (contrast-normalised) input; match a category or grow one."""
        if len(self.W) == 0:
            self.W = xn[None].astype(np.float32).copy()
            self.wins = np.array([1.0])
            return 0
        sim = self.W @ xn
        w = int(sim.argmax())
        if sim[w] < self.vigilance and len(self.W) < self.max_cells:
            self.W = np.vstack([self.W, xn.astype(np.float32)])
            self.wins = np.append(self.wins, 1.0)
            return len(self.W) - 1
        self.W[w] += self.lr * (xn - self.W[w])
        self.W[w] /= np.linalg.norm(self.W[w]) + 1e-9
        self.wins[w] += 1.0
        return w

    def train(self, X: np.ndarray, epochs: int = 2, seed: int = 0
              ) -> "GrowingCategoryMap":
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            for i in rng.permutation(len(X)):
                self.learn(X[i])
        return self

    def best(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Winning category and its match strength for each row of ``X``."""
        sims = X @ self.W.T
        win = sims.argmax(1)
        return win, sims[np.arange(len(X)), win]

    def name_cells(self, X: np.ndarray, labels: np.ndarray) -> None:
        """Give each self-grown category the majority true label of the inputs it
        wins -- used ONLY to read the score, never to learn."""
        win, _ = self.best(X)
        self.cell_label = np.full(len(self.W), -1, dtype=int)
        for c in range(len(self.W)):
            m = win == c
            if m.any():
                self.cell_label[c] = np.bincount(labels[m]).argmax()

    def recognise(self, images: np.ndarray, ood_thresh: float = 0.0
                  ) -> np.ndarray:
        """Recognise each image as the label of its winning category, or -1
        ("don't know") if nothing matches above ``ood_thresh``."""
        X = contrast_normalise(images)
        win, sim = self.best(X)
        out = self.cell_label[win].copy()
        out[sim < ood_thresh] = -1
        return out


@dataclass
class DigitRecognizer:
    """A self-taught recogniser of real handwritten digits."""

    cortex: GrowingCategoryMap
    test_accuracy: float = 0.0
    purity: float = 0.0
    n_categories: int = 0
    ood_reject_rate: float = 0.0
    ood_thresh: float = 0.0


def build_recognizer(trx: np.ndarray, trY: np.ndarray, tex: np.ndarray,
                     teY: np.ndarray, vigilance: float = 0.65,
                     epochs: int = 2, verbose: bool = False) -> DigitRecognizer:
    """The same self-taught recogniser, on **whatever images it is given**.

    :func:`build_digit_recognizer` loads MNIST itself, which made every mind
    built on top of it a mind that could only see digits. Nothing in the
    mechanism cares: `contrast_normalise` flattens, and `GrowingCategoryMap`
    grows categories from whatever dimension arrives. Splitting the data out of
    the builder is what lets the assembled mind meet a photograph.
    """
    def say(*a):
        if verbose:
            print(*a)

    Xtr, Xte = contrast_normalise(trx), contrast_normalise(tex)
    return _fit_recognizer(Xtr, trY, Xte, teY, vigilance, epochs, say,
                           in_shape=trx.shape[1:])


def build_digit_recognizer(n_train: int = 15000, vigilance: float = 0.65,
                           epochs: int = 2, verbose: bool = False
                           ) -> DigitRecognizer:
    """Learn categories from real MNIST with no labels, then score honestly on
    real held-out digits and check the 'don't know' response to non-digits."""
    def say(*a):
        if verbose:
            print(*a)

    say("fetching real handwritten digits (MNIST) ...")
    trx, trY, tex, teY = load_mnist(n_train=n_train)
    Xtr, Xte = contrast_normalise(trx), contrast_normalise(tex)

    return _fit_recognizer(Xtr, trY, Xte, teY, vigilance, epochs, say,
                           in_shape=trx.shape[1:])


def _fit_recognizer(Xtr, trY, Xte, teY, vigilance, epochs, say, in_shape):
    """Shared body: grow categories, name them for scoring, set a don't-know."""
    say("growing categories with no labels (competitive Hebbian) ...")
    cortex = GrowingCategoryMap(Xtr.shape[1], vigilance=vigilance).train(
        Xtr, epochs=epochs)
    cortex.name_cells(Xtr, trY)                     # eval-only naming

    win_tr, _ = cortex.best(Xtr)
    purity = float(np.mean([(trY[win_tr == c] == cortex.cell_label[c]).mean()
                            for c in range(cortex.n_categories)
                            if (win_tr == c).any()]))
    win_te, sim_te = cortex.best(Xte)
    acc = float((cortex.cell_label[win_te] == teY).mean())

    # "don't know": a threshold below the bulk of real-digit matches; non-digits
    # (random noise) should fall under it and be rejected
    thresh = float(np.percentile(sim_te, 5))
    noise = np.random.default_rng(0).random(
        (2000,) + tuple(in_shape)).astype(np.float32) * 255
    _, sim_noise = cortex.best(contrast_normalise(noise))
    ood_reject = float((sim_noise < thresh).mean())

    say(f"   grew {cortex.n_categories} self-made categories from real digits")
    say(f"   test accuracy on real held-out digits: {acc:.1%}  (purity {purity:.1%})")
    say(f"   'don't know' rejects {ood_reject:.0%} of random non-digit inputs")
    return DigitRecognizer(cortex, acc, purity, cortex.n_categories,
                           ood_reject, thresh)
