"""Regression lock for AUDIT.md A1 — the package must import without PyTorch.

`backend.py` once wrote ``@torch.no_grad()`` as a class-body decorator, which
Python evaluates at import time. On a machine without torch that raised
``AttributeError: 'NoneType' has no attribute 'no_grad'`` and took all 60
pure-NumPy modules down with it.

These tests simulate torch's absence even when torch *is* installed, so the
lock keeps working on developer machines that have it. Run with pytest, or
directly:  ``python3 tests/test_import_without_torch.py``
"""
import subprocess
import sys
import textwrap

# Blocks `import torch` (and any submodule) for the child interpreter, then
# imports the package. Parameterised by the exception the blocker raises, so we
# cover both "not installed" and "installed but broken".
_CHILD = textwrap.dedent('''
    import sys

    class _Block:
        def find_module(self, name, path=None):
            return self.find_spec(name, path)
        def find_spec(self, name, path=None, target=None):
            if name == "torch" or name.startswith("torch."):
                raise {exc}("blocked for test")
            return None

    sys.meta_path.insert(0, _Block())
    for mod in [m for m in sys.modules if m == "torch" or m.startswith("torch.")]:
        del sys.modules[mod]

    import neurobrain as nb
    from neurobrain.core import backend

    assert backend._HAVE_TORCH is False, "torch should look absent here"
    assert backend.torch is None, "torch should be None here"
    assert len(nb.__all__) == 247, f"__all__ changed: {{len(nb.__all__)}}"
    assert nb.available_devices() == {{"numpy": True}}, nb.available_devices()

    # the numpy substrate must still be fully usable
    p = nb.Population(64, "regular_spiking")
    import numpy as np
    spikes = sum(int(p.step(np.full(64, 10.0, np.float32), 1.0).sum())
                 for _ in range(40))
    assert spikes > 0, "population did not spike"

    # and TorchPopulation must refuse cleanly rather than crash oddly
    try:
        backend.TorchPopulation(8)
    except ImportError:
        pass
    else:
        raise AssertionError("TorchPopulation should raise ImportError here")

    print("OK", spikes)
''')


def _run(exc: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", _CHILD.format(exc=exc)],
                          capture_output=True, text=True)


def test_imports_when_torch_is_missing():
    """ImportError from `import torch` — torch simply not installed."""
    r = _run("ImportError")
    assert r.returncode == 0, f"import failed without torch:\n{r.stderr}"
    assert r.stdout.startswith("OK"), r.stdout


def test_imports_when_torch_is_broken():
    """OSError from `import torch` — installed but its shared libs are broken.

    This is a real situation (mismatched CUDA, half-finished install) and the
    guard is `except Exception`, so it must survive a non-ImportError too."""
    r = _run("OSError")
    assert r.returncode == 0, f"import failed with a broken torch:\n{r.stderr}"
    assert r.stdout.startswith("OK"), r.stdout


def test_no_grad_is_real_when_torch_is_present():
    """The fix must not have quietly weakened the no-autograd guarantee.

    Skipped when torch is genuinely unavailable."""
    try:
        import torch  # noqa: F401
    except Exception:
        print("skip: torch not installed")
        return
    from neurobrain.core.backend import _no_grad, _HAVE_TORCH
    assert _HAVE_TORCH is True

    @_no_grad
    def f(x):
        return (x * 2).sum()

    out = f(torch.ones(4, requires_grad=True))
    assert not out.requires_grad, "_no_grad is not suppressing autograd"


def test_backend_declares_no_gradient_machinery():
    """The package must contain no autograd/optimiser/loss usage at all."""
    import pathlib, re
    bad = []
    pat = re.compile(r"\.backward\(\)|torch\.optim|nn\.Module|requires_grad_\(")
    for p in pathlib.Path(__file__).resolve().parents[1].joinpath(
            "neurobrain").rglob("*.py"):
        src = p.read_text()
        # strip docstrings/comments: they legitimately *mention* these words
        code = re.sub(r'""".*?"""', "", src, flags=re.S)
        code = re.sub(r"#.*", "", code)
        if pat.search(code):
            bad.append(str(p))
    assert not bad, f"gradient machinery found in: {bad}"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL {name}: {e}")
    print("all passed" if not fails else f"{fails} failed")
    sys.exit(1 if fails else 0)
