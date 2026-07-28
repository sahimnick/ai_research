"""Import-time stub so the package can be imported for structural verification."""
class _Stub:
    def __init__(self, *a, **k): pass
    def __getattr__(self, name): return _Stub()
    def __call__(self, *a, **k):
        if len(a) == 1 and callable(a[0]) and not k:
            return a[0]          # used as a decorator
        return _Stub()
def __getattr__(name): return _Stub()
