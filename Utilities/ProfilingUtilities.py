from contextlib import contextmanager
from time import perf_counter

@contextmanager
def timer(label=""):
    t0 = perf_counter()
    yield
    elapsed = perf_counter() - t0
    print(f"\n  [{label}] elapsed time: {elapsed:.3f}s")