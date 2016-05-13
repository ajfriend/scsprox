import time
from contextlib import contextmanager

@contextmanager
def SimpleTimer():
    elapsed = Elapsed(None)
    start = time.time()
    try:
        yield elapsed
    finally:
        end = time.time()
        elapsed.time = end-start


@contextmanager
def DictTimer(label='time', d=None):
    if d is None:
        d = {}
    start = time.time()
    try:
        yield d
    finally:
        end = time.time()
        d[label] = end-start
