from time import perf_counter
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class _Counter:
    time_elapsed = 0.
    count = 0

    def __repr__(self):
        return f'{{time: {self.time_elapsed:.3e}, count: {self.count}, avg = {self.time_elapsed / self.count:.3e}}}'


_pool = defaultdict(_Counter)


class TimerContext:
    def __init__(self, ids):
        self.ids = ids.split()

    def __enter__(self):
        for idx in self.ids:
            _pool[idx].time_elapsed -= perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        for idx in self.ids:
            _pool[idx].time_elapsed += perf_counter()


def profile(ids):
    return TimerContext(ids)


def show(ids=None):
    if ids is None:
        ids = _pool.keys()
    else:
        ids = ids.split()

    print(" | ".join([f"{key}: {repr(_pool[key])}" for key in ids]))


__all__ = ['profile', 'show', 'TimerContext']
