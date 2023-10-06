import time
import gc
from dataclasses import dataclass
from contextlib import contextmanager


@dataclass()
class Benchmark:
    name: str
    start: float
    end: float

    @property
    def duration_s(self):
        return (self.end - self.start) / 1000000000


@contextmanager
def benchmark(name, print_result=True):
    b = Benchmark(name, 0, 0)
    # gc.disable()
    b.start = time.perf_counter_ns()
    yield b
    b.end = time.perf_counter_ns()
    # gc.enable()
    # if print_result:
    print(f'Benchmark "{name}" took {b.duration_s} s')
