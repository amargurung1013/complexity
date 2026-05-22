import time
import tracemalloc
from collections.abc import Callable

try:
    from app.schemas import BenchmarkResult
except ModuleNotFoundError:
    from schemas import BenchmarkResult

SortAlgorithm = Callable[[list[int]], None]


class BenchmarkRunner:
    def __init__(self, algorithm: SortAlgorithm):
        self.algorithm = algorithm

    def run(self, data: list[int]) -> BenchmarkResult:
        tracemalloc.start()
        start_time = time.perf_counter()

        self.algorithm(data.copy())

        elapsed = time.perf_counter() - start_time
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return BenchmarkResult(
            algorithm=self.algorithm.__name__,
            execution_time_ms=round(elapsed * 1000, 4),
            peak_memory_kb=round(peak_memory / 1024, 4),
            input_size=len(data),
        )
