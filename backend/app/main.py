import math
import random

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

try:
    from app.algorithms import (
        bubble_sort,
        counting_sort,
        heap_sort,
        insertion_sort,
        merge_sort,
        quick_sort,
        selection_sort,
        tim_sort,
    )
    from app.benchmark import BenchmarkRunner, SortAlgorithm
    from app.analyzer_agent import analyze_function, plan_benchmark, repair_benchmark_plan
    from app.custom_runner import CustomAlgorithmError, run_custom_benchmark
    from app.schemas import (
        AutoBenchmarkRequest,
        BenchmarkPlan,
        BenchmarkRequest,
        BenchmarkResult,
        CustomBenchmarkRequest,
        FunctionAnalysis,
    )
except ModuleNotFoundError:
    from algorithms import (
        bubble_sort,
        counting_sort,
        heap_sort,
        insertion_sort,
        merge_sort,
        quick_sort,
        selection_sort,
        tim_sort,
    )
    from benchmark import BenchmarkRunner, SortAlgorithm
    from analyzer_agent import analyze_function, plan_benchmark, repair_benchmark_plan
    from custom_runner import CustomAlgorithmError, run_custom_benchmark
    from schemas import (
        AutoBenchmarkRequest,
        BenchmarkPlan,
        BenchmarkRequest,
        BenchmarkResult,
        CustomBenchmarkRequest,
        FunctionAnalysis,
    )

app = FastAPI(title="complexity API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALGORITHMS: dict[str, SortAlgorithm] = {
    "bubble_sort": bubble_sort,
    "selection_sort": selection_sort,
    "insertion_sort": insertion_sort,
    "merge_sort": merge_sort,
    "quick_sort": quick_sort,
    "heap_sort": heap_sort,
    "counting_sort": counting_sort,
    "tim_sort": tim_sort,
}

def sanitize_code(code: str) -> str:
    """Strip markdown code fences if present, then return clean source."""
    code = code.strip()

    if code.startswith("```"):
        lines = code.splitlines()

        # Remove opening fence line (e.g. ```python or just ```)
        lines = lines[1:]

        # Remove closing fence
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]

        code = "\n".join(lines)

    return code.strip()

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/algorithms")
async def list_algorithms() -> dict[str, list[str]]:
    return {"algorithms": list(ALGORITHMS)}


@app.post("/benchmark", response_model=BenchmarkResult)
async def run_benchmark(request: BenchmarkRequest) -> BenchmarkResult:
    algorithm = ALGORITHMS.get(request.algorithm_name)
    if algorithm is None:
        raise HTTPException(status_code=404, detail="Algorithm not found")

    data = [random.randint(0, 100_000) for _ in range(request.input_size)]
    return BenchmarkRunner(algorithm).run(data)


@app.post("/benchmark/custom", response_model=BenchmarkResult)
async def run_custom_algorithm_benchmark(request: CustomBenchmarkRequest) -> BenchmarkResult:
    try:
        analysis = analyze_function(request) if request.analyze_with_agent else None
        return run_custom_benchmark(request, analysis)
    except TimeoutError as exc:
        raise HTTPException(status_code=408, detail=str(exc)) from exc
    except CustomAlgorithmError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/benchmark/analyze", response_model=BenchmarkResult)
async def analyze_and_benchmark_python(request: AutoBenchmarkRequest) -> BenchmarkResult:
    clean_code = sanitize_code(request.code)

    plan = plan_benchmark(
        clean_code,
        request.analyze_with_agent
    )
    return _run_groq_analysis(request, plan)


def _run_planned_benchmark(
    request: AutoBenchmarkRequest,
    plan,
) -> BenchmarkResult:
    custom_request = CustomBenchmarkRequest(
        algorithm_name=plan.algorithm_name,
        function_name=plan.function_name,
        code=sanitize_code(request.code),
        input_size=request.input_size,
        input_kind=plan.input_kind,
        timeout_seconds=request.timeout_seconds,
        analyze_with_agent=request.analyze_with_agent,
        harness_code=sanitize_code(plan.harness_code) if plan.harness_code else None,
    )

    analysis = analyze_function(custom_request) if request.analyze_with_agent else None
    result = run_custom_benchmark(custom_request, analysis)
    result.plan = plan
    return result


def _run_groq_analysis(
    request: AutoBenchmarkRequest,
    plan: BenchmarkPlan,
) -> BenchmarkResult:
    custom_request = CustomBenchmarkRequest(
        algorithm_name=plan.algorithm_name,
        function_name=plan.function_name,
        code=sanitize_code(request.code),
        input_size=request.input_size,
        input_kind=plan.input_kind,
        timeout_seconds=request.timeout_seconds,
        analyze_with_agent=request.analyze_with_agent,
        harness_code=sanitize_code(plan.harness_code) if plan.harness_code else None,
    )
    analysis = analyze_function(custom_request) if request.analyze_with_agent else None
    theoretical_value = _theoretical_input_size_value(
        analysis,
        request.input_size,
    )

    if analysis is not None:
        if not analysis.input_size_result:
            analysis.input_size_result = (
                f"For input size {request.input_size}, the dominant-term estimate is "
                f"about {theoretical_value:g} operations."
            )
        analysis.notes = [
            *analysis.notes,
            "Result is calculated from Groq's complexity analysis instead of executing a benchmark harness.",
        ][:5]

    return BenchmarkResult(
        algorithm=plan.algorithm_name,
        execution_time_ms=round(theoretical_value, 4),
        peak_memory_kb=0,
        input_size=request.input_size,
        analysis=analysis,
        plan=plan,
    )


def _theoretical_input_size_value(
    analysis: FunctionAnalysis | None,
    input_size: int,
) -> float:
    if analysis and analysis.estimated_operations is not None:
        return max(float(analysis.estimated_operations), 0)

    complexity = (analysis.time_complexity if analysis else "O(n)").lower()
    n = max(input_size, 1)
    log_n = max(math.log2(n), 1)

    if "n!" in complexity:
        return float(math.factorial(min(n, 12)))
    if "2^n" in complexity or "2ⁿ" in complexity:
        return float(2 ** min(n, 40))
    if "n^3" in complexity or "n³" in complexity:
        return float(n ** 3)
    if "n^2" in complexity or "n²" in complexity:
        return float(n ** 2)
    if "n log" in complexity or "nlog" in complexity or "n*log" in complexity:
        return float(n * log_n)
    if "log" in complexity:
        return float(log_n)
    if "1" in complexity and "n" not in complexity:
        return 1.0
    return float(n)


def _timeout_benchmark_result(
    request: AutoBenchmarkRequest,
    plan: BenchmarkPlan,
    message: str,
) -> BenchmarkResult:
    return BenchmarkResult(
        algorithm=plan.algorithm_name,
        execution_time_ms=round(request.timeout_seconds * 1000, 4),
        peak_memory_kb=0,
        input_size=request.input_size,
        plan=plan,
        analysis=FunctionAnalysis(
            summary=(
                f"`{plan.function_name}` exceeded the {request.timeout_seconds:g}s benchmark cap. "
                "The chart uses the timeout cap as the measured upper bound."
            ),
            complexity="Timeout",
            brief=(
                f"The selected function `{plan.function_name}` started running but did not finish "
                "within the configured benchmark window."
            ),
            time_complexity="Exceeded timeout",
            space_complexity="Unknown",
            input_contract=f"Planned input kind: {plan.input_kind}.",
            notes=[message],
            suggestions=[
                "Try lowering recursion depth, adding a stopping condition, or reducing repeated states.",
                "Increase the timeout only if you expect this function to run for a long time.",
            ],
            agent_enabled=plan.agent_enabled,
        ),
    )
