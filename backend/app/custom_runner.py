import ast
import builtins
import inspect
import multiprocessing as mp
import random
import string
import time
import tracemalloc
from queue import Empty
from types import MappingProxyType
from typing import Any

try:
    from app.schemas import BenchmarkResult, CustomBenchmarkRequest, FunctionAnalysis
except ModuleNotFoundError:
    from schemas import BenchmarkResult, CustomBenchmarkRequest, FunctionAnalysis

ALLOWED_IMPORT_ROOTS = {
    "array",
    "bisect",
    "collections",
    "copy",
    "dataclasses",
    "decimal",
    "enum",
    "fractions",
    "functools",
    "heapq",
    "itertools",
    "math",
    "operator",
    "queue",
    "random",
    "statistics",
    "string",
    "typing",
}

BLOCKED_BUILTINS = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "exit",
    "help",
    "input",
    "open",
    "quit",
}


class CustomAlgorithmError(ValueError):
    pass


def _safe_import(name: str, globals_: dict[str, Any] | None = None, locals_: dict[str, Any] | None = None, fromlist: tuple[str, ...] = (), level: int = 0) -> Any:
    root_name = name.split(".", 1)[0]
    if level != 0 or root_name not in ALLOWED_IMPORT_ROOTS:
        raise ImportError(f"Import '{name}' is not allowed in the benchmark sandbox.")
    return builtins.__import__(name, globals_, locals_, fromlist, level)


def _build_builtins() -> MappingProxyType:
    safe_builtins = {
        name: value
        for name, value in vars(builtins).items()
        if name not in BLOCKED_BUILTINS
    }
    safe_builtins["__import__"] = _safe_import
    return MappingProxyType(safe_builtins)


SAFE_GLOBALS = {
    "__builtins__": _build_builtins(),
}


def _compile_setup_code(code: str, filename: str) -> Any:
    tree = ast.parse(code, mode="exec")
    setup_body: list[ast.stmt] = []

    for node in tree.body:
        if isinstance(node, ast.Expr):
            continue

        if isinstance(node, ast.If) and _is_main_guard(node.test):
            continue

        setup_body.append(node)

    setup_tree = ast.Module(body=setup_body, type_ignores=tree.type_ignores)
    ast.fix_missing_locations(setup_tree)
    return compile(setup_tree, filename, "exec")


def _is_main_guard(test: ast.expr) -> bool:
    if not isinstance(test, ast.Compare) or len(test.ops) != 1 or len(test.comparators) != 1:
        return False

    left = test.left
    right = test.comparators[0]
    if not isinstance(left, ast.Name) or left.id != "__name__":
        return False
    if not isinstance(test.ops[0], ast.Eq):
        return False
    return isinstance(right, ast.Constant) and right.value == "__main__"


def validate_custom_code(code: str, function_name: str, skip_name_check: bool = False) -> ast.Module:
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise CustomAlgorithmError(f"Syntax error on line {exc.lineno}: {exc.msg}") from exc

    if not skip_name_check:
        functions = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

        if function_name not in functions:
            raise CustomAlgorithmError(f"Define a function named '{function_name}'.")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
            for name in names:
                root_name = name.split(".", 1)[0]
                if root_name not in ALLOWED_IMPORT_ROOTS:
                    raise CustomAlgorithmError(
                        f"Import '{name}' is blocked. Allowed imports include math, heapq, "
                        "bisect, itertools, functools, collections, statistics, dataclasses, "
                        "typing, and a few other standard algorithm helpers."
                    )

        if isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
            for name in names:
                root_name = name.split(".", 1)[0]
                if root_name not in ALLOWED_IMPORT_ROOTS:
                    raise CustomAlgorithmError(
                        f"Import '{name}' is blocked. Allowed imports include math, heapq, "
                        "bisect, itertools, functools, collections, statistics, dataclasses, "
                        "typing, and a few other standard algorithm helpers."
                    )

    return tree


def validate_imports(code: str) -> None:
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise CustomAlgorithmError(f"Syntax error on line {exc.lineno}: {exc.msg}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue

        for name in names:
            root_name = name.split(".", 1)[0]
            if root_name not in ALLOWED_IMPORT_ROOTS:
                raise CustomAlgorithmError(f"Import '{name}' is not allowed in the benchmark sandbox.")


def make_input(input_kind: str, input_size: int) -> Any:
    if input_kind == "int":
        return input_size
    if input_kind == "string":
        alphabet = string.ascii_lowercase
        return "".join(random.choice(alphabet) for _ in range(input_size))
    if input_kind == "none":
        return None
    if input_kind == "list_int_and_target":
        values = [random.randint(0, 100_000) for _ in range(input_size)]
        if len(values) >= 2:
            target = values[0] + values[-1]
        else:
            target = random.randint(0, 100_000)
        return values, target
    if input_kind == "two_ints":
        return input_size, max(1, input_size // 2)
    if input_kind == "graph_start_end":
        node_count = max(2, min(input_size, 250))
        graph: dict[int, list[tuple[int, int]]] = {node: [] for node in range(node_count)}
        for node in range(node_count - 1):
            graph[node].append((node + 1, random.randint(1, 10)))
            if node + 2 < node_count:
                graph[node].append((node + 2, random.randint(1, 20)))
        return graph, 0, node_count - 1
    if input_kind in {"agent_harness", "signature_args"}:
        return None
    if input_kind == "two_strings":

        letters = string.ascii_lowercase

        s = "".join(
            random.choice(letters)
            for _ in range(input_size)
        )

        t = "".join(
            random.choice(letters)
            for _ in range(
                max(1, input_size // 10)
            )
        )

        return s, t
    return [random.randint(0, 100_000) for _ in range(input_size)]
    

def _value_for_parameter(name: str, input_size: int) -> Any:
    lowered = name.lower()
    if lowered in {
        "n",
        "num",
        "number",
        "size",
        "limit",
        "count",
        "k",
        "budget",
        "capacity",
        "repeat",
        "repeats",
        "scale",
        "factor",
    }:
        return input_size
    if lowered in {"matrix", "table", "grid"}:
        side = max(2, min(input_size, 100))
        return [
            [random.randint(0, 100) for _ in range(10)]
            for _ in range(side)
        ]
    if lowered in {"config", "options", "settings"}:
        return {"weight": 2, "limit": input_size, "scale": 1}
    if lowered in {"start", "source", "src"}:
        return 0
    if lowered in {"end", "target_node", "destination", "dest", "dst"}:
        return max(1, min(input_size, 250) - 1)
    if lowered == "target":
        return random.randint(0, 200_000)
    if lowered in {"s", "text", "string", "word", "chars"}:
        return "".join(random.choice(string.ascii_lowercase) for _ in range(input_size))
    if lowered in {"graph", "network", "grid", "edges", "nodes", "paths"}:
        return make_input("graph_start_end", input_size)[0]
    return [random.randint(0, 100_000) for _ in range(input_size)]


def _signature_args(algorithm: Any, input_size: int) -> tuple[Any, ...]:
    signature = inspect.signature(algorithm)
    args: list[Any] = []
    for parameter in signature.parameters.values():
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        if parameter.default is not inspect.Parameter.empty and len(args) >= 1:
            continue
        args.append(_value_for_parameter(parameter.name, input_size))
    return tuple(args)


def _call_algorithm(algorithm: Any, input_value: Any, input_kind: str, input_size: int) -> Any:
    if input_kind == "signature_args":
        return algorithm(*_signature_args(algorithm, input_size))

    if input_value is None:
        return algorithm()

    if isinstance(input_value, list):
        return algorithm(input_value.copy())

    if isinstance(input_value, tuple):
        copied_args = [item.copy() if isinstance(item, list) else item for item in input_value]
        return algorithm(*copied_args)

    return algorithm(input_value)


def _consume_output(output: Any) -> None:
    if output is None or isinstance(output, (int, float, str, bool, bytes)):
        return

    try:
        iter(output)
    except TypeError:
        return

    # Force lazy iterables to do their work without retaining huge outputs.
    for _ in output:
        pass


def _run_custom_algorithm(
    code: str,
    function_name: str,
    algorithm_name: str,
    input_kind: str,
    harness_code: str | None,
    input_value: Any,
    input_size: int,
    result_queue: mp.Queue,
) -> None:
    try:
        namespace = dict(SAFE_GLOBALS)
        exec(_compile_setup_code(code, "<custom_algorithm>"), namespace, namespace)
        if input_kind == "agent_harness":
            if not harness_code:
                raise CustomAlgorithmError("The benchmark plan did not include harness code.")
            exec(_compile_setup_code(harness_code, "<benchmark_harness>"), namespace, namespace)
            prepare_input = namespace.get("prepare_input")
            run_target = namespace.get("run_target")
            if not callable(prepare_input) or not callable(run_target):
                raise CustomAlgorithmError(
                    "Groq harness must define prepare_input(input_size) and run_target(prepared)."
                )
            prepared_input = prepare_input(input_size)
            algorithm = run_target
        else:
            algorithm = namespace.get(function_name)
            prepared_input = input_value

            if not callable(algorithm):
                raise CustomAlgorithmError(f"'{function_name}' is not callable.")

        tracemalloc.start()
        start_time = time.perf_counter()

        output = (
            algorithm(prepared_input)
            if input_kind == "agent_harness"
            else _call_algorithm(algorithm, prepared_input, input_kind, input_size)
        )
        _consume_output(output)

        elapsed = time.perf_counter() - start_time
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        result_queue.put(
            {
                "ok": True,
                "result": {
                    "algorithm": algorithm_name,
                    "execution_time_ms": round(elapsed * 1000, 4),
                    "peak_memory_kb": round(peak_memory / 1024, 4),
                    "input_size": input_size,
                },
            }
        )
    except Exception as exc:
        result_queue.put({"ok": False, "error": str(exc) or type(exc).__name__})


def run_custom_benchmark(
    request: CustomBenchmarkRequest,
    analysis: FunctionAnalysis | None = None,
) -> BenchmarkResult:
    # Skip function name validation for agent_harness — the harness handles any callable
    skip_name_check = request.input_kind == "agent_harness"
    validate_custom_code(request.code, request.function_name, skip_name_check=skip_name_check)
    if request.harness_code:
        validate_imports(request.harness_code)

    input_value = make_input(request.input_kind, request.input_size)
    result_queue: mp.Queue = mp.Queue(maxsize=1)
    process = mp.Process(
        target=_run_custom_algorithm,
        args=(
            request.code,
            request.function_name,
            request.algorithm_name,
            request.input_kind,
            request.harness_code,
            input_value,
            request.input_size,
            result_queue,
        ),
    )

    process.start()
    process.join(request.timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join(1)
        raise TimeoutError(f"Custom function exceeded {request.timeout_seconds:g}s.")

    try:
        payload = result_queue.get_nowait()
    except Empty as exc:
        raise CustomAlgorithmError("Custom function did not return a benchmark result.") from exc

    if not payload["ok"]:
        raise CustomAlgorithmError(payload["error"])

    return BenchmarkResult(**payload["result"], analysis=analysis)
