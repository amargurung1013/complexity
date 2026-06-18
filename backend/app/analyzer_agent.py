import ast
import json
import os
from typing import TypedDict

from dotenv import load_dotenv

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_groq import ChatGroq
    from langgraph.graph import END, StateGraph
except ImportError:
    ChatGroq = None
    END = "__end__"
    HumanMessage = None
    StateGraph = None
    SystemMessage = None

try:
    from app.schemas import BenchmarkPlan, CustomBenchmarkRequest, FunctionAnalysis
except ModuleNotFoundError:
    from schemas import BenchmarkPlan, CustomBenchmarkRequest, FunctionAnalysis

load_dotenv()


def format_big_o(value):

    superscript_map = {
        "0": "⁰",
        "1": "¹",
        "2": "²",
        "3": "³",
        "4": "⁴",
        "5": "⁵",
        "6": "⁶",
        "7": "⁷",
        "8": "⁸",
        "9": "⁹"
    }

    result = ""
    i = 0

    while i < len(value):

        if value[i] == "^":

            i += 1

            while i < len(value) and value[i].isdigit():

                result += superscript_map.get(
                    value[i],
                    value[i]
                )

                i += 1

            continue

        result += value[i]
        i += 1

    return result

class AnalyzerState(TypedDict):
    code: str
    function_name: str | None
    input_kind: str | None
    plan: BenchmarkPlan | None
    analysis: FunctionAnalysis | None


def _function_arg_names(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> list[str]:
    if isinstance(node, ast.ClassDef):
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "__init__":
                return [arg.arg for arg in item.args.args if arg.arg != "self"]
        return []

    return [arg.arg for arg in node.args.args]


def _infer_input_kind(function_name: str, arg_names: list[str]) -> str:
    if not arg_names:
        return "none"

    first = arg_names[0].lower()
    if len(arg_names) == 1:
        if first in {"n", "num", "number", "size", "limit", "count"}:
            return "int"
        if first in {"s", "text", "string", "word", "chars"}:
            return "string"
        return "list_int"

    second = arg_names[1].lower()
    if len(arg_names) == 2:

    # List + target patterns
        if (
            first in {
                "nums",
                "values",
                "arr",
                "array",
                "items",
                "list"
            }
            and second == "target"
        ):
            return "list_int_and_target"

        # Two string patterns
        if (
            first in {"s", "text", "string", "word"}
            and second in {"t", "pattern", "target", "substring"}
        ):
            return "two_strings"

        # Generic graph/path pattern
        if (
            first in {"graph", "grid", "network"}
            and second in {"start", "source"}
        ):
            return "graph_start_end"

        return "two_ints"

    lowered_args = [name.lower() for name in arg_names]
    if len(arg_names) == 3 and {"start", "end"}.issubset(lowered_args):
        first_arg = lowered_args[0]
        if first_arg in {"graph", "network", "grid", "edges", "nodes", "paths"}:
            return "graph_start_end"

    return "signature_args"


def _fallback_plan(code: str, agent_enabled: bool = False, reason: str | None = None) -> BenchmarkPlan:
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError:
        return BenchmarkPlan(
            algorithm_name="Python function",
            function_name="main",
            input_kind="agent_harness",
            reason=reason or "Could not parse code, so the planner used a default entry point.",
            agent_enabled=agent_enabled,
            harness_code=(
                "def prepare_input(input_size):\n"
                "    return input_size\n\n"
                "def run_target(prepared):\n"
                "    return main(prepared)\n"
            ),
        )

    # Look for public functions AND classes
    callables = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and not node.name.startswith("_")
    ]

    if not callables:
        return BenchmarkPlan(
            algorithm_name="Python function",
            function_name="main",
            input_kind="agent_harness",
            reason=reason or "No public function or class was found, so the planner used `main`.",
            agent_enabled=agent_enabled,
            harness_code=(
                "def prepare_input(input_size):\n"
                "    return input_size\n\n"
                "def run_target(prepared):\n"
                "    return main(prepared)\n"
            ),
        )

    target = callables[-1]
    
    # For classes, generate a harness that instantiates and exercises methods
    if isinstance(target, ast.ClassDef):
        # Try to detect __init__ parameters
        init_params = []
        for node in target.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__init__":
                init_params = [arg.arg for arg in node.args.args if arg.arg != "self"]
                break
        
        # Build constructor call based on detected params
        if not init_params:
            constructor_call = f"{target.name}()"
        elif len(init_params) == 1 and init_params[0].lower() in {"capacity", "size", "maxsize", "limit"}:
            constructor_call = f"{target.name}(capacity=min(len(prepared), 1000))"
        else:
            # Generic: pass input_size as first arg
            constructor_call = f"{target.name}()"
        
        return BenchmarkPlan(
            algorithm_name=target.name.replace("_", " ").title(),
            function_name=target.name,
            input_kind="agent_harness",
            reason=reason or f"Selected class `{target.name}` and generated a harness to exercise its methods.",
            agent_enabled=agent_enabled,
            harness_code=(
                "import random\n\n"
                "def prepare_input(input_size):\n"
                "    random.seed(42)\n"
                "    return [(''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(10)), "
                "random.randint(0, 100000)) for _ in range(input_size)]\n\n"
                "def run_target(prepared):\n"
                f"    obj = {constructor_call}\n"
                "    for key, val in prepared:\n"
                "        if hasattr(obj, 'insert'):\n"
                "            obj.insert(key)\n"
                "        if hasattr(obj, 'search'):\n"
                "            obj.search(key)\n"
                "        if hasattr(obj, 'put'):\n"
                "            obj.put(key, val)\n"
                "        if hasattr(obj, 'get'):\n"
                "            obj.get(key)\n"
                "        if hasattr(obj, 'add'):\n"
                "            obj.add(key)\n"
                "        if hasattr(obj, 'contains'):\n"
                "            obj.contains(key)\n"
            ),
        )
    
    # For functions, use the existing arg-based harness
    arg_names = _function_arg_names(target)
    harness = _build_fallback_harness(target.name, arg_names)

    return BenchmarkPlan(
        algorithm_name=target.name.replace("_", " ").title(),
        function_name=target.name,
        input_kind="agent_harness",
        reason=reason or f"Selected `{target.name}` from the pasted code and generated a harness.",
        agent_enabled=agent_enabled,
        harness_code=harness,
    )


VALID_INPUT_KINDS = {
    "list_int", "int", "string", "none",
    "list_int_and_target", "two_ints",
    "graph_start_end", "signature_args", "agent_harness", "two_strings",
}


def _build_fallback_harness(function_name: str, arg_names: list[str]) -> str:
    """Generate a best-effort harness when Groq is unavailable."""
    if not arg_names:
        return (
            "def prepare_input(input_size):\n"
            "    return None\n\n"
            f"def run_target(prepared):\n"
            f"    return {function_name}()\n"
        )

    # Build per-argument value expressions
    arg_exprs: list[str] = []
    for name in arg_names:
        low = name.lower()
        if low in {"n", "num", "number", "size", "limit", "count", "k", "capacity"}:
            arg_exprs.append("input_size")
        elif low in {"s", "text", "string", "word", "chars", "t", "pattern"}:
            arg_exprs.append(
                '"".join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(input_size))'
            )
        elif low in {"graph", "network", "grid", "edges"}:
            arg_exprs.append(
                "[(i, i+1) for i in range(max(2, min(input_size, 200)) - 1)]"
            )
        elif low in {"start", "source", "src"}:
            arg_exprs.append("0")
        elif low in {"end", "target_node", "destination", "dest"}:
            arg_exprs.append("max(1, min(input_size, 200) - 1)")
        elif low == "target":
            arg_exprs.append("random.randint(0, input_size * 2)")
        else:
            arg_exprs.append("[random.randint(0, 100000) for _ in range(input_size)]")

    if len(arg_exprs) == 1:
        prepare = f"    return {arg_exprs[0]}\n"
        call = f"    return {function_name}(prepared)\n"
    else:
        items = ",\n        ".join(arg_exprs)
        prepare = f"    return (\n        {items},\n    )\n"
        call = f"    return {function_name}(*prepared)\n"

    return (
        "import random\n\n"
        "def prepare_input(input_size):\n"
        "    random.seed(42)\n"
        f"{prepare}\n"
        "def run_target(prepared):\n"
        f"{call}"
    )

def _strip_fences(code: str | None) -> str | None:
    """Remove markdown code fences from a string if present."""
    if not code:
        return code
    code = code.strip()
    if code.startswith("```"):
        lines = code.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        code = "\n".join(lines).strip()
    return code or None


def _parse_plan(raw_text: str, code: str) -> BenchmarkPlan:
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1:
        print(f"[PARSE ERROR] No JSON found in Groq response")
        return _fallback_plan(code, agent_enabled=True, reason="Groq did not return parseable JSON.")

    try:
        payload = json.loads(raw_text[start : end + 1])
        
        if "function_name" not in payload:
            print(f"[PARSE ERROR] Missing 'function_name' in Groq JSON: {payload.keys()}")
            return _fallback_plan(code, agent_enabled=True, reason="Groq JSON missing function_name.")
        
        raw_kind = str(payload.get("input_kind") or "agent_harness")
        input_kind = raw_kind if raw_kind in VALID_INPUT_KINDS else "agent_harness"

        return BenchmarkPlan(
            algorithm_name=str(payload.get("algorithm_name") or "Python function")[:80],
            function_name=str(payload["function_name"])[:80],
            input_kind=input_kind,
            reason=str(payload.get("reason") or "Groq selected the function and generated input."),
            agent_enabled=True,
            harness_code=_strip_fences(payload.get("harness_code")),
        )
    except json.JSONDecodeError as e:
        print(f"[PARSE ERROR] JSON decode failed: {e}")
        return _fallback_plan(code, agent_enabled=True, reason="Groq returned invalid JSON.")
    except Exception as e:
        print(f"[PARSE ERROR] Unexpected error: {e}")
        return _fallback_plan(code, agent_enabled=True, reason="Groq returned an invalid plan.")

def _fallback_analysis(request: CustomBenchmarkRequest, agent_enabled: bool = False) -> FunctionAnalysis:
    readable_name = request.function_name.replace("_", " ")
    family = _classify_from_name(request.function_name)
    return FunctionAnalysis(
        summary=(
            f"`{request.function_name}` is benchmarked with generated {request.input_kind} input "
            f"at size {request.input_size}."
        ),
        complexity=family["time_complexity"],
        brief=f"This appears to run the {readable_name} function and measure its runtime and peak memory.",
        time_complexity=family["time_complexity"],
        space_complexity=family["space_complexity"],
        algorithm_family=family["family"],
        strategy=family["strategy"],
        stability=family["stability"],
        storage_model=family["storage_model"],
        beginner_explanation=(
            f"`{request.function_name}` appears to solve {readable_name}. Run it once with Groq enabled "
            "for a deeper explanation."
        ),
        interview_explanation=(
            f"Describe `{request.function_name}` by its input shape, dominant operation, and measured "
            "growth across increasing input sizes."
        ),
        professor_explanation=(
            f"The selected procedure `{request.function_name}` should be analyzed by recurrence, loop "
            "nesting, and auxiliary state."
        ),
        input_contract=f"Input kind: {request.input_kind}; size: {request.input_size}.",
        notes=[
            "Agent analysis is unavailable until GROQ_API_KEY is configured."
            if not agent_enabled
            else "Agent response could not be parsed, so a static summary was used."
        ],
        suggestions=[
            "Add a clear function name and keep setup code above the function.",
            "Use a smaller input size first when benchmarking unfamiliar code.",
        ],
        agent_enabled=agent_enabled,
    )


def _classify_from_name(function_name: str) -> dict[str, str]:
    lowered = function_name.lower()
    joined = lowered.replace("_", " ")

    if "substring" in lowered or "subarray" in lowered or "window" in lowered:
        return {
            "family": "Sliding window",
            "strategy": "Maintains a moving left/right boundary and updates state as the window changes.",
            "time_complexity": "O(n)",
            "space_complexity": "O(k)",
            "stability": "Not applicable",
            "storage_model": "Uses auxiliary lookup state",
        }
    if "binary" in lowered or "search" in lowered:
        return {
            "family": "Search algorithm",
            "strategy": "Narrows or scans the search space until a target condition is found.",
            "time_complexity": "O(n)",
            "space_complexity": "O(1)",
            "stability": "Not applicable",
            "storage_model": "Usually constant auxiliary storage",
        }
    if "dfs" in lowered or "depth" in joined:
        return {
            "family": "Graph traversal",
            "strategy": "Explores reachable states depth-first using recursion or an explicit stack.",
            "time_complexity": "O(V + E)",
            "space_complexity": "O(V)",
            "stability": "Not applicable",
            "storage_model": "Uses visited state and call/work stack",
        }
    if "bfs" in lowered or "breadth" in joined:
        return {
            "family": "Graph traversal",
            "strategy": "Explores reachable states breadth-first using a queue.",
            "time_complexity": "O(V + E)",
            "space_complexity": "O(V)",
            "stability": "Not applicable",
            "storage_model": "Uses visited state and queue storage",
        }
    if "dp" in lowered or "memo" in lowered or "cache" in lowered:
        return {
            "family": "Dynamic programming",
            "strategy": "Stores overlapping subproblem results to avoid repeated computation.",
            "time_complexity": "Depends on state count",
            "space_complexity": "Depends on memo/table size",
            "stability": "Not applicable",
            "storage_model": "Uses memoization or tabulation storage",
        }

    return {
        "family": "Code-derived algorithm",
        "strategy": "Classified from the selected function name, signature, and benchmark harness.",
        "time_complexity": "Requires measurement",
        "space_complexity": "Requires measurement",
        "stability": "Not applicable",
        "storage_model": "Depends on implementation",
    }


def _parse_analysis(raw_text: str, request: CustomBenchmarkRequest) -> FunctionAnalysis:
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1:
        return _fallback_analysis(request, agent_enabled=True)

    try:
        payload = json.loads(raw_text[start : end + 1])
    except json.JSONDecodeError:
        return _fallback_analysis(request, agent_enabled=True)

    fallback = _classify_from_name(request.function_name)
    time_complexity = format_big_o(str(payload.get("time_complexity") or fallback["time_complexity"]))
    space_complexity = format_big_o(str(payload.get("space_complexity") or fallback["space_complexity"]))
    estimated_operations = payload.get("estimated_operations")
    try:
        estimated_operations = (
            float(str(estimated_operations).replace(",", ""))
            if estimated_operations is not None
            else None
        )
    except ValueError:
        estimated_operations = None

    return FunctionAnalysis(

    summary=str(
        payload.get("summary")
        or payload.get("function_analysis")
        or f"Benchmarks `{request.function_name}`."
    ),

    complexity=time_complexity,

    brief=str(
        payload.get("brief")
        or payload.get("reason")
        or payload.get("summary")
        or payload.get("function_analysis")
        or f"Runs `{request.function_name}`."
    ),

    time_complexity=time_complexity,

    space_complexity=space_complexity,

    algorithm_family=str(
        payload.get("algorithm_family")
        or payload.get("classification")
        or fallback["family"]
    ),

    strategy=str(
        payload.get("strategy")
        or payload.get("operational_invariant")
        or fallback["strategy"]
    ),

    stability=str(
        payload.get("stability")
        or fallback["stability"]
    ),

    storage_model=str(
        payload.get("storage_model")
        or payload.get("memory_model")
        or fallback["storage_model"]
    ),

    beginner_explanation=str(
        payload.get("beginner_explanation")
        or payload.get("eli5")
        or ""
    ),

    interview_explanation=str(
        payload.get("interview_explanation")
        or ""
    ),

    professor_explanation=str(
        payload.get("professor_explanation")
        or ""
    ),

    input_contract=str(
        payload.get("input_contract")
        or f"Selected callable: {request.function_name}; input kind: {request.input_kind}."
    ),

    input_size_result=str(
        payload.get("input_size_result")
        or ""
    ),

    estimated_operations=estimated_operations,

    notes=[str(item) for item in payload.get("notes", [])][:5],

    suggestions=[
        str(item)
        for item in payload.get("suggestions", [])
    ][:5],

    agent_enabled=True,
)


def analyze_function(request: CustomBenchmarkRequest) -> FunctionAnalysis:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or ChatGroq is None or StateGraph is None:
        return _fallback_analysis(request)

    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    def analyze_node(state: AnalyzerState) -> AnalyzerState:
        llm = ChatGroq(api_key=api_key, model=model, temperature=0.1)
        messages = [
            SystemMessage(
                content=(
                    "You are an expert Python algorithm analyst.\n"
                    "The user may paste any Python algorithm code: one function, many functions, "
                    "a class, helper code, or a callable with any number of parameters.\n"
                    "Analyze the actual pasted code and selected callable. Do not assume a fixed "
                    "input shape, a fixed number of variables, or a known LeetCode pattern.\n"
                    "Return ONLY valid JSON with this exact shape:\n"
                    "{\n"
                    "  \"summary\": \"Plain summary of what the algorithm does.\",\n"
                    "  \"time_complexity\": \"Big-O time complexity\",\n"
                    "  \"space_complexity\": \"Big-O space complexity\",\n"
                    "  \"brief\": \"Short reason for the complexity result.\",\n"
                    "  \"algorithm_family\": \"Specific family, e.g. Sorting, Dynamic programming, Graph traversal, Data structure\",\n"
                    "  \"strategy\": \"The core technique or invariant in one precise sentence.\",\n"
                    "  \"stability\": \"Stable, not stable, or not applicable\",\n"
                    "  \"storage_model\": \"How auxiliary memory is used\",\n"
                    "  \"beginner_explanation\": \"Plain language explanation for a beginner\",\n"
                    "  \"interview_explanation\": \"Concise interview answer\",\n"
                    "  \"professor_explanation\": \"Formal CS terminology explanation\",\n"
                    "  \"input_size_result\": \"Mathematical result for the provided input size, e.g. n=1000 gives about 1000000 dominant operations\",\n"
                    "  \"estimated_operations\": 1000000,\n"
                    "  \"notes\": [\"up to five useful notes\"],\n"
                    "  \"suggestions\": [\"up to five improvement or testing suggestions\"]\n"
                    "}\n\n"
                    "Rules:\n"
                    "- Infer parameter roles from the code, not from parameter count.\n"
                    "- Analyze loops, recursion, nested operations, helper calls, and built-in operations.\n"
                    "- State complexity in terms of meaningful variables such as n, m, V, E, rows, or cols.\n"
                    "- If several inputs matter, include them in the Big-O expression.\n"
                    "- For input_size_result, plug the requested input size into the dominant Big-O term.\n"
                    "- For estimated_operations, return a JSON number for that dominant-term estimate.\n"
                    "- If complexity depends on data structure operations or unclear helper behavior, say so briefly.\n"
                    "- Return ONLY JSON\n"
                    "- No markdown"
                )
            ),
            HumanMessage(
                content=(
                    f"Selected callable: {state['function_name']}\n"
                    f"Benchmark input kind: {state['input_kind']}\n\n"
                    f"Requested input size: {request.input_size}\n\n"
                    f"Code:\n{state['code']}"
                )
            ),
        ]
        response = llm.invoke(messages)

        print("\n===== GROQ OUTPUT =====")
        print(response.content)
        print("=======================\n")

        state["analysis"] = _parse_analysis(response.content, request)
        return state

    graph = StateGraph(AnalyzerState)
    graph.add_node("analyze", analyze_node)
    graph.set_entry_point("analyze")
    graph.add_edge("analyze", END)
    app = graph.compile()

    try:
        result = app.invoke(
            {
                "code": request.code,
                "function_name": request.function_name,
                "input_kind": request.input_kind,
                "plan": None,
                "analysis": None,
            }
        )
    except Exception as exc:
        analysis = _fallback_analysis(request)
        analysis.notes = [f"Agent call failed: {exc}"]
        return analysis

    return result.get("analysis") or _fallback_analysis(request, agent_enabled=True)


def plan_benchmark(code: str, use_agent: bool = True) -> BenchmarkPlan:
    api_key = os.getenv("GROQ_API_KEY")
    if not use_agent or not api_key or ChatGroq is None or StateGraph is None:
        return _fallback_plan(code)

    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    def plan_node(state: AnalyzerState) -> AnalyzerState:
        llm = ChatGroq(api_key=api_key, model=model, temperature=0.1)
        messages = [
            SystemMessage(
                content=(
                    "You are a Python benchmark planner. The user pastes arbitrary Python code.\n"
                    "Your job: read the code, understand what it does, pick the best public function "
                    "OR class to benchmark, and write a self-contained harness that exercises it correctly.\n\n"

                    "ALWAYS use input_kind = \"agent_harness\".\n\n"

                    "Return ONLY compact JSON (no markdown, no extra text) with these keys:\n"
                    "  algorithm_name  - human-readable name (e.g. \"LRU Cache\", \"Dijkstra's Shortest Path\")\n"
                    "  function_name   - exact Python name from the code (function or class name)\n"
                    "  input_kind      - always the string \"agent_harness\"\n"
                    "  reason          - one sentence explaining your choice\n"
                    "  harness_code    - plain Python source (NO markdown fences) with two functions:\n"
                    "      prepare_input(input_size: int) -> any\n"
                    "          Build realistic, deterministic input that matches the callable's\n"
                    "          EXACT signature. Scale complexity with input_size.\n"
                    "          Examples:\n"
                    "            - sorting/searching: return a random list of input_size ints\n"
                    "            - graph algorithms: build an adjacency dict/list of ~input_size nodes\n"
                    "            - string algorithms: return a random string of length input_size\n"
                    "            - DP / math: return appropriate int or tuple of ints\n"
                    "            - two-pointer / sliding window: return a list and a target int\n"
                    "            - Union-Find / DSU: return (n, list_of_edge_tuples)\n"
                    "            - data structures (LRU, heap, trie): return a list of operations to perform\n"
                    "          Return a single value or a tuple if the function takes multiple args.\n"
                    "      run_target(prepared) -> any\n"
                    "          Unpack prepared and call the function/class with the correct arguments.\n"
                    "          If prepare_input returns a tuple, unpack it: func(*prepared)\n"
                    "          For classes: instantiate, then call methods in a loop.\n"
                    "          Example for LRUCache:\n"
                    "            cache = LRUCache(capacity)\n"
                    "            for key, val in operations:\n"
                    "                cache.put(key, val)\n"
                    "                cache.get(key)\n\n"

                    "Rules:\n"
                    "  - harness_code must be plain Python — absolutely no ``` fences\n"
                    "  - Only import from: random, string, collections, math, itertools, heapq\n"
                    "  - Do NOT redefine the target function/class inside harness_code\n"
                    "  - The target is already in scope when harness_code runs\n"
                    "  - Make prepare_input deterministic (seed random if you use it)\n"
                    "  - Scale input meaningfully: input_size=100 should be fast, input_size=10000 measurable\n"
                    "  - If the callable takes no arguments, prepare_input returns None and run_target calls it directly\n"
                    "  - For classes, the harness should exercise the main operations (put/get, push/pop, insert/search, etc.)\n"
                )
            ),
            HumanMessage(content=f"Python code:\n{state['code']}"),
        ]
        response = llm.invoke(messages)
        
        print("\n===== GROQ PLAN OUTPUT =====")
        print(response.content)
        print("============================\n")
        
        state["plan"] = _parse_plan(str(response.content), state["code"])
        return state

    graph = StateGraph(AnalyzerState)
    graph.add_node("plan", plan_node)
    graph.set_entry_point("plan")
    graph.add_edge("plan", END)
    app = graph.compile()

    try:
        result = app.invoke(
            {
                "code": code,
                "function_name": None,
                "input_kind": None,
                "plan": None,
                "analysis": None,
            }
        )
    except Exception as exc:
        return _fallback_plan(code, reason=f"Groq planner failed: {exc}")

    plan = result.get("plan") or _fallback_plan(
        code,
        agent_enabled=True
    )

    print("\n===== BENCHMARK PLAN =====")
    print(plan)
    print("==========================\n")

    return plan


def repair_benchmark_plan(code: str, failed_plan: BenchmarkPlan, error: str) -> BenchmarkPlan:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or ChatGroq is None or StateGraph is None:
        repaired = failed_plan.model_copy()
        if "start" in error and "end" in error and failed_plan.input_kind != "graph_start_end":
            repaired.input_kind = "graph_start_end"
            repaired.reason = "The first benchmark call needed start/end arguments, so the local repair planner used graph_start_end input."
            repaired.agent_enabled = False
            return repaired

        if failed_plan.input_kind != "signature_args":
            repaired.input_kind = "signature_args"
            repaired.reason = f"The first benchmark call failed with: {error}. The local repair planner is trying signature_args."
            repaired.agent_enabled = False
            return repaired
        return failed_plan

    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    def repair_node(state: AnalyzerState) -> AnalyzerState:
        llm = ChatGroq(api_key=api_key, model=model, temperature=0.1)
        messages = [
            SystemMessage(
                content=(
                    "You repair a failed Python benchmark plan. The previous harness crashed.\n"
                    "Read the error, read the code, and write a corrected harness.\n\n"
                    "ALWAYS use input_kind = \"agent_harness\".\n\n"
                    "Return ONLY compact JSON (no markdown) with keys:\n"
                    "  algorithm_name, function_name, input_kind, reason, harness_code\n\n"
                    "harness_code must be plain Python (NO ``` fences) with:\n"
                    "  prepare_input(input_size: int) -> any\n"
                    "      Build correct input matching the function's EXACT signature.\n"
                    "      Fix whatever caused the previous error.\n"
                    "  run_target(prepared) -> any\n"
                    "      Unpack and call the function correctly.\n\n"
                    "Rules:\n"
                    "  - Only import from: random, string, collections, math, itertools, heapq\n"
                    "  - Do NOT redefine the target function\n"
                    "  - The target function is already in scope\n"
                    "  - Seed random for determinism\n"
                )
            ),
            HumanMessage(
                content=(
                    f"Failed plan: {failed_plan.model_dump()}\n"
                    f"Error: {error}\n\n"
                    f"Python code:\n{state['code']}"
                )
            ),
        ]
        response = llm.invoke(messages)
        state["plan"] = _parse_plan(str(response.content), state["code"])
        return state

    graph = StateGraph(AnalyzerState)
    graph.add_node("repair", repair_node)
    graph.set_entry_point("repair")
    graph.add_edge("repair", END)
    app = graph.compile()

    try:
        result = app.invoke(
            {
                "code": code,
                "function_name": failed_plan.function_name,
                "input_kind": failed_plan.input_kind,
                "plan": failed_plan,
                "analysis": None,
            }
        )
    except Exception:
        return failed_plan

    return result.get("plan") or failed_plan
