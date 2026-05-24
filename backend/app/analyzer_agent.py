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
            input_kind="signature_args",
            reason=reason or "Could not parse code, so the planner used a default entry point.",
            agent_enabled=agent_enabled,
        )

    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not node.name.startswith("_")
    ]

    if not functions:
        return BenchmarkPlan(
            algorithm_name="Python function",
            function_name="main",
            input_kind="signature_args",
            reason=reason or "No public function was found, so the planner used `main`.",
            agent_enabled=agent_enabled,
        )

    target = functions[-1]
    arg_names = _function_arg_names(target)
    input_kind = _infer_input_kind(target.name, arg_names)

    return BenchmarkPlan(
        algorithm_name=target.name.replace("_", " ").title(),
        function_name=target.name,
        input_kind=input_kind,
        reason=reason
        or f"Selected `{target.name}` from the pasted code and generated `{input_kind}` input.",
        agent_enabled=agent_enabled,
    )


VALID_INPUT_KINDS = {
    "list_int", "int", "string", "none",
    "list_int_and_target", "two_ints",
    "graph_start_end", "signature_args", "agent_harness","two_strings",
}

def _parse_plan(raw_text: str, code: str) -> BenchmarkPlan:
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1:
        return _fallback_plan(code, agent_enabled=True, reason="Groq did not return parseable JSON.")

    try:
        payload = json.loads(raw_text[start : end + 1])
        
        raw_kind = str(payload.get("input_kind") or "list_int")
        input_kind = raw_kind if raw_kind in VALID_INPUT_KINDS else "signature_args"

        return BenchmarkPlan(
            algorithm_name=str(payload.get("algorithm_name") or "Python function")[:80],
            function_name=str(payload["function_name"])[:80],
            input_kind=input_kind,
            reason=str(payload.get("reason") or "Groq selected the function and generated input."),
            agent_enabled=True,
            harness_code=payload.get("harness_code"),
        )
    except Exception:
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

    return FunctionAnalysis(

    summary=str(
        payload.get("function_analysis")
        or payload.get("summary")
        or f"Benchmarks `{request.function_name}`."
    ),

    complexity=str(
        payload.get("time_complexity")
        or fallback["time_complexity"]
    ),

    brief=str(
        payload.get("function_analysis")
        or payload.get("brief")
        or payload.get("summary")
        or f"Runs `{request.function_name}`."
    ),

    time_complexity=str(
        payload.get("time_complexity")
        or fallback["time_complexity"]
    ),

    space_complexity=str(
        payload.get("space_complexity")
        or fallback["space_complexity"]
    ),

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
        or f"Input kind: {request.input_kind}."
    ),

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
                    "Analyze the provided function deeply.\n"
                    "Return ONLY valid JSON.\n\n"

                    "JSON format:\n"

                    "{\n"
                    "  \"function_analysis\": \"Explain what the function does in detail.\",\n"
                    "  \"algorithm_family\": \"Specific family, e.g. Sliding window, Dynamic programming, Graph traversal\",\n"
                    "  \"strategy\": \"The invariant or core technique in one precise sentence.\",\n"
                    "  \"time_complexity\": \"Big-O time complexity\",\n"
                    "  \"space_complexity\": \"Big-O space complexity\",\n"
                    "  \"stability\": \"Stable, not stable, or not applicable\",\n"
                    "  \"storage_model\": \"How auxiliary memory is used\",\n"
                    "  \"beginner_explanation\": \"Plain language explanation for a beginner\",\n"
                    "  \"interview_explanation\": \"Concise interview answer\",\n"
                    "  \"professor_explanation\": \"Formal CS terminology explanation\",\n"
                    "  \"reason\": \"Why the complexities are correct\",\n"
                    "  \"notes\": [\"up to five useful notes\"],\n"
                    "  \"suggestions\": [\"up to five improvement or testing suggestions\"]\n"
                    "}\n\n"

                    "Rules:\n"
                    "- Analyze loops\n"
                    "- Analyze recursion\n"
                    "- Analyze nested operations\n"
                    "- Analyze auxiliary memory\n"
                    "- Return ONLY JSON\n"
                    "- No markdown"
                )
            ),
            HumanMessage(
                content=(
                    f"Function name: {state['function_name']}\n"
                    f"Generated input kind: {state['input_kind']}\n\n"
                    f"Code:\n{state['code']}"
                )
            ),
        ]
        response = llm.invoke(messages)

        print("\n===== GROQ OUTPUT =====")
        print(response.content)
        print("=======================\n")

        raw = response.content
        start = raw.find("{")
        end = raw.rfind("}")

        if start == -1 or end == -1:
            state["analysis"] = _fallback_analysis(request, agent_enabled=True)
            return state

        try:
            payload = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            state["analysis"] = _fallback_analysis(request, agent_enabled=True)
            return state

        formatted_tc = format_big_o(
            payload.get("time_complexity", "Unknown")
        )

        formatted_sc = format_big_o(
            payload.get("space_complexity", "Unknown")
        )

        state["analysis"] = FunctionAnalysis(

            summary=payload.get(
                "function_analysis",
                "No summary"
            ),

            complexity=payload.get(
                "time_complexity",
                "Unknown"
            ),

            brief=payload.get(
                "reason",
                "No analysis"
            ),

            time_complexity=formatted_tc,

            space_complexity=formatted_sc,

            algorithm_family=payload.get(
                "algorithm_family",
                _classify_from_name(request.function_name)["family"]
            ),

            strategy=payload.get(
                "strategy",
                _classify_from_name(request.function_name)["strategy"]
            ),

            stability=payload.get(
                "stability",
                _classify_from_name(request.function_name)["stability"]
            ),

            storage_model=payload.get(
                "storage_model",
                _classify_from_name(request.function_name)["storage_model"]
            ),

            beginner_explanation=payload.get(
                "beginner_explanation",
                ""
            ),

            interview_explanation=payload.get(
                "interview_explanation",
                ""
            ),

            professor_explanation=payload.get(
                "professor_explanation",
                ""
            ),

            input_contract="Automatically generated benchmark input.",

            notes=[str(item) for item in payload.get("notes", [])][:5],

            suggestions=[str(item) for item in payload.get("suggestions", [])][:5],

            agent_enabled=True
        )

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
                    "You are a Python benchmark planner. The user will paste a Python file. "
        "Choose exactly one callable to benchmark and choose generated arguments. "
        "Allowed input_kind values are: list_int, int, string, none, "
        "list_int_and_target, two_ints, signature_args, agent_harness, two_strings. "
        "Return only compact JSON with keys: algorithm_name, function_name, input_kind, "
        "reason, harness_code. Do not include markdown. "
        "Prefer the main algorithm function over tiny helpers. "
        "If a function needs a list and target, use list_int_and_target. "
        "If it needs one n-like integer, use int. "
        "If it needs text, use string. "
        "If it needs two strings (s and t, haystack and needle, text and pattern), use two_strings. "
        "If it takes a graph, adjacency list, grid, network, or any dict/list of edges as input, "
        "ALWAYS use agent_harness — never guess the argument count for graph functions. "
        "If none of those shapes fit exactly, use agent_harness. "
        "For agent_harness write prepare_input(input_size) that builds realistic deterministic "
        "arguments matching the EXACT function signature, and run_target(prepared) that unpacks "
        "and calls the function correctly. "
        "Use only standard Python and existing code names. "
        "When in doubt, prefer agent_harness over guessing."
                )
            ),
            HumanMessage(content=f"Python code:\n{state['code']}"),
        ]
        response = llm.invoke(messages)
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
                    "You repair Python benchmark call plans after a failed run. Return only JSON "
                    "with keys: algorithm_name, function_name, input_kind, reason, harness_code. Allowed "
                    "input_kind values: list_int, int, string, none, list_int_and_target, "
                    "two_ints, graph_start_end, signature_args, agent_harness. If the error says "
                    "missing start and end, usually choose graph_start_end for path/graph optimizers. "
                    "For unusual signatures, choose agent_harness and write prepare_input(input_size) "
                    "and run_target(prepared)."
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
