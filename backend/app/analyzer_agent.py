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


def _function_arg_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
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
        if first in {"nums", "values", "arr", "array", "items", "list"} and second == "target":
            return "list_int_and_target"
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
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_")
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


def _parse_plan(raw_text: str, code: str) -> BenchmarkPlan:
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1:
        return _fallback_plan(code, agent_enabled=True, reason="Groq did not return parseable JSON.")

    try:
        payload = json.loads(raw_text[start : end + 1])
        return BenchmarkPlan(
            algorithm_name=str(payload.get("algorithm_name") or "Python function")[:80],
            function_name=str(payload["function_name"])[:80],
            input_kind=str(payload.get("input_kind") or "list_int"),
            reason=str(payload.get("reason") or "Groq selected the function and generated input."),
            agent_enabled=True,
            harness_code=payload.get("harness_code"),
        )
    except Exception:
        return _fallback_plan(code, agent_enabled=True, reason="Groq returned an invalid plan.")


def _fallback_analysis(request: CustomBenchmarkRequest, agent_enabled: bool = False) -> FunctionAnalysis:
    readable_name = request.function_name.replace("_", " ")
    return FunctionAnalysis(
        summary=(
            f"`{request.function_name}` is benchmarked with generated {request.input_kind} input "
            f"at size {request.input_size}."
        ),
        complexity="Unknown",
        brief=f"This appears to run the {readable_name} function and measure its runtime and peak memory.",
        time_complexity="Unknown",
        space_complexity="Unknown",
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


def _parse_analysis(raw_text: str, request: CustomBenchmarkRequest) -> FunctionAnalysis:
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1:
        return _fallback_analysis(request, agent_enabled=True)

    try:
        payload = json.loads(raw_text[start : end + 1])
    except json.JSONDecodeError:
        return _fallback_analysis(request, agent_enabled=True)

    return FunctionAnalysis(

    summary=str(
        payload.get("function_analysis")
        or payload.get("summary")
        or f"Benchmarks `{request.function_name}`."
    ),

    complexity=str(
        payload.get("time_complexity")
        or "Unknown"
    ),

    brief=str(
        payload.get("function_analysis")
        or payload.get("brief")
        or payload.get("summary")
        or f"Runs `{request.function_name}`."
    ),

    time_complexity=str(
        payload.get("time_complexity")
        or "Unknown"
    ),

    space_complexity=str(
        payload.get("space_complexity")
        or "Unknown"
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
                    "  \"time_complexity\": \"Big-O time complexity\",\n"
                    "  \"space_complexity\": \"Big-O space complexity\",\n"
                    "  \"reason\": \"Why the complexities are correct\"\n"
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

        cleaned = (
            response.content
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

        payload = json.loads(cleaned)

        formatted_tc = format_big_o(
        payload.get(
            "time_complexity",
            "Unknown"
        )
    )

        formatted_sc = format_big_o(
            payload.get(
                "space_complexity",
                "Unknown"
            )
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

            input_contract="Automatically generated benchmark input.",

            notes=[],

            suggestions=[],

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
                    "list_int_and_target, two_ints, graph_start_end, signature_args, agent_harness. "
                    "Return only compact JSON with keys: algorithm_name, function_name, input_kind, "
                    "reason, harness_code. Do not include markdown. "
                    "Prefer the main algorithm function over tiny helpers. If a function needs "
                    "a list and target, use list_int_and_target. If it needs one n-like integer, "
                    "use int. If it needs text, use string. If it needs graph/start/end, path/"
                    "start/end, network/start/end, or grid/start/end, use graph_start_end. "
                    "If none of those shapes fit, use agent_harness and provide harness_code that "
                    "defines exactly two functions: prepare_input(input_size) and run_target(prepared). "
                    "prepare_input builds realistic deterministic arguments. run_target calls the selected "
                    "function/class and returns its result. Use only standard Python and existing code names."
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

    return result.get("plan") or _fallback_plan(code, agent_enabled=True)


def repair_benchmark_plan(code: str, failed_plan: BenchmarkPlan, error: str) -> BenchmarkPlan:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or ChatGroq is None or StateGraph is None:
        if "start" in error and "end" in error:
            repaired = failed_plan.model_copy()
            repaired.input_kind = "graph_start_end"
            repaired.reason = "The first benchmark call needed start/end arguments, so the local repair planner used graph_start_end input."
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
