from pydantic import BaseModel, Field


class FunctionAnalysis(BaseModel):
    summary: str
    complexity: str
    brief: str = ""
    time_complexity: str = "Unknown"
    space_complexity: str = "Unknown"
    algorithm_family: str = "General algorithm"
    strategy: str = "Analyze the code structure and benchmark behavior."
    stability: str = "Not applicable"
    storage_model: str = "Depends on implementation"
    beginner_explanation: str = ""
    interview_explanation: str = ""
    professor_explanation: str = ""
    input_contract: str
    notes: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    agent_enabled: bool = False


class BenchmarkPlan(BaseModel):
    algorithm_name: str
    function_name: str
    input_kind: str = Field(pattern="^(list_int|int|string|none|list_int_and_target|two_ints|graph_start_end|signature_args|agent_harness)$")
    reason: str
    agent_enabled: bool = False
    harness_code: str | None = Field(default=None, max_length=12_000)


class BenchmarkRequest(BaseModel):
    algorithm_name: str
    input_size: int = Field(ge=1, le=100_000)


class CustomBenchmarkRequest(BaseModel):
    algorithm_name: str = Field(min_length=1, max_length=80)
    function_name: str = Field(min_length=1, max_length=80)
    code: str = Field(min_length=1, max_length=30_000)
    input_size: int = Field(ge=1, le=100_000)
    input_kind: str = Field(default="list_int", pattern="^(list_int|int|string|none|list_int_and_target|two_ints|graph_start_end|signature_args|agent_harness)$")
    timeout_seconds: float = Field(default=5, ge=0.25, le=20)
    analyze_with_agent: bool = True
    harness_code: str | None = Field(default=None, max_length=12_000)


class AutoBenchmarkRequest(BaseModel):
    code: str = Field(min_length=1, max_length=30_000)
    input_size: int = Field(default=1_000, ge=1, le=100_000)
    timeout_seconds: float = Field(default=5, ge=0.25, le=20)
    analyze_with_agent: bool = True


class BenchmarkResult(BaseModel):
    algorithm: str
    execution_time_ms: float
    peak_memory_kb: float
    input_size: int
    analysis: FunctionAnalysis | None = None
    plan: BenchmarkPlan | None = None
