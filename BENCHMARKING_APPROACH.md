# Benchmarking Approach: Universal Agent Harness

## The Problem
Users paste arbitrary Python code — functions, classes, algorithms with any signature. The system needs to:
1. Figure out what to benchmark
2. Generate appropriate input
3. Call it correctly
4. Measure performance

## The Solution: Always Use `agent_harness`

Instead of trying to guess which of 10+ input patterns matches the code, **Groq writes the actual benchmark harness**.

### How It Works

1. **User pastes code** (any function or class)
2. **Groq reads it** and generates a `BenchmarkPlan` with:
   - `function_name`: the name to benchmark (function or class)
   - `input_kind`: always `"agent_harness"`
   - `harness_code`: plain Python with two functions:
     - `prepare_input(input_size)` — builds realistic input scaled to `input_size`
     - `run_target(prepared)` — calls the target with correct arguments

3. **Backend executes**:
   ```python
   exec(user_code)           # defines the function/class
   exec(harness_code)        # defines prepare_input and run_target
   prepared = prepare_input(input_size)
   result = run_target(prepared)  # measured
   ```

### Examples

#### Function: `def two_sum(nums, target)`
```python
# Groq generates:
def prepare_input(input_size):
    import random
    random.seed(42)
    nums = [random.randint(0, 100000) for _ in range(input_size)]
    target = nums[0] + nums[-1] if len(nums) >= 2 else 0
    return (nums, target)

def run_target(prepared):
    return two_sum(*prepared)
```

#### Class: `class LRUCache`
```python
# Groq generates:
def prepare_input(input_size):
    import random
    random.seed(42)
    return [(random.randint(0, 10000), random.randint(0, 10000)) 
            for _ in range(input_size)]

def run_target(prepared):
    cache = LRUCache(capacity=min(len(prepared), 1000))
    for key, val in prepared:
        cache.put(key, val)
        cache.get(key)
```

#### Graph: `def dijkstra(graph, start, end)`
```python
# Groq generates:
def prepare_input(input_size):
    import random
    random.seed(42)
    n = max(2, min(input_size, 500))
    graph = {i: [] for i in range(n)}
    for i in range(n - 1):
        graph[i].append((i + 1, random.randint(1, 10)))
        if i + 2 < n:
            graph[i].append((i + 2, random.randint(1, 20)))
    return (graph, 0, n - 1)

def run_target(prepared):
    return dijkstra(*prepared)
```

## Why This Works

✅ **Handles any signature** — Groq reads the actual code  
✅ **Handles classes** — harness instantiates and exercises methods  
✅ **Handles edge cases** — no-arg functions, multiple returns, generators  
✅ **Scales correctly** — `input_size` controls complexity  
✅ **Deterministic** — seeded random for reproducible benchmarks  
✅ **Safe** — harness runs in the same sandbox as user code  

## Fallback (No Groq)

When Groq is unavailable, the system uses heuristics:
- Reads parameter names (`n`, `graph`, `text`, etc.)
- Generates a basic harness
- Works for common patterns

## Key Changes Made

1. **`schemas.py`**: Default `input_kind` is now `"agent_harness"`
2. **`analyzer_agent.py`**: 
   - Planner prompt instructs Groq to always use `agent_harness`
   - Fallback plan generates harnesses for functions and classes
   - Added `_build_fallback_harness()` for no-Groq scenarios
3. **`custom_runner.py`**: 
   - Skip function name validation when `input_kind == "agent_harness"`
   - Allows classes to be benchmarked
4. **`main.py`**: 
   - Sanitize `harness_code` to strip markdown fences from Groq

## Result

Paste **any** Python code → system figures it out → benchmark runs → results displayed.
