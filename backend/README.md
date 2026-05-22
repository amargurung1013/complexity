# AlgoBench Backend

FastAPI service for running sorting algorithm benchmarks.

## Setup

```bash
uv sync
```

Copy `.env.example` to `.env` and add your Groq key to enable the analyzer agent:

```bash
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

## Run

```bash
uv run uvicorn app.main:app --reload
```

If you are already inside `backend/app`, use:

```bash
uv run uvicorn main:app --reload
```

The API starts on `http://127.0.0.1:8000` by default.

## Endpoints

- `GET /health`
- `GET /algorithms`
- `POST /benchmark`
- `POST /benchmark/custom`
- `POST /benchmark/analyze`
