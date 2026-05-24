# complexity Backend

FastAPI service for planning, benchmarking, and explaining Python algorithm functions.

## Setup

```bash
uv sync
```

Add your Groq key to `.env` to enable AI classification and explanations:

```bash
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

## Run

```bash
uv run uvicorn app.main:app --reload
```

The API starts on `http://127.0.0.1:8000` by default.

## Endpoints

- `GET /health`
- `GET /algorithms`
- `POST /benchmark`
- `POST /benchmark/custom`
- `POST /benchmark/analyze`
