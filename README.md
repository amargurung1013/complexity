# AlgoBench

AlgoBench is a small full-stack benchmark dashboard:

- `backend/` contains the FastAPI API and benchmark runner.
- `frontend/` contains the Next.js dashboard UI.

## Project Layout

```text
.
├── backend/          # FastAPI service
│   ├── api/          # API routes, schemas, and benchmark engine
│   ├── pyproject.toml
│   └── uv.lock
└── frontend/         # Next.js app
    ├── app/          # App Router pages and styles
    ├── src/          # Reusable frontend components
    ├── package.json
    └── package-lock.json
```

## Running Locally

Start the API:

```bash
cd backend
uv run uvicorn api.main:app --reload
```

Start the frontend in another terminal:

```bash
cd frontend
npm run dev
```

The dashboard expects the API at `http://127.0.0.1:8000` by default. To use a different API URL, set `NEXT_PUBLIC_API_URL` when starting the frontend.
