# complexity

complexity is a full-stack algorithm analysis workbench. Paste a Python function, run a benchmark, and review complexity trends, memory behavior, execution traces, stack frames, heatmaps, and Groq-generated study notes.

## Project Layout

```text
.
├── backend/
│   ├── app/
│   │   ├── analyzer_agent.py
│   │   ├── benchmark.py
│   │   ├── custom_runner.py
│   │   ├── main.py
│   │   └── schemas.py
│   ├── pyproject.toml
│   └── uv.lock
└── frontend/
    ├── app/
    │   ├── globals.css
    │   ├── layout.tsx
    │   └── page.tsx
    ├── src/
    │   ├── components/
    │   │   ├── AlgoChart.tsx
    │   │   ├── MetricCard.tsx
    │   │   └── Sidebar.tsx
    │   ├── lib/
    │   │   ├── format.ts
    │   │   └── navigation.ts
    │   └── types/
    │       └── benchmark.ts
    ├── package.json
    └── package-lock.json
```

## Running Locally

Start the API:

```bash
cd backend
uv run uvicorn app.main:app --reload
```

Start the frontend in another terminal:

```bash
cd frontend
npm run dev
```

The app expects the API at `http://127.0.0.1:8000` by default. Set `NEXT_PUBLIC_API_URL` to use a different backend URL.
