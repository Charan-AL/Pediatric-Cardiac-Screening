React Frontend for Pediatric Cardiac Screening

Quickstart

1. Install dependencies:

```bash
cd frontend
npm install
```

2. Run the frontend (dev server):

```bash
npm run dev
```

3. Run the backend API (if not already running):

```bash
# from project root
.venv\Scripts\python.exe -m uvicorn inference.api:app --reload --host 0.0.0.0 --port 8000
```

Notes
- The frontend calls `http://localhost:8000/predict`. CORS is already enabled on the backend.
- No backend code was changed.
