# morphometry frontend

The Next.js 15 / React 19 web UI for the morphometry API — DICOM upload, model
job dispatch, and interactive landmark editing of torsion / X-ray / MAD results.
Part of the `morphometry` monorepo; the backend it talks to lives in `../api/`.

## Getting started

```bash
npm install
npm run dev:webpack  # dev server on http://localhost:3000 (USE THIS — see below)
npm run dev          # dev server with Turbopack (breaks the Cornerstone viewer)
npm run build        # production build (webpack)
npm start            # serve the production build
npm run lint         # ESLint
```

**Use `npm run dev:webpack`, not `npm run dev`.** The torsion viewer renders the
NIfTI volumes with **Cornerstone3D**, which needs web workers / WASM wired up by
the `webpack()` hook in `next.config.ts`. `next dev --turbopack` (the plain
`npm run dev`) skips that hook and the viewer silently fails to load volumes.
`npm run build` always uses webpack, so production is unaffected.

## Running against the backend

The UI is just the API's client; nothing renders without the API + worker +
Redis running. From the repo root (venv activated):

```bash
docker start morphometry-redis   # or: docker run -d --name morphometry-redis -p 6379:6379 redis:7
python -m api.tasks.worker       # RQ worker (runs the docker model images)
MORPH_API_CORS_ALLOW_ORIGINS=http://localhost:3000 uvicorn api.main:app --port 8000
```

See the repo root `README.md` → "Running the stack" for the full walkthrough.

## Configuration

The API base URL is read from `NEXT_PUBLIC_MODEL_API` (default
`http://localhost:8000`). Copy `.env.example` to `.env.local` to override.
`NEXT_PUBLIC_*` values are inlined at build time, so rebuild after changing it.

When running against a local API, set the API's CORS to allow this origin:
`MORPH_API_CORS_ALLOW_ORIGINS=http://localhost:3000`. Auth is disabled when
`MORPH_API_API_KEYS` is unset (the dev default); if you enable it, the torsion
viewer reads the key from `NEXT_PUBLIC_API_KEY` and passes it as `?api_key=` to
the volume endpoints.

See `CLAUDE.md` in this directory for architecture and component conventions.
