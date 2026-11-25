# morphometry frontend

The Next.js 15 / React 19 web UI for the morphometry API — DICOM upload, model
job dispatch, and interactive landmark editing of MRI torsion results.
Part of the `morphometry` monorepo; the backend it talks to lives in `../api/`.
The end-to-end guide to the whole web app is [`../README_WEBAPP.md`](../README_WEBAPP.md).

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

`MODEL_API_INTERNAL` (e.g. `http://api:8000`) is a **runtime** variable, not
inlined at build time: server-side rendering uses it to reach the API from inside
the frontend container, where `localhost:8000` is the container itself. It never
affects the browser, which derives `<page hostname>:<NEXT_PUBLIC_MODEL_API_PORT>`
(default 8000); building the production image with a different API port therefore
needs `NEXT_PUBLIC_MODEL_API_PORT` set at build time. The production image
(`frontend/Dockerfile`, Next.js `output: "standalone"`) is built and released only
by `scripts/release.py`; see [`../docs/releasing.md`](../docs/releasing.md) and
[`../deploy/README.md`](../deploy/README.md).

## Version filter

The examinations list has a version dropdown (released versions from `GET /releases`, plus "unreleased" and
"before versioning") and a stage selector (any / segmentation / torsion). Both are URL query parameters
(`?version=…&stage=…`), so a filtered view is linkable; each row shows the release versions of its torsion and
segmentation images. Results are matched by the recorded image digest, so a result from another version that ran
the identical image also matches. The strings exist in German and English.

When running against a local API, set the API's CORS to allow this origin:
`MORPH_API_CORS_ALLOW_ORIGINS=http://localhost:3000`. Users sign in at `/login`
(username/password); create accounts with `python -m api.users`. Auth is only
enforced when the API sets `MORPH_API_AUTH_REQUIRED=true` or an API key — otherwise
the UI works without signing in. A machine `NEXT_PUBLIC_API_KEY` can still be set
and is sent alongside the user token.

See `CLAUDE.md` in this directory for architecture and component conventions.
