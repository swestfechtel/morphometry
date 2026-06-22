# morphometry frontend

The Next.js 15 / React 19 web UI for the morphometry API — DICOM upload, model
job dispatch, and interactive landmark editing of torsion / X-ray / MAD results.
Part of the `morphometry` monorepo; the backend it talks to lives in `../api/`.

## Getting started

```bash
npm install
npm run dev      # dev server on http://localhost:3000
npm run build    # production build
npm start        # serve the production build
npm run lint     # ESLint
```

## Configuration

The API base URL is read from `NEXT_PUBLIC_MODEL_API` (default
`http://localhost:8000`). Copy `.env.example` to `.env.local` to override.
`NEXT_PUBLIC_*` values are inlined at build time, so rebuild after changing it.

When running against a local API, set the API's CORS to allow this origin:
`MORPH_API_CORS_ALLOW_ORIGINS=http://localhost:3000`.

See `CLAUDE.md` in this directory for architecture and component conventions.
