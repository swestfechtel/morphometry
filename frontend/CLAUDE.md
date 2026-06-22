# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Medical morphometry web application for orthopedic analysis. Provides interactive landmark-based measurements on medical images (MRI torsion, X-ray morphometry). **Not intended for clinical use.**

This is the `frontend/` part of the `morphometry` monorepo — the web client for the FastAPI service in `../api/`. See the repository root `README.md` / `CLAUDE.md` for the library and API. All commands below run from this `frontend/` directory.

## Commands

```bash
npm run dev      # Dev server with Turbopack
npm run build    # Production build
npm start        # Production server
npm run lint     # ESLint
```

No test framework is configured.

## Tech Stack

- **Next.js 15** (App Router) with **React 19** and **TypeScript 5**
- **Tailwind CSS v4** with **Flowbite** component library
- Backend API at `http://localhost:8000` (configured in `app/server_config.ts`, overridable via the `NEXT_PUBLIC_MODEL_API` env var)

## Architecture

### Routing

Uses Next.js App Router file-based routing under `app/`:

- `/` — Landing page
- `/examinations` — Server component that fetches and lists all examinations
- `/examinations/[...slug]` — Catch-all route handling both filtered lists (e.g., `/examinations/mr-torsion`) and individual examination detail views (by accession number)
- `/upload` — Client-side file upload form
- `/mad` — MAD (Morphometry Assistant Display) viewer

### Component Pattern

Each examination type has a paired **examination component** + **image component**:

- `TorsionExaminationComponent` + `TorsionImageComponent` — MRI torsion (femoral/tibial angles, Lee/Murphy methods)
- `XrayExaminationComponent` + `XrayImageComponent` — X-ray foot/knee morphometry
- `MadComponent` — MAD analysis viewer

Examination components manage state and API calls. Image components render the image with an SVG overlay for draggable landmarks and handle mouse interaction for landmark editing.

### Data Flow

1. **Upload**: Files posted to `/upload/` → backend returns accession number
2. **Processing**: POST to `/model/segmentation/{id}` or `/model/torsion/{id}` → returns `job_id`
3. **Polling**: `PollingComponent` polls `/jobs/{job_id}` until status is `finished` or `error`
4. **Viewing**: Examination detail fetched from `/examinations/{accession_number}`
5. **Editing**: Landmarks modified via drag → PATCH to `/examinations/{accession_number}`

### State Management

Local `useState` throughout — no global state library. Parent examination components own landmark state and pass `saveChangesCallback`/`setLandmarksCallback` props to image components.

### Shared Utilities (`app/utils.tsx`)

Vector math and medical angle computation functions:
- `computeFemoralTorsion()` / `computeTibialTorsion()` — 3D torsion angles from proximal/distal landmark pairs, side-aware
- `computeHalluxValgusAngle()` — 2D angle from indexed landmark array
- `angleBetweenVectors()` / `angleBetweenVectors2D()` — generic vector angle helpers

### Landmark Data Structures

- **Torsion**: Nested object keyed by `{femur,tibia}.{lee,murphy}.{left,right}` with `[x, y, z]` coordinates
- **X-ray**: Array of `[x, y]` points indexed by anatomical position

### Layout

`app/layout.tsx` is a client component (`'use client'`) providing sticky navbar, dark mode toggle, and footer. Dark mode is managed via local state toggling a CSS class.

### API Configuration

`app/server_config.ts` exports a `model_api` base URL, read from the
`NEXT_PUBLIC_MODEL_API` env var (default `http://localhost:8000`). `NEXT_PUBLIC_*`
values are inlined at build time, so rebuild after changing it; copy
`.env.example` to `.env.local` to override. Components import `server_config`
directly (`@/app/server_config`) for API calls. Path alias `@/*` maps to project
root.