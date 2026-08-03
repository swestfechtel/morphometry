# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Medical morphometry web application for orthopedic analysis. Provides interactive landmark-based measurements on medical images (MRI torsion, X-ray morphometry). **Not intended for clinical use.**

This is the `frontend/` part of the `morphometry` monorepo — the web client for the FastAPI service in `../api/`. See the repository root `README.md` / `CLAUDE.md` for the library and API. All commands below run from this `frontend/` directory.

## Commands

```bash
npm run dev          # Dev server with Turbopack
npm run dev:webpack  # Dev server with webpack — USE THIS for the Cornerstone viewer
npm run build        # Production build (webpack)
npm start            # Production server
npm run lint         # ESLint
```

No test framework is configured.

**Turbopack caveat:** the Cornerstone3D torsion viewer needs webpack (web
workers / WASM + the `webpack()` hook in `next.config.ts`), which
`next dev --turbopack` does not run. Use `npm run dev:webpack` when working on or
testing the viewer. `npm run build` always uses webpack, so production is fine.

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
- `femoralProximalAngle()` / `femoralDistalAngle()` / `tibialProximalAngle()` / `tibialDistalAngle()` — the signed per-reference-line components; the torsion is their difference (femur: prox − dist, tibia: dist − prox). Shown on each reference line's slice in the viewer.
- `computeHalluxValgusAngle()` — 2D angle from indexed landmark array
- `angleBetweenVectors()` / `angleBetweenVectors2D()` — generic vector angle helpers

### Landmark Data Structures

- **Torsion**: Nested object keyed by `{femur,tibia}.{lee,murphy}.{left,right}` with `[x, y, z]` coordinates
- **X-ray**: Array of `[x, y]` points indexed by anatomical position

### Layout

`app/layout.tsx` is a client component (`'use client'`) providing sticky navbar, dark mode toggle, and footer. Dark mode is managed via local state toggling a CSS class.

### Torsion viewer (Cornerstone3D)

The MRI torsion examination is rendered with **Cornerstone3D** (`@cornerstonejs/*`
v5), not pre-rendered PNGs. The viewer lives in `app/components/cornerstone/`:

- `cs-init.ts` — idempotent Cornerstone init + registers the `nifti:` loader, and
  attaches the API key to volume fetches via the loader's `beforeSend` hook.
- `cs-volume-url.ts` — builds the `/examinations/{id}/volume/{image,mask}` URLs
  and the auth token (header + `?api_key=` fallback; `NEXT_PUBLIC_API_KEY`).
- `landmark-mapping.ts` — **pure** voxel↔combined-volume mapping: the hip/knee/
  ankle z-offset handling, the `to/fromLoaderIndex` axis adapter (the single place to
  fix any voxel↔world axis flip), and `buildReferenceLines` (pairs the landmarks into
  the measurement axes with their anatomical labels).
- `reference-line-tool.ts` — a custom Cornerstone annotation tool
  (`TorsionReferenceLine`) subclassing the stock `LengthTool`. It inherits all of
  LengthTool's interaction (hit-testing, handle drag, `ANNOTATION_MODIFIED` events)
  and overrides only rendering: each start/end landmark pair is drawn as ONE bright,
  thick connecting line with large always-visible endpoint dots, a per-endpoint
  anatomical label (e.g. "Femoral head centre"), and the line's signed proximal/distal
  angle (`data.angleLabel`, e.g. "Distal: −3.1°"). No length is computed/shown.
  `renderAnnotation` sees all reference lines at once, so it collects every label and
  runs a vertical-nudge collision pass (`placeLabel`) before drawing, keeping labels
  from overlapping when lines crowd a slice.
- `use-cornerstone-viewport.ts` — renders a single AXIAL native-slice
  **StackViewport** and seeds one reference-line annotation per measurement axis
  (`buildReferenceLines`); dragging either endpoint flows back to React state (live
  torsion recompute) and the PATCH save — both endpoints are rewritten per edit since
  they share the slice. Scrolling the wheel WHILE dragging a handle moves the whole line
  to the adjacent slice (a custom `wheel` handler, because Cornerstone blocks its own
  StackScroll during tool interaction) — both endpoints move together to stay co-planar,
  clamped to the line's hip/knee/ankle sub-volume. Tears everything down on unmount. A
  `ResizeObserver` keeps the Cornerstone canvas in sync when the viewport is resized.
  (VOI derived from the middle slice's actual intensities — see `computeVoiRange`.)
- `cornerstone-torsion-viewer.tsx` — the client component (one axial viewport div,
  sized to fit the screen: the square grows to the column width but is capped at
  `calc(100vh - 9rem)` so the whole image shows without page scrolling), imported via
  `next/dynamic({ ssr: false })` from `torsion-examination-component.tsx`. Renders a
  top-right `TorsionReadout` overlay with the femoral (per method) and tibial torsion
  totals, computed from the live `landmarks` prop so it updates as landmarks are dragged,
  and the "Save changes" button directly below it (shown when `hasChanges`, calls
  `onSave`). The per-reference-line proximal/distal angle is drawn on its own slice by
  the tool. `torsion-examination-component.tsx` lays out an examination-details panel to
  the LEFT of the viewer (the page-level "Examination Details" header is now only
  rendered for non-torsion types). Dark mode is the default (`app/layout.tsx`).
- `stashed/` — retired-but-preserved variants kept as `.txt` (not compiled). The
  3-plane MPR version lives here; see `stashed/README.md`.

**Why one axial stack, not 3-plane MPR:** the torsion MRI is highly anisotropic
(≈0.6 mm in-plane, ≈9 mm through-plane) — a 2D slice stack, not an isotropic
volume. Reslicing it as a volume was fragile: exactly axis-aligned volumes rendered
**black** (a vtk volume-mapper degeneracy), and on gantry-tilted volumes the
landmarks never coincided with the world-aligned MPR planes, so none showed. A
StackViewport renders native slices reliably regardless of orientation, and
landmark probes match by slice. (MPR sag/cor was tried and stashed — it stayed
black on axis-aligned volumes and couldn't show landmarks on tilted ones.)

**Landmark ↔ stack slice:** each reference line's two endpoints share an axial slice,
so a line is drawn in-plane on that slice. Tie the annotation to its acquisition slice
via `metadata.referencedImageId` (frame = combined-volume z), and place both endpoints
with that slice's OWN geometry — `imagePlaneModule`: `IPP + i·columnSpacing·rowCosines
+ j·rowSpacing·columnCosines` (see `stackSliceWorld`/`stackVoxelFromWorld`). Using the
vtk volume's `indexToWorld` instead makes handles render **shifted** on tilted volumes.
Landmarks stay in **voxel space** (so `app/utils.tsx` angle math is unchanged); the
world↔voxel conversion and frame-preserving edit round-trip happen only at the
Cornerstone boundary. On any edit both endpoints are written back (the un-dragged one
maps to the same voxel), keeping the pair on its slice.

**Not yet implemented:** the segmentation labelmap overlay (stack labelmaps need
per-slice *derived* images populated from the mask; the toggle is hidden in
`torsion-examination-component.tsx` until then). The volume-labelmap version is in
the stashed MPR variant.

The API serves the NIfTI volumes (see the root `CLAUDE.md` —
`GET /examinations/{id}/volume/...`, served as GPU-renderable float32).

### API Configuration

`app/server_config.ts` exports a `model_api` base URL, read from the
`NEXT_PUBLIC_MODEL_API` env var (default `http://localhost:8000`). `NEXT_PUBLIC_*`
values are inlined at build time, so rebuild after changing it; copy
`.env.example` to `.env.local` to override. Components import `server_config`
directly (`@/app/server_config`) for API calls. Path alias `@/*` maps to project
root.