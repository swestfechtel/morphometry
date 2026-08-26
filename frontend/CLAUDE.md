# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Medical morphometry web application for orthopedic analysis. Provides interactive landmark-based measurements on MRI torsion examinations. **Not intended for clinical use.**

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
- Backend API base URL resolved in `app/server_config.ts` (same-hostname in the browser, loopback for SSR; overridable via `NEXT_PUBLIC_MODEL_API`)

## Architecture

### Routing

Uses Next.js App Router file-based routing under `app/`:

- `/` — Landing page (static; hero + how-it-works + features, CTAs to upload/examinations)
- `/about` — Static about page (what it measures, the pipeline, tech stack, clinical-use disclaimer); linked from the navbar
- `/examinations` — Server component that fetches and lists all examinations
- `/examinations/[...slug]` — Catch-all route handling both filtered lists (e.g., `/examinations/mr-torsion`) and individual examination detail views (by accession number)
- `/upload` — Client-side file upload form

### Component Pattern

- `TorsionExaminationComponent` — MRI torsion (femoral/tibial angles, Lee/Murphy methods).
  It owns the landmark state and API calls and renders the **Cornerstone3D** viewer
  (`app/components/cornerstone/`), which draws the volume and the draggable reference-line
  landmarks. See the "Torsion viewer (Cornerstone3D)" section below.

### Data Flow

1. **Upload (torsion, two-phase)**: `app/upload/page.tsx` posts the whole examination
   directory to `/upload/torsion/series`; the backend returns a `pending_selection`
   examination with its candidate DICOM series. `SeriesPicker`
   (`app/components/series-picker.tsx`) shows the series as cards with scrollable
   multi-slice previews (`/examinations/{id}/series/{uid}/preview/{i}.png`) and a mode
   toggle — single **whole-leg** series (pick one) or **separate hip/knee/ankle** (assign
   three). On confirm it POSTs `/upload/torsion/select`, which materializes the pick and
   **auto-starts** the full pipeline, returning a `job_id`. A pending examination is
   resumable: the list shows a "Select series" action and the detail route
   (`examinations/[...slug]`) renders `SeriesPicker` for `type: "pending"`.
2. **Processing**: started automatically on series selection, or manually via
   POST to `/model/segmentation/{id}` / `/model/torsion/{id}` → returns `job_id`
3. **Polling**: `PollingComponent` polls `/jobs/{job_id}` until status is `finished` or
   `error`; the picker's callback navigates to the viewer on `finished`
4. **Viewing**: Examination detail fetched from `/examinations/{accession_number}`
5. **Editing**: Landmarks modified via drag → PATCH to `/examinations/{accession_number}`

### State Management

Local `useState` throughout — no global state library. `TorsionExaminationComponent` owns landmark state and passes `saveChangesCallback`/`setLandmarksCallback` props to the Cornerstone viewer.

### Shared Utilities (`app/utils.tsx`)

Vector math and medical angle computation functions:
- `computeFemoralTorsion()` / `computeTibialTorsion()` — 3D torsion angles from proximal/distal landmark pairs, side-aware
- `femoralProximalAngle()` / `femoralDistalAngle()` / `tibialProximalAngle()` / `tibialDistalAngle()` — the signed per-reference-line components; the torsion is their difference (femur: prox − dist, tibia: dist − prox). Shown on each reference line's slice in the viewer.
- `angleBetweenVectors()` — generic 3D vector angle helper

### Landmark Data Structures

- **Torsion**: Nested object keyed by `{femur,tibia}.{lee,murphy}.{left,right}` with `[x, y, z]` coordinates

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

**Segmentation labelmap overlay:** the hip/knee/ankle mask is painted on the axial
slices, toggled by the "Show segmentation" checkbox in
`torsion-examination-component.tsx`. A StackViewport labelmap can't consume the mask
NIfTI directly — it needs a per-slice **derived** Uint8 labelmap image tied to each
source frame (else Cornerstone raises "No derived image found"). So
`use-cornerstone-viewport.ts::setupSegmentation` creates one blank derived image per
frame (`imageLoader.createAndCacheDerivedLabelmapImages(imageIds)`) and fills each
from the aligned mask (`combined.nii.gz`, hip=1/knee=2/ankle=3) loaded through the
**same per-slice loader** as the image, so mask frame *k* overlays image frame *k*
with matching in-plane layout. It then registers a stack segmentation
(`data: { imageIds }`) via `addLabelmapRepresentationToViewport`, sets per-label
colours, and toggles visibility with `config.visibility`. (The stashed MPR variant
used a *volume* labelmap instead, which only renders on volume viewports.)

The API serves the NIfTI volumes (see the root `CLAUDE.md` —
`GET /examinations/{id}/volume/...`, served as GPU-renderable float32).

### Authentication

Username/password login against the API (`POST /auth/login`), managed by
`app/auth.ts` + `app/server-auth.ts`:

- The signed token is stored in a **cookie** (`morph_token`, non-httpOnly), not
  localStorage, so **server components** (the examinations list/detail pages, which
  fetch during SSR) can read it via `serverAuthHeaders()` (`app/server-auth.ts`,
  using `next/headers` `cookies()`) and forward it as a Bearer header. Those pages
  `redirect('/login')` on a 401.
- Client code uses `apiFetch(path, init)` from `app/auth.ts`, which attaches the
  Bearer token (+ optional `X-API-Key`) and, on a 401, clears the session and
  redirects to `/login`. **Route all client API calls through `apiFetch`.** Media
  URLs (volume/preview) get the token as a `?token=` query param (`authQuery`,
  re-exported through `cs-volume-url.ts`), since `<img>` / the Cornerstone loader
  can't always set a header.
- `/login` (`app/login/page.tsx`) posts credentials, calls `setSession`, and
  redirects to `?next=` or `/examinations`. The navbar `AuthNav` shows the current
  user + sign-out. When the API has auth disabled (dev), no 401s fire, so the app
  works without signing in.

### API Configuration

`app/server_config.ts` exposes a `model_api` base URL as a **getter**, resolved
per access because it must differ between SSR and the browser:

- **SSR** (server components fetching during render): `http://localhost:<port>` —
  the API on loopback of the same machine, always reachable.
- **Browser** (client fetches, login, Cornerstone volume/preview URLs): same
  `protocol`+`hostname` as the current page, on the API port. This is what makes
  the UI work when opened via **both** `http://localhost:3000` and
  `http://<lan-ip>:3000`: a page on the LAN IP calls `http://<lan-ip>:8000`, never
  loopback. A hardcoded `localhost` API URL would make the LAN-IP page fetch a
  more-private address, which **Chrome's Private Network Access policy blocks**
  (`…request client is not a secure context and the resource is in more-private
  address space loopback`) — no API-side CORS header can override that.

Resolution order: `NEXT_PUBLIC_MODEL_API` (explicit override, set only when the API
is on a different host than the UI) → browser same-hostname derivation
(`NEXT_PUBLIC_MODEL_API_PORT`, default 8000) → `http://localhost:<port>`.
`NEXT_PUBLIC_*` values are inlined at build time, so rebuild after changing them;
copy `.env.example` to `.env.local` to override. Components import `server_config`
directly (`@/app/server_config`) for API calls. Path alias `@/*` maps to project
root.