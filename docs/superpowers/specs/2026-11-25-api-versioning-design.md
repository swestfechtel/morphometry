# Versioned, archived images for the whole web app — design

Date: 2026-11-25. Status: draft for approval.

## Intent

Every number the web app serves must be traceable to the exact code and model weights that
produced it, and that code must still exist and be runnable later. Today it is neither: the
worker runs `swestfechtel/torsion:latest` / `swestfechtel/nnunet_torsion:latest`
(`api/settings.py:51-52`), mutable tags; nothing records which image a result came from; the
API has no version and runs from a venv; the frontend runs from `npm`; old torsion images
survive only as hand-made local `pre-<sha>` tags. For production the whole app — API and
frontend included — should run from docker images under the same versioning rules.

## Goals

1. **Four images, one shared semantic version** (`MAJOR.MINOR.PATCH`):
   `nnunet_torsion` (segmentation), `torsion` (measurement), `api` (FastAPI + RQ worker),
   `frontend` (Next.js). API and torsion stay separate images for modularity (user's choice).
   A release `1.3.0` names all four as of that release.
2. **Old images are never overwritten.** Every release's images stay pullable from Docker Hub
   under an immutable `:<version>` tag and by `@sha256:` digest.
3. **Every release carries a changelog** against the previous one (`CHANGELOG.md`), and each
   image carries its release's changelog section as a label.
4. **Every examination records its provenance** in a new, separate DB column.
5. **Rules are enforced, not remembered** (option C, user's choice): a local release script is
   the only way to build/push images, plus a docker-free GitHub Actions check.
6. **Production runs from a compose file** that references the four images by version.

## Non-goals

- URL versioning (`/v1/...`): the frontend is the only client.
- Showing provenance in the frontend (follow-up; the API exposes it).
- Building any image in CI (nnUNet cannot be: 79.5 GB, 32 GB of gitignored checkpoints; the
  others could, but the local release script is the single build path — see Approaches).
- TLS / public-internet hardening of the compose deployment; Orthanc and its plugin stay as
  they are, outside compose.
- Re-tagging the existing `pre-<sha>` images; back-filling provenance of old examinations.
- Changing the dev workflow: `npm run dev`, `uvicorn` and the venv worker keep working.

## Anti-goals (failure even if it "works")

- A version tag on Docker Hub that ever points to a different digest than when first pushed.
- A rebuild/re-push of the 79.5 GB nnUNet image when only other code changed.
- The app serving numbers from an image it did not record.

## Constraints (found in the repo)

- No CI exists; the user pushes directly to `main` (no PR flow).
- **torsion**: build context `morphometry/`, 2.3 GB, on Hub; local `:latest` =
  `swestfechtel/torsion@sha256:bb619140…` (built at `2d1fc91`).
- **nnUNet**: build context `api/docker/nnunet_torsion/`; checkpoints gitignored (invisible to
  git and CI). The image the API runs exists **only locally**: local `:latest` id
  `sha256:70229af9…` (built 2026-05-19) ≠ Hub `:latest` (config `sha256:4adf9f3b…`, manifest
  `sha256:69b6e0e0…`, ~9.5 GB compressed).
- **api**: no Dockerfile. It imports the `morphometry` library from the host venv
  (`api/ingest/dicom.py`, `api/tasks/*`, `api/domain/*`, `api/storage/store.py`,
  `api/schemas/docker_io.py`), so the api image must contain the library too.
  The worker starts the nnUNet and torsion containers with `docker run -v <dir>:…`; from inside
  a container that `docker run` goes to the **host** daemon, so every path it passes must be a
  host path. Today those dirs come from `tempfile.TemporaryDirectory()` (`api/tasks/torsion.py`
  ~80 and ~121), i.e. a container-private `/tmp` the host cannot see.
- **frontend**: no Dockerfile; Next.js 15.2.4. The browser side already derives the API URL at
  runtime (same hostname, port `NEXT_PUBLIC_MODEL_API_PORT`, default 8000 —
  `frontend/app/server_config.ts`), but SSR calls `http://localhost:8000`, which inside a
  container is the frontend container itself.
- DB: SQLite via SQLModel; new columns go through `api/db/engine.py::_ADDED_COLUMNS`.
- Docker Hub tags are mutable by default — immutability must be checked before each push.

## Approaches considered

Enforcement:
- **A. Local only** — release script + pre-push hook. Covers all images; relies on the hook.
- **B. GitHub Actions does the release** — strongest for the small images, needs Hub
  credentials in GitHub, cannot cover nnUNet, and would split releases across two build paths.
- **C. Both (chosen)** — local release script as the single build/push path; a docker-free
  GitHub Actions check fails a push whose image-affecting changes lack a version bump and
  changelog entry. The same check runs as an optional pre-push hook.

Versioning scope: **one shared version (chosen)** over per-image versions — a single name for
"what the app was running"; unchanged images are re-tagged, not rebuilt.

Image split: **four images (chosen)** — nnUNet, torsion, api, frontend — over merging torsion
into the api image (one digest for both, but less modular) or leaving the API on the venv
(API code never archived).

## Design

### Version source and changelog

- `VERSION` at repo root (single line). Read by the API via `api/version.py`, passed to
  `FastAPI(version=...)`; baked into every image as a label and, for api/frontend, an env var.
- `CHANGELOG.md` at repo root, Keep-a-Changelog style: `## [1.3.0] - 2026-12-01` sections,
  entries prefixed by image (`nnunet:`, `torsion:`, `api:`, `frontend:`). First section `1.0.0`
  with a pre-versioning-history note (the `pre-<sha>` torsion tags, the superseded Hub nnUNet
  manifest digest).
- Git tag `v<version>` on the release commit.

### Bump rules

- **MAJOR**: breaking HTTP/JSON contract change (removed/renamed field or endpoint, changed sign
  convention).
- **MINOR**: anything that moves measured values (a "do-not-pool" event), new endpoints/fields,
  new frontend features.
- **PATCH**: value-identical fixes/refactors, frontend fixes.
- Machine-checked: if a torsion-path golden (explicit allow-list in `check_version.py`) changed
  since the last release tag, the bump must be at least MINOR.

### What "affects an image" (gated paths) — import closures, user's choice

- **torsion**: the import closure of `morphometry/docker/compute_torsion.py` (every
  `morphometry` module it reaches transitively, computed with `ast`), plus
  `morphometry/Dockerfile` and `morphometry/requirements.txt`. The image still contains the
  whole folder; a release is required only when code it runs changes (48 of the last 142
  commits touched `morphometry/`, mostly research code these images never run).
- **api**: `api/**` excluding `api/docker/`, `api/tests*`, `api/data/`, `api/logs/`,
  `api/*.ipynb`; plus the import closure of the `morphometry` modules `api/` imports; plus the
  api Dockerfile and its requirements file.
- **frontend**: `frontend/**` excluding `node_modules/`, `.next/`, `*.tsbuildinfo`, `.env*`.
- **nnUNet**: `api/docker/nnunet_torsion/**` excluding `checkpoints/`, plus
  `api/docker/nnunet_torsion/checkpoints.sha256` — a committed sha-256 manifest of every
  checkpoint file (32 GB, 103 files), so a swapped checkpoint shows up in git.
- An import the closure walker cannot resolve fails the check loudly rather than shrinking the
  closure. Closures are computed at check time, not committed.

Bump timing (builder-check answer): `VERSION` is bumped **in the commit that makes the gated
change**, keeping CI green. It can then sit ahead of `release/images.json`; settings fall back
to the newest released entry and `/version` reports e.g. `1.4.0-dev, images from 1.3.0`.

### The two new images

- **api** (`api/Dockerfile`, build context repo root with a `.dockerignore` that admits only
  `api/`, `morphometry/`, `VERSION`): base `python:3.10.16-bookworm` as for torsion;
  dependencies from a new fully pinned `api/requirements.txt` (the API's transitive closure,
  versions equal to the root `requirements.txt`, installed `--no-deps` + `pip check`, same
  discipline as `morphometry/requirements.txt`); `docker` CLI installed. One image, two
  compose services: `api` (`uvicorn api.main:app`) and `worker` (`python -m api.tasks.worker`).
- **frontend** (`frontend/Dockerfile`): Next.js `output: "standalone"`, multi-stage, Node
  version pinned to the one the repo builds with. SSR reaches the API through a new **runtime**
  (non-`NEXT_PUBLIC`) env var, e.g. `MODEL_API_INTERNAL=http://api:8000`, read in
  `server_config.ts` on the server path only; the browser path is unchanged. So the frontend
  image is deployment-independent.

### Production compose (`deploy/compose.yml`)

Services `redis` (pinned by digest), `api`, `worker`, `frontend`; images referenced as
`swestfechtel/<image>:${MORPH_VERSION}`. The release script writes the digests per version
into `release/images.json`; the compose file uses version tags (immutable by rule 2).
- Storage dir and SQLite DB on a bind mount at the **same absolute path** inside and outside
  the container (e.g. `/srv/morphometry`), and the worker's job temp dirs are created **under
  that mount** (new setting, e.g. `MORPH_API_JOB_TMP_DIR`, defaulting to the system temp dir
  so the venv workflow is unchanged). This is what makes the worker's `docker run -v` paths
  valid for the host daemon.
- `worker` mounts `/var/run/docker.sock` (it launches the nnUNet and torsion containers on the
  host daemon; GPU access is requested by those `docker run`s as today, the worker itself needs
  none).
- Config via an `.env` next to the compose file (`MORPH_API_*`), as today.

### Release manifest

`release/images.json` (committed), appended by the release script:

```json
{"1.0.0": {"nnunet":   "swestfechtel/nnunet_torsion@sha256:…",
           "torsion":  "swestfechtel/torsion@sha256:bb61…",
           "api":      "swestfechtel/morphometry_api@sha256:…",
           "frontend": "swestfechtel/morphometry_frontend@sha256:…",
           "git_commit": "…", "date": "2026-11-25"}}
```

`api/settings.py` defaults `nnunet_image` / `torsion_image` to the digests of the entry for the
current `VERSION` (or the newest released one, see bump timing); env overrides still win. The
worker thus launches containers by digest, never `:latest`.

### Local release script (`scripts/release.py`) — the only build/push path

Refuses (non-zero exit, clear message) unless all of:
1. working tree clean, on `main`, up to date with `origin/main`;
2. `VERSION` > last `v*` tag, `CHANGELOG.md` has a section for it, the golden rule holds;
3. git tag `v<VERSION>` exists neither locally nor on origin;
4. `:<VERSION>` exists on Docker Hub for none of the four images (`docker manifest inspect`);
5. local checkpoints match `checkpoints.sha256` (only when nnUNet is rebuilt).

Then per image: if its gated paths changed since the last release tag → `docker build` with
OCI labels (`org.opencontainers.image.version`, `.revision` = full git sha, `.created`) and a
`morphometry.changelog` label holding this release's changelog section; otherwise add the
`:<VERSION>` tag to the previous release's digest (no rebuild, no re-upload). Push `:<VERSION>`,
read back the pushed digest, move `:latest` (convenience only). Write the digests into
`release/images.json`, commit `release: v<VERSION>`, tag, push commit + tag. `--dry-run` prints
every decision without building or pushing; `--only <image>` is not offered (a release is
always all four).

### GitHub Actions check (`.github/workflows/version-check.yml`)

On every push to `main` (and PRs): `scripts/check_version.py` (pure git + file reads, no
docker). If any image's gated paths changed since the last `v*` tag it requires a `VERSION`
bump, a matching `CHANGELOG.md` section and the golden rule. It also checks
`morphometry/requirements.txt` and `api/requirements.txt` versions equal the root pins.
Installable as a pre-push hook (`scripts/install_hooks.sh`). Since pushes go straight to
`main`, CI flags rather than prevents; the release script is the hard gate on images.

### Provenance column

- New `Examination.provenance: dict | None` (JSON), added to `_ADDED_COLUMNS`.
- Per stage, because stages can be re-run separately:
  ```json
  {"segmentation": {"api_version": "1.3.0", "image": "swestfechtel/nnunet_torsion@sha256:…",
                    "released": true, "api_image": "…@sha256:…|null",
                    "git_commit": "…", "dirty": false, "ran_at": "…"},
   "torsion":      {…same keys…}}
  ```
  Each task writes only its own key. `api_image` is the worker's own image when it runs in
  compose (passed in via env by compose/release), `null` on the venv; `git_commit`/`dirty` are
  baked in at build for the image, read from the checkout on the venv. Together they cover the
  host-side code that moves values (slab splitting, `transform_coordinate_system()`).
- `image` is resolved **at job time** from the image actually run (`docker image inspect`).
  An image with no registry digest (local dev build) is run, not refused, and recorded by its
  local id with `released: false`; likewise any digest not in `release/images.json`.
- No stale-marking on re-runs: timestamps show a torsion entry older than its segmentation.
- Exposed on the examination endpoints via `api/schemas/examination.py`.

### Version endpoint

`GET /version` (unauthenticated, like `/health`): `{"version", "images_from", "git_commit",
"images": {"nnunet", "torsion"}}` — the configured references.

### Bootstrap (1.0.0)

- torsion: current `:latest` (`bb6191…`) tagged `:1.0.0`.
- nnUNet (user's choice): the **local** image (`70229af9…`) is pushed as `:1.0.0` and Hub
  `:latest` repointed; the superseded Hub image stays reachable by manifest digest, recorded in
  the changelog. Large, slow one-off upload run by the user, not CI.
- api, frontend: first builds, pushed as `:1.0.0`.
- `checkpoints.sha256` generated from the current local checkpoints.

### Docs

`README.md`, `README_WEBAPP.md`, `api/README.md`, `frontend/README.md`/`CLAUDE.md` and
`CLAUDE.md` ("Docker images": four images, the release flow replacing the `pre-<sha>`
convention; "Running the API/UI": the compose path next to the dev path).

## Suggested build order (for the plan)

1. Version source, changelog, `check_version.py` + CI + hook (no images touched).
2. Provenance column, `/version`, digest resolution in `docker_run.py`, job temp dir setting.
3. api and frontend Dockerfiles, `deploy/compose.yml`, SSR env var.
4. `release.py` (`--dry-run` first), checkpoint manifest, then the 1.0.0 bootstrap.

## Left to the builder

Exact CLI flags; finding the last release tag (`git describe --tags --match 'v*'`); changelog
parsing; hook mechanics; image names on Hub for api/frontend (proposed above); the Node
version pin source; test layout (unit tests for `check_version.py`, the closure walker and the
release script's decision logic with git/docker mocked).
