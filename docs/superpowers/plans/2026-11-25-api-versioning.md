# Versioned, Archived Images Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One shared semver for four docker images (nnUNet, torsion, api, frontend), never-overwritten
releases with changelogs, enforced by a local release script plus a GitHub Actions check, and per-stage
result provenance stored in a new DB column.

**Architecture:** A small repo-root `versioning/` package holds the pure logic (version parsing,
changelog parsing, import-closure walker, gated paths per image, release decisions) and is driven by two
thin CLIs, `scripts/check_version.py` (CI + pre-push) and `scripts/release.py` (build/tag/push). The API
gains `api/version.py`, a provenance module called by the worker after each container run, and a
`/version` endpoint. Two new Dockerfiles and `deploy/compose.yml` run the app in production.

**Tech Stack:** Python 3.10 (stdlib `ast`, `subprocess`, `re`), FastAPI/SQLModel, pytest, Docker /
compose, GitHub Actions, Next.js 15 standalone output, Node 22.

**Spec:** `docs/superpowers/specs/2026-11-25-api-versioning-design.md`

## Global Constraints

- Versions are plain `MAJOR.MINOR.PATCH` (regex `^\d+\.\d+\.\d+$`); git tags are `v<version>`.
- Docker Hub repos: `swestfechtel/nnunet_torsion`, `swestfechtel/torsion`,
  `swestfechtel/morphometry_api`, `swestfechtel/morphometry_frontend`. Image keys everywhere:
  `nnunet`, `torsion`, `api`, `frontend`.
- A `:<version>` tag is never pushed if it already exists on Hub (anti-goal: a tag that changes digest).
- The nnUNet image is never rebuilt or re-pushed unless its gated paths changed.
- OCI labels on every built image: `org.opencontainers.image.version`, `org.opencontainers.image.revision`
  (full git sha), `org.opencontainers.image.created`, plus `morphometry.changelog` (this release's section).
- `CHANGELOG.md` entries are prefixed `nnunet:`, `torsion:`, `api:`, `frontend:`.
- Pinned requirement files (`morphometry/requirements.txt`, new `api/requirements.txt`) hold versions
  exactly equal to the root `requirements.txt`.
- Base image for api = `python:3.10.16-bookworm` (same as torsion); Node pinned to `22.14.0`
  (the version the repo currently builds with).
- Dev workflow must keep working unchanged: `uvicorn`, `python -m api.tasks.worker`, `npm run dev`.
- No git push, docker push, or Hub tag change is performed by an implementer: those happen only in
  Task 11, run by the user (outward-facing, irreversible).
- Repo conventions: verbose docstrings with `:param:`/`:return:`; new public behaviour documented in
  README/CLAUDE.md (Task 12).

## Review Focus

1. **A release interrupted after pushing some images** — rerunning must neither overwrite nor dead-lock:
   a pre-existing `:<version>` tag is accepted only when its `org.opencontainers.image.revision` label
   equals HEAD (resume), and refused otherwise. Test in Task 10.
2. **First release with no prior `v*` tag** — every image counts as changed and the version check passes
   without a "previous version"; no crash on `git describe`. Tests in Tasks 3 and 10.
3. **Malformed or non-increasing `VERSION`** (`1.2`, `v1.2.0`, `1.2.0-rc1`, equal to or below the last
   tag) — a clear failure message naming the file and both versions. Test in Task 3.
4. **The worker cannot inspect the image** (docker daemon hiccup, image pruned after the run) —
   provenance records the configured reference with `digest: null, released: false`; the job does not
   fail because of provenance. Test in Task 5.
5. **CI shallow clone** — tags unreachable must not look like "no prior release" silently; the
   workflow fetches full history and `check_version.py` errors if `git tag` lists `v*` tags that
   `git describe` cannot reach. Test in Task 3.

---

## File Structure

| File | Responsibility |
|---|---|
| `VERSION` | the current version, one line |
| `CHANGELOG.md` | release notes, one section per version |
| `versioning/__init__.py` | package marker |
| `versioning/semver.py` | parse/compare versions, find last release tag |
| `versioning/changelog.py` | extract a version's section from `CHANGELOG.md` |
| `versioning/closure.py` | static import closure of a module set |
| `versioning/images.py` | the four `ImageSpec`s and their gated paths; `changed_images()` |
| `versioning/check.py` | the version/changelog/golden/pin rules as one function |
| `versioning/release_plan.py` | pure release decisions (build vs retag, refusals) |
| `versioning/checkpoints.py` | write/verify `checkpoints.sha256` |
| `scripts/check_version.py` | CLI for `versioning.check` |
| `scripts/release.py` | CLI executing a `ReleasePlan` with docker/git |
| `scripts/install_hooks.sh` | installs the pre-push hook |
| `.github/workflows/version-check.yml` | CI |
| `release/images.json` | released digests per version |
| `api/version.py` | version, git commit, dirty flag for the running API |
| `api/release_manifest.py` | read `release/images.json`; default image refs |
| `api/tasks/provenance.py` | resolve image digests, build a stage record |
| `api/routers/version.py` | `GET /version` |
| `api/Dockerfile`, `api/requirements.txt`, `.dockerignore` | api image |
| `frontend/Dockerfile`, `frontend/.dockerignore` | frontend image |
| `deploy/compose.yml`, `deploy/.env.example` | production stack |
| `tests/versioning/` | tests for `versioning/` and the two CLIs |

---

### Task 1: Version source and changelog

**Files:**
- Create: `VERSION` (content `1.0.0\n`), `CHANGELOG.md`, `api/version.py`, `versioning/__init__.py`,
  `versioning/semver.py`, `versioning/changelog.py`
- Modify: `api/main.py` (`FastAPI(title=..., version=get_version(), lifespan=...)`)
- Test: `tests/versioning/test_semver.py`, `tests/versioning/test_changelog.py`, `api/tests/test_version.py`

**Interfaces:**
- Produces:
  - `versioning.semver.Version` — `NamedTuple(major: int, minor: int, patch: int)`, ordered;
    `parse_version(text: str) -> Version` raises `ValueError(f"invalid version {text!r} …")`;
    `str(Version)` → `"1.2.3"`.
  - `versioning.semver.last_release_tag(repo: Path) -> str | None` — newest `v*` tag reachable from
    HEAD (`git describe --tags --abbrev=0 --match 'v[0-9]*'`), `None` if no `v*` tag exists at all;
    raises `RuntimeError("v* tags exist but none is reachable from HEAD (shallow clone?)")` when
    `git tag -l 'v[0-9]*'` is non-empty but describe fails.
  - `versioning.changelog.section(changelog_text: str, version: str) -> str | None` — body of
    `## [<version>] - YYYY-MM-DD` up to the next `## [` heading, stripped; `None` if absent.
  - `api.version.get_version() -> str` (reads repo-root `VERSION`; env `MORPH_VERSION` wins),
    `get_git_commit() -> str | None` (env `MORPH_GIT_COMMIT`, else `git rev-parse HEAD`, else `None`),
    `is_dirty() -> bool | None` (`False` when `MORPH_GIT_COMMIT` is set — image builds are clean by
    construction; else `git status --porcelain` non-empty; `None` if git unavailable).

- [ ] **Step 1: Write failing tests**
  - `test_parse_version_ok`: `parse_version("1.10.0") > parse_version("1.9.3")`.
  - `test_parse_version_rejects` parametrized over `"1.2"`, `"v1.2.0"`, `"1.2.0-rc1"`, `""` → `ValueError`.
  - `test_last_release_tag` on a `tmp_path` git repo (fixture `git_repo` in `tests/versioning/conftest.py`
    that inits a repo with one commit; reuse it in later tasks): no tags → `None`; tag `v1.0.0` → `"v1.0.0"`.
  - `test_section`: text with `## [1.1.0] - 2026-12-01\n- api: x\n\n## [1.0.0] - 2026-11-25\n- y` →
    `section(text, "1.1.0") == "- api: x"`; `section(text, "2.0.0") is None`.
  - `test_api_version_env_override`: with `MORPH_VERSION=9.9.9`, `MORPH_GIT_COMMIT=abc` →
    `get_version() == "9.9.9"`, `get_git_commit() == "abc"`, `is_dirty() is False`.
- [ ] **Step 2: Run** `pytest tests/versioning api/tests/test_version.py -v` — expect import failures.
- [ ] **Step 3: Implement** the three modules; write `CHANGELOG.md` with a header and a `## [1.0.0] - 2026-11-25`
  section: one line per image describing the 1.0.0 state, and a "Pre-versioning history" list of the local
  `swestfechtel/torsion:pre-<sha>` tags (from `docker images`) and the superseded Hub nnUNet manifest
  `sha256:69b6e0e0400daf13221c8ab6b24f472b827c882b6cc0f49ef3d033a2b0b34cff`. Wire `version=` in `api/main.py`.
- [ ] **Step 4: Run** the same command — PASS. Run `pytest api/tests -q` — still green.
- [ ] **Step 5: Commit** `feat(versioning): VERSION, CHANGELOG and version helpers`.

### Task 2: Import closure and gated paths per image

**Files:**
- Create: `versioning/closure.py`, `versioning/images.py`
- Test: `tests/versioning/test_closure.py`, `tests/versioning/test_images.py`

**Interfaces:**
- Produces:
  - `versioning.closure.import_closure(entry_files: list[Path], repo: Path, package: str = "morphometry") -> set[Path]`
    — repo-relative paths of every `package` module reachable from `entry_files` via `import x`,
    `from x import y` (including `y` being a submodule), relative imports, and `__init__.py` of every
    package traversed. Imports inside functions count. Non-`package` imports are ignored. Raises
    `versioning.closure.UnresolvedImport(path, lineno, name)` for a `package` import that maps to no file,
    and for `importlib.import_module`/`__import__` calls whose argument starts with `package`.
  - `versioning.images.ImageSpec` — frozen dataclass: `key: str`, `repo: str` (Hub repo),
    `context: str`, `dockerfile: str`, `gated: Callable[[Path], set[str]]` (repo-relative files).
  - `versioning.images.IMAGES: dict[str, ImageSpec]` — order `nnunet, torsion, api, frontend`
    (the release builds in this order). Gated sets exactly as the spec's "What affects an image":
    - torsion: `import_closure([morphometry/docker/compute_torsion.py])` ∪ `morphometry/Dockerfile`, `morphometry/requirements.txt`, `morphometry/docker/compute_torsion.py`.
    - api: tracked files under `api/` excluding `api/docker/`, `api/tests/`, `api/data/`, `api/logs/`,
      `*.ipynb`, `api/tests.py`; ∪ `import_closure(all api/**/*.py in that set)`; ∪ `api/Dockerfile`,
      `api/requirements.txt`, `.dockerignore`.
    - frontend: tracked files under `frontend/` excluding `node_modules/`, `.next/`, `*.tsbuildinfo`, `.env*`.
    - nnunet: tracked files under `api/docker/nnunet_torsion/` (checkpoints are untracked) — this
      includes `checkpoints.sha256`.
    "Tracked" = `git ls-files` at the ref being examined.
  - `versioning.images.changed_images(repo: Path, base_ref: str | None, head_ref: str = "HEAD") -> dict[str, list[str]]`
    — image key → sorted changed gated files (`git diff --name-only base..head`, gated set computed at
    `head_ref` **and** `base_ref`, union, so deleting a closure module counts). `base_ref=None` → every
    image maps to `["<initial release>"]`.

- [ ] **Step 1: Write failing tests**
  - `test_closure_follows_imports`: fake package in `tmp_path` (`morphometry/__init__.py`, `a.py` importing
    `from morphometry import b`, `b.py` with `from .c import f` inside a function, `c.py`, `unused.py`) →
    closure of `entry.py` (`import morphometry.a`) == `{__init__, a, b, c}`; `unused.py` absent.
  - `test_closure_unresolved_raises`: `from morphometry import missing_mod` where `missing_mod` is neither a
    module nor a name defined in `__init__.py` → `UnresolvedImport`.
  - `test_closure_dynamic_import_raises`: `importlib.import_module("morphometry.x")` → `UnresolvedImport`.
  - `test_real_torsion_closure` (on the real repo): contains `morphometry/measurements/femur.py`,
    `morphometry/image_io.py`; excludes `morphometry/cartilage/thickness.py`.
  - `test_changed_images` on `git_repo` fixture laid out with a mini `morphometry/`, `api/`, `frontend/`
    (monkeypatch `IMAGES` entry points to the mini files): editing `morphometry/unused.py` → `{}`; editing
    a closure file → `torsion` and `api` both listed if `api` imports it; editing `frontend/app/x.ts` →
    `frontend` only; `base_ref=None` → all four.
- [ ] **Step 2: Run** `pytest tests/versioning/test_closure.py tests/versioning/test_images.py -v` — FAIL.
- [ ] **Step 3: Implement.** Resolve `from X import Y` as module `X.Y` if a file exists, else as a name in
  `X` (then `X` alone). Walk with a worklist; parse each file once.
- [ ] **Step 4: Run** — PASS. Print the real torsion closure once (`python -c ...`) and paste its size in
  the commit message body for the reviewer.
- [ ] **Step 5: Commit** `feat(versioning): import closures and gated paths per image`.

### Task 3: The version rules and `check_version.py`

**Files:**
- Create: `versioning/check.py`, `scripts/check_version.py`
- Test: `tests/versioning/test_check.py`

**Interfaces:**
- Consumes: Task 1 (`parse_version`, `last_release_tag`, `section`), Task 2 (`changed_images`).
- Produces:
  - `versioning.check.GOLDEN_ALLOWLIST: tuple[str, ...]` — torsion-path golden files: every
    `tests/golden/*.json` read by `tests/test_compute_torsion_pipeline.py` or by a characterization test of
    a function in the torsion closure. Build it by grepping the golden file names those tests load; list
    them literally with a comment on how it was derived.
  - `versioning.check.check(repo: Path, head_ref: str = "HEAD") -> list[str]` — list of violation
    messages, empty when compliant:
    1. `VERSION` parses (Task 1 message).
    2. if `changed_images(...)` non-empty: `VERSION` > last tag's version
       (`"VERSION 1.2.0 must be greater than the last release v1.2.0 because <image>: <files…> changed"`)
       and `CHANGELOG.md` has a section for it.
    3. if any `GOLDEN_ALLOWLIST` file changed since the last tag: new version's (major, minor) >
       last's (`"golden <file> changed since v1.2.0: bump at least MINOR"`).
    4. every pin in `morphometry/requirements.txt` and `api/requirements.txt` (if present) equals the root
       `requirements.txt` pin for that package (case/`_`-`-` insensitive).
    With no prior tag: rule 2 requires only the changelog section; rule 3 is skipped.
  - `scripts/check_version.py` — `main(argv: list[str] | None = None) -> int`; prints each violation,
    returns 1 if any, else prints `"version check ok (<VERSION>)"` and returns 0. Option `--repo PATH`.

- [ ] **Step 1: Write failing tests** on the `git_repo` fixture (tag `v1.0.0` on the base commit):
  - `test_no_gated_change_passes` (README edit only, VERSION unchanged) → `[]`.
  - `test_gated_change_without_bump_fails` → one message containing `"must be greater"` and the file.
  - `test_bump_without_changelog_fails` → message mentions `CHANGELOG.md`.
  - `test_golden_change_needs_minor`: patch bump + allow-listed golden changed → violation; minor bump → `[]`.
  - `test_pin_mismatch` → violation naming the package and both versions.
  - `test_initial_release` (no tags, changelog has 1.0.0) → `[]`.
  - `test_malformed_version` parametrized (`1.2`, `v1.1.0`, `1.1.0-rc1`) → message contains `"VERSION"`.
  - `test_shallow_unreachable_tags`: tag on a commit not reachable from HEAD (orphan branch) → `check`
    raises/reports the shallow-clone message from Task 1.
  - `test_cli_exit_codes`: `main(["--repo", str(repo)])` returns 0/1 accordingly.
- [ ] **Step 2: Run** `pytest tests/versioning/test_check.py -v` — FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** — PASS; then `python scripts/check_version.py` on the real repo (no tags yet) → ok.
- [ ] **Step 5: Commit** `feat(versioning): version/changelog/golden/pin check`.

### Task 4: CI workflow and pre-push hook

**Files:**
- Create: `.github/workflows/version-check.yml`, `scripts/install_hooks.sh`

- [ ] **Step 1:** Workflow: `on: push: branches: [main]` and `pull_request`; job on `ubuntu-latest`,
  `actions/checkout@v4` with `fetch-depth: 0` and `fetch-tags: true`, `actions/setup-python@v5`
  `python-version: "3.10"`, run `python scripts/check_version.py` (stdlib only — `versioning/` must not
  import third-party packages; assert that in a test: `tests/versioning/test_stdlib_only.py` parses every
  `versioning/*.py` and fails on any non-stdlib top-level import).
- [ ] **Step 2:** `scripts/install_hooks.sh` writes `.git/hooks/pre-push` that runs
  `python scripts/check_version.py || exit 1`; refuses to overwrite an existing non-identical hook.
- [ ] **Step 3: Verify** `bash scripts/install_hooks.sh && .git/hooks/pre-push` exits 0; run
  `pytest tests/versioning -q` — PASS. Validate YAML with `python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/version-check.yml'))"`.
- [ ] **Step 4: Commit** `ci: version check workflow and pre-push hook`.

### Task 5: Provenance column and per-stage records

**Files:**
- Create: `api/tasks/provenance.py`
- Modify: `api/db/models.py` (add `provenance: dict | None = Field(default=None, sa_column=Column(JSON))`),
  `api/db/engine.py` (`_ADDED_COLUMNS["examinations"]["provenance"] = "JSON"`), `api/tasks/torsion.py`
  (`_segment` / `_align` write their stage record), `api/schemas/examination.py`
  (`TorsionDetail.provenance: dict | None = None`), `api/serializers.py::to_detail`
- Test: `api/tests/test_provenance.py`, extend `api/tests/test_tasks.py`, `api/tests/test_db.py`

**Interfaces:**
- Consumes: Task 1 `api.version.*`; Task 6 `api.release_manifest.released_digests()`.
  **Execute Task 6 before Task 5** (the numbering follows the spec's reading order, not dependency order).
- Produces:
  - `api.tasks.provenance.inspect_image(ref: str) -> tuple[str | None, str | None]` — `(repo_digest, image_id)`
    via `docker image inspect --format '{{json .RepoDigests}} {{.Id}}'`; `(None, None)` on any failure
    (logs a warning, never raises). `repo_digest` is the entry matching the ref's repository.
  - `api.tasks.provenance.stage_record(image_ref: str) -> dict` — keys exactly
    `api_version, image, digest, released, api_image, git_commit, dirty, ran_at` where
    `image` = configured ref, `digest` = repo digest or else local image id or `None`,
    `released` = `digest in released_digests()`, `api_image` = env `MORPH_API_SELF_IMAGE` or `None`,
    `ran_at` = UTC ISO-8601.
  - `_segment` sets `ex.provenance = {**(ex.provenance or {}), "segmentation": stage_record(settings.nnunet_image)}`;
    `_align` likewise under `"torsion"` with `settings.torsion_image`. Records are taken right after the
    container returns, before results are persisted.

- [ ] **Step 1: Write failing tests**
  - `test_stage_record_released` (monkeypatch `inspect_image` → `("swestfechtel/torsion@sha256:aa", "sha256:bb")`,
    `released_digests` → `{"swestfechtel/torsion@sha256:aa"}`) → `released is True`, `digest` the repo digest.
  - `test_stage_record_local_build` (`(None, "sha256:bb")`) → `digest == "sha256:bb"`, `released is False`.
  - `test_stage_record_inspect_fails` (`(None, None)`) → `digest is None`, `released is False`, no exception.
  - `test_run_torsion_writes_provenance` (existing `fake_docker_run`; monkeypatch `inspect_image`): after
    `mode="full"`, `ex.provenance` has both keys; after a subsequent `mode="torsion"` run the
    `segmentation` record is unchanged and `torsion.ran_at` is newer.
  - `test_existing_db_gains_provenance_column`: create an `examinations` table without the column via raw
    SQL, run `init_db`, assert the column exists.
  - `test_detail_exposes_provenance` via the API test client.
- [ ] **Step 2: Run** `pytest api/tests/test_provenance.py api/tests/test_tasks.py api/tests/test_db.py -v` — FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `pytest api/tests -q` — PASS.
- [ ] **Step 5: Commit** `feat(api): per-stage provenance column`.

### Task 6: Release manifest, image defaults, job temp dir, `/version`

**Files:**
- Create: `release/images.json` (`{}`), `api/release_manifest.py`, `api/routers/version.py`
- Modify: `api/settings.py`, `api/main.py` (include router, unauthenticated like `health`),
  `api/tasks/torsion.py` (both `tempfile.TemporaryDirectory()` → `dir=settings.job_tmp_dir`),
  `api/.env.example`
- Test: `api/tests/test_release_manifest.py`, `api/tests/test_version.py` (extend)

**Interfaces:**
- Produces:
  - `api.release_manifest.MANIFEST_PATH = <repo>/release/images.json`;
    `load_manifest(path: Path = MANIFEST_PATH) -> dict[str, dict]` (missing file → `{}`);
    `images_for(version: str, manifest: dict) -> tuple[str | None, dict]` — `(images_from, entry)`: the
    entry for `version` if present, else the newest version lower than it, else `(None, {})`;
    `released_digests(manifest: dict | None = None) -> set[str]` — all image refs in all entries.
  - `Settings.nnunet_image` / `torsion_image`: default `None`; new properties
    `resolved_nnunet_image` / `resolved_torsion_image` → env value if set, else the manifest entry from
    `images_for(get_version(), ...)`, else the old `:latest` tag with a logged WARNING. Every caller
    (`api/tasks/docker_run.py`, Task 5) uses the `resolved_*` properties.
  - `Settings.job_tmp_dir: Path | None = None` (env `MORPH_API_JOB_TMP_DIR`); created on use.
  - `GET /version` → `{"version": str, "images_from": str | None, "git_commit": str | None,
    "dirty": bool | None, "images": {"nnunet": str, "torsion": str}}`; `version` gets a `-dev` suffix
    when `images_from != version`.

- [ ] **Step 1: Write failing tests**
  - `images_for("1.4.0", {"1.3.0": {...}, "1.2.0": {...}})` → `("1.3.0", entry_1_3)`;
    exact match wins; empty manifest → `(None, {})`; `"1.2.0"` with only `"1.3.0"` → `(None, {})`.
  - `test_settings_env_wins` / `test_settings_manifest_default` / `test_settings_latest_fallback_warns`.
  - `test_version_endpoint_dev_suffix` (VERSION 1.4.0, manifest has 1.3.0) → `"1.4.0-dev"`,
    `images_from == "1.3.0"`; no auth header needed with `auth_required=True`.
  - `test_job_tmp_dir_used`: set `job_tmp_dir` to `tmp_path/"jobs"`; the fake docker run asserts the mount's
    host path starts with it.
- [ ] **Step 2: Run** `pytest api/tests/test_release_manifest.py api/tests/test_version.py -v` — FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `pytest api/tests -q` — PASS.
- [ ] **Step 5: Commit** `feat(api): /version, manifest-pinned image defaults, job temp dir`.

### Task 7: The api image

**Files:**
- Create: `api/requirements.txt`, `api/Dockerfile`, `.dockerignore` (repo root)
- Test: `tests/versioning/test_api_requirements.py`

**Interfaces:**
- Produces: image built with `docker build -f api/Dockerfile .` from repo root; build args
  `MORPH_VERSION`, `MORPH_GIT_COMMIT`, `MORPH_NNUNET_IMAGE`, `MORPH_TORSION_IMAGE` become `ENV
  MORPH_VERSION`, `MORPH_GIT_COMMIT`, `MORPH_API_NNUNET_IMAGE`, `MORPH_API_TORSION_IMAGE` (so the image
  launches exactly the model images of its own release). Default `CMD` = `uvicorn api.main:app --host
  0.0.0.0 --port 8000`; worker runs with command `python -m api.tasks.worker`.

- [ ] **Step 1: Write `api/requirements.txt`**: the transitive closure of what `api/` imports at runtime
  (derive with `pipdeptree` or `pip show` on the venv, starting from the API's third-party imports +
  `morphometry/requirements.txt`), pinned to the root versions. Test
  `test_api_requirements_closed`: in the venv, `importlib.metadata.requires` of every listed package is
  satisfied by the list (ignoring extras/markers that don't apply on Linux/py3.10). Task 3's pin rule
  covers equality.
- [ ] **Step 2: Write `api/Dockerfile`**: base `python:3.10.16-bookworm`; `apt-get install
  --no-install-recommends docker.io` is too heavy — install only the static docker CLI binary from
  `download.docker.com/linux/static/stable/x86_64/docker-<pinned>.tgz` matching the host's `docker
  version --format '{{.Client.Version}}'`; `libgl1 libxrender1` as in the torsion image; `pip install
  --no-deps -r api/requirements.txt && pip check`; copy `api/`, `morphometry/`, `VERSION`; non-root user
  in group `docker` is **not** done here (socket group id differs per host — compose sets `group_add`).
  `.dockerignore`: deny all, then allow `api/`, `morphometry/`, `VERSION`, and re-deny
  `api/data`, `api/logs`, `api/docker`, `**/__pycache__`, `api/.env`.
- [ ] **Step 3: Verify** `docker build -f api/Dockerfile -t morphometry_api:dev --build-arg MORPH_VERSION=0.0.0
  --build-arg MORPH_GIT_COMMIT=$(git rev-parse HEAD) .` succeeds; `docker run --rm -p 8001:8000
  morphometry_api:dev` then `curl -s localhost:8001/version` shows `"version":"0.0.0-dev"` or `0.0.0`
  and the git commit; `docker run --rm morphometry_api:dev docker --version` prints the pinned CLI;
  `docker run --rm morphometry_api:dev python -c "import morphometry.measurements"` exits 0.
- [ ] **Step 4: Commit** `build(api): api docker image`.

### Task 8: The frontend image

**Files:**
- Modify: `frontend/app/server_config.ts` (server branch: `process.env.MODEL_API_INTERNAL ??
  http://localhost:<port>`), `frontend/next.config.ts` (`output: "standalone"`)
- Create: `frontend/Dockerfile`, `frontend/.dockerignore`

- [ ] **Step 1:** Edit `server_config.ts`: on the server path only, prefer runtime env
  `MODEL_API_INTERNAL`; update the header comment's resolution order. Browser path unchanged.
- [ ] **Step 2:** `frontend/Dockerfile`: multi-stage `node:22.14.0-bookworm-slim`; `npm ci`, `npm run
  build`, copy `.next/standalone`, `.next/static`, `public`; `CMD ["node", "server.js"]`, port 3000;
  `ARG MORPH_VERSION` → `LABEL` + `ENV NEXT_PUBLIC_APP_VERSION` (not displayed yet).
  `.dockerignore`: `node_modules`, `.next`, `.env*`, `*.tsbuildinfo`.
- [ ] **Step 3: Verify** `cd frontend && npm run build` passes locally (standalone output present);
  `docker build -t morphometry_frontend:dev frontend`; run it with `-e MODEL_API_INTERNAL=http://host.docker.internal:8000
  --add-host host.docker.internal:host-gateway -p 3001:3000` against a local API and check
  `curl -s localhost:3001` returns the app HTML and the examination list page renders server-side data
  (the API log shows the SSR request). `npm run dev` still works.
- [ ] **Step 4: Commit** `build(frontend): standalone docker image, runtime SSR API URL`.

### Task 9: Production compose

**Files:**
- Create: `deploy/compose.yml`, `deploy/.env.example`, `deploy/README.md`

- [ ] **Step 1:** Services:
  - `redis`: `redis:7.4.1-bookworm` pinned by digest (resolve with `docker manifest inspect` and write it).
  - `api`: `swestfechtel/morphometry_api:${MORPH_VERSION}`, port 8000, `env_file: .env`,
    `MORPH_API_STORAGE_DIR=${MORPH_DATA_DIR}`, `MORPH_API_REDIS_URL=redis://redis:6379/0`,
    `MORPH_API_SELF_IMAGE=swestfechtel/morphometry_api:${MORPH_VERSION}`, volume
    `${MORPH_DATA_DIR}:${MORPH_DATA_DIR}` (same path both sides).
  - `worker`: same image and env, `command: python -m api.tasks.worker`,
    `MORPH_API_JOB_TMP_DIR=${MORPH_DATA_DIR}/_jobs`, volumes as api plus
    `/var/run/docker.sock:/var/run/docker.sock`, `group_add: ["${DOCKER_GID}"]`.
  - `frontend`: `swestfechtel/morphometry_frontend:${MORPH_VERSION}`, port 3000,
    `MODEL_API_INTERNAL=http://api:8000`.
  `.env.example`: `MORPH_VERSION`, `MORPH_DATA_DIR` (absolute, e.g. `/srv/morphometry`), `DOCKER_GID`
  (`getent group docker | cut -d: -f3`), `MORPH_API_SECRET_KEY`, `MORPH_API_AUTH_REQUIRED`,
  `MORPH_API_CORS_ALLOW_ORIGINS`. `deploy/README.md`: start/upgrade/rollback (= change `MORPH_VERSION`).
- [ ] **Step 2: Verify** `docker compose -f deploy/compose.yml --env-file deploy/.env.example config` is
  valid. Smoke with the Task 7/8 dev images (override image names via a throwaway env file in the
  scratchpad, `MORPH_DATA_DIR` = a scratch dir): `up -d`, `curl localhost:8000/version`,
  `curl localhost:3000`, enqueue a torsion job for a small fixture examination and confirm the worker's
  `docker run` succeeds (mount path visible to the host) — then `down`.
- [ ] **Step 3: Commit** `deploy: production compose stack`.

### Task 10: Release decisions, checkpoint manifest, `release.py`

**Files:**
- Create: `versioning/release_plan.py`, `versioning/checkpoints.py`, `scripts/release.py`
- Test: `tests/versioning/test_release_plan.py`, `tests/versioning/test_checkpoints.py`

**Interfaces:**
- Consumes: Tasks 1–3 (`check`, `changed_images`, `IMAGES`, `section`, `last_release_tag`).
- Produces:
  - `versioning.checkpoints.write_manifest(ckpt_dir: Path, out: Path) -> None` (lines
    `<sha256>  <relative path>`, sorted) and `verify_manifest(ckpt_dir: Path, manifest: Path) -> list[str]`
    (mismatches/missing/extra files; empty = ok). Hash in 1 MiB chunks.
  - `versioning.release_plan.RemoteState` — dataclass: `existing_tags: dict[str, str | None]` (image key →
    revision label of the existing `:<version>` on Hub, `None` if absent), `tag_exists_local: bool`,
    `tag_exists_remote: bool`, `clean: bool`, `on_main: bool`, `up_to_date: bool`.
  - `versioning.release_plan.ImageAction` — dataclass `key, action: Literal["build", "retag", "skip"]`,
    `from_ref: str | None` (retag source = previous release's digest from `release/images.json`), `reason: str`.
  - `versioning.release_plan.plan_release(repo: Path, version: str, head_sha: str, remote: RemoteState,
    manifest: dict) -> list[ImageAction]` — raises `versioning.release_plan.ReleaseRefused(messages: list[str])`
    for: `check()` violations; dirty/not main/not up to date; git tag `v<version>` exists; any
    `existing_tags[k]` not in `(None, head_sha)`. An image whose `:<version>` exists with revision ==
    `head_sha` → `skip` (resume). Else changed since last tag (or no last tag) → `build`; else `retag`.
    `retag` with no previous manifest entry for that image → refusal.
  - `scripts/release.py` — `main(argv) -> int`; flags `--dry-run`, `--adopt`. Checkpoint
    verification always uses `api/docker/nnunet_torsion/checkpoints` (the files `docker build` bakes in; no option). Gathers `RemoteState` (`docker manifest inspect -v`
    + `docker buildx imagetools inspect --format '{{json .Image.Config.Labels}}'` for labels; `git
    ls-remote --tags origin`), prints the plan, and unless `--dry-run`: verifies checkpoints if nnUNet is
    `build`; for each action in `IMAGES` order: `build` → `docker build` with the labels and build args
    (`api` gets `MORPH_NNUNET_IMAGE`/`MORPH_TORSION_IMAGE` = this release's refs resolved so far, which is
    why nnunet and torsion come first), push `:<version>`; `retag` → `docker buildx imagetools create -t
    <repo>:<version> <from_ref>` (registry-side, no download); read back the digest; then move `:latest`
    the same way; write `release/images.json[version] = {nnunet, torsion, api, frontend, git_commit, date}`;
    `git commit -m "release: v<version>"`, `git tag -a v<version> -m <changelog section>`,
    `git push origin main v<version>`.

- [ ] **Step 1: Write failing tests** (pure, no docker): fixtures build a `git_repo` + `RemoteState`:
  - `test_plan_initial_release_builds_all` (no tags, manifest `{}`) → four `build`.
  - `test_plan_retags_unchanged_nnunet` (only a torsion-closure file changed) → nnunet `retag` from the
    manifest digest, torsion `build`, api `build` (shares the closure file), frontend `retag`.
  - `test_plan_refuses_existing_foreign_tag` (`existing_tags["torsion"]="deadbeef"`) → `ReleaseRefused`
    naming torsion.
  - `test_plan_resumes_own_tag` (`existing_tags["nnunet"]=head_sha`) → nnunet `skip`.
  - `test_plan_refuses_dirty`, `test_plan_refuses_existing_git_tag`, `test_plan_refuses_check_violation`.
  - `test_plan_adopts_unlabelled_on_initial_release`: no prior tag, `existing_tags["nnunet"]` present with
    no revision label, `adopt={"nnunet"}` → nnunet `skip`; same without `adopt` → `ReleaseRefused`; same
    with a prior release → `ReleaseRefused` even with `adopt` (so `plan_release` takes
    `adopt: frozenset[str] = frozenset()`, and `RemoteState.existing_tags` distinguishes "absent" from
    "present, unlabelled" — use the sentinel string `"<unlabelled>"` for the latter).
  - `test_checkpoint_manifest_roundtrip`: write, verify → `[]`; modify one byte → one mismatch; add a file →
    one "extra".
- [ ] **Step 2: Run** `pytest tests/versioning/test_release_plan.py tests/versioning/test_checkpoints.py -v` — FAIL.
- [ ] **Step 3: Implement** `release_plan.py`, `checkpoints.py`, then `scripts/release.py` (thin; all
  decisions in `plan_release`).
- [ ] **Step 4: Run** — PASS. Then `python scripts/release.py --dry-run` on the real repo: it must print
  four `build` actions for `1.0.0` (no tags yet) without touching docker state beyond reads.
- [ ] **Step 5: Commit** `feat(versioning): release planner, checkpoint manifest, release.py`.

### Task 11: Bootstrap 1.0.0 (run by the user — pushes to Docker Hub and origin)

This task is a checklist the implementer **prepares and hands to the user**; it is not executed by an agent.

- **Note (decided during review):** only **nnunet** is adopted for 1.0.0 (`--adopt nnunet`). **torsion must be
  BUILT**: its closure changed since the `2d1fc91` image (lazy cartilage re-exports and the head-line
  antetorsion commits). `checkpoints.sha256` must be committed **BEFORE** the `v1.0.0` tag: it is a gated
  nnunet path, so committing it later would force a 79.5 GB nnunet rebuild for 1.0.1.
  Verification always uses `api/docker/nnunet_torsion/checkpoints` (`--checkpoints-dir` was removed).
- [ ] **Step 1:** Generate `api/docker/nnunet_torsion/checkpoints.sha256` with `write_manifest`
  (~32 GB hashed); commit it (`chore: checkpoint manifest`).
- [ ] **Step 2: Decide torsion 1.0.0 = rebuild or reuse.** Hash `/app/morphometry` in the local
  `swestfechtel/torsion:latest` (`docker run --rm --entrypoint sh <img> -c "cd /app/morphometry && find . -name '*.py' | sort | xargs sha256sum"`)
  and compare with HEAD's torsion-closure files. Two commits (`4f4b64c`, `004049a`) touched
  `morphometry/` since that image's build at `2d1fc91`. If any closure file differs, 1.0.0 is a fresh
  build (the default plan); otherwise it may be retagged. Record the result in `CHANGELOG.md` 1.0.0.
- [ ] **Step 3:** For nnUNet, the plan's `build` for 1.0.0 is replaced by **pushing the existing local image**
  (`70229af9…`, user's decision — it is the one the API has been running): `docker tag
  swestfechtel/nnunet_torsion:latest swestfechtel/nnunet_torsion:1.0.0 && docker push …:1.0.0`. Then run
  `release.py`, which sees nnunet's `:1.0.0` — but its revision label is missing, so it would refuse:
  `release.py` gets a `--adopt nnunet` flag for 1.0.0 only (accept an existing `:<version>` without a
  revision label when no prior release exists; test `test_plan_adopts_unlabelled_on_initial_release` in
  Task 10).
- [ ] **Step 4:** User runs `python scripts/release.py --dry-run`, reviews, then `python scripts/release.py --adopt nnunet`.
- [ ] **Step 5:** Verify: `docker manifest inspect` for each `:1.0.0`; `release/images.json` has four
  digests; `git tag` shows `v1.0.0`; CI green on the release commit; `deploy/compose.yml` with
  `MORPH_VERSION=1.0.0` comes up and `/version` reports `1.0.0` with the manifest digests.

### Task 12: Documentation

**Files:**
- Modify: `CLAUDE.md` ("Docker images": four images, release flow replaces `pre-<sha>`; "Running the
  API/UI": compose path; "Tests": `tests/versioning`), `README.md`, `README_WEBAPP.md` (`/version`,
  provenance field, compose deployment, new settings `MORPH_API_JOB_TMP_DIR`, `MORPH_API_SELF_IMAGE`),
  `api/README.md`, `frontend/README.md` (`MODEL_API_INTERNAL`), `api/.env.example`.
- Create: `docs/releasing.md` — the release procedure: bump rules, when to bump (in the change's commit),
  changelog format, `check_version.py`, hook install, `release.py` flags and refusals, resume after an
  interrupted release, rollback via `MORPH_VERSION`.

- [ ] **Step 1:** Write the docs. Keep `CLAUDE.md` additions to a few lines plus a link to `docs/releasing.md`.
- [ ] **Step 2: Verify** `pytest -m "not needs_ct" -q` and `pytest api/tests tests/versioning -q` green;
  `python scripts/check_version.py` ok.
- [ ] **Step 3: Commit** `docs: versioning, releases and compose deployment`.

---

## Addendum: finding results by version (spec Addendum)

### Task 13: Version search API

**Files:**
- Modify: `api/release_manifest.py`, `api/db/repository.py` (`list_examinations`), `api/routers/examinations.py`,
  `api/routers/version.py` (add `/releases`), `api/schemas/examination.py`, `api/serializers.py`
- Test: `api/tests/test_version_search.py`

**Interfaces:**
- Consumes: Task 5 provenance shape (`provenance[stage]["digest"]`, `["released"]`); Task 6 `load_manifest`, `_parse`/version ordering.
- Produces:
  - `api.release_manifest.STAGE_IMAGE = {"segmentation": "nnunet", "torsion": "torsion"}`.
  - `api.release_manifest.stage_digests(version: str, manifest: dict) -> dict[str, str]` — stage → image ref of
    that release; raises `KeyError(version)` when the version is not in the manifest.
  - `api.release_manifest.versions_for_digest(stage: str, digest: str | None, manifest: dict) -> list[str]` —
    ascending release versions whose image for that stage equals `digest`; `[]` for `None`.
  - `api.db.repository.list_examinations(session, *, version_filter: VersionFilter | None = None) -> list[Examination]`
    where `VersionFilter` (dataclass in `api/db/repository.py`): `stages: tuple[str, ...]` (one or both stage
    keys, OR-combined), `mode: Literal["digest", "unreleased", "none"]`, `digests: dict[str, str]` (mode
    `digest` only). Implemented in SQL with `func.json_extract(Examination.provenance, "$.<stage>.digest")`
    / `"$.<stage>.released"` — not by loading every row into Python.
  - `GET /examinations?version=&stage=` — `stage: Literal["any", "torsion", "segmentation"] = "any"`;
    `version` omitted → unfiltered (current behaviour); unknown version → `NotFoundError` (404) whose message
    lists the known versions.
  - `GET /releases` → `{"releases": [{"version": str, "date": str | None, "images": {"nnunet", "torsion",
    "api", "frontend"}}]}`, newest first, no auth.
  - `ExaminationSummary.image_versions: dict[str, list[str]] = {}`.

- [ ] **Step 1: Write failing tests** (seed 5 examinations with provenance records and a 2-release manifest
  where nnunet is identical in 1.0.0 and 1.1.0 and torsion differs; patch `MANIFEST_PATH`):
  - `test_filter_torsion_by_version`: `?version=1.1.0&stage=torsion` → only exams whose torsion digest is 1.1.0's.
  - `test_filter_shared_nnunet_digest`: `?version=1.0.0&stage=segmentation` also returns exams recorded under 1.1.0 (same digest).
  - `test_filter_any_stage` is the OR of both.
  - `test_filter_unreleased` and `test_filter_none` (null provenance and missing stage key both count as none).
  - `test_unknown_version_404` (message contains `1.0.0` and `1.1.0`); `test_bad_stage_422`.
  - `test_no_filter_unchanged` (all rows, same envelope).
  - `test_image_versions_in_summary`: an exam on the shared nnunet digest → `image_versions["segmentation"] == ["1.0.0", "1.1.0"]`.
  - `test_releases_endpoint` newest first, no auth header with `auth_required=True`.
- [ ] **Step 2: Run** `./venv/bin/python -m pytest api/tests/test_version_search.py -v` — FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** `./venv/bin/python -m pytest api/tests -q` — PASS.
- [ ] **Step 5: Commit** `feat(api): find examinations by release version`.

### Task 14: Version search UI

**Files:**
- Modify: `frontend/app/examinations/page.tsx`, `frontend/app/components/list-component.tsx` (row shows versions),
  `frontend/app/types.ts`, `frontend/messages/en.json`, `frontend/messages/de.json`
- Create: `frontend/app/components/version-filter.tsx` (client component)

**Interfaces:**
- Consumes: Task 13 `GET /releases`, `GET /examinations?version=&stage=`, `image_versions` on each summary.
- Produces: the examinations page reads `searchParams.version` / `searchParams.stage` and forwards them to the
  API; `VersionFilter` renders a version `<select>` (options: all; each release newest first; `unreleased`;
  `none` labelled "before versioning") and a stage `<select>` (any/torsion/segmentation), and on change
  navigates with `router.push` to `/examinations?version=…&stage=…` (dropping empty params). Each list row
  shows `torsion: <versions>` / `segmentation: <versions>` compactly (a range `1.0.0–1.1.0` when more than
  one, `–` when empty). An unknown-version 404 from the API shows the API's message instead of crashing.
  Follow the existing Flowbite/Tailwind styling of `filter-component.tsx` and the i18n pattern
  (`useTranslations("examinations")`).

- [ ] **Step 1:** Implement; add the German and English strings.
- [ ] **Step 2: Verify** `cd frontend && npm run lint && npm run build` pass; run the dev server against an API
  (or a stub serving `/releases` and `/examinations`) and check: the dropdown lists releases, selecting one
  changes the URL and the list, a reload keeps the filter, the version column renders, unknown version shows
  the message. Record what was checked.
- [ ] **Step 3: Commit** `feat(frontend): filter examinations by release version`.
