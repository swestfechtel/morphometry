# versioning

The release rules of the web app, as code. Standard library only, so CI runs it without installing
anything.

## The model in one paragraph

All four docker images (`nnunet_torsion`, `torsion`, `morphometry_api`, `morphometry_frontend`) share one
semantic version, held in the repo-root `VERSION`, and each version has a `CHANGELOG.md` section with lines
prefixed `nnunet:`, `torsion:`, `api:` or `frontend:`. Each image has a set of **gated files**: the files
that, if they change, change what that image runs. A commit that changes a gated file must bump
`VERSION` and add the changelog section. A released `:<version>` tag on Docker Hub is never
overwritten. `release/images.json` records each release's image digests, and every examination records
the digest that produced each stage, so results can be searched by version.

## What lives where

| module | does |
|---|---|
| `semver.py` | parse `MAJOR.MINOR.PATCH`, find the last `v*` release tag |
| `changelog.py` | extract one version's section from `CHANGELOG.md` |
| `closure.py` | static import closure: every `morphometry` file reachable from an entry point (`ast`, fails loudly on imports it can't resolve) |
| `images.py` | the four images, their gated files (`IMAGES`), and `changed_images(base, head)` |
| `check.py` | the rules enforced by CI and the pre-push hook (below) |
| `release_plan.py` | per-image `build` / `retag` / `skip` decision for a release, or the list of reasons to refuse |
| `checkpoints.py` | SHA-256 manifest of the gitignored nnUNet checkpoints |
| `_git.py` | git calls with the hook's `GIT_DIR` variables stripped |

The command-line entry points live in `scripts/`:

```bash
python scripts/check_version.py                  # the rules below; CI and the pre-push hook run this
bash scripts/install_hooks.sh                    # once per clone: install the pre-push hook
python scripts/release.py --dry-run              # what a release would do, changes nothing
python scripts/release.py                        # build/retag, push, record, tag v<VERSION>, push
python -m versioning.checkpoints write|verify api/docker/nnunet_torsion/checkpoints \
       api/docker/nnunet_torsion/checkpoints.sha256
```

## The rules (`check.py`)

1. `VERSION` is a plain `MAJOR.MINOR.PATCH`.
2. If any image's gated files changed since the last release tag, `VERSION` is higher than that tag
   and `CHANGELOG.md` has a section for it.
3. A changed torsion golden (measured values moved) needs at least a MINOR bump.
4. Every pin in `morphometry/requirements.txt` and `api/requirements.txt` equals the root
   `requirements.txt` pin. The nnUNet image's `requirements.txt` is deliberately exempt: it is the
   frozen package set of the image that produced the pre-1.0.0 segmentations.

## Gated files, per image

- **nnunet**: everything under `api/docker/nnunet_torsion/`, including `checkpoints.sha256`.
- **torsion**: the import closure of `morphometry/docker/compute_torsion.py`, plus its Dockerfile,
  `requirements.txt` and `.dockerignore`.
- **api**: `api/**` (minus tests, data, logs and `api/docker/`), the `morphometry` closure of the API,
  `api/Dockerfile`, `api/requirements.txt`, root `.dockerignore`.
- **frontend**: `frontend/**`.

Markdown files don't count for api/frontend, except under `frontend/public/` (served to users).
**Granularity is the file**, whatever the import style: an image that imports one function from a
module is gated on the whole module. To keep unrelated edits from forcing releases, split the file or
re-export lazily (see "Conventions worth preserving" in the root `CLAUDE.md`).

## More

- Full procedure, bump rules, resume and rollback: [`docs/releasing.md`](../docs/releasing.md).
- Production stack: [`deploy/README.md`](../deploy/README.md).
- Design and its reasoning: [`docs/superpowers/specs/2026-11-25-api-versioning-design.md`](../docs/superpowers/specs/2026-11-25-api-versioning-design.md).
- Tests: `tests/versioning/`.
