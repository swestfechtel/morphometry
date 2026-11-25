# Releasing

One shared semantic version (`VERSION`, `MAJOR.MINOR.PATCH`) names a release of all four docker
images: `nnunet` (`swestfechtel/nnunet_torsion`), `torsion` (`swestfechtel/torsion`), `api`
(`swestfechtel/morphometry_api`) and `frontend` (`swestfechtel/morphometry_frontend`). Every
examination records which image ran each stage (its provenance), so every number can be traced to the
code that produced it, and that code stays pullable. Design and rationale:
[`docs/superpowers/specs/2026-11-25-api-versioning-design.md`](superpowers/specs/2026-11-25-api-versioning-design.md).

The rules are enforced, not remembered: `scripts/check_version.py` (CI and a pre-push hook) flags a
change that lacks a bump, and `scripts/release.py` is the **only** path that builds or pushes images.

## Bump rules

| Bump | When |
|---|---|
| **MAJOR** | A breaking HTTP/JSON contract change: a removed or renamed field or endpoint, a changed sign convention. |
| **MINOR** | Anything that moves measured values (a *do-not-pool* event), new endpoints or fields, new frontend features. |
| **PATCH** | Value-identical fixes and refactors, frontend fixes. |

**The golden rule (machine-checked).** If a torsion-path golden file (the allow-list
`GOLDEN_ALLOWLIST` in `versioning/check.py`) changed since the last release tag, the bump must be at
least MINOR: a golden re-capture means values moved. When in doubt about MAJOR vs MINOR, ask whether an
existing client would break; when in doubt about MINOR vs PATCH, ask whether any stored number would
differ on re-processing.

## When to bump

**In the commit that makes the change**, not at release time. A change counts when it touches an
image's gated paths:

- **nnunet**: `api/docker/nnunet_torsion/**` (except the untracked `checkpoints/`), including
  `checkpoints.sha256`.
- **torsion**: the import closure of `morphometry/docker/compute_torsion.py`, plus that Dockerfile,
  `.dockerignore` and `morphometry/requirements.txt`. Research code the entry point never reaches
  (cartilage, deformation maps; the lazy re-exports in `morphometry/measurements/__init__.py` keep
  cartilage out) does not count.
- **api**: `api/**` except `api/docker/`, `api/tests*`, `api/data/`, `api/logs/`, notebooks; the
  `morphometry` closure of the `.py` files among them; `api/Dockerfile`, `api/requirements.txt`.
- **frontend**: `frontend/**` except `node_modules/`, `.next/`, `*.tsbuildinfo`, `.env*`.

Bump `VERSION` once per release cycle (the first gated commit after a release raises it; later ones
may reuse the pending version, raising it further only if they need a bigger bump) and add the
matching `CHANGELOG.md` section. `VERSION` may sit ahead of `release/images.json`; the API then reports
e.g. `1.4.0-dev, images from 1.3.0`.

## Changelog

`CHANGELOG.md`, Keep-a-Changelog style, newest first. One section per version, entries prefixed by the
image they affect:

```
## [1.3.0] - 2026-12-01

- torsion: Murphy neck base now ... (moves values; do not pool with 1.2.x).
- api: new `GET /releases`.
- frontend: version filter on the examinations list.
```

The section is required for the current `VERSION`, is embedded in each built image as the
`morphometry.changelog` label, and is the body of the annotated git tag `v<version>`. Notes that
belong to no image go under a `### ` subheading (as the 1.0.0 pre-versioning history does).

## One-time setup: the pre-push hook

```bash
scripts/install_hooks.sh
```

Installs `pre-push` running `scripts/check_version.py`. The hook lives in the hooks directory shared by
all worktrees (`git rev-parse --git-path hooks`), so run it once per clone, not per worktree. It never
overwrites a different existing hook. The same check runs in GitHub Actions
(`.github/workflows/version-check.yml`); because pushes go straight to `main`, CI flags a violation
rather than preventing it.

## `scripts/check_version.py`

```bash
python scripts/check_version.py        # "version check ok (<VERSION>)" or a list of violations, exit 1
```

Pure git and file reads, no docker. It checks: `VERSION` is plain semver; if any image's gated paths
changed since the last `v*` tag, `VERSION` is above that tag and `CHANGELOG.md` has its section; the
golden rule; and that every pin in `morphometry/requirements.txt` and `api/requirements.txt` equals the
root `requirements.txt` pin. An import the closure walker cannot resolve fails loudly instead of
shrinking the closure.

## `scripts/release.py`

Run from a clean `main` that equals `origin/main`, with docker logged in to Docker Hub.

```bash
python scripts/release.py --dry-run    # gather state, print per-image build / retag / skip, change nothing
python scripts/release.py              # execute the plan
python scripts/release.py --adopt nnunet   # 1.0.0 only, see the bootstrap
```

**It refuses** (exit 1, with the reasons listed) unless: the work tree is clean, the branch is `main`
and equals `origin/main`; `check_version` passes (bump above the last tag, changelog section, golden
rule); git tag `v<VERSION>` exists neither locally nor on origin; and `:<VERSION>` on Docker Hub is
either absent or already carries *this commit's* revision label (a resume, below). A `:<VERSION>` that
exists with a different revision is never overwritten: that is the anti-goal this script exists to
prevent. Checkpoints in `api/docker/nnunet_torsion/checkpoints/` must match `checkpoints.sha256` when
nnunet is rebuilt (there is no option to verify another directory).

**The plan.** Per image: `build` if its gated paths changed since the last release tag (or there is no
tag), else `retag`: add `:<VERSION>` to the previous release's digest registry-side, no rebuild and
no upload. In particular the 79.5 GB nnunet image is only rebuilt when its own gated paths change.
Order is nnunet, torsion, api, frontend: the api image is built with this release's nnunet and torsion
digests baked in. Built images carry the OCI labels `version`, `revision` (full git sha), `created` and
`morphometry.changelog`. After pushing, the script reads the digest back, writes
`release/images.json[<version>]`, commits `release: v<version>`, tags `v<version>` and pushes `main` and
the tag; `:latest` of each image is moved last (convenience only; nothing should run `:latest`).

**Resume after an interrupted release.** Re-run the same command at the same commit. Images whose
`:<VERSION>` already carries this commit's revision label are `skip`ped (reused as pushed), the rest
continue. Do not change the commit in between: a different revision on an existing tag is a refusal.

**If the final `git push` fails** (images pushed, `release/images.json` committed and tagged locally):
fix the cause (network, a newer `origin/main`) and push by hand,
`git push origin main v<version>`. If `origin/main` moved, rebase the release commit onto it, which
only touches `release/images.json`; re-run `check_version.py`, and move the local tag to the rebased
commit **only if nothing was pushed yet**. Never re-run `release.py` after the local tag exists: it
refuses, correctly. The images are already published; only the git side needs finishing.

## Rolling back and finding results by version

- **Roll back or upgrade** a deployment: set `MORPH_VERSION` in `deploy/.env` and
  `docker compose -f deploy/compose.yml --env-file deploy/.env up -d`. Old versions stay pullable by
  `:<version>` tag and by digest. Schema additions made by a newer API are not undone: back up
  `MORPH_DATA_DIR` first. See [`deploy/README.md`](../deploy/README.md).
- **Which version produced a number?** `GET /examinations/{id}` carries `provenance` per stage
  (`segmentation`, `torsion`): version, image, digest, `released`, git commit, `dirty`, time.
- **Which examinations came from a version?** `GET /examinations?version=1.2.0&stage=any|torsion|segmentation`
  matches by the image digest actually used (so a result from another version that ran the identical,
  retagged image also matches; each list item's `image_versions` lists every version sharing the
  digest). `version=unreleased` finds local or unknown images, `version=none` finds results with no
  provenance. `GET /releases` lists versions for the UI dropdown, which drives the same query via URL
  parameters.

## Bootstrap: the 1.0.0 release (run once, by hand)

Pushes to Docker Hub and origin; slow (the nnunet upload is large). Nothing here is automated. Status of
the inputs: `VERSION` is `1.0.0` and `CHANGELOG.md` has its section already.

1. **Hook.** `scripts/install_hooks.sh`.
2. **Torsion is built, not reused.** The local `swestfechtel/torsion:latest` was built at `2d1fc91`; the
   torsion closure has changed since (lazy cartilage re-exports, head-line antetorsion), so `release.py`
   plans `build` for it. (To confirm, hash `/app/morphometry` in the old image as in
   `docs/evidence/torsion_image_rebuilds.md` and compare with HEAD; any difference is decisive.) Update
   the changelog 1.0.0 `torsion:` line if you want the build's contents described precisely. The old
   `bb619140...` image stays as the pre-versioning image.
3. **Generate the checkpoint manifest and commit it BEFORE the tag.**
   ```bash
   python -m versioning.checkpoints write api/docker/nnunet_torsion/checkpoints api/docker/nnunet_torsion/checkpoints.sha256
   git add api/docker/nnunet_torsion/checkpoints.sha256 && git commit -m "chore: checkpoint manifest"
   ```
   (~32 GB hashed.) `checkpoints.sha256` is a gated nnunet path: committing it **after** `v1.0.0` would
   count as an nnunet change and force a 79.5 GB rebuild for 1.0.1.
4. **Adopt the existing nnunet image as 1.0.0** (only nnunet is adopted). The local image `70229af9...`
   is the one the API has been running; push it as the release image:
   ```bash
   docker tag swestfechtel/nnunet_torsion:latest swestfechtel/nnunet_torsion:1.0.0
   docker push swestfechtel/nnunet_torsion:1.0.0
   ```
   It has no revision label, so `release.py` would refuse it without `--adopt nnunet`. The superseded Hub
   `:latest` stays reachable by its manifest digest (recorded in the changelog).
5. **Push `main`** (so `HEAD` equals `origin/main`) and make sure the work tree is clean.
6. **Dry run:** `python scripts/release.py --dry-run`. Expect nnunet `skip` (adopted), torsion, api and
   frontend `build`. Read the refusals if any.
7. **Release:** `python scripts/release.py --adopt nnunet`. If interrupted, re-run it (see resume).
8. **Verify:** `docker manifest inspect swestfechtel/<image>:1.0.0` for each of the four; `release/images.json`
   holds four digests; `git tag` shows `v1.0.0` and it is on origin; CI is green on the release commit;
   bring up `deploy/compose.yml` with `MORPH_VERSION=1.0.0` and check that `GET /version` reports `1.0.0`
   with the manifest digests.

After 1.0.0, `--adopt` is refused (it applies to an initial release only).
