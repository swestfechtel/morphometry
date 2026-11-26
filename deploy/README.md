# Production deployment

One compose file runs the whole stack: `redis`, the FastAPI `api`, the RQ `worker` and the Next.js
`frontend`. All four are versioned together by `MORPH_VERSION`; the api image also pins the exact
nnUNet / torsion model images of its release (baked in at build time).

The api image does **not** contain `release/images.json` (an image can never carry its own manifest
entry: the entry is written after the image is pushed). Compose bind-mounts this checkout's
`release/images.json` read-only into the api and worker (`MORPH_API_RELEASE_MANIFEST`); it drives
`GET /version`'s `images_from`, `GET /releases` and the version search. **The deploy checkout must be at
or after the `release: v<MORPH_VERSION>` commit**, so `git pull` before switching `MORPH_VERSION`. A
missing file makes `up` fail; an outdated one only makes `/version` report `<version>-dev` and leaves the
new version out of `/releases` and the search.

## Prerequisites

- Docker with the compose plugin. The worker launches the model containers through the host's
  docker socket; the nnUNet step needs the NVIDIA runtime (`--gpus all`).
- A host data directory (`MORPH_DATA_DIR`, absolute path). It is bind-mounted at the **same path**
  inside the api and worker containers, because the worker passes job directories to `docker run -v`
  and the host daemon resolves them, not the container.

## Start

```bash
git pull                                # release/images.json must contain MORPH_VERSION
cp deploy/.env.example deploy/.env      # set MORPH_VERSION, MORPH_DATA_DIR, DOCKER_GID, secrets
mkdir -p "$MORPH_DATA_DIR"
docker compose -f deploy/compose.yml --env-file deploy/.env up -d
curl localhost:8000/version             # the running release and its model images
```

`DOCKER_GID` is `getent group docker | cut -d: -f3`. The UI is on port 3000, the API on 8000
(`MORPH_API_PORT` / `MORPH_FRONTEND_PORT` change the host ports).
**Leave `MORPH_API_PORT` alone**: the browser calls the API at `<same host>:8000` unless the frontend image was built with `NEXT_PUBLIC_MODEL_API_PORT`, so a different published port breaks the UI. Create a user with
`docker compose -f deploy/compose.yml --env-file deploy/.env exec api python -m api.users create <name>`.

## Upgrade and roll back

Both are the same operation: `git pull` this checkout (for the new `release/images.json`), change
`MORPH_VERSION` in `deploy/.env` and re-run

```bash
docker compose -f deploy/compose.yml --env-file deploy/.env up -d
```

Compose pulls the new tags and recreates the containers; data under `MORPH_DATA_DIR` and the redis
volume persist. Check `GET /version` afterwards. Database schema changes are applied by the api on
start, but a rollback does not undo them: back up `MORPH_DATA_DIR` before upgrading.
