# Production deployment

One compose file runs the whole stack: `redis`, the FastAPI `api`, the RQ `worker` and the Next.js
`frontend`. All four are versioned together by `MORPH_VERSION`; the api image also pins the exact
nnUNet / torsion model images of its release (`release/images.json`, baked in at build time).

## Prerequisites

- Docker with the compose plugin. The worker launches the model containers through the host's
  docker socket; the nnUNet step needs the NVIDIA runtime (`--gpus all`).
- A host data directory (`MORPH_DATA_DIR`, absolute path). It is bind-mounted at the **same path**
  inside the api and worker containers, because the worker passes job directories to `docker run -v`
  and the host daemon resolves them, not the container.

## Start

```bash
cp deploy/.env.example deploy/.env      # set MORPH_VERSION, MORPH_DATA_DIR, DOCKER_GID, secrets
mkdir -p "$MORPH_DATA_DIR"
docker compose -f deploy/compose.yml --env-file deploy/.env up -d
curl localhost:8000/version             # the running release and its model images
```

`DOCKER_GID` is `getent group docker | cut -d: -f3`. The UI is on port 3000, the API on 8000
(`MORPH_API_PORT` / `MORPH_FRONTEND_PORT` change the host ports). Create a user with
`docker compose -f deploy/compose.yml --env-file deploy/.env exec api python -m api.users create <name>`.

## Upgrade and roll back

Both are the same operation: change `MORPH_VERSION` in `deploy/.env` and re-run

```bash
docker compose -f deploy/compose.yml --env-file deploy/.env up -d
```

Compose pulls the new tags and recreates the containers; data under `MORPH_DATA_DIR` and the redis
volume persist. Check `GET /version` afterwards. Database schema changes are applied by the api on
start, but a rollback does not undo them: back up `MORPH_DATA_DIR` before upgrading.
