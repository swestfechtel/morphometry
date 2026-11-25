# API image (swestfechtel/morphometry_api): FastAPI service + RQ worker (same image, the worker
# overrides the command with `python -m api.tasks.worker`).
# Build from the REPO ROOT:  docker build -f api/Dockerfile -t morphometry_api:dev .
#
# Base pinned to the venv's interpreter, exactly as morphometry/Dockerfile (see the comments there).
FROM python:3.10.16-bookworm

WORKDIR /app

# libgl1 / libxrender1: importing the measurements pulls in pyvista -> vtk -> libGL (nothing
# renders). Same reasoning as the torsion image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# The worker launches the nnunet_torsion and torsion images with `docker run`, so the image needs
# a docker CLI talking to the host socket (mounted by compose). Only the static client binary is
# installed (docker.io from apt is far heavier); the version matches the development host's client.
ARG DOCKER_CLI_VERSION=29.9.1
RUN curl -fsSL "https://download.docker.com/linux/static/stable/x86_64/docker-${DOCKER_CLI_VERSION}.tgz" \
        | tar -xz -C /usr/local/bin --strip-components=1 docker/docker \
    && docker --version

# Fully pinned, complete-closure dependencies (see api/requirements.txt): --no-deps so a missing
# package fails the build rather than resolving unpinned; `pip check` verifies the closure.
# Copied before the code so editing code does not invalidate the dependency layer.
COPY api/requirements.txt /app/api/requirements.txt
RUN pip3 install --upgrade pip \
    && pip3 install --no-cache-dir --no-deps -r /app/api/requirements.txt \
    && pip3 check

COPY api /app/api
COPY morphometry /app/morphometry
COPY VERSION /app/VERSION

# Release provenance and the model images this release launches. Docker images carry no .git, so
# api/version.py reads these from the environment. The model-image tags are passed by the release
# script so the API launches exactly the images of its own release.
ARG MORPH_VERSION=0.0.0-dev
ARG MORPH_GIT_COMMIT=unknown
ARG MORPH_NNUNET_IMAGE=swestfechtel/nnunet_torsion:latest
ARG MORPH_TORSION_IMAGE=swestfechtel/torsion:latest
ENV MORPH_VERSION=${MORPH_VERSION} \
    MORPH_GIT_COMMIT=${MORPH_GIT_COMMIT} \
    MORPH_API_NNUNET_IMAGE=${MORPH_NNUNET_IMAGE} \
    MORPH_API_TORSION_IMAGE=${MORPH_TORSION_IMAGE}

EXPOSE 8000

# Runs as root: access to the docker socket depends on the host's docker gid, which compose
# handles with `group_add`; a non-root user is deliberately not created here.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
