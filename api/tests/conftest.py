"""Shared pytest fixtures for the API tests.

Provides isolated settings backed by temp directories and helpers to fake the
docker subprocess, so the HTTP + orchestration logic can be tested without a GPU,
Redis, or docker. App/client/db fixtures are added as those layers land.
"""
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from api.settings import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """A Settings instance pointed entirely at temp directories."""
    return Settings(
        storage_dir=tmp_path / "data",
        log_dir=tmp_path / "logs",
        redis_url="redis://localhost:6379/15",
        api_keys=["test-key"],
        cors_allow_origins=["http://testclient"],
    )


@pytest.fixture
def runtime(tmp_path: Path, monkeypatch):
    """Point the cached runtime (settings/engine/store) at temp dirs via env vars."""
    monkeypatch.setenv("MORPH_API_STORAGE_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MORPH_API_DATABASE_URL", f"sqlite:///{tmp_path / 'api.db'}")
    monkeypatch.setenv("MORPH_API_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("MORPH_API_API_KEYS", "test-key")
    monkeypatch.setenv("MORPH_API_CORS_ALLOW_ORIGINS", "http://testclient")
    import api.runtime as rt
    rt.reset()
    yield rt
    rt.reset()


@pytest.fixture
def client(runtime):
    """A TestClient on a fresh app wired to temp deps + an eager (in-process) queue."""
    from fastapi.testclient import TestClient

    from api.deps import get_queue
    from api.main import create_app
    from api.tasks.queue import EagerQueue

    app = create_app()
    app.dependency_overrides[get_queue] = lambda: EagerQueue()
    with TestClient(app) as test_client:
        test_client.headers.update({"X-API-Key": "test-key"})
        yield test_client


@pytest.fixture
def fake_docker_run():
    """Factory for a ``subprocess.run`` replacement that fakes the model containers.

    The returned callable inspects the docker command's mount target to decide
    whether it's the segmentation container (writes tiny mask .nii.gz) or the
    torsion container (writes results/landmarks/errors json), and returns a
    CompletedProcess with the configured return code.
    """
    import subprocess

    def _factory(returncode: int = 0, malformed: bool = False):
        def _run(cmd, *args, **kwargs):
            # locate the host path of the bind mount (-v host:container)
            mount = next((c for c in cmd if isinstance(c, str) and ":/app/" in c), None)
            host_dir = Path(mount.split(":")[0]) if mount else None
            if returncode == 0 and host_dir is not None:
                if "/app/mnt" in mount:  # segmentation container
                    for region in ("hip", "knee", "ankle"):
                        out = host_dir / region / "output"
                        out.mkdir(parents=True, exist_ok=True)
                        img = nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.int16), np.eye(4))
                        nib.save(img, out / f"{region}.nii.gz")
                else:  # torsion container (/app/temp)
                    if malformed:
                        (host_dir / "results.json").write_text("{ not json")
                    else:
                        (host_dir / "results.json").write_text(json.dumps({
                            "femoral_torsion_left": 1.0, "femoral_torsion_right": 2.0,
                            "femoral_torsion_left_murphy": 3.0, "femoral_torsion_right_murphy": 4.0,
                            "tibial_torsion_left": 5.0, "tibial_torsion_right": 6.0,
                        }))
                        (host_dir / "landmarks.json").write_text(json.dumps({"femur": {}, "tibia": {}}))
                        (host_dir / "errors.json").write_text(json.dumps({"errors": []}))
            return subprocess.CompletedProcess(cmd, returncode, stdout=b"fake container log")
        return _run

    return _factory
