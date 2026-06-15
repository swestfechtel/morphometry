"""Thin wrappers around the two model containers.

Centralizes the ``docker run`` invocations (previously inline in
model_controller.py) so the image tags / shm-size come from settings and the
blocking subprocess + return-code handling lives in one place. This runs in the
RQ worker process, where blocking is fine.
"""
import logging
import subprocess

from api.settings import Settings

logger = logging.getLogger("api")


def _run(cmd: list[str], timeout: int | None) -> str:
    """Run a docker command, raising RuntimeError on non-zero exit; return stdout text."""
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    log = proc.stdout.decode("utf-8", errors="replace") if proc.stdout else ""
    if proc.returncode != 0:
        logger.error("Container %s exited with code %s. Logs:\n%s", cmd[-1], proc.returncode, log)
        raise RuntimeError(f"Container {cmd[-1]} failed with exit code {proc.returncode}.")
    logger.debug("Container %s exited 0.", cmd[-1])
    return log


def run_segmentation_container(mount_dir: str, settings: Settings) -> None:
    """Run the GPU nnUNet segmentation container against ``mount_dir`` (mounted at /app/mnt)."""
    _run([
        "docker", "run", "--rm",
        "--runtime=nvidia", "--gpus", "all",
        "--shm-size", settings.docker_shm_size,
        "--group-add", "root",
        "-v", f"{mount_dir}:/app/mnt:rw,Z",
        settings.nnunet_image,
    ], timeout=settings.job_timeout)


def run_torsion_container(mount_dir: str, settings: Settings) -> None:
    """Run the CPU torsion container against ``mount_dir`` (mounted at /app/temp)."""
    _run([
        "docker", "run", "--rm",
        "--shm-size", settings.docker_shm_size,
        "-u", "root",
        "-v", f"{mount_dir}:/app/temp:rw,z",
        settings.torsion_image,
    ], timeout=settings.job_timeout)
