"""File-based image store.

Large images are kept as ``.nii.gz`` files under ``STORAGE_DIR/{examination_id}/``
(never in the DB or pickled into RAM). The DB row records the relative paths
returned here. Encoded slice PNGs are cached on disk and base64-read on demand,
so the API never re-renders images in the request path.

Layout per examination:
    {id}/source/{original,transformed,hip,knee,ankle}.nii.gz
    {id}/masks/{hip,knee,ankle}.nii.gz
    {id}/encoded/image_000.png … seg_000.png …
Incoming Orthanc instances are staged under STORAGE_DIR/_incoming/{accession}/.
"""
import base64
import shutil
from pathlib import Path

from morphometry.image_io import Image, Segmentation


class Store:
    """Filesystem-backed image store rooted at ``storage_dir``."""

    def __init__(self, storage_dir: Path):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    # --- paths ---------------------------------------------------------------
    def examination_dir(self, examination_id: str) -> Path:
        return self.storage_dir / examination_id

    def abspath(self, rel_path: str) -> Path:
        return self.storage_dir / rel_path

    # --- volumes / masks -----------------------------------------------------
    def save_volume(self, examination_id: str, kind: str, image: Image, *, subdir: str = "source") -> str:
        """Save an Image/Segmentation as .nii.gz; return its storage-relative path."""
        target_dir = self.examination_dir(examination_id) / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"{kind}.nii.gz"
        image.save_image(str(path))
        return str(path.relative_to(self.storage_dir))

    def save_mask(self, examination_id: str, kind: str, segmentation: Segmentation) -> str:
        return self.save_volume(examination_id, kind, segmentation, subdir="masks")

    def load_image(self, rel_path: str) -> Image:
        image = Image("nibabel")
        image.read_image(str(self.abspath(rel_path)))
        return image

    def load_segmentation(self, rel_path: str) -> Segmentation:
        segmentation = Segmentation("nibabel")
        segmentation.read_image(str(self.abspath(rel_path)))
        return segmentation

    # --- encoded slice PNGs --------------------------------------------------
    def save_encoded(self, examination_id: str, images_b64: list[str], seg_b64: list[str]) -> dict:
        """Decode base64 slice PNGs to disk; return {'image': [...], 'segmentation': [...]} rel paths."""
        target_dir = self.examination_dir(examination_id) / "encoded"
        target_dir.mkdir(parents=True, exist_ok=True)

        def _write(prefix: str, items: list[str]) -> list[str]:
            paths = []
            for i, b64 in enumerate(items):
                path = target_dir / f"{prefix}_{i:04d}.png"
                path.write_bytes(base64.b64decode(b64))
                paths.append(str(path.relative_to(self.storage_dir)))
            return paths

        return {"image": _write("image", images_b64), "segmentation": _write("seg", seg_b64)}

    def load_encoded_b64(self, rel_paths: list[str]) -> list[str]:
        """Read PNG files and return them base64-encoded (UI contract)."""
        return [base64.b64encode(self.abspath(p).read_bytes()).decode("ascii") for p in rel_paths]

    # --- lifecycle -----------------------------------------------------------
    def delete_examination(self, examination_id: str) -> None:
        shutil.rmtree(self.examination_dir(examination_id), ignore_errors=True)

    # --- orthanc incoming staging -------------------------------------------
    def incoming_dir(self, accession: str) -> Path:
        return self.storage_dir / "_incoming" / accession

    def stage_incoming(self, accession: str, instance_uid: str, data: bytes) -> Path:
        """Write a received DICOM instance to the per-accession staging dir (idempotent by UID)."""
        directory = self.incoming_dir(accession)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{instance_uid}.dcm"
        path.write_bytes(data)
        return path

    def incoming_files(self, accession: str) -> list[Path]:
        directory = self.incoming_dir(accession)
        return sorted(directory.glob("*.dcm")) if directory.exists() else []

    def clear_incoming(self, accession: str) -> None:
        shutil.rmtree(self.incoming_dir(accession), ignore_errors=True)

    def pending_accessions(self) -> list[str]:
        incoming = self.storage_dir / "_incoming"
        return [p.name for p in incoming.iterdir() if p.is_dir()] if incoming.exists() else []
