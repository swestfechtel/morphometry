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
import os
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

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

    def save_combined_mask(self, examination_id: str, combined_array, affine) -> str:
        """Save a combined label volume (built from the region masks) as masks/combined.nii.gz.

        :param examination_id: Owning examination id.
        :param combined_array: The concatenated integer label volume (see
            :func:`api.domain.masks.combine_region_masks`).
        :param affine: The transformed volume's affine, so the mask shares its grid.
        :return: The storage-relative path of the written file.
        """
        seg = Segmentation.from_nibabel(nib.Nifti1Image(combined_array.astype(np.int16), affine))
        return self.save_mask(examination_id, "combined", seg)

    def ensure_combined_mask(self, examination_id: str, mask_paths: dict, transformed_rel: str) -> str:
        """Return the combined-mask rel path, building it from region masks if absent.

        New examinations get the combined mask written by the worker. For older
        examinations (segmented before the combined mask existed) this lazily
        builds and caches it from the per-region masks, aligned to the transformed
        volume. Built without importing matplotlib (see :mod:`api.domain.masks`).

        :param examination_id: Owning examination id.
        :param mask_paths: The examination's ``mask_paths`` dict (needs hip/knee/ankle).
        :param transformed_rel: Rel path of the transformed source volume (for the affine).
        :return: Storage-relative path of masks/combined.nii.gz.
        """
        existing = (mask_paths or {}).get("combined")
        if existing and self.abspath(existing).exists():
            return existing
        from api.domain.masks import combine_region_masks
        hip = self.load_segmentation(mask_paths["hip"])
        knee = self.load_segmentation(mask_paths["knee"])
        ankle = self.load_segmentation(mask_paths["ankle"])
        affine = self.load_image(transformed_rel).affine
        return self.save_combined_mask(examination_id, combine_region_masks(hip, knee, ankle), affine)

    # GPU-renderable dtypes for the Cornerstone/WebGL viewer. float64 (and int64)
    # have no WebGL texture type, so vtk.js textures them as an empty (black)
    # volume — with no error. Everything narrower than 32-bit floats is fine.
    _DISPLAY_SAFE_DTYPES = frozenset(
        {"uint8", "int8", "uint16", "int16", "int32", "uint32", "float32"})

    def ensure_display_volume(self, examination_id: str, transformed_rel: str) -> str:
        """Return a rel path to a GPU-renderable copy of the transformed volume.

        The stored ``transformed.nii.gz`` is typically ``float64`` (a byproduct of
        resampling), which WebGL cannot texture, so Cornerstone renders it black.
        This casts the volume to ``float32`` — same shape and affine, so landmark
        voxel indices and the aligned labelmap are unaffected — and caches it as
        ``source/display.nii.gz``. Volumes already in a supported dtype are served
        as-is (no copy).

        :param examination_id: Owning examination id.
        :param transformed_rel: Rel path of the stored transformed source volume.
        :return: Storage-relative path of a GPU-renderable volume (the original
            when already safe, else the cached float32 copy).
        """
        img = nib.load(str(self.abspath(transformed_rel)))
        if str(img.get_data_dtype()) in self._DISPLAY_SAFE_DTYPES:
            return transformed_rel
        target_dir = self.examination_dir(examination_id) / "source"
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / "display.nii.gz"
        rel = str(path.relative_to(self.storage_dir))
        if path.exists():
            return rel
        # np.asanyarray(dataobj) applies scl_slope/inter, matching what the JS
        # NIfTI reader and nibabel both see; affine (spacing/orientation) is kept.
        data = np.asanyarray(img.dataobj).astype(np.float32)
        # Write atomically: the viewer's loader fetches the volume URL twice
        # concurrently on first load, so a reader must never see a half-written
        # file. Write to a unique temp path, then os.replace (atomic on POSIX).
        # Temp name must keep the .nii.gz suffix so nibabel infers the format.
        tmp = target_dir / f".display.{os.getpid()}.nii.gz"
        nib.save(nib.Nifti1Image(data, img.affine), str(tmp))
        os.replace(tmp, path)
        return rel

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
