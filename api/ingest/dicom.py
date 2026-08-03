"""Turn uploaded DICOM into stored images + an Examination row.

Ported from the old ``FileController.save_files`` / ``save_torsion_series`` /
``TorsionExamination.split_series``, but writing ``.nii.gz`` to the Store and a
lightweight row to the DB instead of pickling a god-object. These functions do
blocking image work and are meant to be called via ``run_in_threadpool`` from the
async endpoints (or directly from the worker for the Orthanc path).
"""
import logging
import random
import string
from datetime import datetime
from io import BytesIO
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
import ruptures

from morphometry.image_io import Image

from api.db import repository
from api.db.engine import session_scope
from api.db.models import Examination
from api.errors import DuplicateError, IngestError, NotFoundError
from api.runtime import get_engine, get_settings, get_store
from api.schemas.enums import ExaminationStatus, ExaminationType

logger = logging.getLogger("api")


def _random_accession() -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=10))


def _study_fields(metadata) -> dict:
    """Extract the small subset of DICOM tags the app displays/stores."""
    def _fmt(tag, in_fmt, out_fmt):
        try:
            return datetime.strptime(metadata[tag].value, in_fmt).strftime(out_fmt)
        except Exception:  # noqa: BLE001 - missing/odd tags are non-fatal
            return None

    def _val(tag):
        try:
            return str(metadata[tag].value)
        except Exception:  # noqa: BLE001
            return None

    return {
        "study_date": _fmt((0x0008, 0x0020), "%Y%m%d", "%Y-%m-%d"),
        "study_time": _fmt((0x0008, 0x0030), "%H%M%S", "%H:%M"),
        "study_description": _val((0x0008, 0x1030)),
        "patient_name": "Anonymised",
        "dicom_metadata": {
            "study_date": _fmt((0x0008, 0x0020), "%Y%m%d", "%Y-%m-%d"),
            "study_description": _val((0x0008, 0x1030)),
            "accession_number": _val((0x0008, 0x0050)),
        },
    }


def _accession(metadata) -> str:
    accession = ""
    try:
        accession = str(metadata.AccessionNumber)
    except Exception:  # noqa: BLE001
        pass
    if not accession:
        accession = _random_accession()
        logger.warning("No accession number in metadata; using generated id %s", accession)
    return accession


def _materialize(image: Image) -> Image:
    """Copy an image's data into a fresh in-memory NIfTI.

    ``dicom_to_nibabel`` returns lazy proxies backed by a temp file that ingest
    cleans up; without this, ``nib.save`` later re-reads the deleted file. Call
    while the temp file still exists (before cleanup).
    """
    return Image.from_nibabel(nib.Nifti1Image(np.ascontiguousarray(image.array), image.affine))


def _slice_footprint(arr: np.ndarray) -> np.ndarray:
    """Per-slice count of foreground (intensity >= 50) voxels along the z-axis."""
    cleaned = np.where(arr < 50, 0, arr)
    return np.array([np.count_nonzero(cleaned[:, :, z]) for z in range(arr.shape[2])])


def _orient_superoinferior(transformed: Image) -> Image:
    """Ensure the LPI z-axis truly runs superior (hip) -> inferior (ankle).

    ``transform_coordinate_system`` reorients to LPI using *only* the DICOM
    affine. A few series carry bad orientation metadata (acquired/stored
    upside-down): the affine reports LPI while the voxel data actually runs
    ankle -> knee -> hip. The affine-based transform cannot detect this, and
    ``_split_volume`` (which assumes low z = hip) then mislabels ankle as hip,
    breaking everything downstream.

    Detect it from the anatomy rather than the metadata: the proximal-thigh /
    pelvis cross-section is far larger than the ankle, so if the low-z end has a
    smaller foreground footprint than the high-z end, the volume is flipped.
    Correct it by reversing the data along z while **keeping the affine** — the
    affine still labels z as Inferior, so the repeated ``transform_coordinate_
    system`` calls on the derived masks downstream stay no-ops and the fix
    sticks. (A nibabel-consistent flip would relabel z to Superior and be undone
    by the next re-transform.)

    :param transformed: the LPI-reoriented whole-leg volume.
    :return: the same image, or a z-reversed copy if it was upside-down.
    """
    footprint = _slice_footprint(transformed.array)
    n = len(footprint)
    q = max(1, n // 4)
    superior_area = float(footprint[:q].mean())   # low z  -> expected hip (large)
    inferior_area = float(footprint[-q:].mean())  # high z -> expected ankle (small)
    if superior_area >= inferior_area:
        return transformed
    logger.warning(
        "Whole-leg volume looks upside-down (superior footprint %.0f < inferior %.0f); "
        "reversing z so hip is at low z. Check the source DICOM orientation metadata.",
        superior_area, inferior_area,
    )
    flipped = np.ascontiguousarray(transformed.array[:, :, ::-1])
    return Image.from_nibabel(nib.Nifti1Image(flipped, affine=transformed.affine))


def _split_volume(transformed: Image) -> dict[str, Image]:
    """Split a stacked whole-leg volume into hip/knee/ankle via changepoint detection."""
    arr = transformed.array
    num_pixels = _slice_footprint(arr)
    # ruptures returns [bkp1, bkp2, len(signal)] for n_bkps=2; the trailing length is ignored
    breakpoints = ruptures.KernelCPD().fit_predict(num_pixels, 2)
    knee_hip, ankle_knee = breakpoints[0], breakpoints[1]
    affine = transformed.affine
    return {
        "hip": Image.from_nibabel(nib.Nifti1Image(arr[:, :, :knee_hip], affine=affine)),
        "knee": Image.from_nibabel(nib.Nifti1Image(arr[:, :, knee_hip:ankle_knee], affine=affine)),
        "ankle": Image.from_nibabel(nib.Nifti1Image(arr[:, :, ankle_knee:], affine=affine)),
    }


def _check_duplicate(accession: str) -> None:
    settings = get_settings()
    with session_scope(get_engine()) as session:
        existing = repository.get_examination(session, accession)
    if existing is None:
        return
    if settings.on_duplicate == "replace":
        logger.warning("Replacing existing examination %s", accession)
        get_store().delete_examination(accession)
        with session_scope(get_engine()) as session:
            repository.delete_examination(session, accession)
    else:
        raise DuplicateError(f"Examination {accession} already exists")


def _as_float32(image: Image) -> Image:
    """Return ``image`` cast to float32 (same shape/affine), or unchanged if already so.

    Resampling in ``transform_coordinate_system`` leaves the transformed volume as
    float64, which WebGL cannot texture — the Cornerstone viewer would render it
    black. Persisting float32 keeps the served volume GPU-renderable (and halves its
    size) so the API needn't convert it on the fly. float32 has ample precision for
    MRI intensities.

    :param image: The volume to normalise.
    :return: A float32 :class:`~morphometry.image_io.Image` (new object only if a
        cast was needed).
    """
    if image.array.dtype == np.float32:
        return image
    return Image.from_nibabel(nib.Nifti1Image(image.array.astype(np.float32), affine=image.affine))


def _persist_torsion(accession: str, study: dict, original: Image, transformed: Image,
                     regions: dict[str, Image]) -> str:
    """Save volumes + insert the examination row; return the examination id."""
    store = get_store()
    transformed = _as_float32(transformed)  # keep the served volume GPU-renderable
    source_paths = {
        "original": store.save_volume(accession, "original", original),
        "transformed": store.save_volume(accession, "transformed", transformed),
        "hip": store.save_volume(accession, "hip", regions["hip"]),
        "knee": store.save_volume(accession, "knee", regions["knee"]),
        "ankle": store.save_volume(accession, "ankle", regions["ankle"]),
    }
    knee_offset = int(regions["hip"].shape[2])
    ankle_offset = knee_offset + int(regions["knee"].shape[2])
    with session_scope(get_engine()) as session:
        repository.upsert_examination(session, Examination(
            id=accession,
            examination_type=ExaminationType.TORSION.value,
            status=ExaminationStatus.UNPROCESSED.value,
            shape=list(transformed.shape),
            knee_offset=knee_offset,
            ankle_offset=ankle_offset,
            source_paths=source_paths,
            **study,
        ))
    return accession


def _ingest_whole_leg_dir(dicom_dir: Path, accession: str, metadata) -> str:
    """Convert one stacked whole-leg series to volumes, split it, and persist the row.

    The duplicate check and accession derivation are the caller's responsibility, so
    this can be reused both for a fresh upload (:func:`ingest_torsion_from_dir`) and to
    materialize an already-staged, user-selected series (:func:`materialize_torsion_selection`).
    """
    nib_image, tmp = Image.dicom_to_nibabel(str(dicom_dir))
    try:
        original = _materialize(Image.from_nibabel(nib_image))  # read before tmp is removed
        transformed = original.copy()
        transformed.transform_coordinate_system()
        transformed = _orient_superoinferior(transformed)  # fix upside-down DICOMs
    finally:
        tmp.cleanup()

    regions = _split_volume(transformed)
    return _persist_torsion(accession, _study_fields(metadata), original, transformed, regions)


def ingest_torsion_from_dir(dicom_dir: Path) -> str:
    """Ingest a single stacked DICOM series (UI single upload / Orthanc finalize)."""
    metadata = Image.read_dicom_metadata(str(dicom_dir))
    accession = _accession(metadata)
    _check_duplicate(accession)
    return _ingest_whole_leg_dir(dicom_dir, accession, metadata)


def _ingest_multi_dirs(hip_dir: Path, knee_dir: Path, ankle_dir: Path, accession: str, metadata) -> str:
    """Concatenate three already-split region series into a volume and persist the row.

    Duplicate check / accession derivation are the caller's responsibility (see
    :func:`_ingest_whole_leg_dir` for the rationale).
    """
    images, tmps = {}, []
    try:
        for region, directory in (("hip", hip_dir), ("knee", knee_dir), ("ankle", ankle_dir)):
            nib_image, tmp = Image.dicom_to_nibabel(str(directory))
            tmps.append(tmp)
            img = Image.from_nibabel(nib_image)
            img.transform_coordinate_system()
            images[region] = _materialize(img)  # read before tmps are removed

        shapes = [images[r].array.shape[:2] for r in ("hip", "knee", "ankle")]
        if len(set(shapes)) != 1:
            raise IngestError(f"In-plane shapes differ between series: {shapes}")

        combined = np.concatenate([images[r].array for r in ("hip", "knee", "ankle")], axis=2)
        transformed = Image.from_nibabel(nib.Nifti1Image(combined, affine=images["hip"].affine))
    finally:
        for tmp in tmps:
            tmp.cleanup()

    return _persist_torsion(accession, _study_fields(metadata), transformed.copy(), transformed, images)


def ingest_torsion_multi_from_dirs(hip_dir: Path, knee_dir: Path, ankle_dir: Path) -> str:
    """Ingest three separate DICOM series (hip/knee/ankle), already split."""
    metadata = Image.read_dicom_metadata(str(hip_dir))
    accession = _accession(metadata)
    _check_duplicate(accession)
    return _ingest_multi_dirs(hip_dir, knee_dir, ankle_dir, accession, metadata)


# --- series enumeration + previews (two-phase upload) ------------------------

def _hdr_str(ds, name: str) -> str | None:
    value = getattr(ds, name, None)
    return str(value) if value not in (None, "") else None


def _hdr_int(ds, name: str) -> int | None:
    try:
        return int(getattr(ds, name))
    except (TypeError, ValueError):
        return None


def _group_series(source_dir: Path) -> dict[str, dict]:
    """Group every parseable image-DICOM file under ``source_dir`` by SeriesInstanceUID.

    Non-DICOM files (and non-image DICOM objects such as DICOMDIR) are silently
    discarded — they either fail to parse or lack a SeriesInstanceUID / pixel
    geometry. Files within a series are sorted by InstanceNumber.

    :param source_dir: A directory tree of mixed uploaded files.
    :return: ``{series_uid: {"files_sorted": [Path, ...], "header": Dataset}}``.
    """
    groups: dict[str, dict] = {}
    for path in sorted(Path(source_dir).rglob("*")):
        if not path.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True)
        except Exception:  # noqa: BLE001 - anything unreadable as DICOM is junk to discard
            continue
        uid = getattr(ds, "SeriesInstanceUID", None)
        # require an image series: has a series UID and pixel geometry (excludes DICOMDIR etc.)
        if not uid or not (getattr(ds, "Rows", None) and getattr(ds, "Columns", None)):
            continue
        group = groups.setdefault(str(uid), {"files": [], "header": ds})
        instance = getattr(ds, "InstanceNumber", None)
        order = int(instance) if instance is not None else len(group["files"])
        group["files"].append((order, path))
    for group in groups.values():
        group["files"].sort(key=lambda item: item[0])
        group["files_sorted"] = [p for _, p in group["files"]]
    return groups


def _array_to_png(arr: np.ndarray) -> bytes | None:
    """Window a 2D slice by its 1st–99th intensity percentile and encode it as a PNG."""
    from PIL import Image as PILImage

    data = np.squeeze(np.asarray(arr)).astype(np.float64)
    if data.ndim == 3:  # multi-frame / RGB slice -> take the middle frame
        data = data[data.shape[0] // 2]
    if data.ndim != 2:
        return None
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return None
    lo, hi = float(np.percentile(finite, 1)), float(np.percentile(finite, 99))
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        hi = lo + 1.0
    u8 = (np.clip((data - lo) / (hi - lo), 0.0, 1.0) * 255).astype(np.uint8)
    buffer = BytesIO()
    PILImage.fromarray(u8, mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


def _render_series_previews(files: list[Path], max_slices: int = 16) -> list[bytes]:
    """Render up to ``max_slices`` evenly-sampled preview PNGs across a DICOM series.

    Uses SimpleITK (GDCM) to decode each slice, matching the volume-read path, so
    compressed transfer syntaxes work. Unreadable slices are skipped.
    """
    import SimpleITK as sitk

    n = len(files)
    if n == 0:
        return []
    count = min(max_slices, n)
    indices = [round(i * (n - 1) / (count - 1)) for i in range(count)] if count > 1 else [0]
    previews: list[bytes] = []
    for i in indices:
        try:
            arr = sitk.GetArrayFromImage(sitk.ReadImage(str(files[i])))
        except Exception:  # noqa: BLE001 - skip a slice we cannot decode
            continue
        png = _array_to_png(arr)
        if png is not None:
            previews.append(png)
    return previews


def _replace_for_enumerate(accession: str) -> None:
    """Clear any prior state for this accession before staging a fresh enumeration.

    A half-finished pending upload for the same study is always replaceable; a
    fully-ingested examination respects the ``on_duplicate`` policy.
    """
    with session_scope(get_engine()) as session:
        existing = repository.get_examination(session, accession)
    if existing is None:
        return
    if existing.status == ExaminationStatus.PENDING_SELECTION.value:
        get_store().delete_examination(accession)
        with session_scope(get_engine()) as session:
            repository.delete_examination(session, accession)
        return
    _check_duplicate(accession)  # raises or replaces per settings.on_duplicate


def enumerate_torsion_series(source_dir: Path) -> str:
    """Enumerate all DICOM series in an uploaded examination directory (phase 1).

    Groups files by series, discards non-DICOM junk, stages each series' raw DICOM
    and renders per-series preview slices, and records a ``pending_selection``
    examination row carrying the candidate-series metadata. No volumes are produced
    yet — that happens in :func:`materialize_torsion_selection` once the user picks.

    :param source_dir: Directory of the whole uploaded examination (mixed files).
    :return: The examination id (accession number).
    """
    groups = _group_series(source_dir)
    if not groups:
        raise IngestError("No DICOM image series found in the upload")

    first_header = next(iter(groups.values()))["header"]
    accession = _accession(first_header)
    _replace_for_enumerate(accession)

    store = get_store()
    series_meta: list[dict] = []
    for uid, group in groups.items():
        files = group["files_sorted"]
        for index, path in enumerate(files):
            store.stage_series_file(accession, uid, index, path.read_bytes())
        previews = _render_series_previews(files)
        for index, png in enumerate(previews):
            store.save_preview(accession, uid, index, png)
        header = group["header"]
        series_meta.append({
            "uid": uid,
            "description": _hdr_str(header, "SeriesDescription"),
            "modality": _hdr_str(header, "Modality"),
            "instances": len(files),
            "rows": _hdr_int(header, "Rows"),
            "cols": _hdr_int(header, "Columns"),
            "preview_count": len(previews),
        })

    with session_scope(get_engine()) as session:
        repository.upsert_examination(session, Examination(
            id=accession,
            examination_type=ExaminationType.TORSION.value,
            status=ExaminationStatus.PENDING_SELECTION.value,
            series=series_meta,
            **_study_fields(first_header),
        ))
    return accession


def _require_staged(series_dir: Path, accession: str) -> None:
    if not series_dir.exists() or not any(series_dir.glob("*.dcm")):
        raise NotFoundError(f"Series {series_dir.name} not staged for examination {accession}")


def materialize_torsion_selection(accession: str, mode: str, selection: dict) -> str:
    """Materialize the user-selected series into a processable examination (phase 2).

    :param accession: The pending examination id.
    :param mode: ``"whole_leg"`` (one series, auto-split) or ``"regions"`` (three
        already-split hip/knee/ankle series).
    :param selection: For ``whole_leg`` ``{"series_uid": ...}``; for ``regions``
        ``{"hip": uid, "knee": uid, "ankle": uid}``.
    :return: The examination id (unchanged accession), now ``unprocessed``.
    """
    store = get_store()
    if mode == "whole_leg":
        series_dir = store.series_incoming_dir(accession, selection["series_uid"])
        _require_staged(series_dir, accession)
        metadata = Image.read_dicom_metadata(str(series_dir))
        _ingest_whole_leg_dir(series_dir, accession, metadata)
    elif mode == "regions":
        dirs = {region: store.series_incoming_dir(accession, selection[region])
                for region in ("hip", "knee", "ankle")}
        for series_dir in dirs.values():
            _require_staged(series_dir, accession)
        metadata = Image.read_dicom_metadata(str(dirs["hip"]))
        _ingest_multi_dirs(dirs["hip"], dirs["knee"], dirs["ankle"], accession, metadata)
    else:
        raise IngestError(f"Unknown selection mode: {mode}")

    store.clear_series_staging(accession)  # staged DICOM + previews now consumed
    return accession
