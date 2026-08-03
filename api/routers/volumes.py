"""NIfTI volume streaming endpoints for the Cornerstone3D viewer.

These serve the raw ``.nii.gz`` image and (combined) mask volumes so the UI can
render a real 3D volume instead of pre-encoded PNG slices. They live on their own
router (separate from the JSON examinations router) because they use
``require_volume_access`` — which accepts the API key from either the header or an
``?api_key=`` query param — since Cornerstone's loader can't set a custom header.

``FileResponse`` gives byte-range support (HTTP 206) for free, which the loader
and the browser use when streaming the gzip volume.

The route paths deliberately end in ``.nii.gz``: Cornerstone's
``createNiftiImageIdsAndCacheMetadata`` decides whether to gunzip the response
*solely* from ``new URL(url).pathname.endsWith('.gz')`` — the ``Content-Type`` /
``Content-Disposition`` are ignored. Without the ``.gz`` path suffix the loader
feeds raw gzip bytes to the NIfTI header parser, reads garbage dimensions and
fails with "Array buffer allocation failed". A ``?api_key=`` query param does not
affect ``pathname``, so the fallback-auth URL still decompresses correctly.
"""
from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
from sqlmodel import Session

from api.db import repository
from api.deps import get_session, get_store, require_volume_access
from api.errors import NotFoundError
from api.storage.store import Store

router = APIRouter(prefix="/examinations", tags=["volumes"],
                   dependencies=[Depends(require_volume_access)])

_NIFTI_MEDIA_TYPE = "application/gzip"


@router.get("/{examination_id}/volume/image.nii.gz")
def get_image_volume(examination_id: str, session: Session = Depends(get_session),
                     store: Store = Depends(get_store)) -> FileResponse:
    """Stream the transformed (combined hip+knee+ankle) source volume as ``.nii.gz``."""
    row = repository.get_examination(session, examination_id)
    if row is None:
        raise NotFoundError(f"Examination {examination_id} not found")
    rel = (row.source_paths or {}).get("transformed")
    if not rel or not store.abspath(rel).exists():
        raise NotFoundError(f"Image volume not available for {examination_id}")
    # Serve a GPU-renderable dtype: the transformed volume is often float64, which
    # WebGL can't texture (Cornerstone would render it black). Cached float32 copy.
    rel = store.ensure_display_volume(examination_id, rel)
    return FileResponse(store.abspath(rel), media_type=_NIFTI_MEDIA_TYPE, filename="image.nii.gz")


@router.get("/{examination_id}/volume/mask.nii.gz")
def get_mask_volume(examination_id: str, session: Session = Depends(get_session),
                    store: Store = Depends(get_store)) -> FileResponse:
    """Stream the combined label mask (aligned to the image volume) as ``.nii.gz``.

    Built by the worker for new examinations; lazily built and cached from the
    per-region masks for examinations processed before the combined mask existed.
    """
    row = repository.get_examination(session, examination_id)
    if row is None:
        raise NotFoundError(f"Examination {examination_id} not found")
    mask_paths = row.mask_paths or {}
    transformed = (row.source_paths or {}).get("transformed")
    if not all(mask_paths.get(r) for r in ("hip", "knee", "ankle")) or not transformed:
        raise NotFoundError(f"Mask volume not available for {examination_id}")
    rel = store.ensure_combined_mask(examination_id, mask_paths, transformed)
    return FileResponse(store.abspath(rel), media_type=_NIFTI_MEDIA_TYPE, filename="mask.nii.gz")


@router.get("/{examination_id}/series/{series_uid}/preview/{index}.png")
def get_series_preview(examination_id: str, series_uid: str, index: int,
                       session: Session = Depends(get_session),
                       store: Store = Depends(get_store)) -> FileResponse:
    """Stream one preview slice PNG of a candidate series on a pending examination.

    Served here (with header-or-query-param auth) because the UI loads previews via
    ``<img>`` tags, which cannot set a custom header.
    """
    row = repository.get_examination(session, examination_id)
    if row is None:
        raise NotFoundError(f"Examination {examination_id} not found")
    if series_uid not in {(s or {}).get("uid") for s in (row.series or [])}:
        raise NotFoundError(f"Series {series_uid} not found for {examination_id}")
    path = store.preview_path(examination_id, series_uid, index)
    if not path.exists():
        raise NotFoundError(f"Preview {index} not available for series {series_uid}")
    return FileResponse(path, media_type="image/png")
