"""Unit tests for ingest helpers + an end-to-end ingest against synthetic DICOM."""
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from morphometry.image_io import Image
from api.ingest.dicom import _materialize, _split_volume


def _write_dicom_series(directory: Path, n_slices: int = 18, rows: int = 32, cols: int = 32) -> None:
    """Write a minimal but valid CT DICOM series SITK can assemble into a volume.

    The in-plane footprint changes across three z-thirds so the changepoint split
    finds two breakpoints (hip/knee/ankle).
    """
    directory.mkdir(parents=True, exist_ok=True)
    series_uid, study_uid = generate_uid(), generate_uid()
    footprints = [28, 12, 20]  # square side per third
    for i in range(n_slices):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(str(directory / f"{i:03d}.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.SeriesInstanceUID = series_uid
        ds.StudyInstanceUID = study_uid
        ds.Modality = "CT"
        ds.AccessionNumber = "TESTACC1"
        ds.StudyDate = "20240102"
        ds.StudyTime = "153000"
        ds.StudyDescription = "MRT Beinachsenmessung"
        ds.PatientName = "Anon"
        ds.Rows, ds.Columns = rows, cols
        ds.PixelSpacing = [1.0, 1.0]
        ds.SliceThickness = 1.0
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [0.0, 0.0, float(i)]
        ds.InstanceNumber = i + 1
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        side = footprints[min(i // (n_slices // 3), 2)]
        arr = np.zeros((rows, cols), dtype=np.uint16)
        arr[:side, :side] = 1000
        ds.PixelData = arr.tobytes()
        ds.save_as(str(directory / f"{i:03d}.dcm"), enforce_file_format=True)


def test_materialize_survives_backing_file_deletion(tmp_path):
    """An image loaded from a file must remain saveable after that file is deleted.

    Reproduces the dicom_to_nibabel proxy bug: ingest cleans up the temp file
    before saving, so the image must be materialized in memory first.
    """
    src = tmp_path / "src.nii.gz"
    nib.save(nib.Nifti1Image(np.arange(2 * 3 * 4, dtype=np.int16).reshape(2, 3, 4), np.eye(4)), src)
    proxy = Image("nibabel")
    proxy.read_image(str(src))

    materialized = _materialize(proxy)
    src.unlink()  # backing file gone (as after tmp.cleanup())

    out = tmp_path / "out.nii.gz"
    materialized.save_image(str(out))  # must not raise FileNotFoundError
    assert out.exists()
    reloaded = Image("nibabel")
    reloaded.read_image(str(out))
    assert reloaded.array.shape == (2, 3, 4)


def test_ingest_torsion_multi_from_dirs_end_to_end(runtime, tmp_path):
    """Full real multi-series ingest: three synthetic DICOM series -> row + files."""
    from api.db import repository
    from api.db.engine import session_scope
    from api.ingest.dicom import ingest_torsion_multi_from_dirs

    dirs = {}
    for region in ("hip", "knee", "ankle"):
        d = tmp_path / region
        _write_dicom_series(d, n_slices=6)  # same in-plane shape across series (required)
        dirs[region] = d

    accession = ingest_torsion_multi_from_dirs(dirs["hip"], dirs["knee"], dirs["ankle"])

    store, engine = runtime.get_store(), runtime.get_engine()
    with session_scope(engine) as s:
        row = repository.get_examination(s, accession)
        assert row is not None and row.status == "unprocessed"
        assert set(row.source_paths) == {"original", "transformed", "hip", "knee", "ankle"}
        assert row.knee_offset == 6 and row.ankle_offset == 12  # 6 slices per series
        paths = dict(row.source_paths)
    for rel in paths.values():
        assert store.abspath(rel).exists()


def _stacked_volume():
    """A (16,16,30) volume with three z-regions of different in-plane footprints."""
    arr = np.zeros((16, 16, 30), dtype=np.float64)
    arr[:14, :14, 0:10] = 100   # hip: large footprint
    arr[:6, :6, 10:20] = 100    # knee: small footprint
    arr[:10, :10, 20:30] = 100  # ankle: medium footprint
    return Image.from_nibabel(nib.Nifti1Image(arr, np.eye(4)))


def test_ingest_torsion_from_dir_end_to_end(runtime, tmp_path):
    """Full real ingest: synthetic DICOM series -> row + .nii.gz files (no docker)."""
    from api.db import repository
    from api.db.engine import session_scope
    from api.ingest.dicom import ingest_torsion_from_dir

    series_dir = tmp_path / "series"
    _write_dicom_series(series_dir)

    accession = ingest_torsion_from_dir(series_dir)
    assert accession == "TESTACC1"

    store, engine = runtime.get_store(), runtime.get_engine()
    with session_scope(engine) as s:  # read attributes while the session is open
        row = repository.get_examination(s, accession)
        assert row is not None
        assert row.status == "unprocessed"
        assert row.study_description == "MRT Beinachsenmessung"
        assert set(row.source_paths) == {"original", "transformed", "hip", "knee", "ankle"}
        assert row.knee_offset and row.ankle_offset and row.knee_offset < row.ankle_offset
        source_paths = dict(row.source_paths)

    # every recorded volume file actually exists and reloads
    for rel in source_paths.values():
        assert store.abspath(rel).exists()
    assert store.load_image(source_paths["transformed"]).array.ndim == 3


def test_split_volume_returns_three_regions_summing_to_input():
    regions = _split_volume(_stacked_volume())
    assert set(regions) == {"hip", "knee", "ankle"}
    z = [regions[r].array.shape[2] for r in ("hip", "knee", "ankle")]
    assert all(n > 0 for n in z), z          # no empty sub-volume (the old unpacking bug)
    assert sum(z) == 30                        # partitions the full stack
