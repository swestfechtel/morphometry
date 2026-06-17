"""Unit tests for ingest helpers (no real DICOM needed)."""
import nibabel as nib
import numpy as np

from morphometry.image_io import Image
from api.ingest.dicom import _materialize, _split_volume


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


def _stacked_volume():
    """A (16,16,30) volume with three z-regions of different in-plane footprints."""
    arr = np.zeros((16, 16, 30), dtype=np.float64)
    arr[:14, :14, 0:10] = 100   # hip: large footprint
    arr[:6, :6, 10:20] = 100    # knee: small footprint
    arr[:10, :10, 20:30] = 100  # ankle: medium footprint
    return Image.from_nibabel(nib.Nifti1Image(arr, np.eye(4)))


def test_split_volume_returns_three_regions_summing_to_input():
    regions = _split_volume(_stacked_volume())
    assert set(regions) == {"hip", "knee", "ankle"}
    z = [regions[r].array.shape[2] for r in ("hip", "knee", "ankle")]
    assert all(n > 0 for n in z), z          # no empty sub-volume (the old unpacking bug)
    assert sum(z) == 30                        # partitions the full stack
