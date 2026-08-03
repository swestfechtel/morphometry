"""Tests for the NIfTI volume streaming endpoints used by the Cornerstone viewer."""
import nibabel as nib

from api.tests.test_api import _seed_examination

GZIP_MAGIC = b"\x1f\x8b"


def _load_nifti(content: bytes, tmp_path):
    """Load a .nii.gz response body into a nibabel image via a temp file."""
    path = tmp_path / "vol.nii.gz"
    path.write_bytes(content)
    return nib.load(str(path))


def test_image_volume_served(client, runtime):
    _seed_examination(runtime, status="processed", with_volumes=True)
    resp = client.get("/examinations/ACC1/volume/image.nii.gz")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/gzip"
    assert resp.content[:2] == GZIP_MAGIC


def test_mask_volume_built_and_aligned(client, runtime, tmp_path):
    # with_volumes gives source transformed + hip/knee/ankle; with_masks gives region masks.
    _seed_examination(runtime, status="processed", with_volumes=True, with_masks=True)
    resp = client.get("/examinations/ACC1/volume/mask.nii.gz")
    assert resp.status_code == 200
    assert resp.content[:2] == GZIP_MAGIC
    combined = _load_nifti(resp.content, tmp_path)
    # three 4x4x4 region masks concatenated along z -> 4x4x12, labels within {0,1,2,3}
    assert combined.shape == (4, 4, 12)
    assert set(int(v) for v in set(combined.get_fdata().ravel())) <= {0, 1, 2, 3}


def test_mask_volume_404_without_masks(client, runtime):
    _seed_examination(runtime, status="processed", with_volumes=True)  # no masks
    assert client.get("/examinations/ACC1/volume/mask.nii.gz").status_code == 404


def test_volume_404_unknown_examination(client, runtime):
    assert client.get("/examinations/NOPE/volume/image.nii.gz").status_code == 404


def test_volume_range_request_206(client, runtime):
    _seed_examination(runtime, status="processed", with_volumes=True)
    resp = client.get("/examinations/ACC1/volume/image.nii.gz", headers={"Range": "bytes=0-19"})
    assert resp.status_code == 206
    assert len(resp.content) == 20
    assert resp.headers["accept-ranges"] == "bytes"


def test_volume_auth_via_query_param(client, runtime):
    """The loader fallback: the key may arrive as ?api_key= instead of a header."""
    _seed_examination(runtime, status="processed", with_volumes=True)
    # strip the default header so only the query param carries the key
    no_header = {"X-API-Key": ""}
    assert client.get("/examinations/ACC1/volume/image.nii.gz",
                      params={"api_key": "test-key"}, headers=no_header).status_code == 200
    assert client.get("/examinations/ACC1/volume/image.nii.gz",
                      params={"api_key": "wrong"}, headers=no_header).status_code == 401
    assert client.get("/examinations/ACC1/volume/image.nii.gz", headers=no_header).status_code == 401
