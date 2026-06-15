"""Unit tests for the file-based image Store."""
import base64

import nibabel as nib
import numpy as np

from morphometry.image_io import Image
from api.storage.store import Store


def _make_image(value=1):
    arr = np.full((4, 4, 3), value, dtype=np.int16)
    return Image.from_nibabel(nib.Nifti1Image(arr, np.eye(4)))


def test_volume_roundtrip(tmp_path):
    store = Store(tmp_path)
    rel = store.save_volume("ACC1", "hip", _make_image(7))
    assert rel == "ACC1/source/hip.nii.gz"
    assert store.abspath(rel).exists()
    loaded = store.load_image(rel)
    assert loaded.array.shape == (4, 4, 3)
    assert int(loaded.array[0, 0, 0]) == 7


def test_encoded_roundtrip(tmp_path):
    store = Store(tmp_path)
    pngs = [base64.b64encode(b"\x89PNG-fake-%d" % i).decode() for i in range(3)]
    paths = store.save_encoded("ACC1", pngs, pngs[:2])
    assert len(paths["image"]) == 3 and len(paths["segmentation"]) == 2
    assert store.load_encoded_b64(paths["image"]) == pngs


def test_incoming_staging(tmp_path):
    store = Store(tmp_path)
    store.stage_incoming("ACC9", "uid1", b"dicom-a")
    store.stage_incoming("ACC9", "uid2", b"dicom-b")
    store.stage_incoming("ACC9", "uid1", b"dicom-a")  # idempotent overwrite
    assert len(store.incoming_files("ACC9")) == 2
    assert store.pending_accessions() == ["ACC9"]
    store.clear_incoming("ACC9")
    assert store.incoming_files("ACC9") == []
    assert store.pending_accessions() == []


def test_delete_examination_removes_dir(tmp_path):
    store = Store(tmp_path)
    store.save_volume("ACCX", "hip", _make_image())
    assert store.examination_dir("ACCX").exists()
    store.delete_examination("ACCX")
    assert not store.examination_dir("ACCX").exists()
