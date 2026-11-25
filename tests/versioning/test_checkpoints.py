"""Tests for the nnUNet checkpoint manifest (``versioning.checkpoints``)."""
from pathlib import Path

from versioning.checkpoints import verify_manifest, write_manifest


def test_checkpoint_manifest_roundtrip(tmp_path: Path):
    """Write → verify is clean; one changed byte is one mismatch; one new file is one extra."""
    ckpt = tmp_path / "checkpoints"
    (ckpt / "hip" / "fold_0").mkdir(parents=True)
    (ckpt / "knee").mkdir()
    (ckpt / "hip" / "fold_0" / "checkpoint_final.pth").write_bytes(b"\x00\x01weights" * 1000)
    (ckpt / "knee" / "plans.json").write_text("{}\n")
    manifest = tmp_path / "checkpoints.sha256"

    write_manifest(ckpt, manifest)
    lines = manifest.read_text().splitlines()
    assert [line.split("  ", 1)[1] for line in lines] == ["hip/fold_0/checkpoint_final.pth", "knee/plans.json"]
    assert all(len(line.split("  ", 1)[0]) == 64 for line in lines)
    assert verify_manifest(ckpt, manifest) == []

    weights = ckpt / "hip" / "fold_0" / "checkpoint_final.pth"
    data = bytearray(weights.read_bytes())
    data[5] ^= 0xFF
    weights.write_bytes(bytes(data))
    problems = verify_manifest(ckpt, manifest)
    assert len(problems) == 1 and "mismatch" in problems[0] and "checkpoint_final.pth" in problems[0]

    write_manifest(ckpt, manifest)
    (ckpt / "ankle.pth").write_bytes(b"new")
    problems = verify_manifest(ckpt, manifest)
    assert len(problems) == 1 and "extra" in problems[0] and "ankle.pth" in problems[0]


def test_checkpoint_manifest_missing_file(tmp_path: Path):
    """A file listed in the manifest but absent on disk is reported as missing."""
    ckpt = tmp_path / "checkpoints"
    ckpt.mkdir()
    (ckpt / "a.pth").write_bytes(b"a")
    manifest = tmp_path / "checkpoints.sha256"
    write_manifest(ckpt, manifest)
    (ckpt / "a.pth").unlink()
    problems = verify_manifest(ckpt, manifest)
    assert len(problems) == 1 and "missing" in problems[0] and "a.pth" in problems[0]
