"""Central, environment-overridable locations of sample segmentation data.

The characterization tests need real segmentation masks to run. Those masks are
not part of the repository, so every path here can be overridden with an
environment variable and the fixtures in ``conftest.py`` skip cleanly when the
data is absent. This keeps the suite runnable on machines without the data.
"""
import os
from pathlib import Path

# A single Augsburg case with separate hip / knee / ankle MRI segmentations.
# hip_seg: femur=1 (femur-only, 16 slices); knee_seg: femur=1, tibia=2;
# ankle_seg: tibia=1, fibula=2.
AUGSBURG_PA000001 = Path(
    os.environ.get(
        "MORPH_AUGSBURG_PA000001",
        "/home/simon/Data/Augsburg_large/preprocessed/PA000001",
    )
)

# The Augsburg reader-study directory: one ``<patient>/{hip,knee,ankle}_seg_test.nii.gz`` folder
# per patient (hip: femur=1; knee: femur=1, tibia=2). Used for femoral-torsion cases whose
# femoral head lies level with or posterior to the neck point in the image, which PA000001
# never exercises.
AUGSBURG_READER_STUDY = Path(
    os.environ.get(
        "MORPH_AUGSBURG_READER_STUDY",
        "/home/simon/Data/Augsburg_large/reader_study_augsburg",
    )
)

# Directory of NaKo hip MRI segmentations (isotropic; femur=1, cartilage=2,
# acetabulum=3 — both image sides present in one mask).
NAKO_SAMPLE_DIR = Path(
    os.environ.get(
        "MORPH_NAKO_SAMPLE_DIR",
        "/home/simon/Data/NaKo_sample/segmentations",
    )
)

# Directory of ``dfg_knees`` knee segmentations, one file per scan named
# ``K<subject>_<0|30|60|90>_<belastet|unbelastet>.nii.gz`` (labels: 1 = femur bone,
# 2 = tibia bone, 3 = femoral cartilage, 4 = tibial cartilage, 5 = patella). Used by the
# knee-cartilage subregion / thickness tests (``needs_dfg_knees``).
DFG_KNEES_DIR = Path(
    os.environ.get(
        "MORPH_DFG_KNEES_DIR",
        "/home/simon/Data/dfg_knees/monai_segmentations/labels/original",
    )
)

# The scans the knee-cartilage goldens are pinned on: a clean extended (0 deg) knee and the
# 90 deg scan of the same subject and load, which exercises the flexed-knee code paths
# (femoral frame, posterior-condyle zones, the shared in-plane frame).
DFG_KNEES_GOLDEN_SCAN = "K1_0_unbelastet"
DFG_KNEES_GOLDEN_SCAN_FLEXED = "K1_90_unbelastet"


def dfg_knee_path(name: str) -> Path:
    """Path of one ``dfg_knees`` scan inside :data:`DFG_KNEES_DIR`.

    :param name: The scan name without suffix, e.g. ``'K1_0_unbelastet'``.
    :return: ``DFG_KNEES_DIR / '<name>.nii.gz'`` (which need not exist).
    """
    return DFG_KNEES_DIR / f"{name}.nii.gz"


# A whole-leg CT segmentation (femur=1, tibia=2, fibula=3, patella=5, hip=7),
# containing both legs (split at shape[0]//2). MORPH_CT_SAMPLE may point at a
# single .nii.gz file; otherwise a fixed case from the NMDID verification sample
# directory is used (chosen deterministically for stable goldens).
_CT_DEFAULT_DIR = Path(
    "/home/simon/sshfs/hpcproject/workspace_simon/nmdid/verification_sample/ground_truth"
)
_CT_DEFAULT_CASE = "case-105094.nii.gz"


def _resolve_ct_sample():
    env = os.environ.get("MORPH_CT_SAMPLE")
    if env:
        p = Path(env)
        return p if p.exists() else None
    candidate = _CT_DEFAULT_DIR / _CT_DEFAULT_CASE
    return candidate if candidate.exists() else None


CT_WHOLE_LEG = _resolve_ct_sample()
