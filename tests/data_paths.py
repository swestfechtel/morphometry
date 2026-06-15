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

# Directory of NaKo hip MRI segmentations (isotropic; femur=1, cartilage=2,
# acetabulum=3 — both image sides present in one mask).
NAKO_SAMPLE_DIR = Path(
    os.environ.get(
        "MORPH_NAKO_SAMPLE_DIR",
        "/home/simon/Data/NaKo_sample/segmentations",
    )
)

# A whole-leg CT segmentation (femur=1, tibia=2, fibula=3, patella=5, hip=7).
# Location is uncertain on most machines; set MORPH_CT_SAMPLE to enable the
# CT (`*_ct`) characterization tests.
CT_WHOLE_LEG = Path(os.environ["MORPH_CT_SAMPLE"]) if os.environ.get("MORPH_CT_SAMPLE") else None
