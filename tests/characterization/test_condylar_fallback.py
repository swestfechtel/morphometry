"""The posterior-condylar fallback on the real Augsburg legs whose notch search fails.

``find_notch`` resolves the intercondylar gap on 89 of the 100 Augsburg reader-study tibial halves.
On the other 11 the posterior half of the biggest-convex-area section holds no discontinuity wider
than 2 voxels at all (measured: maximum gap 0-2 voxels, against a median 15 on the 89), so the
reference line is built from the fallback divider (``get_condylar_divider``). The old fallback
rotated those sections by 140-181 degrees and reported the anterior cortex: its endpoints sat a
median 40.5 voxels (max 51) anterior of the posterior-most mask voxel of their own column, and on 5
of the 11 it raised ``RuntimeError('No notch found.')`` instead of returning anything.

The legs below are three of those 11, picked to cover both image sides, both previously-failing and
previously-returning cases, and the largest observed change:

* ``PA000077`` image-left -- raised ``RuntimeError`` before;
* ``PA000347`` image-left -- returned 29.49 degrees before, 39.95 now;
* ``PA000717`` image-right -- returned 28.15 degrees before, 44.43 now (largest change).

**Do not pool tibial torsion rows produced before and after this change** -- see CLAUDE.md.
"""
import warnings

import numpy as np
import pytest

from morphometry.image_io import split_image_left_right
from morphometry.measurements.tibia import calculate_tibial_torsion, get_tibial_torsion_landmarks
from morphometry.utils import find_notch, get_layer_with_biggest_convex_area
from tests import data_paths
from tests.conftest import _load_seg, assert_golden

pytestmark = pytest.mark.needs_augsburg_reader_study

# (patient, image side); "left"/"right" is the image side, not the patient side
FALLBACK_LEGS = [('PA000077', 'left'), ('PA000347', 'left'), ('PA000717', 'right')]
OTHER = {'left': 'right', 'right': 'left'}
TIBIA_LABEL_KNEE, TIBIA_LABEL_ANKLE, FIBULA_LABEL = 2, 1, 2


@pytest.fixture(scope='module')
def reader_study_knee_ankle():
    """A loader for one reader-study patient's knee and ankle halves.

    The shared ``augsburg_reader_study`` fixture loads hip and knee only; the tibial torsion also
    needs the ankle, so this fixture loads the pair itself (LPI, de-outliered) and caches nothing
    beyond the module scope.
    :return: A function ``load(patient) -> {'knee': {side: Segmentation}, 'ankle': {...}}`` that
        skips the test when the patient's files are absent.
    """
    base = data_paths.AUGSBURG_READER_STUDY

    def load(patient: str) -> dict:
        files = {n: base / patient / f'{n}_seg_test.nii.gz' for n in ('knee', 'ankle')}
        if not all(p.exists() for p in files.values()):
            pytest.skip(f'Augsburg reader-study patient {patient} not found under {base}')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            images = {n: _load_seg(p) for n, p in files.items()}
        return {n: dict(zip(('left', 'right'), split_image_left_right(image)))
                for n, image in images.items()}

    return load


def _search_frame(knee_array: np.ndarray, side: str) -> np.ndarray:
    """The array the condylar search actually sees for this image side.

    A right half is measured as the left-right reflection of a left one
    (``morphometry.measurements.tibia._tibial_torsion_points``), so its landmark search runs on the
    reversed array.
    :param knee_array: The knee half's voxel array.
    :param side: Image side, 'left' or 'right'.
    :return: The (possibly reversed) array.
    """
    return knee_array if side == 'left' else knee_array[::-1]


def _cortex_distances(section: np.ndarray, *points):
    """Distance of each point to the posterior and the anterior cortex of an axial section.

    Per column of the section, the posterior cortex is the mask voxel with the largest coronal
    index and the anterior cortex the one with the smallest. The posterior condylar endpoints must
    lie on the posterior cortex; the old fallback put them on the anterior one.
    :param section: A 2D binary axial section.
    :param points: Points to score (only x and y are used).
    :return: List of ``(distance to posterior cortex, distance to anterior cortex)`` in voxels.
    """
    columns = np.nonzero(section.any(axis=1))[0]
    posterior = np.array([[x, np.nonzero(section[x])[0].max()] for x in columns], dtype=float)
    anterior = np.array([[x, np.nonzero(section[x])[0].min()] for x in columns], dtype=float)
    out = []
    for point in points:
        p = np.asarray(point, dtype=float)[:2]
        out.append((float(np.linalg.norm(posterior - p, axis=1).min()),
                    float(np.linalg.norm(anterior - p, axis=1).min())))
    return out


@pytest.mark.parametrize('patient,side', FALLBACK_LEGS)
def test_leg_really_exercises_the_fallback(reader_study_knee_ankle, patient, side):
    """These legs' reference sections hold no resolvable intercondylar gap.

    Without this the other tests here would silently stop testing the fallback if the notch search
    ever started succeeding on them.
    """
    knee = reader_study_knee_ankle(patient)['knee'][side]
    mask = np.where(_search_frame(knee.array, side) == TIBIA_LABEL_KNEE, 1, 0)
    layer = get_layer_with_biggest_convex_area(mask)
    assert find_notch(mask[:, :, layer], percentage=0.5, thresh=2) is None


@pytest.mark.parametrize('patient,side', FALLBACK_LEGS)
def test_fallback_condylar_points_lie_on_the_posterior_cortex(reader_study_knee_ankle, patient, side):
    """The reported condylar endpoints sit on the posterior cortex, not the anterior one.

    Scored against an independent cue (the posterior-most mask voxel per column), not against the
    rotation that produced the points. The 2-voxel allowance covers the back-rotation and the final
    ``uint16`` cast of the swept endpoint.
    """
    images = reader_study_knee_ankle(patient)
    knee, ankle = images['knee'][side], images['ankle'][side]
    landmarks = get_tibial_torsion_landmarks(knee.array, ankle.array, TIBIA_LABEL_KNEE,
                                             TIBIA_LABEL_ANKLE, FIBULA_LABEL, side)
    start, end = landmarks['knee_start'], landmarks['knee_end']
    assert int(start[2]) == int(end[2])
    mask = np.where(knee.array == TIBIA_LABEL_KNEE, 1, 0)
    section = mask[:, :, int(start[2])]
    for posterior, anterior in _cortex_distances(section, start, end):
        assert posterior <= 2, f'condylar endpoint {posterior:.1f} voxels off the posterior cortex'
        assert anterior > 10, 'condylar endpoint sits on the anterior cortex'


@pytest.mark.parametrize('patient,side', FALLBACK_LEGS)
def test_fallback_tibial_torsion_golden(reader_study_knee_ankle, patient, side):
    """Pin the tibial torsion of each fallback leg (all three were wrong or absent before)."""
    images = reader_study_knee_ankle(patient)
    knee, ankle = images['knee'][side], images['ankle'][side]
    angle = calculate_tibial_torsion(knee.array, ankle.array, TIBIA_LABEL_KNEE, TIBIA_LABEL_ANKLE,
                                     FIBULA_LABEL, side)
    assert np.isfinite(angle)
    assert_golden(f'tibial_torsion_fallback_{patient}_{side}', angle)


@pytest.mark.parametrize('patient,side', FALLBACK_LEGS)
def test_fallback_is_mirror_equivariant(reader_study_knee_ankle, patient, side):
    """A fallback leg measured as the reflection of itself gives the same torsion.

    As in ``test_ccd_tibial_equivariance.py`` the angle assertion is near-tautological while the
    right side is measured by reflecting the half and running the left-side code -- it guards
    against a ``side == 'right'`` branch being reintroduced. It has real content for the fallback
    specifically: the old code chose its frame from the section's longest contour chord and could
    pick a different one on a half and on its reflection.
    """
    images = reader_study_knee_ankle(patient)
    knee, ankle = images['knee'][side], images['ankle'][side]
    args = (TIBIA_LABEL_KNEE, TIBIA_LABEL_ANKLE, FIBULA_LABEL)
    reference = calculate_tibial_torsion(knee.array, ankle.array, *args, side)
    mirrored = calculate_tibial_torsion(knee.array[::-1], ankle.array[::-1], *args, OTHER[side])
    assert mirrored == pytest.approx(reference, abs=1e-9)
