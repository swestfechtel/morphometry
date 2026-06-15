"""End-to-end characterization of the docker torsion pipeline.

Mirrors morphometry/docker/compute_torsion.py exactly (load -> LPI -> remove
outliers -> split L/R -> femoral Lee/Murphy + tibial torsion per side, with the
right-side coordinate shift) and asserts the results dict and the nested landmark
structure the REST layer consumes. Guards the swethfechtel/torsion:latest rebuild.
"""
import nibabel as nib
import numpy as np
import pytest

from morphometry.image_io import Image, Segmentation
from morphometry.measurements.femur import calculate_femoral_torsion, get_femoral_torsion_landmarks
from morphometry.measurements.tibia import calculate_tibial_torsion, get_tibial_torsion_landmarks

pytestmark = pytest.mark.needs_augsburg


def test_compute_torsion_pipeline(augsburg_pa000001):
    hip_mask, knee_mask, ankle_mask = (augsburg_pa000001[k] for k in ("hip", "knee", "ankle"))
    x_ratio = abs(hip_mask.spacing[2]) / 2 * abs(hip_mask.spacing[0])

    left_hip = Image.from_nibabel(nib.Nifti1Image(hip_mask.array[:hip_mask.array.shape[0] // 2], hip_mask.affine, hip_mask.header))
    right_hip = Image.from_nibabel(nib.Nifti1Image(hip_mask.array[hip_mask.array.shape[0] // 2:], hip_mask.affine, hip_mask.header))
    left_knee = knee_mask.array[:knee_mask.array.shape[0] // 2]
    right_knee = knee_mask.array[knee_mask.array.shape[0] // 2:]
    left_ankle = ankle_mask.array[:ankle_mask.array.shape[0] // 2]
    right_ankle = ankle_mask.array[ankle_mask.array.shape[0] // 2:]

    results = {}
    landmarks_final = {'femur': {'Lee': {}, 'Murphy': {}}, 'tibia': {}}

    # femoral, image-left half (stored as patient-right)
    results['femoral_torsion_right'] = calculate_femoral_torsion(left_hip, left_knee, side='left', method='lee', x_ratio=x_ratio)
    landmarks_final['femur']['Lee']['right'] = get_femoral_torsion_landmarks(left_hip, left_knee, side='left', method='lee', x_ratio=x_ratio)
    results['femoral_torsion_right_murphy'] = calculate_femoral_torsion(left_hip, left_knee, side='left', method='murphy', x_ratio=x_ratio)
    landmarks_final['femur']['Murphy']['right'] = get_femoral_torsion_landmarks(left_hip, left_knee, side='left', method='murphy', x_ratio=x_ratio)

    # femoral, image-right half (stored as patient-left), with the sagittal shift
    results['femoral_torsion_left'] = calculate_femoral_torsion(right_hip, right_knee, side='right', method='lee', x_ratio=x_ratio)
    lm = get_femoral_torsion_landmarks(right_hip, right_knee, side='right', method='lee', x_ratio=x_ratio)
    for key in ('hip_start', 'hip_end'):
        lm[key][0] += left_hip.array.shape[0]
    landmarks_final['femur']['Lee']['left'] = lm

    # tibial torsion both sides
    results['tibial_torsion_right'] = calculate_tibial_torsion(left_knee, left_ankle, tibia_label_knee=2, tibia_label_ankle=1, fibula_label=2, side='left')
    landmarks_final['tibia']['right'] = get_tibial_torsion_landmarks(left_knee, left_ankle, tibia_label_knee=2, tibia_label_ankle=1, fibula_label=2, side='left')

    # results dict shape + finiteness
    assert set(results) == {'femoral_torsion_right', 'femoral_torsion_right_murphy',
                            'femoral_torsion_left', 'tibial_torsion_right'}
    for key, value in results.items():
        assert np.isfinite(value), f'{key} is not finite'

    # femoral landmark dicts carry the 4 expected keys as length-3 coordinates
    for method in ('Lee', 'Murphy'):
        lm = landmarks_final['femur'][method]['right']
        assert set(lm) == {'hip_start', 'hip_end', 'knee_start', 'knee_end'}
        assert all(np.asarray(lm[k]).shape == (3,) for k in lm)

    # tibial landmark dict keys
    assert set(landmarks_final['tibia']['right']) == {'knee_start', 'knee_end', 'ankle_start', 'ankle_end'}

    # the right-image-half femoral landmarks were shifted into the right half
    assert landmarks_final['femur']['Lee']['left']['hip_start'][0] >= left_hip.array.shape[0]
