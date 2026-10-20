"""End-to-end characterization of the docker torsion pipeline.

Mirrors morphometry/docker/compute_torsion.py exactly (load -> LPI -> remove
outliers -> split L/R -> femoral Lee/Murphy + tibial torsion per side, with the
right-side coordinate shift) and asserts the results dict and the nested landmark
structure the REST layer consumes. Guards the swethfechtel/torsion:latest rebuild.
"""
import numpy as np
import pytest

from morphometry.image_io import split_image_left_right
from morphometry.measurements.femur import calculate_femoral_torsion, get_femoral_torsion_landmarks
from morphometry.measurements.tibia import calculate_tibial_torsion, get_tibial_torsion_landmarks

pytestmark = pytest.mark.needs_augsburg


def test_compute_torsion_pipeline(augsburg_pa000001):
    hip_mask, knee_mask, ankle_mask = (augsburg_pa000001[k] for k in ("hip", "knee", "ankle"))
    x_ratio = abs(hip_mask.spacing[2]) / 2 * abs(hip_mask.spacing[0])

    left_hip, right_hip = split_image_left_right(hip_mask)
    left_knee = knee_mask.array[:knee_mask.array.shape[0] // 2]
    right_knee = knee_mask.array[knee_mask.array.shape[0] // 2:]
    left_ankle = ankle_mask.array[:ankle_mask.array.shape[0] // 2]
    right_ankle = ankle_mask.array[ankle_mask.array.shape[0] // 2:]
    # the pipeline hands the bare knee/ankle halves their parent image's spacing so the
    # measurements can check the equal-in-plane-spacing precondition of the index-space angle
    knee_spacing, ankle_spacing = knee_mask.spacing, ankle_mask.spacing

    results = {}
    landmarks_final = {'femur': {'Lee': {}, 'Murphy': {}}, 'tibia': {}}

    # femoral, image-left half (stored as patient-right)
    results['femoral_torsion_right'] = calculate_femoral_torsion(left_hip, left_knee, side='left', method='lee', x_ratio=x_ratio,
                                                                 knee_spacing=knee_spacing)
    landmarks_final['femur']['Lee']['right'] = get_femoral_torsion_landmarks(left_hip, left_knee, side='left', method='lee', x_ratio=x_ratio,
                                                                             knee_spacing=knee_spacing)
    results['femoral_torsion_right_murphy'] = calculate_femoral_torsion(left_hip, left_knee, side='left', method='murphy', x_ratio=x_ratio,
                                                                        knee_spacing=knee_spacing)
    landmarks_final['femur']['Murphy']['right'] = get_femoral_torsion_landmarks(left_hip, left_knee, side='left', method='murphy', x_ratio=x_ratio,
                                                                                knee_spacing=knee_spacing)

    # femoral, image-right half (stored as patient-left), with the sagittal shift
    results['femoral_torsion_left'] = calculate_femoral_torsion(right_hip, right_knee, side='right', method='lee', x_ratio=x_ratio,
                                                                knee_spacing=knee_spacing)
    lm = get_femoral_torsion_landmarks(right_hip, right_knee, side='right', method='lee', x_ratio=x_ratio,
                                       knee_spacing=knee_spacing)
    for key in ('hip_start', 'hip_end'):
        lm[key][0] += left_hip.array.shape[0]
    landmarks_final['femur']['Lee']['left'] = lm

    # tibial torsion both sides
    results['tibial_torsion_right'] = calculate_tibial_torsion(left_knee, left_ankle, tibia_label_knee=2, tibia_label_ankle=1, fibula_label=2, side='left',
                                                               knee_spacing=knee_spacing, ankle_spacing=ankle_spacing)
    landmarks_final['tibia']['right'] = get_tibial_torsion_landmarks(left_knee, left_ankle, tibia_label_knee=2, tibia_label_ankle=1, fibula_label=2, side='left',
                                                                     knee_spacing=knee_spacing, ankle_spacing=ankle_spacing)

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
