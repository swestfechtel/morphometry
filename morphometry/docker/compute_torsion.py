import traceback
import json

import nibabel as nib
import numpy as np

from morphometry.femur import get_lee_neck_qc, get_murphy_neck_qc
from morphometry.knee import get_condylar_line_qc
from morphometry.image_io import Segmentation, split_image_left_right
from morphometry.measurements.femur import calculate_femoral_torsion, get_femoral_torsion_landmarks
from morphometry.measurements.tibia import calculate_tibial_torsion, get_tibial_torsion_landmarks

from collections import defaultdict

if __name__ == '__main__':
    hip_mask = Segmentation.from_nibabel(nib.load('/app/temp/hip_segmentation.nii.gz'))
    knee_mask = Segmentation.from_nibabel(nib.load('/app/temp/knee_segmentation.nii.gz'))
    ankle_mask = Segmentation.from_nibabel(nib.load('/app/temp/ankle_segmentation.nii.gz'))

    hip_mask.transform_coordinate_system()
    knee_mask.transform_coordinate_system()
    ankle_mask.transform_coordinate_system()

    x_ratio = abs(hip_mask.spacing[2]) / 2 * abs(hip_mask.spacing[0])

    hip_mask.remove_outliers()
    knee_mask.remove_outliers()
    ankle_mask.remove_outliers()

    # the hip halves are used through their geometry, so split them with the correct per-half affine
    left_hip, right_hip = split_image_left_right(hip_mask)
    left_knee = knee_mask.array[:knee_mask.array.shape[0] // 2]
    right_knee = knee_mask.array[knee_mask.array.shape[0] // 2:]
    left_ankle = ankle_mask.array[:ankle_mask.array.shape[0] // 2]
    right_ankle = ankle_mask.array[ankle_mask.array.shape[0] // 2:]

    # The knee and ankle halves are bare arrays, so they carry no geometry of their own; pass the
    # parent images' spacing so the torsion measurements can check the equal-in-plane-spacing
    # precondition of their index-space angle composition (the hip halves are Images and are
    # checked automatically). An in-plane anisotropic acquisition is refused with a ValueError,
    # which the per-measurement handlers below record as an error instead of aborting the run.
    knee_spacing = knee_mask.spacing
    ankle_spacing = ankle_mask.spacing

    results = dict()
    landmarks_final = {
        'femur': {
            'Lee': {
                'left': defaultdict(list),
                'right': defaultdict(list)
            },
            'Murphy': {
                'left': defaultdict(list),
                'right': defaultdict(list)
            }
        },
        'tibia': {
            'left': defaultdict(list),
            'right': defaultdict(list)
        }
    }
    errors = list()

    try:
        torsion = calculate_femoral_torsion(left_hip, left_knee, side='left', method='lee', x_ratio=x_ratio,
                                            knee_spacing=knee_spacing)
        landmarks = get_femoral_torsion_landmarks(left_hip, left_knee, side='left', method='lee', x_ratio=x_ratio,
                                                  knee_spacing=knee_spacing)
        results['femoral_torsion_right'] = torsion

        for k, v in landmarks.items():
            landmarks[k] = v.tolist()

        landmarks_final['femur']['Lee']['right'] = landmarks

    except (RuntimeError, AssertionError, ValueError) as e:
        errors.append(traceback.format_exc())
        results['femoral_torsion_right'] = np.nan
        landmarks_final['femur']['Lee']['right']['hip_start'] = [0, 0, 0]
        landmarks_final['femur']['Lee']['right']['hip_end'] = [0, 0, 0]
        landmarks_final['femur']['Lee']['right']['knee_start'] = [0, 0, 0]
        landmarks_final['femur']['Lee']['right']['knee_end'] = [0, 0, 0]

    try:
        torsion = calculate_femoral_torsion(left_hip, left_knee, side='left', method='murphy', x_ratio=x_ratio,
                                            knee_spacing=knee_spacing)
        landmarks = get_femoral_torsion_landmarks(left_hip, left_knee, side='left', method='murphy', x_ratio=x_ratio,
                                                  knee_spacing=knee_spacing)
        results['femoral_torsion_right_murphy'] = torsion

        for k, v in landmarks.items():
            landmarks[k] = v.tolist()

        landmarks_final['femur']['Murphy']['right'] = landmarks

    except (RuntimeError, AssertionError, ValueError) as e:
        errors.append(traceback.format_exc())
        results['femoral_torsion_right_murphy'] = np.nan
        landmarks_final['femur']['Murphy']['right']['hip_start'] = [0, 0, 0]
        landmarks_final['femur']['Murphy']['right']['hip_end'] = [0, 0, 0]
        landmarks_final['femur']['Murphy']['right']['knee_start'] = [0, 0, 0]
        landmarks_final['femur']['Murphy']['right']['knee_end'] = [0, 0, 0]

    # QC of the two proximal femoral-neck landmarks, in the ';'-joined name:value format the batch
    # scripts use: morphometry.femur.murphy_neck_base_qc for Murphy's neck base and
    # morphometry.femur.lee_neck_axis_qc for Lee's neck ring. Empty when the landmark looks sound,
    # otherwise the corresponding femoral_torsion_* is still reported but should not be trusted
    # without looking at the mask. These keys are written to results.json but NOTHING IN THE API
    # READS THEM: api/schemas/docker_io.py::ResultsModel has extra="ignore", and
    # api/tasks/torsion.py reads results.json inside a TemporaryDirectory and persists only
    # model_dump(), while the Job record carries no result payload at all -- so on the API path the
    # flags are computed and discarded, and they never reach the UI. They are only visible when
    # this container is run by hand with a mounted output dir. Surfacing them in GET /jobs/{id}
    # needs an API schema and DB change.
    # Neither get_murphy_neck_qc nor get_lee_neck_qc ever raises (both report a failed landmark as
    # a flag), which is why these calls can sit outside the try blocks without risking the whole
    # container.
    # The Lee key carries an explicit '_lee_' although the Lee ANGLE's key does not (it is the bare
    # 'femoral_torsion_<side>', Lee being the default method): naming it
    # 'femoral_torsion_<side>_qc_flags' would read as the row's QC rather than as one landmark's,
    # and the pair of keys is clearer side by side.
    results['femoral_torsion_right_murphy_qc_flags'] = get_murphy_neck_qc(left_hip, side='left', x_ratio=x_ratio)
    results['femoral_torsion_right_lee_qc_flags'] = get_lee_neck_qc(left_hip, side='left', x_ratio=x_ratio)

    try:
        torsion = calculate_femoral_torsion(right_hip, right_knee, side='right', method='lee', x_ratio=x_ratio,
                                            knee_spacing=knee_spacing)
        landmarks = get_femoral_torsion_landmarks(right_hip, right_knee, side='right', method='lee', x_ratio=x_ratio,
                                                  knee_spacing=knee_spacing)
        landmarks['hip_start'][0] += left_hip.array.shape[0]  # shift to the right image side
        landmarks['hip_end'][0] += left_hip.array.shape[0]
        landmarks['knee_start'][0] += left_knee.shape[0]
        landmarks['knee_end'][0] += left_knee.shape[0]

        results['femoral_torsion_left'] = torsion

        for k, v in landmarks.items():
            landmarks[k] = v.tolist()

        landmarks_final['femur']['Lee']['left'] = landmarks

    except (RuntimeError, AssertionError, ValueError) as e:
        errors.append(traceback.format_exc())
        results['femoral_torsion_left'] = np.nan
        landmarks_final['femur']['Lee']['left']['hip_start'] = [0, 0, 0]
        landmarks_final['femur']['Lee']['left']['hip_end'] = [0, 0, 0]
        landmarks_final['femur']['Lee']['left']['knee_start'] = [0, 0, 0]
        landmarks_final['femur']['Lee']['left']['knee_end'] = [0, 0, 0]

    try:
        torsion = calculate_femoral_torsion(right_hip, right_knee, side='right', method='murphy', x_ratio=x_ratio,
                                            knee_spacing=knee_spacing)
        landmarks = get_femoral_torsion_landmarks(right_hip, right_knee, side='right', method='murphy', x_ratio=x_ratio,
                                                  knee_spacing=knee_spacing)
        landmarks['hip_start'][0] += left_hip.array.shape[0]  # shift to the right image side
        landmarks['hip_end'][0] += left_hip.array.shape[0]
        landmarks['knee_start'][0] += left_knee.shape[0]
        landmarks['knee_end'][0] += left_knee.shape[0]

        results['femoral_torsion_left_murphy'] = torsion

        for k, v in landmarks.items():
            landmarks[k] = v.tolist()

        landmarks_final['femur']['Murphy']['left'] = landmarks

    except (RuntimeError, AssertionError, ValueError) as e:
        errors.append(traceback.format_exc())
        results['femoral_torsion_left_murphy'] = np.nan
        landmarks_final['femur']['Murphy']['left']['hip_start'] = [0, 0, 0]
        landmarks_final['femur']['Murphy']['left']['hip_end'] = [0, 0, 0]
        landmarks_final['femur']['Murphy']['left']['knee_start'] = [0, 0, 0]
        landmarks_final['femur']['Murphy']['left']['knee_end'] = [0, 0, 0]

    results['femoral_torsion_left_murphy_qc_flags'] = get_murphy_neck_qc(right_hip, side='right', x_ratio=x_ratio)
    results['femoral_torsion_left_lee_qc_flags'] = get_lee_neck_qc(right_hip, side='right', x_ratio=x_ratio)

    try:
        torsion = calculate_tibial_torsion(left_knee, left_ankle, tibia_label_knee=2, tibia_label_ankle=1, fibula_label=2,
                                           side='left',
                                           knee_spacing=knee_spacing, ankle_spacing=ankle_spacing)
        landmarks = get_tibial_torsion_landmarks(left_knee, left_ankle, tibia_label_knee=2, tibia_label_ankle=1, fibula_label=2,
                                                 side='left',
                                                 knee_spacing=knee_spacing, ankle_spacing=ankle_spacing)
        results['tibial_torsion_right'] = torsion

        for k, v in landmarks.items():
            landmarks[k] = v.tolist()

        landmarks_final['tibia']['right'] = landmarks

    except (RuntimeError, AssertionError, ValueError) as e:
        errors.append(traceback.format_exc())
        results['tibial_torsion_right'] = np.nan
        landmarks_final['tibia']['right']['knee_start'] = [0, 0, 0]
        landmarks_final['tibia']['right']['knee_end'] = [0, 0, 0]
        landmarks_final['tibia']['right']['ankle_start'] = [0, 0, 0]
        landmarks_final['tibia']['right']['ankle_end'] = [0, 0, 0]

    try:
        torsion = calculate_tibial_torsion(right_knee, right_ankle, tibia_label_knee=2, tibia_label_ankle=1, fibula_label=2,
                                           side='right',
                                           knee_spacing=knee_spacing, ankle_spacing=ankle_spacing)
        landmarks = get_tibial_torsion_landmarks(right_knee, right_ankle, tibia_label_knee=2, tibia_label_ankle=1, fibula_label=2,
                                                 side='right',
                                                 knee_spacing=knee_spacing, ankle_spacing=ankle_spacing)
        landmarks['knee_start'][0] += left_knee.shape[0]
        landmarks['knee_end'][0] += left_knee.shape[0]
        landmarks['ankle_start'][0] += left_ankle.shape[0]
        landmarks['ankle_end'][0] += left_ankle.shape[0]

        results['tibial_torsion_left'] = torsion

        for k, v in landmarks.items():
            landmarks[k] = v.tolist()

        landmarks_final['tibia']['left'] = landmarks

    except (RuntimeError, AssertionError, ValueError) as e:
        errors.append(traceback.format_exc())
        results['tibial_torsion_left'] = np.nan
        landmarks_final['tibia']['left']['knee_start'] = [0, 0, 0]
        landmarks_final['tibia']['left']['knee_end'] = [0, 0, 0]
        landmarks_final['tibia']['left']['ankle_start'] = [0, 0, 0]
        landmarks_final['tibia']['left']['ankle_end'] = [0, 0, 0]

    # QC of the two posterior condylar reference lines, one per bone and image half. Only the
    # in-plane DIRECTION of each line enters the angles, so a short line is badly conditioned in
    # the one quantity that is used: one voxel of landmark error turns a 9.5-voxel line by 6
    # degrees. The femoral line is read by both femoral_torsion_* methods, the tibial one by
    # tibial_torsion_*, and a flagged line keeps every one of those values -- the flag only marks
    # the row as needing a look at the mask.
    #
    # The values are in in-plane VOXELS, where the Murphy and Lee flags above are in millimetres
    # and in head radii respectively; see morphometry.knee.knee_condylar_line_qc for why (a
    # millimetre value is not monotone in the verdict across in-plane spacings).
    #
    # The knee halves here are bare arrays, so the parent's spacing is passed explicitly, exactly
    # as it is to the torsion measurements above -- otherwise the in-plane isotropy precondition
    # the line length rests on could not be checked (backlog task #10).
    #
    # Keys are the PATIENT side, like every other key here, so the image-left half carries the
    # '_right' name. get_condylar_line_qc never raises (a failed search is reported as a flag,
    # including the bare IndexError get_knee_reference_line throws on an empty bone label, which
    # is NOT in this module's (RuntimeError, AssertionError, ValueError) handler tuple), which is
    # why these calls can sit outside the try blocks without risking the whole container.
    #
    # As with the Murphy and Lee flags: NOTHING IN THE API READS THESE KEYS (ResultsModel has
    # extra="ignore" and api/tasks/torsion.py persists only model_dump()), so on the API path they
    # are computed and discarded. They are visible only when this container is run by hand with a
    # mounted output dir.
    for patient_side, knee_half, image_side in (('right', left_knee, 'left'),
                                                ('left', right_knee, 'right')):
        for bone, label in (('femur', 1), ('tibia', 2)):
            results[f'{bone}_condylar_line_{patient_side}_qc_flags'] = get_condylar_line_qc(
                knee_half, bone, segmentation_label=label, side=image_side,
                spacing=knee_spacing)

    errors = {'errors': errors}

    json.dump(errors, open('/app/temp/errors.json', 'w'))
    json.dump(results, open('/app/temp/results.json', 'w'))
    json.dump(landmarks_final, open('/app/temp/landmarks.json', 'w'))

    print(errors)
