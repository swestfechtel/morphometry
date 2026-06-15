import traceback
import json

import nibabel as nib
import numpy as np

from morphometry.image_io import Image, Segmentation
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

    left_hip = hip_mask.array[:hip_mask.array.shape[0] // 2]
    right_hip = hip_mask.array[hip_mask.array.shape[0] // 2:]
    left_knee = knee_mask.array[:knee_mask.array.shape[0] // 2]
    right_knee = knee_mask.array[knee_mask.array.shape[0] // 2:]
    left_ankle = ankle_mask.array[:ankle_mask.array.shape[0] // 2]
    right_ankle = ankle_mask.array[ankle_mask.array.shape[0] // 2:]

    left_hip = nib.Nifti1Image(left_hip, hip_mask.affine, hip_mask.header)
    left_hip = Image.from_nibabel(left_hip)
    right_hip = nib.Nifti1Image(right_hip, hip_mask.affine, hip_mask.header)
    right_hip = Image.from_nibabel(right_hip)

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
        torsion = calculate_femoral_torsion(left_hip, left_knee, side='left', method='lee', x_ratio=x_ratio)
        landmarks = get_femoral_torsion_landmarks(left_hip, left_knee, side='left', method='lee', x_ratio=x_ratio)
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
        torsion = calculate_femoral_torsion(left_hip, left_knee, side='left', method='murphy', x_ratio=x_ratio)
        landmarks = get_femoral_torsion_landmarks(left_hip, left_knee, side='left', method='murphy', x_ratio=x_ratio)
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

    try:
        torsion = calculate_femoral_torsion(right_hip, right_knee, side='right', method='lee', x_ratio=x_ratio)
        landmarks = get_femoral_torsion_landmarks(right_hip, right_knee, side='right', method='lee', x_ratio=x_ratio)
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
        torsion = calculate_femoral_torsion(right_hip, right_knee, side='right', method='murphy', x_ratio=x_ratio)
        landmarks = get_femoral_torsion_landmarks(right_hip, right_knee, side='right', method='murphy', x_ratio=x_ratio)
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

    try:
        torsion = calculate_tibial_torsion(left_knee, left_ankle, tibia_label_knee=2, tibia_label_ankle=1, fibula_label=2, side='left')
        landmarks = get_tibial_torsion_landmarks(left_knee, left_ankle, tibia_label_knee=2, tibia_label_ankle=1, fibula_label=2, side='left')
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
        torsion = calculate_tibial_torsion(right_knee, right_ankle, tibia_label_knee=2, tibia_label_ankle=1, fibula_label=2, side='right')
        landmarks = get_tibial_torsion_landmarks(right_knee, right_ankle, tibia_label_knee=2, tibia_label_ankle=1, fibula_label=2, side='right')
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

    errors = {'errors': errors}

    json.dump(errors, open('/app/temp/errors.json', 'w'))
    json.dump(results, open('/app/temp/results.json', 'w'))
    json.dump(landmarks_final, open('/app/temp/landmarks.json', 'w'))

    print(errors)
