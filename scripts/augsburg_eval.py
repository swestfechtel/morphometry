import re
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from morphometry.femur import calculate_femoral_torsion
from morphometry.tibia import calculate_tibial_torsion
from morphometry.knee import calculate_knee_rotation_angle
from morphometry.whole_leg import calculate_mikulicz_deviation
from morphometry.ankle import calculate_pma_angle
from morphometry.hip import calculate_ccd


if __name__ == '__main__':
    r = re.compile(r'PA\d+')
    for file in Path('/home/simon/Downloads/Augsburg_2/huefte').iterdir():
        patient = r.search(file.name)[0]
        try:
            hip = sitk.ReadImage(f'/home/simon/Data/nnUnet_raw/Dataset001_AugsburgHip/labelsTr/t1_tse_tra_Huften_bds_{patient}.nii.gz')
            knee = sitk.ReadImage(f'/home/simon/Data/nnUnet_raw/Dataset002_AugsburgKnee/labelsTr/t1_tse_tra_Knie_{patient}.nii.gz')
            ankle = sitk.ReadImage(f'/home/simon/Data/nnUnet_raw/Dataset003_AugsburgAnkle/labelsTr/t1_tse_tra_OSG_{patient}.nii.gz')
        except RuntimeError as e:
            print(f'Patient {patient} could not be found.', e)
            continue

        x_ratio = abs(hip.GetSpacing()[2]) / 2 * abs(hip.GetSpacing()[0])

        hip_mask = sitk.GetArrayFromImage(hip)
        knee_mask = sitk.GetArrayFromImage(knee)
        ankle_mask = sitk.GetArrayFromImage(ankle)

        hip_mask = hip_mask[::-1]
        hip_mask = hip_mask[:, ::-1]
        knee_mask = knee_mask[::-1]
        ankle_mask = ankle_mask[::-1]

        print(hip_mask.shape, knee_mask.shape, ankle_mask.shape)

        left_hip = hip_mask[:, :, :hip_mask.shape[2] // 2]
        right_hip = hip_mask[:, :, hip_mask.shape[2] // 2:]
        left_knee = knee_mask[:, :, :knee_mask.shape[2] // 2]
        right_knee = knee_mask[:, :, knee_mask.shape[2] // 2:]
        left_ankle = ankle_mask[:, :, :ankle_mask.shape[2] // 2]
        right_ankle = ankle_mask[:, :, ankle_mask.shape[2] // 2:]

        try:
            femoral_torsion_left = calculate_femoral_torsion(left_hip, left_knee, side='left', x_ratio=x_ratio, plot=False)
            femoral_torsion_right = calculate_femoral_torsion(right_hip, right_knee, side='right', x_ratio=x_ratio,
                                                              plot=False)

            tibial_torsion_left = calculate_tibial_torsion(left_knee, left_ankle, tibia_label_knee=2, tibia_label_ankle=1,
                                                           fibula_label=2, side='left', plot=False)
            tibial_torsion_right = calculate_tibial_torsion(right_knee, right_ankle, tibia_label_knee=2,
                                                            tibia_label_ankle=1, fibula_label=2, side='right', plot=False)
        except (ValueError, IndexError) as e:
            print(f'Patient {patient} could not be processed.', e)
            continue

        print(f'Patient {patient}:')
        print(f'Femoral torsion (right patient side): {femoral_torsion_left}°')
        print(f'Femoral torsion (left patient side): {femoral_torsion_right}°')
        print(f'Tibial torsion (right patient side): {tibial_torsion_left}°')
        print(f'Tibial torsion (left patient side): {tibial_torsion_right}°')
        print('--------------------------------------------------')