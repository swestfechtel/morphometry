import sys
sys.path.append('/home/simon/Work/morpohmetry')

import SimpleITK as sitk
import pandas as pd
import numpy as np
from pathlib import Path
from morphometry.femur import calculate_femoral_torsion
from morphometry.tibia import calculate_tibial_torsion
from morphometry.knee import calculate_knee_rotation_angle
from morphometry.whole_leg import calculate_mikulicz_deviation
from morphometry.ankle import calculate_pma_angle
from morphometry.hip import calculate_ccd
from morphometry.utils import correct_axis_ordering
from matplotlib import pyplot as plt
from tqdm import tqdm


if __name__ == '__main__':
    plot = False
    patients = [x.name for x in Path('/home/simon/Data/Augsburg_large/preprocessed/').iterdir()]
    index = pd.MultiIndex.from_product([patients, ['right', 'left']], names=['patient', 'side'])
    df = pd.DataFrame(columns=['CCD (actual)', 'CCD (projected)', 'AT', 'TT', 'KRA'], index=index)

    for patient in tqdm(patients):
        try:
            hip = sitk.ReadImage(f'/home/simon/Data/Augsburg_large/preprocessed/{patient}/hip_seg.nii.gz')
            knee = sitk.ReadImage(f'/home/simon/Data/Augsburg_large/preprocessed/{patient}/knee_seg.nii.gz')
            ankle = sitk.ReadImage(f'/home/simon/Data/Augsburg_large/preprocessed/{patient}/ankle_seg.nii.gz')
        except RuntimeError as e:
            print(f'Patient {patient} could not be found.')
            continue

        try:
            hip = correct_axis_ordering(hip)
            knee = correct_axis_ordering(knee)
            ankle = correct_axis_ordering(ankle)
        except (RuntimeError, AssertionError) as e:
            print(f'Patient {patient} could not be processed.', e)
            continue

        x_ratio = abs(hip.GetSpacing()[2]) / 2 * abs(hip.GetSpacing()[0])

        hip_mask = sitk.GetArrayFromImage(hip)
        knee_mask = sitk.GetArrayFromImage(knee)
        ankle_mask = sitk.GetArrayFromImage(ankle)

        left_hip = hip_mask[:, :, :hip_mask.shape[2] // 2]
        right_hip = hip_mask[:, :, hip_mask.shape[2] // 2:]
        left_knee = knee_mask[:, :, :knee_mask.shape[2] // 2]
        right_knee = knee_mask[:, :, knee_mask.shape[2] // 2:]
        left_ankle = ankle_mask[:, :, :ankle_mask.shape[2] // 2]
        right_ankle = ankle_mask[:, :, ankle_mask.shape[2] // 2:]

        try:
            if plot:
                femoral_torsion_left, fig = calculate_femoral_torsion(left_hip, left_knee, side='left', x_ratio=x_ratio, plot=plot)
                fig.savefig(f'/home/simon/Downloads/Augsburg/figures/{patient}_at_left.png')
                plt.close(fig)

                femoral_torsion_right, fig = calculate_femoral_torsion(right_hip, right_knee, side='right', x_ratio=x_ratio,
                                                                  plot=plot)
                fig.savefig(f'/home/simon/Downloads/Augsburg/figures/{patient}_at_right.png')
                plt.close(fig)

                tibial_torsion_left, fig = calculate_tibial_torsion(left_knee, left_ankle, tibia_label_knee=2, tibia_label_ankle=1,
                                                               fibula_label=2, side='left', plot=plot)
                fig.savefig(f'/home/simon/Downloads/Augsburg/figures/{patient}_tt_left.png')
                plt.close(fig)

                tibial_torsion_right, fig = calculate_tibial_torsion(right_knee, right_ankle, tibia_label_knee=2,
                                                                tibia_label_ankle=1, fibula_label=2, side='right', plot=plot)
                fig.savefig(f'/home/simon/Downloads/Augsburg/figures/{patient}_tt_right.png')
                plt.close(fig)
            else:
                femoral_torsion_left = calculate_femoral_torsion(left_hip, left_knee, side='left', x_ratio=x_ratio,
                                                                 plot=plot)
                femoral_torsion_right = calculate_femoral_torsion(right_hip, right_knee, side='right', x_ratio=x_ratio,
                                                                  plot=plot)

                tibial_torsion_left = calculate_tibial_torsion(left_knee, left_ankle, tibia_label_knee=2,
                                                               tibia_label_ankle=1,
                                                               fibula_label=2, side='left', plot=plot)
                tibial_torsion_right = calculate_tibial_torsion(right_knee, right_ankle, tibia_label_knee=2,
                                                                tibia_label_ankle=1, fibula_label=2, side='right',
                                                                plot=plot)
            ccd_left = calculate_ccd(left_hip, 'left', 1, False, x_ratio)
            ccd_right = calculate_ccd(right_hip, 'right', 1, False, x_ratio)

            kra_left = calculate_knee_rotation_angle(left_knee, 1, 2, False)
            kra_right = calculate_knee_rotation_angle(right_knee, 1, 2, False)
        except (ValueError, IndexError, RuntimeError) as e:
            print(f'Patient {patient} could not be processed.', e)
            continue

        df.loc[(patient, 'left'), 'CCD (actual)'] = ccd_right[0]
        df.loc[(patient, 'right'), 'CCD (actual)'] = ccd_left[0]
        df.loc[(patient, 'left'), 'CCD (projected)'] = ccd_right[1]
        df.loc[(patient, 'right'), 'CCD (projected)'] = ccd_left[1]
        df.loc[(patient, 'left'), 'AT'] = femoral_torsion_right
        df.loc[(patient, 'right'), 'AT'] = femoral_torsion_left
        df.loc[(patient, 'left'), 'TT'] = tibial_torsion_right
        df.loc[(patient, 'right'), 'TT'] = tibial_torsion_left
        df.loc[(patient, 'left'), 'KRA'] = kra_right
        df.loc[(patient, 'right'), 'KRA'] = kra_left

    df = df.apply(lambda x: np.round(x, 1))
    df.to_excel('/home/simon/Data/Augsburg_large/results.xlsx')
    print(f'{df.shape[0] - df.dropna().shape[0]} patients could not be processed.')
