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
from multiprocessing import Pool


def process_patient(patient):
    try:
        hip = sitk.ReadImage(f'/home/simon/Data/Augsburg_large/preprocessed/{patient}/hip_seg.nii.gz')
        knee = sitk.ReadImage(f'/home/simon/Data/Augsburg_large/preprocessed/{patient}/knee_seg.nii.gz')
        ankle = sitk.ReadImage(f'/home/simon/Data/Augsburg_large/preprocessed/{patient}/ankle_seg.nii.gz')
    except RuntimeError as e:
        print(f'Patient {patient} could not be found.')
        return {'patient': patient}

    try:
        hip = correct_axis_ordering(hip)
        knee = correct_axis_ordering(knee)
        ankle = correct_axis_ordering(ankle)
    except (RuntimeError, AssertionError) as e:
        print(f'Patient {patient} could not be processed.', e)
        return {'patient': patient}

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
        at_lee_left, fig = calculate_femoral_torsion(left_hip, left_knee, side='left', x_ratio=x_ratio, plot=True)
        fig.savefig(f'/home/simon/Data/Augsburg_large/figures/{patient}_lee_left.png')
        plt.close(fig)
    except (ValueError, IndexError, RuntimeError):
        at_lee_left = np.nan

    try:
        at_lee_right, fig = calculate_femoral_torsion(right_hip, right_knee, side='right', x_ratio=x_ratio, plot=True)
        fig.savefig(f'/home/simon/Data/Augsburg_large/figures/{patient}_lee_right.png')
        plt.close(fig)
    except (ValueError, IndexError, RuntimeError):
        at_lee_right = np.nan

    try:
        at_murphy_left, fig = calculate_femoral_torsion(left_hip, left_knee, 'left', 'murphy', x_ratio=x_ratio,
                                                   hip_image=hip, plot=True)
        fig.savefig(f'/home/simon/Data/Augsburg_large/figures/{patient}_murphy_left.png')
        plt.close(fig)
    except (ValueError, IndexError, RuntimeError):
        at_murphy_left = np.nan

    try:
        at_murphy_right, fig = calculate_femoral_torsion(right_hip, right_knee, 'right', 'murphy', x_ratio=x_ratio,
                                                    hip_image=hip, plot=True)
        fig.savefig(f'/home/simon/Data/Augsburg_large/figures/{patient}_murphy_right.png')
        plt.close(fig)
    except (ValueError, IndexError, RuntimeError):
        at_murphy_right = np.nan

    try:
        tt_left = calculate_tibial_torsion(left_knee, left_ankle, tibia_label_knee=2,
                                           tibia_label_ankle=1,
                                           fibula_label=2, side='left', plot=False)
    except (ValueError, IndexError, RuntimeError):
        tt_left = np.nan

    try:
        tt_right = calculate_tibial_torsion(right_knee, right_ankle, tibia_label_knee=2,
                                            tibia_label_ankle=1, fibula_label=2, side='right',
                                            plot=False)
    except (ValueError, IndexError, RuntimeError):
        tt_right = np.nan

    try:
        ccd_left = calculate_ccd(left_hip, 'left', 1, False, x_ratio)
    except (ValueError, IndexError, RuntimeError):
        ccd_left = (np.nan, np.nan)

    try:
        ccd_right = calculate_ccd(right_hip, 'right', 1, False, x_ratio)
    except (ValueError, IndexError, RuntimeError):
        ccd_right = (np.nan, np.nan)

    try:
        kra_left = calculate_knee_rotation_angle(left_knee, 1, 2, False)
    except (ValueError, IndexError, RuntimeError):
        kra_left = np.nan

    try:
        kra_right = calculate_knee_rotation_angle(right_knee, 1, 2, False)
    except (ValueError, IndexError, RuntimeError):
        kra_right = np.nan

    return {'patient': patient, 'at_lee_left': at_lee_left, 'at_lee_right': at_lee_right, 'at_murphy_left': at_murphy_left, 'at_murphy_right': at_murphy_right, 'tt_left': tt_left, 'tt_right': tt_right, 'ccd_left': ccd_left, 'ccd_right': ccd_right, 'kra_left': kra_left, 'kra_right': kra_right}


if __name__ == '__main__':
    plot = False
    patients = [x.name for x in Path('/home/simon/Data/Augsburg_large/preprocessed/').iterdir()]

    with Pool() as pool:
        res = pool.map(process_patient, patients)

    index = pd.MultiIndex.from_product([patients, ['right', 'left']], names=['Patient', 'Side'])
    df = pd.DataFrame(columns=['CCD (actual)', 'CCD (projected)', 'AT (Lee)', 'AT (Murphy)', 'TT', 'KRA'], index=index)

    for r in tqdm(res):
        patient = r['patient']
        if not 'at_lee_left' in r.keys():
            continue

        df.loc[(patient, 'left'), 'CCD (actual)'] = r['ccd_right'][0]
        df.loc[(patient, 'right'), 'CCD (actual)'] = r['ccd_left'][0]
        df.loc[(patient, 'left'), 'CCD (projected)'] = r['ccd_right'][1]
        df.loc[(patient, 'right'), 'CCD (projected)'] = r['ccd_left'][1]
        df.loc[(patient, 'left'), 'AT (Lee)'] = r['at_lee_right']
        df.loc[(patient, 'right'), 'AT (Lee)'] = r['at_lee_left']
        df.loc[(patient, 'left'), 'AT (Murphy)'] = r['at_murphy_right']
        df.loc[(patient, 'right'), 'AT (Murphy)'] = r['at_murphy_left']
        df.loc[(patient, 'left'), 'TT'] = r['tt_right']
        df.loc[(patient, 'right'), 'TT'] = r['tt_left']
        df.loc[(patient, 'left'), 'KRA'] = r['kra_right']
        df.loc[(patient, 'right'), 'KRA'] = r['kra_left']

    df = df.apply(lambda x: np.round(x, 1))
    df.to_excel('/home/simon/Data/Augsburg_large/results.xlsx')
    print(f'{df.shape[0] - df.dropna().shape[0]} patients have missing values.')
