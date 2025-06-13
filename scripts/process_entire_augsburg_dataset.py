import sys
sys.path.append('/home/simon/Work/morphometry')


import pandas as pd
import numpy as np
import nibabel as nib
import traceback
from pathlib import Path
from morphometry.femur import calculate_femoral_torsion
from morphometry.tibia import calculate_tibial_torsion
from morphometry.knee import calculate_knee_rotation_angle
from morphometry.whole_leg import calculate_mikulicz_deviation, calculate_bone_length
from morphometry.ankle import calculate_pma_angle
from morphometry.hip import calculate_ccd
from morphometry.image_io import Segmentation
from matplotlib import pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool


def process_patient(patient):
    try:
        hip = Segmentation('nibabel')
        hip.read_image(f'/home/simon/Data/Augsburg_large/reader_study_augsburg/{patient}/hip_seg_test.nii.gz')
        knee = Segmentation('nibabel')
        knee.read_image(f'/home/simon/Data/Augsburg_large/reader_study_augsburg/{patient}/knee_seg_test.nii.gz')
        ankle = Segmentation('nibabel')
        ankle.read_image(f'/home/simon/Data/Augsburg_large/reader_study_augsburg/{patient}/ankle_seg_test.nii.gz')
    except FileNotFoundError as e:
        print(f'Patient {patient} could not be found.')
        return {'patient': patient}

    hip.transform_coordinate_system()
    knee.transform_coordinate_system()
    ankle.transform_coordinate_system()

    hip.remove_outliers()
    knee.remove_outliers()
    ankle.remove_outliers()

    x_ratio = abs(hip.spacing[2]) / 2 * abs(hip.spacing[0])

    hip_mask = hip.array
    knee_mask = knee.array
    ankle_mask = ankle.array

    left_hip = hip_mask[:hip_mask.shape[0] // 2]
    right_hip = hip_mask[hip_mask.shape[0] // 2:]
    left_knee = knee_mask[:knee_mask.shape[0] // 2]
    right_knee = knee_mask[knee_mask.shape[0] // 2:]
    left_ankle = ankle_mask[:ankle_mask.shape[0] // 2]
    right_ankle = ankle_mask[ankle_mask.shape[0] // 2:]

    left_hip = nib.Nifti1Image(left_hip, hip.affine, hip.header)
    left_hip = Segmentation.from_nibabel(left_hip)
    right_hip = nib.Nifti1Image(right_hip, hip.affine, hip.header)
    right_hip = Segmentation.from_nibabel(right_hip)
    left_knee = nib.Nifti1Image(left_knee, knee.affine, knee.header)
    left_knee = Segmentation.from_nibabel(left_knee)
    right_knee = nib.Nifti1Image(right_knee, knee.affine, knee.header)
    right_knee = Segmentation.from_nibabel(right_knee)
    left_ankle = nib.Nifti1Image(left_ankle, ankle.affine, ankle.header)
    left_ankle = Segmentation.from_nibabel(left_ankle)
    right_ankle = nib.Nifti1Image(right_ankle, ankle.affine, ankle.header)
    right_ankle = Segmentation.from_nibabel(right_ankle)

    try:
        at_lee_left, fig = calculate_femoral_torsion(left_hip, left_knee.array, side='left', x_ratio=x_ratio, plot=True)
        fig.savefig(f'/home/simon/Data/Augsburg_large/figures/training_without_finetuning/femoral_torsion/{patient}_lee_left.png')
        plt.close(fig)
    except (ValueError, IndexError, RuntimeError) as e:
        print(patient, f'Failed to calculate femoral torsion (lee) for the left side.', e)
        print(traceback.format_exc())
        at_lee_left = np.nan

    try:
        at_lee_right, fig = calculate_femoral_torsion(right_hip, right_knee.array, side='right', x_ratio=x_ratio, plot=True)
        fig.savefig(f'/home/simon/Data/Augsburg_large/figures/training_without_finetuning/femoral_torsion/{patient}_lee_right.png')
        plt.close(fig)
    except (ValueError, IndexError, RuntimeError) as e:
        print(patient, f'Failed to calculate femoral torsion (lee) for the right side.', e)
        print(traceback.format_exc())
        at_lee_right = np.nan

    try:
        at_murphy_left, fig = calculate_femoral_torsion(left_hip, left_knee.array, 'left', 'murphy', x_ratio=x_ratio,
                                                 plot=True)
        fig.savefig(f'/home/simon/Data/Augsburg_large/figures/training_without_finetuning/femoral_torsion/{patient}_murphy_left.png')
        plt.close(fig)
    except (ValueError, IndexError, RuntimeError) as e:
        print(patient, f'Failed to calculate femoral torsion (murphy) for the left side.', e)
        print(traceback.format_exc())
        at_murphy_left = np.nan

    try:
        at_murphy_right, fig = calculate_femoral_torsion(right_hip, right_knee.array, 'right', 'murphy', x_ratio=x_ratio,
                                                    plot=True)
        fig.savefig(f'/home/simon/Data/Augsburg_large/figures/training_without_finetuning/femoral_torsion/{patient}_murphy_right.png')
        plt.close(fig)
    except (ValueError, IndexError, RuntimeError) as e:
        print(patient, f'Failed to calculate femoral torsion (murphy) for the right side.', e)
        print(traceback.format_exc())
        at_murphy_right = np.nan

    try:
        tt_left, fig = calculate_tibial_torsion(left_knee.array, left_ankle.array, tibia_label_knee=2,
                                           tibia_label_ankle=1,
                                           fibula_label=2, side='left', plot=True)
        fig.savefig(f'/home/simon/Data/Augsburg_large/figures/training_without_finetuning/tibial_torsion/{patient}_tt_left.png')
        plt.close(fig)
    except (ValueError, IndexError, RuntimeError) as e:
        print(patient, f'Failed to calculate tibial torsion for the left side.', e)
        tt_left = np.nan

    try:
        tt_right, fig = calculate_tibial_torsion(right_knee.array, right_ankle.array, tibia_label_knee=2,
                                            tibia_label_ankle=1, fibula_label=2, side='right',
                                            plot=True)
        fig.savefig(f'/home/simon/Data/Augsburg_large/figures/training_without_finetuning/tibial_torsion/{patient}_tt_right.png')
        plt.close(fig)
    except (ValueError, IndexError, RuntimeError) as e:
        print(patient, f'Failed to calculate tibial torsion for the right side.', e)
        tt_right = np.nan

    fig, ax = plt.subplots(ncols=2, figsize=(20, 10))

    try:
        ccd_left_actual, ccd_left_projected = calculate_ccd(left_hip, left_knee, 'left', 1, False, x_ratio, plot=ax[0])
        ccd_left = (ccd_left_actual, ccd_left_projected)
    except (ValueError, IndexError, RuntimeError) as e:
        print(patient, f'Failed to calculate CCD for the left side.', e)
        ccd_left = (np.nan, np.nan)

    try:
        ccd_left_actual, ccd_left_projected = calculate_ccd(right_hip, right_knee, 'right', 1, False, x_ratio, plot=ax[1])
        ccd_right = (ccd_left_actual, ccd_left_projected)
    except (ValueError, IndexError, RuntimeError) as e:
        print(patient, f'Failed to calculate CCD for the right side.', e)
        ccd_right = (np.nan, np.nan)

    ax[0].set_title(f'CCD right: {ccd_left[0]:.1f}°')
    ax[1].set_title(f'CCD left: {ccd_right[0]:.1f}°')
    fig.savefig(f'/home/simon/Data/Augsburg_large/figures/training_without_finetuning/ccd/{patient}_ccd.png')
    plt.close(fig)

    try:
        kra_left, fig = calculate_knee_rotation_angle(left_knee.array, 1, 2, 'left', True)
        fig.savefig(f'/home/simon/Data/Augsburg_large/figures/training_without_finetuning/kra/{patient}_kra_left.png')
        plt.close(fig)
    except (ValueError, IndexError, RuntimeError) as e:
        print(patient, f'Failed to calculate KRA for the left side.', e)
        kra_left = np.nan

    try:
        kra_right, fig = calculate_knee_rotation_angle(right_knee.array, 1, 2, 'right', True)
        fig.savefig(f'/home/simon/Data/Augsburg_large/figures/training_without_finetuning/kra/{patient}_kra_right.png')
        plt.close(fig)
    except (ValueError, IndexError, RuntimeError) as e:
        print(patient, f'Failed to calculate KRA for the right side.', e)
        kra_right = np.nan

    try:
        ll_left = calculate_bone_length(left_hip, left_ankle)
    except (ValueError, IndexError, RuntimeError) as e:
        print(patient, f'Failed to calculate leg length for the left side.', e)
        ll_left = np.nan

    try:
        ll_right = calculate_bone_length(right_hip, right_ankle)
    except (ValueError, IndexError, RuntimeError) as e:
        print(patient, f'Failed to calculate leg length for the right side.', e)
        ll_right = np.nan

    try:
        fl_left = calculate_bone_length(left_hip, left_knee)
    except (ValueError, IndexError, RuntimeError) as e:
        print(patient, f'Failed to calculate femur length for the left side.', e)
        fl_left = np.nan

    try:
        fl_right = calculate_bone_length(right_hip, right_knee)
    except (ValueError, IndexError, RuntimeError) as e:
        print(patient, f'Failed to calculate femur length for the right side.', e)
        fl_right = np.nan

    try:
        tl_left = calculate_bone_length(left_knee, left_ankle)
    except (ValueError, IndexError, RuntimeError) as e:
        print(patient, f'Failed to calculate tibia length for the left side.', e)
        tl_left = np.nan

    try:
        tl_right = calculate_bone_length(right_knee, right_ankle)
    except (ValueError, IndexError, RuntimeError) as e:
        print(patient, f'Failed to calculate tibia length for the right side.', e)
        tl_right = np.nan

    try:
        mld_left = calculate_mikulicz_deviation(left_hip, left_knee, left_ankle, 'left', x_ratio=x_ratio)
    except (ValueError, IndexError, RuntimeError) as e:
        print(patient, f'Failed to calculate MLD for the left side.', e)
        mld_left = np.nan

    try:
        mld_right = calculate_mikulicz_deviation(right_hip, right_knee, right_ankle, 'right', x_ratio=x_ratio)
    except (ValueError, IndexError, RuntimeError) as e:
        print(patient, f'Failed to calculate MLD for the right side.', e)
        mld_right = np.nan

    return {'patient': patient, 'at_lee_left': at_lee_left, 'at_lee_right': at_lee_right, 'at_murphy_left': at_murphy_left, 'at_murphy_right': at_murphy_right, 'tt_left': tt_left, 'tt_right': tt_right, 'ccd_left': ccd_left, 'ccd_right': ccd_right,
            'kra_left': kra_left, 'kra_right': kra_right, 'll_left': ll_left, 'll_right': ll_right, 'fl_left': fl_left, 'fl_right': fl_right, 'tl_left': tl_left, 'tl_right': tl_right, 'mld_left': mld_left, 'mld_right': mld_right}


if __name__ == '__main__':
    plot = False
    patients = [x.name for x in Path('/home/simon/Data/Augsburg_large/reader_study_augsburg/').iterdir()]

    with Pool() as pool:
        res = pool.map(process_patient, patients)

    index = pd.MultiIndex.from_product([patients, ['right', 'left']], names=['Patient', 'Side'])
    df = pd.DataFrame(columns=['CCD (actual)', 'CCD (projected)', 'AT (Lee)', 'AT (Murphy)', 'TT', 'KRA', 'LL', 'FL', 'TL', 'LLD', 'MLD'], index=index)

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
        df.loc[(patient, 'left'), 'LL'] = r['ll_right']
        df.loc[(patient, 'right'), 'LL'] = r['ll_left']
        df.loc[(patient, 'left'), 'FL'] = r['fl_right']
        df.loc[(patient, 'right'), 'FL'] = r['fl_left']
        df.loc[(patient, 'left'), 'TL'] = r['tl_right']
        df.loc[(patient, 'right'), 'TL'] = r['tl_left']
        df.loc[(patient, 'left'), 'LLD'] = r['ll_right'] - r['ll_left']
        df.loc[(patient, 'right'), 'LLD'] = r['ll_right'] - r['ll_left']  # convention: if left patient leg is longer, LLD is positive
        df.loc[(patient, 'left'), 'MLD'] = r['mld_right']
        df.loc[(patient, 'right'), 'MLD'] = r['mld_left']

    df = df.apply(lambda x: np.round(x, 1))
    df.to_excel('/home/simon/Data/Augsburg_large/results_no_finetuning.xlsx')
    print(f'{df.shape[0] - df.dropna().shape[0]} patients have missing values.')
