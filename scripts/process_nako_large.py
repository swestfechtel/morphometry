import os
import multiprocessing
import sys
import argparse

sys.path.append('/home/sw521914/Work/morphometry')

import pandas as pd
import numpy as np
import nibabel as nib

from pathlib import Path
from morphometry.measurements.hip import calculate_ccd, calculate_anteversion, calculate_acetabular_anteversion, \
    calculate_alpha_angle, calculate_acetabular_depth, calculate_center_edge_angle, \
    calculate_cartilage_thickness_knn, calculate_femoral_offset, calculate_femoral_offset_projected
from morphometry.image_io import Segmentation


def f(patient):
    mask = Segmentation('nibabel')
    mask.read_image(patient)
    mask.transform_coordinate_system()
    mask.remove_outliers()

    mask_left = mask.array[:mask.array.shape[0] // 2]
    mask_right = mask.array[mask.array.shape[0] // 2:]

    mask_left = nib.Nifti1Image(mask_left, mask.affine, mask.header)
    mask_left = Segmentation.from_nibabel(mask_left)
    mask_right = nib.Nifti1Image(mask_right, mask.affine, mask.header)
    mask_right = Segmentation.from_nibabel(mask_right)

    try:
        # ccd_left, _ = calculate_ccd(mask_left, None, 'left', 1, isotropic=True, plot=p)
        _, ccd_left = calculate_ccd(mask_left, None, 'left', 1, isotropic=True, plot=False)  # use projected ccd
    except Exception as e:
        print(f"Error calculating CCD for left side of patient {patient.name}: {e}")
        ccd_left = np.nan

    try:
        # ccd_right, _ = calculate_ccd(mask_right, None, 'right', 1, isotropic=True, plot=p)
        _, ccd_right = calculate_ccd(mask_right, None, 'right', 1, isotropic=True, plot=False)
    except Exception as e:
        print(f"Error calculating CCD for right side of patient {patient.name}: {e}")
        ccd_right = np.nan

    try:
        fat_left = calculate_anteversion(mask_left, 'left', 1, isotropic=True, plot=False)
    except Exception as e:
        print(f"Error calculating FAT for left side of patient {patient.name}: {e}")
        fat_left = np.nan

    try:
        fat_right = calculate_anteversion(mask_right, 'right', 1, isotropic=True, plot=False)
    except Exception as e:
        print(f"Error calculating FAT for right side of patient {patient.name}: {e}")
        fat_right = np.nan

    try:
        aa_left = calculate_alpha_angle(mask_left.array, 'left', 1, isotropic=True, plot=False)
    except Exception as e:
        print(f"Error calculating AA for left side of patient {patient.name}: {e}")
        aa_left = (np.nan, np.nan)
    # calculate_alpha_angle(mask_left.array, 'left', 1, isotropic=True)

    try:
        aa_right = calculate_alpha_angle(mask_right.array, 'right', 1, isotropic=True, plot=False)
    except Exception as e:
        print(f"Error calculating AA for right side of patient {patient.name}: {e}")
        aa_right = (np.nan, np.nan)

    try:
        aav = calculate_acetabular_anteversion(mask, 1, 3, isotropic=False, plot=False)
    except Exception as e:
        print(f"Error calculating AAV for patient {patient.name}: {e}")
        aav = [np.nan, np.nan]

    try:
        cea = calculate_center_edge_angle(mask, 1, 3, project=True, isotropic=True, plot=False)
    except Exception as e:
        print(f"Error calculating CEA for left side of patient {patient.name}: {e}")
        cea = [np.nan, np.nan]

    try:
        offset_left = calculate_femoral_offset_projected(mask_left, None, 'left', 1, isotropic=True)
    except Exception as e:
        print(f"Error calculating femoral offset for left side of patient {patient.name}: {e}")
        offset_left = np.nan

    try:
        offset_right = calculate_femoral_offset_projected(mask_right, None, 'right', 1, isotropic=True)
    except Exception as e:
        print(f"Error calculating femoral offset for right side of patient {patient.name}: {e}")
        offset_right = np.nan

    try:
        cartilage_thickness_left = calculate_cartilage_thickness_knn(mask_left.array, 2)
    except Exception as e:
        print(f"Error calculating cartilage thickness for left side of patient {patient.name}: {e}")
        cartilage_thickness_left = (np.nan, np.nan, np.nan, np.nan)

    try:
        cartilage_thickness_right = calculate_cartilage_thickness_knn(mask_right.array, 2)
    except Exception as e:
        print(f"Error calculating cartilage thickness for right side of patient {patient.name}: {e}")
        cartilage_thickness_right = (np.nan, np.nan, np.nan, np.nan)

    return {'patient': patient.name.split('.')[0], 'ccd_left': ccd_left, 'ccd_right': ccd_right,'fat_left': fat_left, 'fat_right': fat_right,
            'aa_left': aa_left, 'aa_right': aa_right, 'aav_left': aav[0], 'aav_right': aav[1],
            'cea_left': cea[0], 'cea_right': cea[1], 'offset_left': offset_left, 'offset_right': offset_right,
            'cartilage_thickness_left': cartilage_thickness_left,'cartilage_thickness_right': cartilage_thickness_right}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process Nako Large image')
    parser.add_argument('--chunk', type=int, default=0, help='Chunk number to process')
    args = parser.parse_args()
    chunk = args.chunk

    patients = [x.name.split('.')[0] for x in os.scandir(f'/hpcwork/p0021834/workspace_simon/nako/chunk_{chunk}_segmentations')]
    iterables = [patients, ['right', 'left']]
    index = pd.MultiIndex.from_product(iterables, names=['Patient', 'Side'])
    df = pd.DataFrame(columns=['CCD', 'AT_murphy', 'AA_anterior', 'AA_posterior', 'AAV', 'CE', 'Offset', 'Cartilage_thickness'], index=index)
    patients = [patient for patient in Path(f'/hpcwork/p0021834/workspace_simon/nako/chunk_{chunk}_segmentations').iterdir() if patient.suffix == '.gz']

    with multiprocessing.Pool() as pool:
        res = pool.map(f, patients)

    for x in res:
        patient = x['patient']
        df.loc[(patient, 'right'), 'CCD'] = round(x['ccd_left'], 1)
        df.loc[(patient, 'left'), 'CCD'] = round(x['ccd_right'], 1)
        df.loc[(patient, 'right'), 'AT_murphy'] = round(x['fat_left'], 1)
        df.loc[(patient, 'left'), 'AT_murphy'] = round(x['fat_right'], 1)
        df.loc[(patient, 'right'), 'AA_anterior'] = round(x['aa_left'][0], 1)
        df.loc[(patient, 'left'), 'AA_anterior'] = round(x['aa_right'][0], 1)
        df.loc[(patient, 'left'), 'AA_posterior'] = round(x['aa_right'][1], 1)
        df.loc[(patient, 'right'), 'AA_posterior'] = round(x['aa_left'][1], 1)
        df.loc[(patient, 'right'), 'AAV'] = round(x['aav_left'], 1)
        df.loc[(patient, 'left'), 'AAV'] = round(x['aav_right'], 1)
        df.loc[(patient, 'right'), 'CE'] = round(x['cea_left'], 1)
        df.loc[(patient, 'left'), 'CE'] = round(x['cea_right'], 1)
        df.loc[(patient, 'right'), 'Offset'] = round(x['offset_left'], 1)
        df.loc[(patient, 'left'), 'Offset'] = round(x['offset_right'], 1)
        df.loc[(patient, 'right'), 'Cartilage_thickness (mean)'] = round(x['cartilage_thickness_left'][0], 1)
        df.loc[(patient, 'left'), 'Cartilage_thickness (mean)'] = round(x['cartilage_thickness_right'][0], 1)
        df.loc[(patient, 'right'), 'Cartilage_thickness (std)'] = round(x['cartilage_thickness_left'][1], 1)
        df.loc[(patient, 'left'), 'Cartilage_thickness (std)'] = round(x['cartilage_thickness_right'][1], 1)
        df.loc[(patient, 'right'), 'Cartilage_thickness (min)'] = round(x['cartilage_thickness_left'][2], 1)
        df.loc[(patient, 'left'), 'Cartilage_thickness (min)'] = round(x['cartilage_thickness_right'][2], 1)
        df.loc[(patient, 'right'), 'Cartilage_thickness (max)'] = round(x['cartilage_thickness_left'][3], 1)
        df.loc[(patient, 'left'), 'Cartilage_thickness (max)'] = round(x['cartilage_thickness_right'][3], 1)

    # df.to_excel('/home/sw521914/Data/nako/eval.xlsx')
    df.to_csv(f'/home/sw521914/Data/nako/results_chunk_{chunk}.csv')
