import os
import re
import SimpleITK as sitk
import pandas as pd
import numpy as np
from pathlib import Path
from morphometry.hip import calculate_ccd, calculate_anteversion, calculate_acetabular_anteversion, calculate_alpha_angle, calculate_acetabular_depth, calculate_center_edge_angle, calculate_cartilage_thickness_knn
from morphometry.utils import correct_axis_ordering
from tqdm import tqdm


if __name__ == '__main__':
    patients = [x.name for x in os.scandir('/home/simon/Downloads/Hamburg/simon/')]
    iterables = [patients, ['right', 'left']]
    index = pd.MultiIndex.from_product(iterables, names=['Patient', 'Side'])
    df = pd.DataFrame(columns=['CCD', 'FAT', 'AA', 'AAV', 'AD', 'CT'], index=index)
    for patient in tqdm(patients):
        try:
            mask = sitk.ReadImage(f'/home/simon/Downloads/Hamburg/simon/{patient}/Segmentation.seg.nrrd')
        except RuntimeError as e:
            print(f'Patient {patient} could not be found.', e)
            continue

        try:
            mask = correct_axis_ordering(mask)
        except RuntimeError as e:
            print(f'Patient {patient} could not be corrected.', e)
            continue

        mask_np = sitk.GetArrayFromImage(mask)
        # x_ratio = abs(mask.GetSpacing()[2]) / 2 * abs(mask.GetSpacing()[0])
        mask_left = mask_np[:, :, :mask_np.shape[2] // 2]
        mask_right = mask_np[:, :, mask_np.shape[2] // 2:]

        ccd_left = calculate_ccd(mask_left, 'left', 1, isotropic=True)[0]
        # ccd_right = calculate_ccd(mask_right, 'right', 1, isotropic=True)[0]
        fat_left = calculate_anteversion(mask_left, 'left', 1, isotropic=True)
        # fat_right = calculate_anteversion(mask_right, 'right', 1)
        aa_left = calculate_alpha_angle(mask_left, 'left', 1, isotropic=True)
        # aa_right = calculate_alpha_angle(mask_right, 'right', 1)
        # aav_left = calculate_acetabular_anteversion(mask_left, 1, 3)
        # aav_right = calculate_acetabular_anteversion(mask_right, 1, 3)
        ad_left = calculate_acetabular_depth(mask_left, 'left', 1, 3, isotropic=True)
        # ad_right = calculate_acetabular_depth(mask_right, 'right', 1, 3)
        ct_left = calculate_cartilage_thickness_knn(mask_left, 2)
        # ct_right = calculate_cartilage_thickness_knn(mask_right, 2)
        df.loc[(patient, 'right'), 'CCD'] = round(ccd_left, 1)
        df.loc[(patient, 'right'), 'FAT'] = round(fat_left, 1)
        df.loc[(patient, 'right'), 'AA'] = round(aa_left, 1)
        # df.loc[(patient, 'right'), 'AAV'] = aav_left
        df.loc[(patient, 'right'), 'AD'] = round(ad_left, 1)
        df.loc[(patient, 'right'), 'CT'] = round(ct_left, 1)
        # df.loc[(patient, 'left'), 'CCD'] = ccd_right
        # df.loc[(patient, 'left'), 'FAT'] = fat_right
        # df.loc[(patient, 'left'), 'AA'] = aa_right
        # df.loc[(patient, 'left'), 'AAV'] = aav_right
        # df.loc[(patient, 'left'), 'AD'] = ad_right
        # df.loc[(patient, 'left'), 'CT'] = ct_right

    df.to_excel('/home/simon/Downloads/Hamburg/eval.xlsx')