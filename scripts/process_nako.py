import os
import re
import multiprocessing
import sys

sys.path.append('/home/simon/Work/morphometry')

import pandas as pd
import numpy as np
import nibabel as nib
import pyvista as pv

from pathlib import Path
from morphometry.hip import calculate_ccd, calculate_anteversion, calculate_acetabular_anteversion, \
    calculate_alpha_angle, calculate_acetabular_depth, calculate_center_edge_angle, \
    calculate_cartilage_thickness_knn, calculate_femoral_offset, calculate_femoral_offset_projected
from morphometry.image_io import Segmentation
from matplotlib import pyplot as plt


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

    p = pv.Plotter(off_screen=True)

    try:
        # ccd_left, _ = calculate_ccd(mask_left, None, 'left', 1, isotropic=True, plot=p)
        _, ccd_left = calculate_ccd(mask_left, None, 'left', 1, isotropic=True, plot=p)
    except Exception as e:
        print(f"Error calculating CCD for left side of patient {patient.name}: {e}")
        ccd_left = np.nan

    try:
        # ccd_right, _ = calculate_ccd(mask_right, None, 'right', 1, isotropic=True, plot=p)
        _, ccd_right = calculate_ccd(mask_right, None, 'right', 1, isotropic=True, plot=p)
    except Exception as e:
        print(f"Error calculating CCD for right side of patient {patient.name}: {e}")
        ccd_right = np.nan

    p.camera.azimuth = 45
    p.camera.elevation = -25
    p.add_text(f'CCD right: {ccd_left:.1f}°', position='upper_left', font_size=20, color='black')
    p.add_text(f'CCD left: {ccd_right:.1f}°', position='upper_right', font_size=20, color='black')
    p.export_html(f'/home/simon/Data/NaKo_sample/plots/ccd/{patient.name}_ccd.html')
    p.screenshot(f'/home/simon/Data/NaKo_sample/plots/ccd/{patient.name}_ccd.png')
    p.close()
    """
    ax[0].set_title(f'CCD right: {ccd_left:.1f}°')
    ax[1].set_title(f'CCD left: {ccd_right:.1f}°')
    fig.savefig(f'/home/simon/Data/NaKo_sample/plots/ccd/{patient.name}.png')
    plt.close(fig)
    """

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
    try:
        fat_left = calculate_anteversion(mask_left, 'left', 1, isotropic=True, plot=(ax[0,0], ax[1,0]))
    except Exception as e:
        print(f"Error calculating FAT for left side of patient {patient.name}: {e}")
        fat_left = np.nan

    try:
        fat_right = calculate_anteversion(mask_right, 'right', 1, isotropic=True, plot=(ax[0, 1], ax[1,1]))
    except Exception as e:
        print(f"Error calculating FAT for right side of patient {patient.name}: {e}")
        fat_right = np.nan

    ax[0,0].set_title(f'Antetorsion right: {fat_left:.1f}°')
    ax[0,1].set_title(f'Antetorsion left: {fat_right:.1f}°')
    fig.savefig(f'/home/simon/Data/NaKo_sample/plots/anteversion/{patient.name}.png')
    plt.close(fig)

    fig, ax = plt.subplots(ncols=2, figsize=(20, 10))

    try:
        aa_left = calculate_alpha_angle(mask_left.array, 'left', 1, isotropic=True, plot=ax[0])
    except Exception as e:
        print(f"Error calculating AA for left side of patient {patient.name}: {e}")
        aa_left = (np.nan, np.nan)
    # calculate_alpha_angle(mask_left.array, 'left', 1, isotropic=True)

    try:
        aa_right = calculate_alpha_angle(mask_right.array, 'right', 1, isotropic=True, plot=ax[1])
    except Exception as e:
        print(f"Error calculating AA for right side of patient {patient.name}: {e}")
        aa_right = (np.nan, np.nan)

    ax[0].set_title(f'Alpha angle right: {aa_left}°')
    ax[1].set_title(f'Alpha angle left: {aa_right}°')
    fig.savefig(f'/home/simon/Data/NaKo_sample/plots/alpha_angle/{patient.name}.png')
    plt.close(fig)

    try:
        aav = calculate_acetabular_anteversion(mask.array, 1, 3, isotropic=True, plot=True,
                                               fp=f'/home/simon/Data/NaKo_sample/plots/acetabular_anteversion/{patient.name}.png')
    except Exception as e:
        print(f"Error calculating AAV for patient {patient.name}: {e}")
        aav = [np.nan, np.nan]

    try:
        cea = calculate_center_edge_angle(mask.array, 1, 3, isotropic=True, plot=True, fp=f'/home/simon/Data/NaKo_sample/plots/center_edge/{patient.name}.png')
    except Exception as e:
        print(f"Error calculating CEA for left side of patient {patient.name}: {e}")
        cea = [np.nan, np.nan]

    p = pv.Plotter(off_screen=True)

    try:
        # offset_left = calculate_femoral_offset(mask_left, None, 'left', 1, isotropic=True, plot=p)
        offset_left = calculate_femoral_offset_projected(mask_left, None, 'left', 1, True, False)
    except Exception as e:
        print(f"Error calculating femoral offset for left side of patient {patient.name}: {e}")
        offset_left = np.nan

    try:
        # offset_right = calculate_femoral_offset(mask_right, None, 'right', 1, isotropic=True, plot=p)
        offset_right = calculate_femoral_offset_projected(mask_right, None, 'right', 1, True, False)
    except Exception as e:
        print(f"Error calculating femoral offset for right side of patient {patient.name}: {e}")
        offset_right = np.nan

    p.camera.azimuth = 45
    p.camera.elevation = -25
    p.add_text(f'Offset right: {offset_left:.1f}mm', position='upper_left', font_size=20, color='black')
    p.add_text(f'Offset left: {offset_right:.1f}mm', position='upper_right', font_size=20, color='black')
    p.export_html(f'/home/simon/Data/NaKo_sample/plots/offset/{patient.name}_offset.html')
    p.screenshot(f'/home/simon/Data/NaKo_sample/plots/offset/{patient.name}_offset.png')
    p.close()

    return {'patient': patient.name.split('.')[0], 'ccd_left': ccd_left, 'ccd_right': ccd_right,'fat_left': fat_left, 'fat_right': fat_right,
            'aa_left': aa_left, 'aa_right': aa_right, 'aav_left': aav[0], 'aav_right': aav[1],
            'cea_left': cea[0], 'cea_right': cea[1], 'offset_left': offset_left, 'offset_right': offset_right}



if __name__ == '__main__':
    patients = [x.name.split('.')[0] for x in os.scandir('/home/simon/Data/NaKo_sample/segmentations')]
    iterables = [patients, ['right', 'left']]
    index = pd.MultiIndex.from_product(iterables, names=['Patient', 'Side'])
    df = pd.DataFrame(columns=['CCD', 'AT_murphy', 'AA_anterior', 'AA_posterior', 'AAV', 'CE', 'Offset'], index=index)
    patients = [patient for patient in Path('/home/simon/Data/NaKo_sample/segmentations').iterdir() if patient.suffix == '.gz']
    pv.start_xvfb()

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

    print(df)
    df.to_excel('/home/simon/Data/NaKo_sample/eval.xlsx')