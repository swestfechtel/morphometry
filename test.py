from morphometry import hip
from pathlib import Path
import SimpleITK as sitk
import numpy as np


if __name__ == '__main__':
    segmentations = ['/home/simon/Data/Hamburg/100000_30_PD_FS_SPC_COR/100000_30_PD_FS_SPC_COR.nii',
                     '/home/simon/Data/Hamburg/100001_30_PD_FS_SPC_COR/100001_30_PD_FS_SPC_COR.nii',
                ]
    for seg in segmentations:
        print(seg)
        mask = sitk.ReadImage(str(seg))
        mask_array = sitk.GetArrayFromImage(mask)
        mask_array = np.swapaxes(mask_array, 0, 1)
        left_mask = mask_array[:, :, :mask_array.shape[2] // 2]
        right_mask = mask_array[:, :, mask_array.shape[2] // 2:]

        print(f'CCD (L): {hip.calc_ccd(left_mask)}')
        print(f'CCD (R): {hip.calc_ccd(right_mask)}')
        print(f'Alpha angle (L): {hip.calc_alpha_angle(left_mask)}')
        print(f'Alpha angle (R): {hip.calc_alpha_angle(right_mask)}')
        print(f'Acetabular anteversion: {hip.calc_acetabular_anteversion(mask_array)}')
        print(f'Acetabular depth: {hip.calc_acetabular_depth(mask_array)}')
        print(f'Center edge angle: {hip.calc_center_edge_angle(mask_array)}')
