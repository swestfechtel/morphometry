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

        print(f'Anteversion (L): {hip.calc_anteversion(left_mask, side="left", segmentation_label=1)}')
        print(f'Anteversion (R): {hip.calc_anteversion(right_mask, side="right", segmentation_label=1)}')
        print(f'CCD (L): {hip.calc_ccd(left_mask, side="left", segmentation_label=1)}')
        print(f'CCD (R): {hip.calc_ccd(right_mask, side="right", segmentation_label=1)}')
        print(f'Alpha angle (L): {hip.calc_alpha_angle(left_mask, side="left", segmentation_label=1)}')
        print(f'Alpha angle (R): {hip.calc_alpha_angle(right_mask, side="right", segmentation_label=1)}')
        print(f'Acetabular anteversion: {hip.calc_acetabular_anteversion(mask_array)}')
        print(f'Acetabular depth: {hip.calc_acetabular_depth(mask_array)}')
        print(f'Center edge angle: {hip.calc_center_edge_angle(mask_array)}')
        print(f'Mininum distance between femoral head and acetabulum (L): {hip.get_min_distance_between_femoral_head_and_acetabulum(left_mask)}')
        print(f'Mininum distance between femoral head and acetabulum (R): {hip.get_min_distance_between_femoral_head_and_acetabulum(right_mask, side="right")}')
