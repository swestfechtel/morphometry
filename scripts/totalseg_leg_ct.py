import os
import tempfile
from pathlib import Path
import SimpleITK as sitk
import numpy as np
import json
import torch
from totalsegmentator.python_api import totalsegmentator


if __name__ == '__main__':
    assert torch.cuda.is_available()
    metadata = dict()
    for patient in Path('/mnt/ocean_storage/data/UKA/LegCT/core/data').iterdir():
        metadata[patient.name] = dict()
        for s1 in patient.iterdir():
            for s2 in s1.iterdir():
                for series in s2.iterdir():

                    metadata[patient.name][series.name] = dict()
                    input_path = series

                    output_path = f'/mnt/ocean_storage/data/UKA/LegCT/totalsegmentator_segmentations/{patient.name}/{series.name}/'
                    Path(output_path).mkdir(parents=True, exist_ok=True)

                    try:
                        totalsegmentator(input_path, os.path.join(output_path, 'total'), ml=True, task='total', roi_subset=['femur_left', 'femur_right'])
                        totalsegmentator(input_path, os.path.join(output_path, 'appendicular') , ml=True, task='appendicular_bones', roi_subset=None)
                    except:
                        print(f"Error processing {input_path}.")
                        continue

                    total_mask = sitk.ReadImage(f'{output_path}/total.nii')
                    appendicular_mask = sitk.ReadImage(f'{output_path}/appendicular.nii')
                    total_mask_array = sitk.GetArrayFromImage(total_mask)
                    appendicular_mask_array = sitk.GetArrayFromImage(appendicular_mask)
                    total_mask_array_unique = np.unique(total_mask_array)
                    appendicular_mask_array_unique = np.unique(appendicular_mask_array)
                    metadata[patient.name][series.name]['femur_left'] = 75 in total_mask_array_unique
                    metadata[patient.name][series.name]['femur_right'] = 76 in total_mask_array_unique
                    metadata[patient.name][series.name]['tibia'] = 2 in appendicular_mask_array_unique
                    metadata[patient.name][series.name]['fibula'] = 3 in appendicular_mask_array_unique
                    metadata[patient.name][series.name]['tarsal'] = 4 in appendicular_mask_array_unique
        break
    print(metadata)
    with open('/mnt/ocean_storage/data/UKA/LegCT/totalsegmentator_segmentations/metadata/bone_screening.json', 'w') as f:
        json.dump(metadata, f, indent=4)
