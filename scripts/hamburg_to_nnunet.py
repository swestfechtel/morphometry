from pathlib import Path
import SimpleITK as sitk
from dicom_to_nifti import dicom_to_nifti
import re
import shutil
import tempfile
import zipfile
from tqdm import tqdm
import os
import numpy as np


if __name__ == '__main__':
    segmentation_pattern = r'Segmentation.*'

    for study in tqdm(Path('/home/simon/Data/Hamburg/').iterdir()):
        study_name = study.name
        study_id = study_name.split('_')[0]

        if study_name == 'old_batch' or study_name == 'MRTs überarbeitet und neu':
            continue

        if not '_30_' in study_name:
            study_name = study_name.replace('_PD_', '_30_PD_')

        valid_segmentation_found = False
        for file in study.iterdir():
            if re.match(segmentation_pattern, file.name):
                if not file.name.endswith('.nrrd'):
                    print(f'Renaming {file.name} to {file.with_suffix(".nrrd").name}')
                    file = shutil.move(file, file.with_suffix('.nrrd'))

                segmentation = sitk.ReadImage(file)
                segmentation_array = sitk.GetArrayFromImage(segmentation)
                # segmentation_array = np.where(segmentation_array == 2, 0, segmentation_array)
                # segmentation_array = np.where(segmentation_array == 3, 2, segmentation_array)
                segmentation_array = np.where(segmentation_array > 3, 0, segmentation_array)

                if np.unique(segmentation_array).tolist() != [0, 1, 2, 3]:
                    print(f'Unexpected values in segmentation {study_name}: {np.unique(segmentation_array).tolist()}')
                    continue

                tmp = sitk.GetImageFromArray(segmentation_array)
                tmp.CopyInformation(segmentation)
                sitk.WriteImage(tmp, f'/home/simon/Data/nnUnet_raw/Dataset029_HamburgHip/labelsTr/{study_name}.nii.gz')
                valid_segmentation_found = True
                break

        if not valid_segmentation_found:
            print(f'No valid segmentation found for {study_name}')
            continue

        with tempfile.TemporaryDirectory() as tmpdirname:
            with zipfile.ZipFile(f'/home/simon/Data/NaKo_PD_FS_SPC_COR/{study_name}.zip', 'r') as zip_ref:
                zip_ref.extractall(tmpdirname)

            dicom_to_nifti(f'{tmpdirname}/{study_id}_30/PD_FS_SPC_COR/', f'/home/simon/Data/nnUnet_raw/Dataset029_HamburgHip/imagesTr/{study_name}_0000.nii.gz', single_file=False)
