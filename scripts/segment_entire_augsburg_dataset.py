import os
import subprocess
from tqdm import tqdm


if __name__ == '__main__':
    os.environ['nnUNet_raw'] = '/home/simon/Data/nnUnet_raw'
    os.environ['nnUNet_preprocessed'] = '/home/simon/Data/nnUnet_preprocessed'
    os.environ['nnUNet_results'] = '/home/simon/Data/nnUnet_results'

    patients = [x.name for x in os.scandir('/home/simon/Data/Augsburg_large/preprocessed/')]
    for patient in tqdm(patients):
        try:
            subprocess.run(['python', '/home/simon/Work/nnUNet/nnunetv2/inference/predict_from_raw_data.py', '-i',
                            f'/home/simon/Data/Augsburg_large/preprocessed/{patient}/hip.nii.gz', '-o',
                            f'/home/simon/Data/Augsburg_large/preprocessed/{patient}/hip_seg.nii.gz', '-m', 'hip'])

            subprocess.run(['python', '/home/simon/Work/nnUNet/nnunetv2/inference/predict_from_raw_data.py', '-i',
                            f'/home/simon/Data/Augsburg_large/preprocessed/{patient}/knee.nii.gz', '-o',
                            f'/home/simon/Data/Augsburg_large/preprocessed/{patient}/knee_seg.nii.gz', '-m', 'knee'])

            subprocess.run(['python', '/home/simon/Work/nnUNet/nnunetv2/inference/predict_from_raw_data.py', '-i',
                            f'/home/simon/Data/Augsburg_large/preprocessed/{patient}/ankle.nii.gz', '-o',
                            f'/home/simon/Data/Augsburg_large/preprocessed/{patient}/ankle_seg.nii.gz', '-m', 'ankle'])
        except RuntimeError as e:
            print(f'Failed to segment patient {patient}.', e)
            continue
