import torch
import logging
import traceback
import tempfile
import string
import random
import uuid
import asyncio

import nibabel as nib

from api.examination import TorsionExamination
from api.file_controller import FileController

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.paths import nnUNet_results
from batchgenerators.utilities.file_and_folder_operations import join

from morphometry.image_io import Image, Segmentation
from morphometry.femur import calculate_femoral_torsion
from morphometry.tibia import calculate_tibial_torsion


class ModelJob:
    """
    Abstract class for a model job. Subclass must implement the execute method.
    """
    identifier: str
    running: bool
    file_controller: FileController

    def __init__(self, file_controller: FileController):
        # self.identifier = str(uuid.uuid1())
        self.file_controller = file_controller
        self.running = False

    async def dummy_job(self):
        import asyncio
        import time
        self.running = True
        # await asyncio.sleep(10)
        time.sleep(10)
        self.running = False


class TorsionModelJob(ModelJob):

    def execute(self, examination: TorsionExamination):
        """
        Compute segmentation and torsional alignment for a TorsionExamination object.
        :param examination: A TorsionExamination object to process.
        :return: The job.
        """
        self.compute_segmentation(examination)
        self.compute_torsional_alignment(examination)

        return self

    def compute_segmentation(self, examination: TorsionExamination):
        """
        Compute segmentation masks for hip, knee and ankle.
        :param examination: A TorsionExamination object to compute segmentations for.
        :return: The job
        """
        logger = logging.getLogger('api')
        logger.info(f'Created segmentation job for {examination.identifier} with identifier {self.identifier}.')
        self.running = True
        examination.status = 'running'
        self.identifier = examination.identifier  # TODO will become a problem if more than one job is running at the same time for the same examination

        def predict_array(image: Image, model_folder: str) -> Segmentation:
            """
            Compute segmentation mask for a single Image.

            :param image: The Image to segment.
            :param model_folder: The folder containing the trained model.
            :return: The segmentation mask.
            """
            predictor = nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=True,
                perform_everything_on_device=True,
                device=torch.device('cuda', 0),
                verbose=False,
                verbose_preprocessing=False,
                allow_tqdm=True
            )
            predictor.initialize_from_trained_model_folder(
                join(nnUNet_results, model_folder, 'nnUNetTrainer__nnUNetPlans__3d_fullres'), use_folds=('all',),
                checkpoint_name='checkpoint_best.pth')

            with tempfile.TemporaryDirectory() as tmpdir:  # have to use predict_from_files because predicting from arrays requires SimpleITK axis ordering and I don't want to deal with that
                # https://github.com/MIC-DKFZ/nnUNet/tree/master/nnunetv2/inference#predicting-a-single-npy-array
                temp_image = f'{tmpdir}/temp_image.nii.gz'
                image.save_image(temp_image)
                predictor.predict_from_files([[temp_image]], [f'{tmpdir}/temp_segmentation.nii.gz'])
                seg = nib.load(f'{tmpdir}/temp_segmentation.nii.gz')
                seg = Segmentation.from_nibabel(seg)
                seg.transform_coordinate_system()

            return seg

        examination.split_series()

        hip_mask = predict_array(examination.hip, 'Dataset008_TorsionHipTrainVal')
        hip_mask.axcodes = examination.hip.axcodes
        examination.hip_mask = hip_mask

        knee_mask = predict_array(examination.knee, 'Dataset021_TorsionKneeTrainVal')
        knee_mask.axcodes = examination.knee.axcodes
        examination.knee_mask = knee_mask

        ankle_mask = predict_array(examination.ankle, 'Dataset022_TorsionAnkleTrainVal')
        ankle_mask.axcodes = examination.ankle.axcodes
        examination.ankle_mask = ankle_mask

        examination.encode_images()

        logger.info(f'Finished segmentation job for {examination.identifier} with identifier {self.identifier}.')
        examination.status = 'segmented'
        self.file_controller.update_examination(examination)

        return self

    def compute_torsional_alignment(self, examination: TorsionExamination):
        """
        Compute the torsional alignment for a TorsionExamination object.
        :param examination: A TorsionExamination object to compute torsional alignment for.
        :return: The job.
        """
        logger = logging.getLogger('api')
        logger.info(f'Created torsional alignment job for {examination.identifier} with identifier {self.identifier}.')
        self.running = True
        examination.status = 'running'
        self.identifier = examination.identifier

        x_ratio = abs(examination.hip_mask.spacing[2]) / 2 * abs(examination.hip_mask.spacing[0])

        examination.hip_mask.remove_outliers()
        examination.knee_mask.remove_outliers()
        examination.ankle_mask.remove_outliers()

        hip_mask = examination.hip_mask.array
        knee_mask = examination.knee_mask.array
        ankle_mask = examination.ankle_mask.array

        left_hip = hip_mask[:hip_mask.shape[0] // 2]
        right_hip = hip_mask[hip_mask.shape[0] // 2:]
        left_knee = knee_mask[:knee_mask.shape[0] // 2]
        right_knee = knee_mask[knee_mask.shape[0] // 2:]
        left_ankle = ankle_mask[:ankle_mask.shape[0] // 2]
        right_ankle = ankle_mask[ankle_mask.shape[0] // 2:]

        left_hip = nib.Nifti1Image(left_hip, examination.hip.affine, examination.hip.header)
        left_hip = Image.from_nibabel(left_hip)
        right_hip = nib.Nifti1Image(right_hip, examination.hip.affine, examination.hip.header)
        right_hip = Image.from_nibabel(right_hip)

        try:
            torsion, landmarks = calculate_femoral_torsion(left_hip, left_knee, side='left', method='lee', x_ratio=x_ratio, plot=False, return_landmarks=True)
            examination.femoral_torsion_right = torsion
            for k, v in landmarks.items():
                landmarks[k] = v.tolist()
            examination.landmarks['femur']['right'] = landmarks
        except (RuntimeError, AssertionError, ValueError) as e:
            logger.error(traceback.format_exc())

        try:
            torsion, landmarks = calculate_femoral_torsion(right_hip, right_knee, side='right', method='lee', x_ratio=x_ratio, plot=False, return_landmarks=True)
            landmarks['hip_start'][0] += examination.hip_mask.shape[0] // 2  # shift to the right image side
            landmarks['hip_end'][0] += examination.hip_mask.shape[0] // 2
            landmarks['knee_start'][0] += examination.knee_mask.shape[0] // 2
            landmarks['knee_end'][0] += examination.knee_mask.shape[0] // 2
            examination.femoral_torsion_left = torsion
            for k, v in landmarks.items():
                landmarks[k] = v.tolist()
            examination.landmarks['femur']['left'] = landmarks
        except (RuntimeError, AssertionError, ValueError) as e:
            logger.error(traceback.format_exc())

        try:
            torsion, landmarks = calculate_tibial_torsion(left_knee, left_ankle, tibia_label_knee=2, tibia_label_ankle=1, fibula_label=2, side='left', plot=False, return_landmarks=True)
            examination.tibial_torsion_right = torsion
            for k, v in landmarks.items():
                landmarks[k] = v.tolist()
            examination.landmarks['tibia']['right'] = landmarks
        except (RuntimeError, AssertionError, ValueError) as e:
            logger.error(traceback.format_exc())

        try:
            torsion, landmarks = calculate_tibial_torsion(right_knee, right_ankle, tibia_label_knee=2, tibia_label_ankle=1, fibula_label=2, side='right', plot=False, return_landmarks=True)
            landmarks['knee_start'][0] += examination.knee_mask.shape[0] // 2
            landmarks['knee_end'][0] += examination.knee_mask.shape[0] // 2
            landmarks['ankle_start'][0] += examination.ankle_mask.shape[0] // 2
            landmarks['ankle_end'][0] += examination.ankle_mask.shape[0] // 2
            examination.tibial_torsion_left = torsion
            for k, v in landmarks.items():
                landmarks[k] = v.tolist()
            examination.landmarks['tibia']['left'] = landmarks
        except (RuntimeError, AssertionError, ValueError) as e:
            logger.error(traceback.format_exc())

        logger.info(f'Finished torsional alignment job for {examination.identifier} with identifier {self.identifier}.')

        examination.status = 'processed'
        self.file_controller.update_examination(examination)

        return self
