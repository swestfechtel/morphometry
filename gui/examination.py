import pydicom
import ruptures
import torch
import subprocess
import base64
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.paths import nnUNet_results
from batchgenerators.utilities.file_and_folder_operations import join
from morphometry.femur import calculate_femoral_torsion
from morphometry.tibia import calculate_tibial_torsion
from morphometry.utils import correct_axis_ordering
from typing import Union
from io import BytesIO
from matplotlib.figure import Figure


class Examination:
    """
    Data class for examinations. Holds image data, metadata and computed morphometric values.
    """
    def __init__(self):
        self.image = None
        self.metadata = None
        self.morphometry = None
        self.hip = None
        self.knee = None
        self.ankle = None
        self.hip_mask = None
        self.knee_mask = None
        self.ankle_mask = None
        self.femoral_torsion_left = None
        self.femoral_torsion_right = None
        self.tibial_torsion_left = None
        self.tibial_torsion_right = None
        self.marked_stack = None


    def read_dicom_series(self, directory: str):
        """
        Read a DICOM series from a directory.
        :param directory: The directory containing the DICOM series.
        :return:
        """
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(directory)
        reader.SetFileNames(dicom_names)
        self.image = reader.Execute()

    def read_dicom_metadata(self, directory: str):
        """
        Read metadata from a DICOM series.
        :param directory: The directory containing the DICOM series.
        :return:
        """
        for file in Path(directory).iterdir():
            dataset = pydicom.dcmread(file)
            break  # just take a random file, should be the same for all

        self.metadata = dataset

    def get_metadata(self, dicom_tag: str):
        """
        Get a specific metadata tag from the DICOM series.
        :param dicom_tag: A DICOM tag in the format '0010,0010'.
        :return:
        """
        first_part, second_part = dicom_tag.split(',')
        return self.metadata[f'0x{first_part}', f'0x{second_part}'].value

    def split_series(self):
        """
        Split a stacked series into hip, knee and ankle series.
        :return:
        """
        cpd = ruptures.KernelCPD()
        image_array = sitk.GetArrayFromImage(self.image)
        image_array = np.where(image_array < 50, 0, image_array)
        num_pixels = np.array([np.count_nonzero(x) for x in image_array])
        breakpoints = cpd.fit_predict(num_pixels, 2)  # two breakpoints: ankle-knee, knee-hip
        self.ankle = self.image[:, :, :breakpoints[0]]
        self.knee = self.image[:, :, breakpoints[0]:breakpoints[1]]
        self.hip = self.image[:, :, breakpoints[1]:]

        self.hip = correct_axis_ordering(self.hip)  # get all axes into expected ordering
        self.knee = correct_axis_ordering(self.knee)
        self.ankle = correct_axis_ordering(self.ankle)

    def compute_segmentations(self):
        """
        Compute segmentations for hip, knee and ankle.
        :return:
        """

        def predict_array(image: sitk.Image, model_folder: str) -> np.ndarray:
            """
            Compute segmentation mask for a single nifti file.

            :param image: The SimpleITK image array to segment.
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

            img = sitk.GetArrayFromImage(image)
            img = torch.unsqueeze(torch.tensor(img), 0).numpy()  # have to unsqueeze because nnunet expects 4dim array
            tmp = image.GetSpacing()
            props = {
                'sitk_stuff': {
                    'spacing': image.GetSpacing(),
                    'origin': image.GetOrigin(),
                    'direction': image.GetDirection()
                },
                'spacing': [tmp[2], tmp[1], tmp[0]]  # need to reverse order here to align with numpy array
            }
            return predictor.predict_from_list_of_npy_arrays(img, None, props, None)[0]  # only one image, so need to index 0 because method returns a list...

        self.hip_mask = predict_array(self.hip, 'Dataset008_TorsionHipTrainVal')
        self.knee_mask = predict_array(self.knee, 'Dataset021_TorsionKneeTrainVal')
        self.ankle_mask = predict_array(self.ankle, 'Dataset022_TorsionAnkleTrainVal')

    def compute_torsional_alignment(self) -> Union[None, list]:
        """
        Compute torsional alignment for femur and tibia.
        :return: If exceptions occurred during computation, return a list of exceptions.
        """
        x_ratio = abs(self.hip.GetSpacing()[2]) / 2 * abs(self.hip.GetSpacing()[0])

        left_hip = self.hip_mask[:, :, :self.hip_mask.shape[2] // 2]
        right_hip = self.hip_mask[:, :, self.hip_mask.shape[2] // 2:]
        left_knee = self.knee_mask[:, :, :self.knee_mask.shape[2] // 2]
        right_knee = self.knee_mask[:, :, self.knee_mask.shape[2] // 2:]
        left_ankle = self.ankle_mask[:, :, :self.ankle_mask.shape[2] // 2]
        right_ankle = self.ankle_mask[:, :, self.ankle_mask.shape[2] // 2:]

        exceptions = list()
        try:
            ft_left, hip_mask_left, proximal_knee_mask_left = calculate_femoral_torsion(left_hip, left_knee, side='left', method='lee', x_ratio=x_ratio, plot=False, mark_mask=True)
            self.femoral_torsion_left = ft_left
        except (RuntimeError, AssertionError, ValueError) as e:
            hip_mask_left, proximal_knee_mask_left = left_hip, left_knee
            exceptions.append(e)

        try:
            ft_right, hip_mask_right, proximal_knee_mask_right = calculate_femoral_torsion(right_hip, right_knee, side='right', method='lee', x_ratio=x_ratio, plot=False, mark_mask=True)
            self.femoral_torsion_right = ft_right
        except (RuntimeError, AssertionError, ValueError) as e:
            hip_mask_right, proximal_knee_mask_right = right_hip, right_knee
            exceptions.append(e)

        try:
            tt_left, ankle_mask_left, distal_knee_mask_left = calculate_tibial_torsion(left_knee, left_ankle, tibia_label_knee=2, tibia_label_ankle=1, fibula_label=2, side='left', plot=False, mark_mask=True)
            self.tibial_torsion_left = tt_left
        except (RuntimeError, AssertionError, ValueError) as e:
            ankle_mask_left, distal_knee_mask_left = left_ankle, left_knee
            exceptions.append(e)

        try:
            tt_right, ankle_mask_right, distal_knee_mask_right = calculate_tibial_torsion(right_knee, right_ankle, tibia_label_knee=2, tibia_label_ankle=1, fibula_label=2, side='right', plot=False, mark_mask=True)
            self.tibial_torsion_right = tt_right
        except (RuntimeError, AssertionError, ValueError) as e:
            ankle_mask_right, distal_knee_mask_right = right_ankle, right_knee
            exceptions.append(e)

        # Combine masks
        num_layers = self.hip_mask.shape[0] + self.knee_mask.shape[0] + self.ankle_mask.shape[0]
        l_r_split = self.hip_mask.shape[2] // 2
        hip_ends = self.hip_mask.shape[0]
        knee_ends = hip_ends + self.knee_mask.shape[0]
        combined_mask_marked = np.zeros((num_layers, self.hip_mask.shape[1], self.hip_mask.shape[2]), dtype=np.uint8)
        combined_mask_marked[0:hip_ends, :, :l_r_split] = hip_mask_left
        combined_mask_marked[0:hip_ends, :, l_r_split:] = hip_mask_right
        combined_mask_marked[hip_ends:knee_ends, :, :l_r_split] = proximal_knee_mask_left
        combined_mask_marked[hip_ends:knee_ends, :, l_r_split:] = proximal_knee_mask_right
        # combined_mask_marked[self.hip_mask.shape[0]:self.knee_mask.shape[0], :, :l_r_split] += np.where(distal_knee_mask_left < 1, 0, distal_knee_mask_left)
        # combined_mask_marked[self.hip_mask.shape[0]:self.knee_mask.shape[0], :, l_r_split:] += np.where(distal_knee_mask_right < 1, 0, distal_knee_mask_right)
        combined_mask_marked[knee_ends:, :, :l_r_split] = ankle_mask_left
        combined_mask_marked[knee_ends:, :, l_r_split:] = ankle_mask_right

        self.marked_stack = combined_mask_marked

        if len(exceptions) > 0:
            return exceptions

        return None

    def to_base64(self) -> bytes:
        """
        Convert the marked stack to a base64 encoded string.
        :return: The base64 encoded string.
        """
        layers = [] * self.marked_stack.shape[0]
        for i, layer in enumerate(self.marked_stack):
            fig = Figure(figsize=(20, 20))
            ax = fig.subplots()
            ax.imshow(layer, cmap='gray')
            ax.axis('off')
            buffer = BytesIO()
            fig.savefig(buffer, format='png', bbox_inches='tight')
            layers[i] = base64.b64encode(buffer.getbuffer()).decode('ascii')

        return layers

    def get_torsion_values(self) -> dict:
        """
        Get the computed torsion values.
        :return: A dictionary containing the torsion values.
        """
        return {'femoral_torsion_left': self.femoral_torsion_left, 'femoral_torsion_right': self.femoral_torsion_right,
                'tibial_torsion_left': self.tibial_torsion_left, 'tibial_torsion_right': self.tibial_torsion_right}

    def get_array(self, part: str = None) -> np.ndarray:
        """
        Get the image data as a numpy array.
        :param part: The part of the image to return. Can be 'hip', 'knee' or 'ankle'. If None, return the whole image.
        :return: The image data as a numpy array.
        """
        if part is None:
            return sitk.GetArrayFromImage(self.image)

        assert hasattr(self, part), f'Part {part} does not exist.'
        return sitk.GetArrayFromImage(getattr(self, part))

    def write_segmentation(self, part: str, filename: str):
        """
        Write a segmentation mask to a file.
        :param part: The part of the image to write. Can be 'hip', 'knee' or 'ankle'.
        :param filename: The filename to write to.
        :return:
        """
        attribute = f'{part}_mask'
        assert hasattr(self, attribute), f'Part {part} does not exist.'
        img = sitk.GetImageFromArray(getattr(self, attribute))
        img.CopyInformation(getattr(self, part))
        sitk.WriteImage(img, filename)

    def read_segmentation(self, part: str, filename: str):
        """
        Read a segmentation mask from a file.
        :param part: The part of the image to read. Can be 'hip', 'knee' or 'ankle'.
        :param filename: The filename to read from.
        :return:
        """
        attribute = f'{part}_mask'
        assert hasattr(self, attribute), f'Part {part} does not exist.'
        img = sitk.ReadImage(filename)
        mask = sitk.GetArrayFromImage(img)
        setattr(self, attribute, mask)