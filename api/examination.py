import pydicom
import ruptures
import base64
import multiprocessing
import datetime

import nibabel as nib
import numpy as np

from morphometry.image_io import Image, Segmentation

from io import BytesIO

from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap, BoundaryNorm

from dataclasses import dataclass
from copy import deepcopy
from collections import defaultdict
from typing import Tuple, Union

masking_value = -1


@dataclass
class Examination(object):
    identifier: str
    original_image: Image
    transformed_image: Image
    metadata: pydicom.FileDataset
    status: str
    image_b64: list[str]

    def __init__(self, identifier: str = None, original_image: Image = None, transformed_image: Image = None, metadata: pydicom.FileDataset = None):
        self.identifier = identifier
        self.original_image = original_image
        self.transformed_image = transformed_image
        self.metadata = metadata
        self.status = 'unprocessed'
        self.image_b64 = None

    def copy(self):
        """
        Create a deep copy of the examination.
        :return: A deep copy of the examination.
        """
        return deepcopy(self)

    @property
    def study_date(self):
        """
        Get the study date from the metadata.
        :return: The study date.
        """
        tmp = self.metadata[0x0008, 0x0020].value
        return datetime.datetime.strptime(tmp, '%Y%m%d').strftime('%Y-%m-%d')

    @property
    def study_time(self):
        """
        Get the study time from the metadata.
        :return: The study time.
        """
        tmp = self.metadata[0x0008, 0x0030].value
        return datetime.datetime.strptime(tmp, '%H%M%S').strftime('%H:%M')

    @property
    def study_description(self):
        """
        Get the study description from the metadata.
        :return: The study description.
        """
        return self.metadata[0x0008, 0x1030].value

    @property
    def accession_number(self):
        """
        Get the accession number from the metadata.
        :return: The accession number.
        """
        return self.metadata[0x0008, 0x0050].value

    @property
    def patient_name(self) -> pydicom.dataelem.PersonName:
        """
        Get the patient name from the metadata.
        :return: The patient name.
        """
        return self.metadata[0x0010, 0x0010].value

    def encode_images(self):
        """
        Encode the images and segmentation masks to base64 strings.
        :return: A list of base64 encoded images and segmentation masks.
        """
        layers = [self.transformed_image.array[:, :, i] for i in range(self.transformed_image.shape[-1])]

        with multiprocessing.Pool() as pool:
            self.image_b64 = pool.map(encode_figure, layers)


class TorsionExamination(Examination):
    hip: Image
    knee: Image
    ankle: Image
    hip_mask: Segmentation
    knee_mask: Segmentation
    ankle_mask: Segmentation
    femoral_torsion_left: float
    femoral_torsion_right: float
    tibial_torsion_left: float
    tibial_torsion_right: float
    landmarks: defaultdict
    image_segmentation_b64: list[str]

    def __init__(self, examination: Examination):
        super().__init__(examination.identifier, examination.original_image, examination.transformed_image, examination.metadata)
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
        self.landmarks = defaultdict(dict)
        self.image_segmentation_b64 = None

    def split_series(self):
        """
        Split a stacked series into hip, knee and ankle series.
        :return:
        """
        cpd = ruptures.KernelCPD()
        image_array = self.transformed_image.array
        image_array = np.where(image_array < 50, 0, image_array)  # outlier removal
        num_pixels = np.array([np.count_nonzero(image_array[:, :, x]) for x in range(image_array.shape[2])])
        breakpoints = cpd.fit_predict(num_pixels, 2)  # two breakpoints: ankle-knee, knee-hip

        hip = nib.Nifti1Image(self.transformed_image.array[:, :, :breakpoints[0]], affine=self.transformed_image.affine)
        self.hip = Image.from_nibabel(hip)

        knee = nib.Nifti1Image(self.transformed_image.array[:, :, breakpoints[0]:breakpoints[1]], affine=self.transformed_image.affine)
        self.knee = Image.from_nibabel(knee)

        ankle = nib.Nifti1Image(self.transformed_image.array[:, :, breakpoints[1]:], affine=self.transformed_image.affine)
        self.ankle = Image.from_nibabel(ankle)

    def get_torsion_values(self) -> dict:
        """
        Get the computed torsion values.
        :return: A dictionary containing the torsion values.
        """
        return {'femoral_torsion_left': self.femoral_torsion_left, 'femoral_torsion_right': self.femoral_torsion_right,
                'tibial_torsion_left': self.tibial_torsion_left, 'tibial_torsion_right': self.tibial_torsion_right}

    def encode_images(self):
        """
        Encode the images and segmentation masks to base64 strings.
        :return: A list of base64 encoded images and segmentation masks.
        """
        super().encode_images()

        tmp = self.ankle_mask.array.copy()
        tmp = np.where(tmp == 2, 3, tmp)
        tmp = np.where(tmp == 1, 2, tmp)
        segmented_array = np.concatenate((self.hip_mask.array, self.knee_mask.array, tmp), axis=2)
        image_layers = [self.transformed_image.array[:, :, i] for i in range(self.transformed_image.shape[-1])]
        segmentation_layers = [segmented_array[:, :, i] for i in range(segmented_array.shape[-1])]
        layers = [(image_layers[i], segmentation_layers[i]) for i in range(len(image_layers))]

        with multiprocessing.Pool() as pool:
            self.image_segmentation_b64 = pool.map(encode_figure, layers)


def encode_figure(layer: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]) -> str:
    image_layer = layer[0] if isinstance(layer, tuple) else layer
    segmentation_layer = layer[1] if isinstance(layer, tuple) else None

    fig = Figure(figsize=(20, 20))
    ax = fig.subplots()

    ax.imshow(image_layer.T, cmap='gray')
    if segmentation_layer is not None:
        colours = ['white', 'yellow', 'purple', 'cyan']
        cmap = ListedColormap(colours)

        bounds = [-1, 0.5, 1.5, 2.5, 3.5]
        norm = BoundaryNorm(bounds, cmap.N)

        ax.imshow(np.where(segmentation_layer == 0, np.nan, segmentation_layer).T, cmap=cmap, norm=norm, alpha=0.5)

    ax.axis('off')
    buffer = BytesIO()
    fig.savefig(buffer, format='png', transparent=True, bbox_inches='tight')
    return base64.b64encode(buffer.getbuffer()).decode('ascii')