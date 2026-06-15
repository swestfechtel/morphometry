import pydicom
import ruptures
import base64
import multiprocessing
import datetime

import nibabel as nib
import numpy as np
import morphometry.image_io as mio

from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap, BoundaryNorm

from dataclasses import dataclass
from copy import deepcopy
from collections import defaultdict
from typing import Tuple, Union
from io import BytesIO
from PIL import Image as PILImage

masking_value = -1


@dataclass
class Examination(object):
    identifier: str
    metadata: pydicom.FileDataset
    status: str

    def __init__(self, identifier: str = None, metadata: pydicom.FileDataset = None):
        self.identifier = identifier
        self.metadata = metadata
        self.status = 'unprocessed'

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


class TorsionExamination(Examination):
    original_image: mio.Image
    transformed_image: mio.Image
    hip: mio.Image
    knee: mio.Image
    ankle: mio.Image
    hip_mask: mio.Segmentation
    knee_mask: mio.Segmentation
    ankle_mask: mio.Segmentation
    femoral_torsion_left: float
    femoral_torsion_right: float
    femoral_torsion_left_murphy: float
    femoral_torsion_right_murphy: float
    tibial_torsion_left: float
    tibial_torsion_right: float
    landmarks: dict
    image_segmentation_b64: list[str]
    image_b64: list[str]

    def __init__(self, examination: Examination):
        super().__init__(examination.identifier, metadata=examination.metadata)
        self.hip = None
        self.knee = None
        self.ankle = None
        self.hip_mask = None
        self.knee_mask = None
        self.ankle_mask = None
        self.femoral_torsion_left = None
        self.femoral_torsion_right = None
        self.femoral_torsion_left_murphy = None
        self.femoral_torsion_right_murphy = None
        self.tibial_torsion_left = None
        self.tibial_torsion_right = None
        self.landmarks = None
        self.image_segmentation_b64 = None

    def split_series(self):
        """
        Split a stacked series into hip, knee and ankle series.

        If ``hip``, ``knee`` and ``ankle`` are already populated (e.g. because the
        examination was created from three separate series via
        :meth:`FileController.save_torsion_series`), this method is a no-op.
        :return:
        """
        if self.hip is not None and self.knee is not None and self.ankle is not None:
            return

        cpd = ruptures.KernelCPD()
        image_array = self.transformed_image.array
        image_array = np.where(image_array < 50, 0, image_array)  # outlier removal
        num_pixels = np.array([np.count_nonzero(image_array[:, :, x]) for x in range(image_array.shape[2])])
        breakpoints = cpd.fit_predict(num_pixels, 2)  # two breakpoints: ankle-knee, knee-hip

        hip = nib.Nifti1Image(self.transformed_image.array[:, :, :breakpoints[0]], affine=self.transformed_image.affine)
        self.hip = mio.Image.from_nibabel(hip)

        knee = nib.Nifti1Image(self.transformed_image.array[:, :, breakpoints[0]:breakpoints[1]], affine=self.transformed_image.affine)
        self.knee = mio.Image.from_nibabel(knee)

        ankle = nib.Nifti1Image(self.transformed_image.array[:, :, breakpoints[1]:], affine=self.transformed_image.affine)
        self.ankle = mio.Image.from_nibabel(ankle)

    def get_torsion_values(self) -> dict:
        """
        Get the computed torsion values.
        :return: A dictionary containing the torsion values.
        """
        def safe_value(value: float) -> float:
            if value is None:
                return 0
            return 0 if np.isnan(value) else value

        return {
            'femoral_torsion_left': safe_value(self.femoral_torsion_left),
            'femoral_torsion_right': safe_value(self.femoral_torsion_right),
            'femoral_torsion_left_murphy': safe_value(self.femoral_torsion_left_murphy),
            'femoral_torsion_right_murphy': safe_value(self.femoral_torsion_right_murphy),
            'tibial_torsion_left': safe_value(self.tibial_torsion_left),
            'tibial_torsion_right': safe_value(self.tibial_torsion_right)
        }

    def encode_images(self):
        """
        Encode the images and segmentation masks to base64 strings.
        :return: A list of base64 encoded images and segmentation masks.
        """
        layers = [self.transformed_image.array[:, :, i] for i in range(self.transformed_image.shape[-1])]

        with multiprocessing.Pool() as pool:
            self.image_b64 = pool.map(encode_figure, layers)

        tmp = self.ankle_mask.array.copy()
        tmp = np.where(tmp == 2, 3, tmp)
        tmp = np.where(tmp == 1, 2, tmp)
        segmented_array = np.concatenate((self.hip_mask.array, self.knee_mask.array, tmp), axis=2)
        image_layers = [self.transformed_image.array[:, :, i] for i in range(self.transformed_image.shape[-1])]
        segmentation_layers = [segmented_array[:, :, i] for i in range(segmented_array.shape[-1])]
        layers = [(image_layers[i], segmentation_layers[i]) for i in range(len(image_layers))]

        with multiprocessing.Pool() as pool:
            self.image_segmentation_b64 = pool.map(encode_figure, layers)


class XRayExamination(Examination):
    image: PILImage.Image
    landmarks: dict

    def __init__(self, examination: Examination):
        super().__init__(examination.identifier, metadata=examination.metadata)
        self._image = None
        self._landmarks = None

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, value: PILImage.Image):
        self._image = value

    @property
    def landmarks(self):
        return self._landmarks

    @landmarks.setter
    def landmarks(self, value: dict):
        if not isinstance(value, dict):
            raise TypeError("Landmarks must be a dictionary.")
        self._landmarks = value

    def to_base64(self) -> str:
        buffer = BytesIO()
        self.image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('ascii')


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