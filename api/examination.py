import pydicom
import ruptures
import base64

import nibabel as nib
import numpy as np

from morphometry.image_io import Image, Segmentation

from io import BytesIO

from matplotlib.figure import Figure
from matplotlib import colors, cm

from dataclasses import dataclass
from copy import deepcopy
from collections import defaultdict

masking_value = -1


@dataclass
class Examination(object):
    identifier: str
    original_image: Image
    transformed_image: Image
    metadata: pydicom.FileDataset
    status: str

    def __init__(self, identifier: str = None, original_image: Image = None, transformed_image: Image = None, metadata: pydicom.FileDataset = None):
        self.identifier = identifier
        self.original_image = original_image
        self.transformed_image = transformed_image
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
        return self.metadata[0x0008, 0x0020].value

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


def encode_figure(layer: np.ndarray, segmentation: bool = False) -> str:
    fig = Figure(figsize=(20, 20))
    ax = fig.subplots()
    if segmentation:
        layer = np.ma.masked_where(layer < 4, layer)
        cmap = colors.ListedColormap(['yellow', 'purple', 'lightblue', 'red'])
        # cmap.set_under('k', alpha=0)
        bounds = [1, 2, 3, 4, 5]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        ax.imshow(layer.T, cmap='viridis')
    else:
        ax.imshow(layer.T, cmap='gray')

    ax.axis('off')
    buffer = BytesIO()
    fig.savefig(buffer, format='png', transparent=True, bbox_inches='tight')
    return base64.b64encode(buffer.getbuffer()).decode('ascii')