import tempfile
import os
import pydicom

import SimpleITK as sitk
import nibabel as nib
import numpy as np

from typing import Union
from scipy.ndimage import label
from deprecated import deprecated
from pathlib import Path
from copy import deepcopy
from typing import Tuple


class Image(object):
    """
    Data class for Image I/O.
    """
    type: str
    image: Union[sitk.Image, nib.Nifti1Image]
    axcodes: tuple
    metadata: pydicom.FileDataset

    def __init__(self, image_type: str, metadata: pydicom.FileDataset = None, image: Union[sitk.Image, nib.Nifti1Image] = None, axcodes: tuple = None):
        """
        Instantiate empty image.
        :param image_type: In which format the image is stored. Either 'sitk' or 'nibabel'.
        :param image: Image to be stored.
        :param axcodes: Axes codes to transform.
        """
        assert image_type in ['sitk', 'nibabel'], 'Image type must be either "sitk" or "nibabel".'
        self.type = image_type
        self.metadata = metadata
        self.image = image
        self.axcodes = axcodes

        if image_type == 'nibabel' and image is not None:
            self.image.get_fdata()  # make sure to load the image data into memory in case the underlying file gets deleted later on

    def copy(self):
        """
        Create a deep copy of the examination.
        :return: A deep copy of the examination.
        """
        return deepcopy(self)

    @classmethod
    def from_sitk(cls, image: sitk.Image):
        """
        Instantiate image from SimpleITK image.
        :param image: SimpleITK image.
        """
        return cls('sitk', image=image)

    @classmethod
    def from_nibabel(cls, image: nib.Nifti1Image):
        """
        Instantiate image from nibabel image.
        :param image: nibabel image.
        """
        return cls('nibabel', image=image)

    def read_image(self, filepath: str | Path):
        """
        Read image from disk.
        :param filepath: Path to image file.
        """
        if self.type == 'sitk':
            self.image = sitk.ReadImage(filepath)
        elif self.type == 'nibabel':
            self.image = nib.load(filepath)
            self.image.get_fdata()  # make sure to load the image data into memory in case the underlying file gets deleted later on

    """
    @property
    def metadata(self):
        return self.metadata

    @metadata.setter
    def metadata(self, metadata: pydicom.FileDataset):
        self.metadata = metadata
    """

    @property
    def array(self):
        return self.image.get_fdata() if self.type == 'nibabel' else sitk.GetArrayFromImage(self.image)

    @property
    def affine(self):
        return self.image.affine if self.type == 'nibabel' else self.image.GetDirection()

    @property
    def header(self):
        return self.image.header if self.type == 'nibabel' else {k: self.image.GetMetaData(k) for k in self.image.GetMetaDataKeys()}

    @property
    def shape(self):
        return self.image.shape if self.type == 'nibabel' else self.image.GetSize()

    @property
    def spacing(self):
        return self.image.header.get_zooms() if self.type == 'nibabel' else self.image.GetSpacing()

    @property
    def origin(self):
        return self.image.affine[:3, 3] if self.type == 'nibabel' else self.image.GetOrigin()

    @property
    def direction(self):
        return self.image.affine[:3, :3] if self.type == 'nibabel' else self.image.GetDirection()

    @deprecated
    def get_array(self):
        """
        Get image data as numpy array.
        :return:
        """
        if self.type == 'nibabel':
            return self.image.get_fdata()
        else:
            return sitk.GetArrayFromImage(self.image)

    @deprecated
    def get_affine(self):
        """
        Get image affine.
        :return:
        """
        if self.type == 'nibabel':
            return self.image.affine
        else:
            return self.image.GetDirection()

    @deprecated
    def get_header(self):
        """
        Get image header.
        :return:
        """
        if self.type == 'nibabel':
            return self.image.header
        else:
            return {k: self.image.GetMetaData(k) for k in self.image.GetMetaDataKeys()}

    @deprecated
    def get_shape(self):
        """
        Get image size.
        :return:
        """
        if self.type == 'nibabel':
            return self.image.shape
        else:
            return self.image.GetSize()

    @deprecated
    def get_spacing(self):
        """
        Get image spacing.
        :return:
        """
        if self.type == 'nibabel':
            return self.image.header.get_zooms()
        else:
            return self.image.GetSpacing()

    @deprecated
    def get_origin(self):
        """
        Get image origin.
        :return:
        """
        if self.type == 'nibabel':
            return self.image.affine[:3, 3]
        else:
            return self.image.GetOrigin()

    @deprecated
    def get_direction(self):
        """
        Get image direction.
        :return:
        """
        if self.type == 'nibabel':
            return self.image.affine[:3, :3]
        else:
            return self.image.GetDirection()

    @deprecated
    def get_transversal_axis(self) -> int:
        """
        Get the dimension (axis) of the image that corresponds to the transversal plane.
        :return: The dimension of the image that corresponds to the transversal plane.
        """
        assert self.axcodes is not None, 'Image has no coordinate system.'
        assert 'I' in self.axcodes or 'S' in self.axcodes, f'Image has invalid coordinate system: {self.axcodes}'

        axstring = ''.join(self.axcodes)
        if (pos:=axstring.find('I')) < 0:  # if 'I' is not in axcodes, look for S
            pos = axstring.find('S')

        return pos

    @deprecated
    def get_coronal_axis(self) -> int:
        """
        Get the dimension (axis) of the image that corresponds to the coronal plane.
        :return: The dimension of the image that corresponds to the coronal plane.
        """
        assert self.axcodes is not None, 'Image has no coordinate system.'
        assert 'A' in self.axcodes or 'P' in self.axcodes, f'Image has invalid coordinate system: {self.axcodes}'

        axstring = ''.join(self.axcodes)
        if (pos:=axstring.find('A')) < 0:  # if 'A' is not in axcodes, look for P
            pos = axstring.find('P')

        return pos

    @deprecated
    def get_sagittal_axis(self) -> int:
        """
        Get the dimension (axis) of the image that corresponds to the sagittal plane.
        :return: The dimension of the image that corresponds to the sagittal plane.
        """
        assert self.axcodes is not None, 'Image has no coordinate system.'
        assert 'L' in self.axcodes or 'R' in self.axcodes, f'Image has invalid coordinate system: {self.axcodes}'

        axstring = ''.join(self.axcodes)
        if (pos:=axstring.find('L')) < 0:  # if 'L' is not in axcodes, look for R
            pos = axstring.find('R')

        return pos

    @deprecated
    def get_transversal_axis_direction(self) -> int:
        """
        Get the direction of the transversal plane, i.e. if indices increase from superior to inferior or vice versa.
        :return: +1 if indices increase from superior to inferior, -1 if indices increase from inferior to superior.
        """
        assert self.axcodes is not None, 'Image has no coordinate system.'
        assert 'I' in self.axcodes or 'S' in self.axcodes, f'Image has invalid coordinate system: {self.axcodes}'

        axstring = ''.join(self.axcodes)
        return 1 if axstring.find('S') < 0 else -1

    @deprecated
    def get_coronal_axis_direction(self) -> int:
        """
        Get the direction of the coronal plane, i.e. if indices increase from anterior to posterior or vice versa.
        :return: +1 if indices increase from anterior to posterior, -1 if indices increase from posterior to anterior.
        """
        assert self.axcodes is not None, 'Image has no coordinate system.'
        assert 'A' in self.axcodes or 'P' in self.axcodes, f'Image has invalid coordinate system: {self.axcodes}'

        axstring = ''.join(self.axcodes)
        return 1 if axstring.find('A') < 0 else -1

    @deprecated
    def get_sagittal_axis_direction(self) -> int:
        """
        Get the direction of the sagittal plane, i.e. if indices increase from left to right patient side or vice versa.
        :return: +1 if indices increase from right to left patient side, -1 if indices increase from left to right patient side.
        """
        assert self.axcodes is not None, 'Image has no coordinate system.'
        assert 'L' in self.axcodes or 'R' in self.axcodes, f'Image has invalid coordinate system: {self.axcodes}'

        axstring = ''.join(self.axcodes)
        return 1 if axstring.find('R') < 0 else -1

    def get_axes(self) -> dict:
        """
        Get axis indices and directions

        :return: A dictionary with the axes and their directions.
        """
        d = dict()
        d['transversal'] = self.get_transversal_axis()
        d['coronal'] = self.get_coronal_axis()
        d['sagittal'] = self.get_sagittal_axis()
        d['transversal_direction'] = self.get_transversal_axis_direction()
        d['coronal_direction'] = self.get_coronal_axis_direction()
        d['sagittal_direction'] = self.get_sagittal_axis_direction()
        return d

    def transform_coordinate_system(self, axcodes: tuple = ('R', 'P', 'I')):
        """
        Transform the image into a standard coordinate system.
        Currently, this is 'LPI', i.e. the first axis is right-left (patient side!), the second axis is anterior-posterior
        and the third axis is superior-inferior.
        :param axcodes: Axes codes to transform.
        :return:
        """
        assert 'I' in axcodes or 'S' in axcodes, f'Axcodes must contain either "I"(nferior) or "S"(uperior), got {axcodes} instead.'
        assert 'A' in axcodes or 'P' in axcodes, f'Axcodes must contain either "A"(nterior) or "P"(osterior), got {axcodes} instead.'
        assert 'L' in axcodes or 'R' in axcodes, f'Axcodes must contain either "L"(eft) or "R"(ight), got {axcodes} instead.'

        if self.type == 'nibabel':
            data = self.image.get_fdata()
            affine = self.image.affine
            orientation = nib.orientations.io_orientation(affine)
            std_orientation = nib.orientations.axcodes2ornt(axcodes)

            transform = nib.orientations.ornt_transform(orientation, std_orientation)
            reoriented_data = nib.orientations.apply_orientation(data, transform)
            new_affine = affine @ nib.orientations.inv_ornt_aff(transform, data.shape)
            self.image = nib.Nifti1Image(reoriented_data, new_affine)
        else:
            raise NotImplementedError('SimpleITK not implemented yet.')

        self.axcodes = axcodes

    def save_image(self, filepath: str):
        """
        Save image to disk.
        :param filepath: Path to save the image to.
        """
        if self.type == 'nibabel':
            nib.save(self.image, filepath)
        else:
            sitk.WriteImage(self.image, filepath)

    def transform_index_to_physical_point(self, index: Union[tuple, list, np.ndarray]):
        """
        Transform index to physical point.
        :param index: Index to transform.
        :return:
        """
        if self.type == 'nibabel':
            affine = self.image.affine
            M = affine[:3, :3]
            abc = affine[:3, 3]
            return M.dot(index) + abc
        else:
            return self.image.TransformIndexToPhysicalPoint(index)

    @staticmethod
    def dicom_to_nibabel(dicom_directory: str) -> Tuple[nib.Nifti1Image, tempfile.TemporaryDirectory]:
        """
        Convert a DICOM series to a nibabel NIfTI image using SimpleITK and nibabel.
        :param dicom_directory: The directory containing the dicom series.
        :return: A Nibabel NIfTI image and a temporary directory containing the converted image.
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            if os.path.isdir(dicom_directory):
                reader = sitk.ImageSeriesReader()
                dicom_names = reader.GetGDCMSeriesFileNames(dicom_directory)
                reader.SetFileNames(dicom_names)
                image = reader.Execute()
            else:
                reader = sitk.ImageFileReader()
                reader.SetFileName(dicom_directory)
                image = reader.Execute()

            tmp = tempfile.TemporaryDirectory()
            sitk.WriteImage(image, os.path.join(tmp.name, 'temp.nii.gz'))
            nib_image = nib.load(os.path.join(tmp.name, 'temp.nii.gz'))

        return nib_image, tmp

    @staticmethod
    def read_dicom_metadata(directory: str) -> pydicom.FileDataset:
        """
        Read metadata from a DICOM series.
        :param directory: The directory containing the DICOM series.
        :return:
        """
        for file in Path(directory).iterdir():
            dataset = pydicom.dcmread(file)
            break  # just take a random file, should be the same for all

        return dataset

class Segmentation(Image):

    def remove_outliers(self, threshold_ratio: float = 0.1):
        """
        Remove outliers from segmentation mask.
        :param threshold_ratio: Threshold ratio for outlier removal. Percentage of the mask size, where all components
        with less voxels are removed.
        :return:
        """
        if self.type == 'nibabel':
            data = self.get_array()
            structure = np.ones((3, 3, 3), dtype=bool)
            cleaned_data = np.zeros_like(data)
            unique_labels = np.unique(data)

            for label_value in unique_labels:
                if label_value == 0:
                    continue

                label_mask = (data == label_value)
                labeled_array, _ = label(label_mask, structure=structure)
                sizes = np.bincount(labeled_array.ravel())
                sizes[0] = 0
                threshold = threshold_ratio * np.count_nonzero(label_mask)
                large_components = sizes > threshold
                cleaned_data[large_components[labeled_array]] = label_value

            self.image = nib.Nifti1Image(cleaned_data, self.get_affine())
        else:
            raise NotImplementedError('SimpleITK not implemented yet.')
