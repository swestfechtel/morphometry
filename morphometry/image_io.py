import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Tuple
from typing import Union

import SimpleITK as sitk
import nibabel as nib
import numpy as np
import pydicom
from scipy.ndimage import label


class Image(object):
    """
    Data class for Image I/O.
    """
    type: str
    image: Union[sitk.Image, nib.Nifti1Image]
    axcodes: tuple
    metadata: pydicom.FileDataset

    def __init__(self, image_type: str, metadata: pydicom.FileDataset = None,
                 image: Union[sitk.Image, nib.Nifti1Image] = None, axcodes: tuple = None):
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
        return self.image.header if self.type == 'nibabel' else {k: self.image.GetMetaData(k) for k in
                                                                 self.image.GetMetaDataKeys()}

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

    def transform_coordinate_system(self, axcodes: tuple = ('L', 'P', 'I')):
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

    def transform_physical_point_to_index(self, point: Union[tuple, list, np.ndarray]):
        """
        Transform physical point to index.
        :param point: Physical point to transform.
        :return:
        """
        if self.type == 'nibabel':
            affine = self.image.affine
            M = affine[:3, :3]
            abc = affine[:3, 3]
            return np.linalg.inv(M).dot(np.array(point) - abc)
        else:
            return self.image.TransformPhysicalPointToIndex(point)

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
            data = self.array
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

            self.image = nib.Nifti1Image(cleaned_data, self.affine)
        else:
            raise NotImplementedError('SimpleITK not implemented yet.')


def split_ct_image(segmentation: Segmentation, image: Image = None, femur_label: int = 1, hip_label: int = 7, patella_label: int = 5, fibula_label: int = 3) -> Tuple[
    Tuple[Image, Image, Image], Tuple[Segmentation, Segmentation, Segmentation]]:
    """
    Split a CT image to extract the hip, knee and ankle regions.
    :param image: A whole-body CT image.
    :param segmentation: A segmentation mask corresponding to the CT image.
    :param femur_label: Label of the femur in the segmentation mask.
    :param hip_label: Label of the hip in the segmentation mask.
    :param patella_label: Label of the patella in the segmentation mask.
    :param fibula_label: Label of the fibula in the segmentation mask.
    :return: CT images of the hip, knee and ankle regions.
    """
    hip_start = np.min(np.argwhere(segmentation.array == hip_label)[:, 2])  # get the most proximal slice of the hip bone
    femur_end = np.max(np.argwhere(segmentation.array == femur_label)[:, 2])  # get the most distal slice of the femur
    num_femur_slices = femur_end - hip_start + 1
    hip_end = hip_start + int(0.3 * num_femur_slices)

    knee_start = np.min(np.argwhere(segmentation.array == patella_label)[:, 2])  # get the most proximal slice of the patella
    knee_end = np.min(np.argwhere(segmentation.array == fibula_label)[:, 2])  # get the most proximal slice of the fibula

    ankle_end = np.max(np.argwhere(segmentation.array == fibula_label)[:, 2])  # get the most distal slice of the fibula
    num_fibula_slices = ankle_end - knee_end + 1
    ankle_start = ankle_end - int(0.3 * num_fibula_slices)

    print(f"Hip slices: {hip_start} to {hip_end}, Knee slices: {knee_start} to {knee_end}, Ankle slices: {ankle_start} to {ankle_end}")

    if image is not None:
        hip_image = Image(image.type)
        knee_image = Image(image.type)
        ankle_image = Image(image.type)

        hip_image.image = nib.Nifti1Image(image.array[:, :, hip_start:hip_end], image.affine,
                                          image.header) if image.type == 'nibabel' else sitk.RegionOfInterest(image.image,
                                                                                                              size=[
                                                                                                                  image.shape[
                                                                                                                      0],
                                                                                                                  image.shape[
                                                                                                                      1],
                                                                                                                  hip_end - hip_start],
                                                                                                              index=[0, 0,
                                                                                                                     hip_start])
        knee_image.image = nib.Nifti1Image(image.array[:, :, knee_start:knee_end], image.affine,
                                           image.header) if image.type == 'nibabel' else sitk.RegionOfInterest(image.image,
                                                                                                               size=[
                                                                                                                   image.shape[
                                                                                                                       0],
                                                                                                                   image.shape[
                                                                                                                       1],
                                                                                                                   knee_end - knee_start],
                                                                                                               index=[0, 0,
                                                                                                                      knee_start])
        ankle_image.image = nib.Nifti1Image(image.array[:, :, ankle_start:ankle_end], image.affine,
                                            image.header) if image.type == 'nibabel' else sitk.RegionOfInterest(image.image,
                                                                                                                size=[
                                                                                                                    image.shape[
                                                                                                                        0],
                                                                                                                    image.shape[
                                                                                                                        1],
                                                                                                                    ankle_end - ankle_start],
                                                                                                                index=[0, 0,
                                                                                                                       ankle_start])

        hip_image.metadata = image.metadata
        knee_image.metadata = image.metadata
        ankle_image.metadata = image.metadata

    hip_segmentation = Segmentation(segmentation.type)
    knee_segmentation = Segmentation(segmentation.type)
    ankle_segmentation = Segmentation(segmentation.type)

    hip_segmentation.image = nib.Nifti1Image(segmentation.array[:, :, hip_start:hip_end], segmentation.affine,
                                             segmentation.header) if segmentation.type == 'nibabel' else sitk.RegionOfInterest(
        segmentation.image, size=[segmentation.shape[0], segmentation.shape[1], hip_end - hip_start],
        index=[0, 0, hip_start])
    knee_segmentation.image = nib.Nifti1Image(segmentation.array[:, :, knee_start:knee_end], segmentation.affine,
                                              segmentation.header) if segmentation.type == 'nibabel' else sitk.RegionOfInterest(
        segmentation.image, size=[segmentation.shape[0], segmentation.shape[1], knee_end - knee_start],
        index=[0, 0, knee_start])
    ankle_segmentation.image = nib.Nifti1Image(segmentation.array[:, :, ankle_start:ankle_end], segmentation.affine,
                                               segmentation.header) if segmentation.type == 'nibabel' else sitk.RegionOfInterest(
        segmentation.image, size=[segmentation.shape[0], segmentation.shape[1], ankle_end - ankle_start],
        index=[0, 0, ankle_start])

    hip_segmentation.metadata = segmentation.metadata
    knee_segmentation.metadata = segmentation.metadata
    ankle_segmentation.metadata = segmentation.metadata

    if image is None:
        return None, (hip_segmentation, knee_segmentation, ankle_segmentation)

    return (hip_image, knee_image, ankle_image), (hip_segmentation, knee_segmentation, ankle_segmentation)
