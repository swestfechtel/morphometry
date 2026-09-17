import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Tuple
from typing import Union

import SimpleITK as sitk
import nibabel as nib
import logging
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
    def dicom_files_to_nibabel(files) -> Tuple[nib.Nifti1Image, tempfile.TemporaryDirectory]:
        """Convert an explicit, ordered list of DICOM slice files to a nibabel NIfTI.

        Like :meth:`dicom_to_nibabel`, but takes the exact files — already sorted into
        slice order — rather than a directory. This lets a caller convert a single
        *contiguous* slab of a multi-slab acquisition on its own, so SimpleITK derives
        the true within-slab z-spacing. Handing SimpleITK a whole gapped series
        (e.g. a torsion scan with separate hip/knee/ankle stations) instead forces all
        slices onto one uniform grid whose spacing is the gap-averaged extent / (n-1),
        which stretches each slab and corrupts every superior-inferior distance.

        :param files: DICOM file paths in slice order (one contiguous slab).
        :return: A nibabel NIfTI image and the ``TemporaryDirectory`` backing it (keep
            it alive until the image data has been copied out; the caller cleans it up).
        """
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames([str(f) for f in files])
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


def _crop_axis(source, axis: int, start: int, end: int):
    """
    Crop an Image / Segmentation to the index range ``[start, end)`` along one index axis.

    The crop keeps the source geometry: index ``i`` of the crop maps to the same physical
    point as the source's index ``i`` advanced by ``start`` along ``axis``. For nibabel the
    affine's translation is advanced by ``start`` voxels along ``axis`` (the linear
    direction/spacing block is unchanged); ``sitk.RegionOfInterest`` sets the origin of the
    region on its own. Without that shift every ``transform_index_to_physical_point`` on
    the crop would be off by ``start`` voxels along ``axis`` -- a constant translation that
    leaves in-crop distances intact but misplaces the crop in the scanner frame.

    The axis is an *index* axis as used by ``transform_index_to_physical_point``, i.e. the
    nibabel array axis and the SimpleITK ``(x, y, z)`` index axis. (``Image.array`` of a
    SimpleITK image is ``(z, y, x)``, so index axis 0 there is the array's last axis.)
    :param source: The Image or Segmentation to crop.
    :param axis: The index axis to crop along (0, 1 or 2).
    :param start: First voxel of the crop along ``axis`` (inclusive).
    :param end: Last voxel of the crop along ``axis`` (exclusive).
    :return: The underlying nibabel / SimpleITK image of the crop.
    """
    if source.type == 'nibabel':
        offset = np.zeros(3)
        offset[axis] = start
        affine = source.affine.copy()
        affine[:3, 3] = affine[:3, :3].dot(offset) + affine[:3, 3]
        slicer = [slice(None)] * 3
        slicer[axis] = slice(start, end)
        return nib.Nifti1Image(source.array[tuple(slicer)], affine, source.header)

    size = list(source.shape)
    size[axis] = end - start
    index = [0, 0, 0]
    index[axis] = start
    return sitk.RegionOfInterest(source.image, size=size, index=index)


def _crop_z(source, start: int, end: int):
    """
    Crop an Image / Segmentation along the superior-inferior (third) axis.

    Thin wrapper around :func:`_crop_axis` with ``axis=2``: the crop's affine is advanced
    by ``start`` slices so that index ``0`` of the crop maps to the same physical point as
    index ``start`` of the source.
    :param source: The Image or Segmentation to crop.
    :param start: First slice of the crop (inclusive).
    :param end: Last slice of the crop (exclusive).
    :return: The underlying nibabel / SimpleITK image of the crop.
    """
    return _crop_axis(source, 2, start, end)


def crop_image(source, axis: int, start: int, end: int):
    """
    Crop an Image / Segmentation to ``[start, end)`` along one index axis, keeping its geometry correct.

    Public wrapper around :func:`_crop_axis`: index ``i`` of the crop maps to the same physical
    point as the source's index ``i`` advanced by ``start`` along ``axis``, so sub-volumes can be
    handled in physical space without knowing where they were cut from.
    :param source: The Image or Segmentation to crop (nibabel or SimpleITK backed). It is not modified.
    :param axis: The index axis to crop along (0, 1 or 2; for SimpleITK the ``(x, y, z)`` index axis).
    :param start: First voxel of the crop along ``axis`` (inclusive).
    :param end: Last voxel of the crop along ``axis`` (exclusive).
    :return: The crop, of the same class and backend as ``source``, carrying over its
    ``metadata`` and ``axcodes``.
    """
    crop = type(source)(source.type, metadata=source.metadata, axcodes=source.axcodes)
    crop.image = _crop_axis(source, axis, start, end)
    return crop


def split_image_left_right(source):
    """
    Split an Image / Segmentation into its image-left and image-right halves, with correct geometry.

    The split is at ``shape[0] // 2`` along the first index axis (the left-right axis once
    the image is in LPI via ``transform_coordinate_system``): the left half holds indices
    ``[0, half)``, the right half ``[half, shape[0])``, so for odd widths the right half is
    one voxel wider. "left" / "right" refer to the *image* side, not the patient side.

    Geometry guarantee: the left half keeps the parent's geometry unchanged, and index
    ``i`` of the right half maps to the same physical point as the parent's index
    ``i + [half, 0, 0]`` (its origin is advanced by ``half`` voxels, see
    :func:`crop_image`); spacing and direction are unchanged on both halves. The
    translation matters for everything that goes through physical space: wrapping the
    right half in the parent's affine made the right half's coordinates overlay the left
    half, so every right-side physical point landed on the left leg, ``half * spacing_x``
    away from its true position (195 mm on the NMDID whole-leg CT test case, where the two
    femoral head centres came out ~17 mm apart instead of ~195 mm). Distances and angles
    measured within one half are translation-invariant and unaffected, and callers that
    move between the halves in *index* space must still add ``half`` to right-half indices
    themselves.

    The pure array counterpart without any geometry is ``morphometry.geometry.split_left_right``.
    :param source: The Image or Segmentation to split (nibabel or SimpleITK backed). It is not modified.
    :return: ``(left, right)``, each of the same class and backend as ``source``, carrying
    over its ``metadata`` and ``axcodes``.
    """
    half = source.shape[0] // 2
    return crop_image(source, 0, 0, half), crop_image(source, 0, half, source.shape[0])


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

    logging.debug("Hip slices: %s to %s, Knee slices: %s to %s, Ankle slices: %s to %s",
                  hip_start, hip_end, knee_start, knee_end, ankle_start, ankle_end)

    if image is not None:
        hip_image = Image(image.type)
        knee_image = Image(image.type)
        ankle_image = Image(image.type)

        hip_image.image = _crop_z(image, hip_start, hip_end)
        knee_image.image = _crop_z(image, knee_start, knee_end)
        ankle_image.image = _crop_z(image, ankle_start, ankle_end)

        hip_image.metadata = image.metadata
        knee_image.metadata = image.metadata
        ankle_image.metadata = image.metadata

    hip_segmentation = Segmentation(segmentation.type)
    knee_segmentation = Segmentation(segmentation.type)
    ankle_segmentation = Segmentation(segmentation.type)

    hip_segmentation.image = _crop_z(segmentation, hip_start, hip_end)
    knee_segmentation.image = _crop_z(segmentation, knee_start, knee_end)
    ankle_segmentation.image = _crop_z(segmentation, ankle_start, ankle_end)

    hip_segmentation.metadata = segmentation.metadata
    knee_segmentation.metadata = segmentation.metadata
    ankle_segmentation.metadata = segmentation.metadata

    if image is None:
        return None, (hip_segmentation, knee_segmentation, ankle_segmentation)

    return (hip_image, knee_image, ankle_image), (hip_segmentation, knee_segmentation, ankle_segmentation)
