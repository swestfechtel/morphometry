import SimpleITK as sitk
import nibabel as nib
import numpy as np
from typing import Union


class Image:
    """
    Data class for Image I/O.
    """
    def __init__(self, image_type: str):
        """
        Instantiate empty image.
        :param image_type: In which format the image is stored. Either 'sitk' or 'nibabel'.
        """
        assert image_type in ['sitk', 'nibabel'], 'Image type must be either "sitk" or "nibabel".'
        self.type = image_type
        self.image = None

    def __init__(self, image: sitk.Image):
        """
        Instantiate image from SimpleITK image.
        :param image: SimpleITK image.
        """
        self.type = 'sitk'
        self.image = image

    def __init__(self, image: nib.Nifti1Image):
        """
        Instantiate image from nibabel image.
        :param image: nibabel image.
        """
        self.type = 'nibabel'
        self.image = image

    def read_image(self, filepath: str):
        """
        Read image from disk.
        :param filepath: Path to image file.
        """
        if self.type == 'sitk':
            self.image = sitk.ReadImage(filepath)
        elif self.type == 'nibabel':
            self.image = nib.load(filepath)

    def transform_coordinate_system(self):
        """
        Transform the image into a standard coordinate system.
        Currently, this is 'IPR', i.e. the first axis is superior-inferior, the second axis is anterior-posterior
        and the third axis is left-right.
        :return:
        """
        if self.type == 'nibabel':
            data = self.image.get_fdata()
            affine = self.image.affine
            orientation = nib.orientations.io_orientation(affine)
            std_orientation = nib.orientations.axcodes2ornt(('I', 'P', 'R'))

            transform = nib.orientations.ornt_transform(orientation, std_orientation)
            reoriented_data = nib.orientations.apply_orientation(data, transform)
            new_affine = affine @ nib.orientations.inv_ornt_aff(transform, data.shape)
            self.image = nib.Nifti1Image(reoriented_data, new_affine)
        else:
            raise NotImplementedError('SimpleITK not implemented yet.')

    def get_array(self):
        """
        Get image data as numpy array.
        :return:
        """
        if self.type == 'nibabel':
            return self.image.get_fdata()
        else:
            return sitk.GetArrayFromImage(self.image)

    def get_affine(self):
        """
        Get image affine.
        :return:
        """
        if self.type == 'nibabel':
            return self.image.affine
        else:
            return self.image.GetDirection()

    def get_header(self):
        """
        Get image header.
        :return:
        """
        if self.type == 'nibabel':
            return self.image.header
        else:
            return self.image.GetMetaData()

    def get_size(self):
        """
        Get image size.
        :return:
        """
        if self.type == 'nibabel':
            return self.image.shape
        else:
            return self.image.GetSize()

    def get_spacing(self):
        """
        Get image spacing.
        :return:
        """
        if self.type == 'nibabel':
            return self.image.header.get_zooms()
        else:
            return self.image.GetSpacing()

    def get_origin(self):
        """
        Get image origin.
        :return:
        """
        if self.type == 'nibabel':
            return self.image.affine[:3, 3]
        else:
            return self.image.GetOrigin()

    def get_direction(self):
        """
        Get image direction.
        :return:
        """
        if self.type == 'nibabel':
            return self.image.affine[:3, :3]
        else:
            return self.image.GetDirection()

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