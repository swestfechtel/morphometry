import numpy as np
from morphometry.knee import get_knee_reference_line
from morphometry.bresenham import bresenhamline
from morphometry.utils import calculate_angle_between_vectors
from scipy.ndimage import center_of_mass
from skimage.measure import regionprops, label
from typing import Tuple
from matplotlib import pyplot as plt
from morphometry.utils import draw_line


def get_layer_with_largest_diameter(segmentation_mask: np.ndarray) -> int:
    """
    Return the layer with the biggest ellipse covering the mask area.

    For each layer, the diameter of an ellipse covering an area equivalent to the mask area is calculated. The layer
    with the biggest diameter is returned.
    :param segmentation_mask: A 3D segmentation mask.
    :return: The layer index with the biggest mask area.
    """

    # one entry per axial layer. This was sized on shape[0] (the sagittal axis), so a mask with
    # fewer sagittal columns than axial layers raised IndexError, and an empty mask returned the
    # last layer index (argsort of all zeros) as if it held the widest tibia. The array is kept at
    # max(shape[0], shape[2]) rather than shape[2] on purpose: argsort's default sort is not
    # stable, so with tied maximum diameters the winning layer depends on the array's length, and
    # the trailing zeros of the old length keep every previously-valid input bit-identical. The
    # padding can never win: an all-zero array is refused below.
    diameter = np.zeros(max(segmentation_mask.shape[0], segmentation_mask.shape[2]))
    # save diameters of the layers
    for i in range(segmentation_mask.shape[2]):
        if len(np.nonzero(segmentation_mask[:, :, i])[0]) != 0:
            props = regionprops(label(segmentation_mask[:, :, i]))
            if props.__len__() > 1:
                i_biggest = 0
                for j in range(props.__len__()):
                    if props[j].equivalent_diameter > props[
                            i_biggest].equivalent_diameter:
                        i_biggest = j
                diameter[i] = props[i_biggest].equivalent_diameter
            else:
                diameter[i] = props[0].equivalent_diameter

    if not np.any(diameter > 0):
        raise ValueError('get_layer_with_largest_diameter: the mask is empty on every axial layer')

    # find index of the layer with the biggest diameter
    indices = np.argsort(diameter)
    return indices[-1]


def _label_centroid(mask_2d: np.ndarray, name: str) -> Tuple[float, float]:
    """
    Return the centre of mass of every voxel of one label on one axial layer, refusing an empty layer.

    The centroid deliberately averages **all** components of the label on the slice (legacy
    behaviour, unchanged). Taking the largest connected component instead was tried for the NMDID
    review's defect #9 on 2026-11-16 and withdrawn the same day: it repairs a small fragment of
    another leg (188320 L) but on legs crossed at the ankle the *other* leg's bone is the larger
    piece -- on the CT golden fixture case-105094 image-left both the larger tibia (3157 vs 1171
    voxels) and the larger fibula (722 vs 158) on the reference layer are the right leg's, so the
    tibial torsion moved 66.84 -> 71.33 degrees, from one wrong leg mixture to the wrong leg. The
    fix is 3-D identity on the whole volume, before the split
    (:func:`morphometry.geometry.remove_other_halfs_bones`, applied by ``scripts/process_nmdid.py``);
    callers that split a whole-body volume themselves should do the same.
    :param mask_2d: A 2D binary mask (one axial layer of one label).
    :param name: The bone's name, used in the error message.
    :return: The (sagittal, coronal) centre of mass, in index units.
    :raises ValueError: If the label is absent from this layer (formerly a silent NaN centroid).
    """
    if not np.any(mask_2d):
        raise ValueError(f'get_distal_reference_line: no {name} voxels on the reference layer')
    return center_of_mass(mask_2d)


def get_distal_reference_line(segmentation_mask: np.ndarray, tibia_label: int, fibula_label: int) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Get the distal reference line of a segmentation mask for calculating the tibial torsion.

    Finds the layer with the biggest diameter of the tibia and calculates the center of mass of the tibia and fibula
    on that layer. The reference line connects both center of mass points. Each centre of mass
    averages every voxel of the label on the layer, so fragments of another leg -- or a whole
    crossed leg -- drag it unless the caller removed them first (see :func:`_label_centroid`); a label absent from the
    layer raises ``ValueError`` (since 2026-11-16; formerly a NaN centroid).
    :param segmentation_mask: A 3D segmentation mask of the tibia and fibula.
    :param tibia_label: The segmentation label of the tibia.
    :param fibula_label: The segmentation label of the fibula.
    :return: The layer and start and end points of the reference line.
    """
    tibia_mask = np.where(segmentation_mask == tibia_label, 1, 0)
    fibula_mask = np.where(segmentation_mask == fibula_label, 1, 0)
    # find index of the layer with the biggest diameter of the tibia
    layer_index = get_layer_with_largest_diameter(tibia_mask)

    # centre of mass of all tibia and fibula voxels on the layer (an absent label raises; see
    # _label_centroid for why the largest component is NOT taken)
    com_tibia = _label_centroid(tibia_mask[:, :, layer_index], 'tibia')
    com_fibula = _label_centroid(fibula_mask[:, :, layer_index], 'fibula')

    # transform points from layer mask to 3D mask
    com_tibia = np.array([com_tibia[0], com_tibia[1], layer_index])
    com_fibula = np.array([com_fibula[0], com_fibula[1], layer_index])

    return layer_index, com_tibia, com_fibula
