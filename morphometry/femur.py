import numpy as np
from morphometry.hip import get_femoral_head_center


def get_proximal_reference_line(segmentation_mask: np.array, side: str = 'left', segmentation_label: int = 1) -> np.array:
    """
    Get the proximal reference line of a segmentation mask for calculating the femoral torsion.
    :param segmentation_mask: A segmentation mask of the proximal femur.
    :param side: Side of the image (not patient!), either 'left' or 'right'.
    :param segmentation_label: The label of the segmentation mask.
    :return:
    """
    assert side in ['left', 'right'], 'Side must be either "left" or "right"'
    segmentation_mask = np.where(segmentation_mask == segmentation_label, 1, 0)

    radius, center = get_femoral_head_center(segmentation_mask, side=side, segmentation_label=segmentation_label)