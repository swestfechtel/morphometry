"""Render axial slices (and segmentation overlays) to base64 PNGs.

Ported from the old ``TorsionExamination.encode_images`` (which ran in the GET
request path). This now runs worker-side; callers persist the returned base64
lists to disk via the Store. The matplotlib + multiprocessing work stays here,
off the API event loop.
"""
import base64
import multiprocessing
from io import BytesIO
from typing import Tuple, Union

import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure

from morphometry.image_io import Image, Segmentation


def encode_figure(layer: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]) -> str:
    """Render one axial slice (optionally with a segmentation overlay) to a base64 PNG."""
    image_layer = layer[0] if isinstance(layer, tuple) else layer
    segmentation_layer = layer[1] if isinstance(layer, tuple) else None

    fig = Figure(figsize=(20, 20))
    ax = fig.subplots()
    ax.imshow(image_layer.T, cmap="gray")

    if segmentation_layer is not None:
        cmap = ListedColormap(["white", "yellow", "purple", "cyan"])
        norm = BoundaryNorm([-1, 0.5, 1.5, 2.5, 3.5], cmap.N)
        ax.imshow(np.where(segmentation_layer == 0, np.nan, segmentation_layer).T, cmap=cmap, norm=norm, alpha=0.5)

    ax.axis("off")
    buffer = BytesIO()
    fig.savefig(buffer, format="png", transparent=True, bbox_inches="tight")
    return base64.b64encode(buffer.getbuffer()).decode("ascii")


def encode_torsion_images(transformed_image: Image, hip_mask: Segmentation, knee_mask: Segmentation,
                          ankle_mask: Segmentation, pool_size: int | None = None) -> Tuple[list[str], list[str]]:
    """Encode the grayscale slices and the segmentation-overlay slices.

    :param transformed_image: The LPI-oriented full volume.
    :param hip_mask: Hip segmentation (labels relabeled into the combined volume).
    :param knee_mask: Knee segmentation.
    :param ankle_mask: Ankle segmentation (tibia/fibula relabeled to match).
    :param pool_size: Multiprocessing pool size (None = default).
    :return: ``(image_b64, segmentation_b64)`` lists, one entry per axial slice.
    """
    image = transformed_image.array
    image_layers = [image[:, :, i] for i in range(image.shape[-1])]

    with multiprocessing.Pool(pool_size) as pool:
        image_b64 = pool.map(encode_figure, image_layers)

    relabelled_ankle = ankle_mask.array.copy()
    relabelled_ankle = np.where(relabelled_ankle == 2, 3, relabelled_ankle)
    relabelled_ankle = np.where(relabelled_ankle == 1, 2, relabelled_ankle)
    segmented = np.concatenate((hip_mask.array, knee_mask.array, relabelled_ankle), axis=2)
    pairs = [(image_layers[i], segmented[:, :, i]) for i in range(len(image_layers))]

    with multiprocessing.Pool(pool_size) as pool:
        segmentation_b64 = pool.map(encode_figure, pairs)

    return image_b64, segmentation_b64
