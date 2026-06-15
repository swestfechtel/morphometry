"""Knee cartilage thickness measurement entry points.

The heavy landmark / subregion machinery lives in the ``Tibia`` and ``Femur``
classes in ``morphometry.cartilage.knee`` (kept there as region-specific
machinery, analogous to the ``get_*`` helpers). These thin functional wrappers
provide the public measurement surface in ``measurements``; the classes are also
re-exported for callers that need the intermediate landmarks.
"""
from morphometry.cartilage.knee import Tibia, Femur

__all__ = [
    "calculate_tibial_cartilage_thickness",
    "calculate_femoral_cartilage_thickness",
    "calculate_knee_cartilage_thickness",
    "Tibia",
    "Femur",
]


def calculate_tibial_cartilage_thickness(image, cartilage_label: int, method: str = 'knn') -> dict:
    """
    Calculate tibial cartilage thickness per subregion.
    :param image: An Image of the knee cartilage segmentation.
    :param cartilage_label: The label of the tibial cartilage.
    :param method: Thickness method, 'knn' or 'mesh'.
    :return: A dict mapping each subregion to its per-point thickness map.
    """
    return Tibia(image, cartilage_label).calculate_thickness(method)


def calculate_femoral_cartilage_thickness(image, tibia: Tibia, cartilage_label: int, method: str = 'knn') -> dict:
    """
    Calculate femoral cartilage thickness per subregion.

    Requires a ``Tibia`` whose landmarks have been computed (e.g. via
    :func:`calculate_tibial_cartilage_thickness` or ``Tibia.calculate_thickness``),
    because the femoral subregions are defined relative to the tibial plateau.
    :param image: An Image of the knee cartilage segmentation.
    :param tibia: A Tibia instance with computed landmarks.
    :param cartilage_label: The label of the femoral cartilage.
    :param method: Thickness method, 'knn' or 'mesh'.
    :return: A dict mapping each subregion to its per-point thickness map.
    """
    return Femur(image, cartilage_label).calculate_thickness(tibia, method)


def calculate_knee_cartilage_thickness(image, femur_label: int, tibia_label: int, method: str = 'knn') -> dict:
    """
    Calculate both tibial and femoral cartilage thickness in one call.

    Mirrors the standard workflow (build the tibia, then the femur relative to it).
    :param image: An Image of the knee cartilage segmentation.
    :param femur_label: The label of the femoral cartilage.
    :param tibia_label: The label of the tibial cartilage.
    :param method: Thickness method, 'knn' or 'mesh'.
    :return: ``{'tibia': <tibia subregions>, 'femur': <femur subregions>}``.
    """
    tibia = Tibia(image, tibia_label)
    tibia_results = tibia.calculate_thickness(method)
    femur_results = Femur(image, femur_label).calculate_thickness(tibia, method)
    return {'tibia': tibia_results, 'femur': femur_results}
