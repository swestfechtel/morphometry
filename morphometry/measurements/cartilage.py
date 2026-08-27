"""Knee cartilage thickness measurement entry points.

The heavy landmark / subregion machinery lives in the ``Tibia`` and ``Femur``
classes in ``morphometry.cartilage.knee`` (kept there as region-specific
machinery, analogous to the ``get_*`` helpers). These thin functional wrappers
provide the public measurement surface in ``measurements``; the classes are also
re-exported for callers that need the intermediate landmarks.
"""
from typing import Optional

import pyvista as pv

from morphometry.cartilage.knee import (
    Tibia,
    Femur,
    plot_knee_segments,
    plot_knee_thickness,
    find_thinnest_area,
    plot_thinnest_point,
)

__all__ = [
    "calculate_tibial_cartilage_thickness",
    "calculate_femoral_cartilage_thickness",
    "calculate_knee_cartilage_thickness",
    "calculate_thinnest_cartilage_area",
    "Tibia",
    "Femur",
    "plot_knee_segments",
    "plot_knee_thickness",
    "find_thinnest_area",
    "plot_thinnest_point",
]


def calculate_tibial_cartilage_thickness(image, cartilage_label: int, method: str = 'knn',
                                         plot: bool | pv.Plotter = False) -> dict:
    """
    Calculate tibial cartilage thickness per subregion.
    :param image: An Image of the knee cartilage segmentation.
    :param cartilage_label: The label of the tibial cartilage.
    :param method: Thickness method, 'knn' or 'mesh'.
    :param plot: Pass a PyVista ``Plotter`` to draw the tibial thickness heatmap into it
        (the articular surface coloured by thickness). ``False`` disables plotting.
    :return: A dict mapping each subregion to its per-point thickness map.
    """
    tibia = Tibia(image, cartilage_label)
    result = tibia.calculate_thickness(method)
    if isinstance(plot, pv.Plotter):
        tibia.plot_thickness(result, plotter=plot, show=False)
    return result


def calculate_femoral_cartilage_thickness(image, tibia: Tibia, cartilage_label: int, method: str = 'knn',
                                          plot: bool | pv.Plotter = False) -> dict:
    """
    Calculate femoral cartilage thickness per subregion.

    Requires a ``Tibia`` whose landmarks have been computed (e.g. via
    :func:`calculate_tibial_cartilage_thickness` or ``Tibia.calculate_thickness``),
    because the femoral subregions are defined relative to the tibial plateau.
    :param image: An Image of the knee cartilage segmentation.
    :param tibia: A Tibia instance with computed landmarks.
    :param cartilage_label: The label of the femoral cartilage.
    :param method: Thickness method, 'knn' or 'mesh'.
    :param plot: Pass a PyVista ``Plotter`` to draw the femoral thickness heatmap into it.
        ``False`` disables plotting.
    :return: A dict mapping each subregion to its per-point thickness map.
    """
    femur = Femur(image, cartilage_label)
    result = femur.calculate_thickness(tibia, method)
    if isinstance(plot, pv.Plotter):
        femur.plot_thickness(result, tibia=tibia, plotter=plot, show=False)
    return result


def calculate_knee_cartilage_thickness(image, femur_label: int, tibia_label: int, method: str = 'knn',
                                       plot: bool | pv.Plotter = False) -> dict:
    """
    Calculate both tibial and femoral cartilage thickness in one call.

    Mirrors the standard workflow (build the tibia, then the femur relative to it).
    :param image: An Image of the knee cartilage segmentation.
    :param femur_label: The label of the femoral cartilage.
    :param tibia_label: The label of the tibial cartilage.
    :param method: Thickness method, 'knn' or 'mesh'.
    :param plot: Pass a PyVista ``Plotter`` to draw a combined tibia+femur thickness
        heatmap (both articular surfaces, shared colour scale). ``False`` disables plotting.
    :return: ``{'tibia': <tibia subregions>, 'femur': <femur subregions>}``.
    """
    tibia = Tibia(image, tibia_label)
    tibia_results = tibia.calculate_thickness(method)
    femur = Femur(image, femur_label)
    femur_results = femur.calculate_thickness(tibia, method)
    if isinstance(plot, pv.Plotter):
        plot_knee_thickness(tibia, femur, tibia_results, femur_results, plotter=plot, show=False)
    return {'tibia': tibia_results, 'femur': femur_results}


def calculate_thinnest_cartilage_area(image, femur_label: int, tibia_label: int,
                                      method: str = 'knn', neighbourhood_mm: float = 3.0,
                                      radius_mm: Optional[float] = None,
                                      plot: bool | pv.Plotter = False) -> dict:
    """
    Find the thinnest area of the tibiofemoral cartilage in one call.

    Computes the tibial and femoral thickness maps and locates the thinnest neighbourhood
    of the combined (contact-region) cartilage, robust to single-voxel outliers. See
    :func:`morphometry.cartilage.knee.find_thinnest_area` for the algorithm.

    :param image: An Image of the knee cartilage segmentation.
    :param femur_label: The label of the femoral cartilage.
    :param tibia_label: The label of the tibial cartilage.
    :param method: Thickness method, 'knn' or 'mesh'.
    :param neighbourhood_mm: Diameter (mm) of the median-smoothing neighbourhood.
    :param radius_mm: Radius (mm) of the refinement disc (defaults to ``neighbourhood_mm / 2``).
    :param plot: Pass a PyVista ``Plotter`` to render the cartilage with an arrow marking
        the thinnest point (thickness-heatmap base). ``False`` disables plotting.
    :return: The result dict from :func:`find_thinnest_area`.
    """
    tibia = Tibia(image, tibia_label)
    tibia_results = tibia.calculate_thickness(method)
    femur = Femur(image, femur_label)
    femur_results = femur.calculate_thickness(tibia, method)
    thinnest = find_thinnest_area(tibia_results, femur_results, image,
                                  neighbourhood_mm=neighbourhood_mm, radius_mm=radius_mm)
    if isinstance(plot, pv.Plotter):
        plot_thinnest_point(tibia, femur, tibia_results, femur_results, thinnest,
                            plotter=plot, show=False)
    return thinnest
