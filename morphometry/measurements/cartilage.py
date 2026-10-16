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
    find_thinnest_area_by_compartment,
    split_tibia_thickness_by_compartment,
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
    "find_thinnest_area_by_compartment",
    "split_tibia_thickness_by_compartment",
    "plot_thinnest_point",
]


def calculate_tibial_cartilage_thickness(image, cartilage_label: int, method=None,
                                         plot: bool | pv.Plotter = False,
                                         tibia_bone_label: int = 2) -> dict:
    """
    Calculate tibial cartilage thickness per subregion (Wirth & Eckstein 2008, section II-C).

    :param image: An Image of the knee cartilage segmentation.
    :param cartilage_label: The label of the tibial cartilage.
    :param method: Removed; must be ``None``. There is one thickness definition (the
        area-weighted surface distance), so the old ``'knn'`` / ``'mesh'`` values raise -- see
        :meth:`morphometry.cartilage.knee.Tibia.calculate_thickness`.
    :param plot: Pass a PyVista ``Plotter`` to draw the tibial thickness heatmap into it
        (the subchondral bone area coloured by thickness). ``False`` disables plotting.
    :param tibia_bone_label: The label of the tibial bone. The tibial subregions (Wirth &
        Eckstein 2008) are defined on its subchondral surface, so it must be present.
    :return: The :class:`~morphometry.cartilage.thickness.ThicknessMaps` of the tibia: a
        ``{subregion: {(u, v): thickness_mm}}`` dict keyed in the shared in-plane frame, with
        the per-subregion ``statistics`` (aMe, tab_aMe, aMax, aMav, aSD, aMin, aMiv) attached.
    """
    tibia = Tibia(image, cartilage_label, bone_label=tibia_bone_label)
    result = tibia.calculate_thickness(method)
    if isinstance(plot, pv.Plotter):
        tibia.plot_thickness(result, plotter=plot, show=False)
    return result


def calculate_femoral_cartilage_thickness(image, tibia: Optional[Tibia] = None,
                                          cartilage_label: int = 3,
                                          method=None, plot: bool | pv.Plotter = False,
                                          femur_bone_label: int = 1) -> dict:
    """
    Calculate femoral cartilage thickness per subregion (Wirth & Eckstein 2008, section II-C).

    :param image: An Image of the knee cartilage segmentation.
    :param tibia: The knee's ``Tibia``, used only for the **shared in-plane frame** the maps
        are keyed in (the tibial plateau plane); the femoral subregions are derived from the
        femur alone. Without one the maps are keyed in a femoral frame and cannot be overlapped
        with tibial maps -- pass a ``Tibia`` whenever the two bones' maps will be combined (as
        :func:`calculate_knee_cartilage_thickness` does). Anything other than ``None`` or a
        ``Tibia`` raises ``TypeError``, which catches
        ``calculate_femoral_cartilage_thickness(image, 3)``.
    :param cartilage_label: The label of the femoral cartilage (``dfg_knees``: 3).
    :param method: Removed; must be ``None`` (see
        :func:`calculate_tibial_cartilage_thickness`).
    :param plot: Pass a PyVista ``Plotter`` to draw the femoral thickness heatmap into it.
        ``False`` disables plotting.
    :param femur_bone_label: The label of the femoral bone. The femoral subregions are defined
        on its subchondral surface and the femoral frame is built from it, so it must be
        present.
    :return: The :class:`~morphometry.cartilage.thickness.ThicknessMaps` of the femur (see
        :func:`calculate_tibial_cartilage_thickness`).
    """
    femur = Femur(image, cartilage_label, bone_label=femur_bone_label)
    result = femur.calculate_thickness(tibia, method=method)
    if isinstance(plot, pv.Plotter):
        femur.plot_thickness(result, plotter=plot, show=False)
    return result


def calculate_knee_cartilage_thickness(image, femur_label: int, tibia_label: int, method=None,
                                       plot: bool | pv.Plotter = False,
                                       tibia_bone_label: int = 2,
                                       femur_bone_label: int = 1) -> dict:
    """
    Calculate both tibial and femoral cartilage thickness in one call.

    Both bones' maps are keyed in the knee's shared in-plane frame (the tibial plateau plane),
    so they can be overlapped (:func:`find_thinnest_area`).

    :param image: An Image of the knee cartilage segmentation.
    :param femur_label: The label of the femoral cartilage.
    :param tibia_label: The label of the tibial cartilage.
    :param method: Removed; must be ``None`` (see
        :func:`calculate_tibial_cartilage_thickness`).
    :param plot: Pass a PyVista ``Plotter`` to draw a combined tibia+femur thickness
        heatmap (both subchondral bone areas, shared colour scale). ``False`` disables plotting.
    :param tibia_bone_label: The label of the tibial bone, on whose subchondral surface the
        tibial subregions are defined (must be present).
    :param femur_bone_label: The label of the femoral bone, on whose subchondral surface the
        femoral subregions are defined (must be present).
    :return: ``{'tibia': <tibia ThicknessMaps>, 'femur': <femur ThicknessMaps>}``.
    """
    tibia = Tibia(image, tibia_label, bone_label=tibia_bone_label)
    tibia_results = tibia.calculate_thickness(method)
    femur = Femur(image, femur_label, bone_label=femur_bone_label)
    femur_results = femur.calculate_thickness(tibia, method=method)
    if isinstance(plot, pv.Plotter):
        plot_knee_thickness(tibia, femur, tibia_results, femur_results, plotter=plot, show=False)
    return {'tibia': tibia_results, 'femur': femur_results}


def calculate_thinnest_cartilage_area(image, femur_label: int, tibia_label: int,
                                      method=None, neighbourhood_mm: float = 3.0,
                                      radius_mm: Optional[float] = None,
                                      plot: bool | pv.Plotter = False,
                                      by_compartment: bool = False,
                                      tibia_bone_label: int = 2,
                                      femur_bone_label: int = 1) -> dict:
    """
    Find the thinnest area of the tibiofemoral cartilage in one call.

    Computes the tibial and femoral thickness maps and locates the thinnest neighbourhood
    of the combined (contact-region) cartilage, robust to single-voxel outliers. See
    :func:`morphometry.cartilage.knee.find_thinnest_area` for the algorithm and
    :func:`morphometry.cartilage.knee.find_thinnest_area_by_compartment` for the
    per-compartment variant.

    :param image: An Image of the knee cartilage segmentation.
    :param femur_label: The label of the femoral cartilage.
    :param tibia_label: The label of the tibial cartilage.
    :param method: Removed; must be ``None`` (see
        :func:`calculate_tibial_cartilage_thickness`).
    :param neighbourhood_mm: Diameter (mm) of the median-smoothing neighbourhood.
    :param radius_mm: Radius (mm) of the refinement disc (defaults to ``neighbourhood_mm / 2``).
    :param plot: Pass a PyVista ``Plotter`` to render the cartilage with an arrow marking
        the thinnest point (one arrow per compartment when ``by_compartment``;
        thickness-heatmap base). ``False`` disables plotting.
    :param by_compartment: ``False`` (default) searches the whole knee and returns a single
        result. ``True`` searches the patient-left and patient-right tibiofemoral
        compartments separately.
    :param tibia_bone_label: The label of the tibial bone, on whose subchondral surface the
        tibial subregions are defined (must be present).
    :param femur_bone_label: The label of the femoral bone, on whose subchondral surface the
        femoral subregions are defined (must be present).
    :return: With ``by_compartment=False``, the result dict from :func:`find_thinnest_area`.
        With ``by_compartment=True``, ``{'left': result, 'right': result}`` keyed by
        **patient** side, from :func:`find_thinnest_area_by_compartment`.
    """
    tibia = Tibia(image, tibia_label, bone_label=tibia_bone_label)
    tibia_results = tibia.calculate_thickness(method)
    femur = Femur(image, femur_label, bone_label=femur_bone_label)
    femur_results = femur.calculate_thickness(tibia, method=method)
    search = find_thinnest_area_by_compartment if by_compartment else find_thinnest_area
    thinnest = search(tibia_results, femur_results, image,
                      neighbourhood_mm=neighbourhood_mm, radius_mm=radius_mm)
    if isinstance(plot, pv.Plotter):
        plot_thinnest_point(tibia, femur, tibia_results, femur_results, thinnest,
                            plotter=plot, show=False)
    return thinnest
