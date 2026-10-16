import itertools
import logging
import warnings

import pyvista as pv
import numpy as np
from typing import Optional, Tuple
from scipy.ndimage import distance_transform_edt, generic_filter, label, minimum_filter
from morphometry.image_io import Image
from morphometry.cartilage.surface import (
    array_indices_to_world, transfer_qc, TRANSFER_FAR_MM, TRANSFER_MAX_FAR_FRACTION,
)
from morphometry.cartilage.tibia_regions import (
    subdivide_tibia, anatomical_tibia_names, tibia_display_label, TIBIA_REGION_KEYS,
)
from morphometry.cartilage.femur_regions import (
    subdivide_femur, anatomical_femur_names, FEMUR_REGION_KEYS, ZONE_THICKNESS_KEYS,
)
from morphometry.cartilage.thickness import (
    DEFAULT_GRID_MM, KneeFrame, ThicknessMaps, knee_frame_from_femur, knee_frame_from_tibia,
    measure_thickness, thickness_frame,
)

logger = logging.getLogger(__name__)

# The old voxel-column thickness ('mesh' = normal ray cast, 'knn' = one-directional nearest
# neighbour) is gone: the area-weighted surface distances of Wirth & Eckstein are now the only
# definition (docs/wirth_eckstein_2008_plan.md step 4), so there is nothing for a 'method'
# argument to select. The parameter is kept, rejecting every value, so that an old call site
# fails loudly instead of silently meaning something else.
# How far a thinnest-point candidate must be from each bone's own cartilage-map boundary --
# **off by default**, and deliberately so. The rim artefact this exists for had a specific
# cause, now fixed: the refinement step searched the raw map over the whole overlap, escaping
# the full-window restriction the neighbourhood-centre search obeys, and so handed back a rim
# cell from inside the disc. Measured over the 93 clean dfg_knees, the femoral thickness at the
# returned point went from a median 0.62 mm (67% below 1 mm; tibial 0.71 mm, 79%) to 1.51 mm
# (11%) when the refinement was made to obey the same restriction.
#
# A margin can be stacked on top for the residual, but the data cannot choose its value: every
# indicator improves monotonically with it, because eroding the search region simply drives the
# answer towards the middle of the contact. Over the cohort, raising it to 0 / 2 / 3 mm takes
# the share of results with a femoral thickness below 1 mm from 14.0 to 11.1 to 5.7%, and the
# share whose arc position sits in the posterior tenth of the condyle from 15.4 to 12.9 to
# 10.0% -- with no knee anywhere in those curves. Worse, for the flexion analysis the erosion
# pushes the answer away from both ends of the condyle, i.e. towards the null, so a value
# chosen to make the artefact indicators look good would manufacture its own result. That is
# the user's call, not a library default. Note also that anything up to 1.5 mm is a no-op at
# the default 3 mm neighbourhood and 0.5 mm grid: the full-window rule already excludes three
# cells of boundary.
DEFAULT_EDGE_MARGIN_MM = 0.0

_METHOD_REMOVED = (
    "the 'method' argument is gone: cartilage thickness is now the area-weighted surface "
    "distance between the subchondral bone area and the articular surface (Wirth & Eckstein "
    "2008, section II-C), the only definition there is. Drop the argument. The old voxel-column "
    "methods ('mesh', 'knn') are only reachable from git history, and their numbers must not be "
    "pooled with the new ones.")

# Display label per tibial region code ('cLT', ..., 'pRT'; patient side).
_TIBIA_DISPLAY_LABELS = tuple(tibia_display_label(key) for key in TIBIA_REGION_KEYS)


class Tibia:
    """
    Tibial cartilage of one knee: subregions (Wirth & Eckstein 2008) and thickness.

    The subregions are defined on the tibial subchondral bone area (tAB) by
    :func:`morphometry.cartilage.tibia_regions.subdivide_tibia` and transferred to the
    cartilage voxels by nearest tAB triangle. Regions are keyed by **patient side**
    (``clt`` ... ``plt`` = patient-left plate, ``crt`` ... ``prt`` = patient-right plate);
    with a known ``laterality`` :attr:`anatomical_names` maps them to the paper's
    medial/lateral names.
    """

    def __init__(self, image: Image, cartilage_label: int, outlier_ratio: float = 0.1,
                 bone_label: int = 2, laterality: Optional[str] = None,
                 central_fraction: float = 0.2):
        """
        :param image: The (LPI-oriented) knee ``Segmentation``.
        :param cartilage_label: Label of the tibial cartilage (``dfg_knees``: 4).
        :param outlier_ratio: Cartilage components smaller than this fraction of the total
            cartilage volume are dropped from the point cloud (see
            :func:`_remove_mask_outliers`); ``0`` disables it.
        :param bone_label: Label of the tibial bone (``dfg_knees``: 2), on whose surface the
            subregions are defined.
        :param laterality: ``'left'`` / ``'right'`` knee or ``None``; only adds the
            medial/lateral names (:attr:`anatomical_names`).
        :param central_fraction: Share of each plate's tAB in its central region (0.2).
        """
        anatomical_tibia_names(laterality)  # validate early
        self.image = image
        self.cartilage_label = cartilage_label
        self.bone_label = bone_label
        self.laterality = laterality
        self.central_fraction = central_fraction
        self.outlier_ratio = outlier_ratio
        self.subdivision = None
        # PlateFrame of the patient-left / patient-right plate once calculate_landmarks ran.
        self.left_landmarks, self.right_landmarks = None, None
        self.point_cloud = None
        # Display label ('cLT', ...) of every point_cloud row, raw (before fragment removal).
        self.point_labels = None
        # Distance QC of the region transfer (see calculate_landmarks), None until then.
        self.transfer_qc = None
        self.clt, self.ilt, self.elt, self.alt, self.plt = None, None, None, None, None
        self.crt, self.irt, self.prt, self.ert, self.art = None, None, None, None, None

    @property
    def anatomical_names(self) -> Optional[dict]:
        """
        The paper's medial/lateral names of the patient-side region keys.

        :return: ``{'clt': 'cMT', ...}`` for a right knee, ``{'clt': 'cLT', ...}`` for a
            left knee, ``None`` without a laterality (see
            :func:`morphometry.cartilage.tibia_regions.anatomical_tibia_names`).
        """
        return anatomical_tibia_names(self.laterality)

    def get_surface_points(self):
        """
        Build the tibial cartilage voxel point cloud (:attr:`point_cloud`).

        The cleaned cartilage mask's voxel indices; the subregion labels are transferred onto
        these points (:meth:`calculate_landmarks`) and they are what :meth:`plot_segments`
        draws. The thickness itself is measured on the bone and cartilage *surfaces*, not on
        these voxels (:mod:`morphometry.cartilage.thickness`).
        """
        cartilage = np.where(self.image.array == self.cartilage_label, 1, 0)
        cartilage = _remove_mask_outliers(cartilage, self.outlier_ratio)
        self.point_cloud = np.argwhere(cartilage).astype(float)

    def calculate_landmarks(self):
        """
        Subdivide the tibial subchondral bone and label every cartilage voxel.

        Runs :func:`morphometry.cartilage.tibia_regions.subdivide_tibia` (stored as
        :attr:`subdivision`; the two plates' frames also as :attr:`left_landmarks` /
        :attr:`right_landmarks`, patient side) and gives each ``point_cloud`` voxel the
        region of its nearest tAB triangle (world mm), stored as display labels in
        :attr:`point_labels`. The voxel-to-triangle distances are summarised in
        :attr:`transfer_qc` (:func:`~morphometry.cartilage.tibia_regions.transfer_qc`); a
        WARNING is logged on the ``morphometry`` logger when more than
        ``TRANSFER_MAX_FAR_FRACTION`` (1%) of the voxels are farther than
        ``TRANSFER_FAR_MM`` (6 mm), which flags broken segmentations.
        """
        if self.point_cloud is None:
            self.get_surface_points()
        self.subdivision = subdivide_tibia(self.image, bone_label=self.bone_label,
                                           cartilage_label=self.cartilage_label,
                                           central_fraction=self.central_fraction,
                                           laterality=self.laterality)
        self.left_landmarks = self.subdivision.plates['left']
        self.right_landmarks = self.subdivision.plates['right']
        codes, distances = self.subdivision.nearest_regions(
            array_indices_to_world(self.image, self.point_cloud))
        self.point_labels = np.asarray(_TIBIA_DISPLAY_LABELS, dtype=object)[codes]
        # QC: cartilage far from every tAB triangle is not explained by the subdivision
        # (broken segmentation, plate missing from the bone coverage). Warn, do not fail.
        self.transfer_qc = transfer_qc(distances)
        if self.transfer_qc['far_fraction'] > TRANSFER_MAX_FAR_FRACTION:
            logger.warning(
                'Tibia: %.1f%% of the tibial cartilage voxels are more than %.0f mm from the '
                'subchondral bone area they were assigned to (max %.1f mm); the segmentation '
                'is probably broken and the tibial subregions unreliable.',
                100 * self.transfer_qc['far_fraction'], TRANSFER_FAR_MM, self.transfer_qc['max_mm'])

    def classify_point(self, point: np.ndarray) -> str:
        """
        Classify a point as belonging to a specific region of the tibia.

        The point gets the region of the nearest tAB triangle (see
        :meth:`TibiaSubdivision.region_codes_at`). For the cartilage voxels themselves use
        the precomputed :attr:`point_labels` instead of calling this per point.

        :param point: A length-3 index into ``image.array``.
        :return: The display label of the region (``'cLT'`` ... ``'pRT'``, patient side).
        """
        if self.subdivision is None:
            self.calculate_landmarks()
        code = self.subdivision.region_codes_at(array_indices_to_world(self.image, point))[0]
        return _TIBIA_DISPLAY_LABELS[code]

    def extract_subregions(self):
        """
        Extract the subregions of the tibia as voxel point clouds (``self.clt`` etc.).
        """
        if self.point_labels is None:
            self.calculate_landmarks()
        for key in TIBIA_REGION_KEYS:
            setattr(self, key, self.point_cloud[self.point_labels == tibia_display_label(key)])

        # Safety net kept from the voxel-based scheme: a subregion is one anatomical patch,
        # so drop small disconnected fragments (a no-op when the nearest-triangle transfer
        # gives contiguous regions, which it does on clean scans).
        for attr in TIBIA_REGION_KEYS:
            setattr(self, attr, _remove_subregion_fragments(getattr(self, attr)))

    def knee_frame(self, grid_mm: float = DEFAULT_GRID_MM) -> KneeFrame:
        """
        The shared in-plane frame of this knee: the tibial plateau plane.

        Both bones' thickness maps are keyed in it, so that their overlap stays over the
        tibiofemoral contact at every flexion angle; pass it to
        :meth:`Femur.calculate_thickness` (or pass this ``Tibia``, which does the same).

        :param grid_mm: Side length of one map cell in mm.
        :return: The :class:`~morphometry.cartilage.thickness.KneeFrame`
            (``source='tibia'``); the subdivision is computed on demand.
        """
        if self.subdivision is None:
            self.calculate_landmarks()
        return knee_frame_from_tibia(self.subdivision, grid_mm=grid_mm)

    def calculate_thickness(self, method=None, grid_mm: float = DEFAULT_GRID_MM,
                            frame: KneeFrame = None) -> ThicknessMaps:
        """
        Calculate the tibial cartilage thickness per subregion (Wirth & Eckstein 2008, II-C).

        The thickness is the area-weighted Euclidean distance between the tibial subchondral
        bone area and the articular cartilage surface, in both directions, as implemented by
        :func:`~morphometry.cartilage.thickness.measure_thickness`. Denuded tAB triangles count
        as 0.0 mm in the ``tab_*`` statistics.

        :param method: Removed; must be ``None``. See the module-level note: there is only one
            thickness definition now, so the old ``'mesh'`` / ``'knn'`` values raise rather
            than silently meaning something else.
        :param grid_mm: Side length of one in-plane map cell in mm (ignored when ``frame`` is
            given).
        :param frame: The shared in-plane frame to key the maps in; ``None`` derives it from
            this tibia (:meth:`knee_frame`), which is the normal case -- the tibial plateau is
            the shared frame.
        :return: A :class:`~morphometry.cartilage.thickness.ThicknessMaps`: a
            ``{subregion: {(u, v): thickness_mm}}`` dict (patient-side keys ``'clt'`` ...
            ``'prt'``) carrying the per-subregion ``statistics``, the ``frame`` and the
            per-triangle thicknesses.
        :raises ValueError: If ``method`` is given, or the subdivision / articular surface
            cannot be built.
        """
        if method is not None:
            raise ValueError(f'Tibia.calculate_thickness: {_METHOD_REMOVED}')
        if self.point_cloud is None:
            self.get_surface_points()
        if self.subdivision is None:
            self.calculate_landmarks()
        self.extract_subregions()
        if frame is None:
            frame = knee_frame_from_tibia(self.subdivision, grid_mm=grid_mm)
        # +1: the tibial plateau faces up, towards the femur.
        return measure_thickness(self.image, self.subdivision, TIBIA_REGION_KEYS,
                                 bone_label=self.bone_label,
                                 cartilage_label=self.cartilage_label, frame=frame, bone='tibia',
                                 facing_sign=+1)

    def _ensure_subregions(self):
        """
        Lazily compute the surface points, landmarks and subregion voxel clouds
        needed for visualisation, if they have not been computed yet.

        Idempotent: does nothing once the subregions are available.
        """
        if self.clt is not None:
            return
        if self.point_cloud is None:
            self.get_surface_points()
        if self.left_landmarks is None:
            self.calculate_landmarks()
        self.extract_subregions()

    def plot_segments(self, plotter: pv.Plotter = None, show: bool = None) -> pv.Plotter:
        """
        Render the raw voxel volume of every tibial subregion as coloured cube
        glyphs in a single 3D scene, one distinct colour per subregion.

        Subregions (and the landmarks they depend on) are computed on demand.

        :param plotter: An existing PyVista ``Plotter`` to draw into. When ``None`` a
            new plotter is created (and shown, unless ``show`` overrides this).
        :param show: Whether to call ``plotter.show()`` before returning. Defaults to
            ``True`` only when this method created the plotter, ``False`` otherwise.
        :return: The ``Plotter`` the segments were drawn into.
        """
        created = plotter is None
        if plotter is None:
            plotter = pv.Plotter()
        self._ensure_subregions()

        drawn = False
        for attr, label, color in TIBIA_SUBREGIONS:
            drawn |= _add_voxel_glyphs(plotter, self.image, getattr(self, attr), color, label)

        if drawn:
            plotter.add_legend(bcolor='white')
        if show or (show is None and created):
            plotter.show()
        return plotter

    def plot_regions(self, plotter: pv.Plotter = None, show: bool = None,
                     show_bone: bool = True, show_frames: bool = True) -> pv.Plotter:
        """
        Render the subdivided tibial subchondral bone area (tAB), one colour per region.

        This shows the subdivision itself (on the bone surface, triangles cut exactly),
        unlike :meth:`plot_segments`, which shows the cartilage voxels it was transferred to.

        :param plotter: An existing PyVista ``Plotter`` to draw into (new one if ``None``).
        :param show: Whether to call ``plotter.show()`` (see :meth:`plot_segments`).
        :param show_bone: Also draw the rest of the tibial bone surface, translucent grey.
        :param show_frames: Mark each plate's centre of gravity S and draw its internal
            (towards the other plate, black) and anterior (white) directions.
        :return: The ``Plotter`` the regions were drawn into.
        """
        created = plotter is None
        if plotter is None:
            plotter = pv.Plotter()
        if self.subdivision is None:
            self.calculate_landmarks()
        sub = self.subdivision
        if show_bone and sub.bone_mesh is not None:
            plotter.add_mesh(sub.bone_mesh, color='lightgrey', opacity=0.25)
        codes = np.asarray(sub.mesh.cell_data['region'])
        colors = {attr: color for attr, _, color in TIBIA_SUBREGIONS}
        names = sub.anatomical_names
        for code, key in enumerate(TIBIA_REGION_KEYS):
            if not (codes == code).any():
                continue
            label = tibia_display_label(key) + (f' ({names[key]})' if names else '')
            part = sub.mesh.extract_cells(np.flatnonzero(codes == code))
            plotter.add_mesh(part, color=colors[key], label=label)
        if show_frames:
            for frame in sub.plates.values():
                length = 2.0 * float(frame.semi_axes.max())
                plotter.add_mesh(pv.Sphere(radius=0.8, center=frame.centroid), color='black')
                plotter.add_mesh(pv.Arrow(frame.centroid, frame.ml_direction, scale=length), color='black')
                plotter.add_mesh(pv.Arrow(frame.centroid, frame.anterior_direction, scale=length), color='white')
        plotter.add_legend(bcolor='white')
        if show or (show is None and created):
            plotter.show()
        return plotter

    def plot_thickness(self, thicknesses: dict, plotter: pv.Plotter = None, show: bool = None,
                       cmap: str = 'viridis', clim: Tuple[float, float] = None,
                       show_scalar_bar: bool = True) -> pv.Plotter:
        """
        Render the tibial subchondral bone area (tAB) coloured by cartilage thickness.

        This is the paper's Fig. 4(b) / 5(b) view: the thickness as it was measured, one value
        per tAB triangle, on the surface the subregions are defined on (denuded triangles read
        0 mm). It needs the :class:`~morphometry.cartilage.thickness.ThicknessMaps` that
        :meth:`calculate_thickness` returned, whose ``triangle_thickness`` is aligned with the
        subdivision mesh; a plain dict of per-position maps is rendered from the maps instead,
        as triangle-less points, which is only a fallback.

        :param thicknesses: The result of :meth:`calculate_thickness`.
        :param plotter: An existing PyVista ``Plotter`` to draw into (new one if ``None``).
        :param show: Whether to call ``plotter.show()`` (see :meth:`plot_segments`).
        :param cmap: A matplotlib colormap name used for the thickness heatmap.
        :param clim: ``(min, max)`` colour limits in mm. Derived from the data when ``None``.
        :param show_scalar_bar: Whether to draw the thickness colour bar.
        :return: The ``Plotter`` the heatmap was drawn into.
        """
        created = plotter is None
        if plotter is None:
            plotter = pv.Plotter()
        if self.subdivision is None:
            self.calculate_landmarks()
        _add_thickness_meshes(plotter, _thickness_meshes(self.subdivision, thicknesses),
                              cmap, clim, show_scalar_bar)
        if show or (show is None and created):
            plotter.show()
        return plotter


class Femur:
    """
    Femoral cartilage of one knee: subregions (Wirth & Eckstein 2008) and thickness.

    The subregions are defined on the femoral subchondral bone area (tAB) by
    :func:`morphometry.cartilage.femur_regions.subdivide_femur` and transferred to the
    cartilage voxels by nearest tAB triangle. Regions are keyed by **patient side**
    (``iclf``, ``cclf``, ``eclf`` = the thirds of the patient-left weight-bearing condyle,
    ``icrf``, ``ccrf``, ``ecrf`` the patient-right ones, plus the in-house extension zones
    ``alf`` / ``arf`` for the trochlea and ``plf`` / ``prf`` for the posterior condyles);
    with a known ``laterality`` :attr:`anatomical_names` maps them to the paper's
    medial/lateral names.

    The subdivision is built from the femur alone -- its shaft axis, posterior condylar line
    and intercondylar notch -- so it no longer needs a :class:`Tibia` and no longer moves with
    knee flexion. The ``tibia`` arguments of :meth:`calculate_thickness` and the plotting
    methods are kept only so existing callers keep working; they are ignored.
    """

    def __init__(self, image: Image, cartilage_label: int, outlier_ratio: float = 0.1,
                 bone_label: int = 1, laterality: Optional[str] = None,
                 weightbearing_fraction: float = 0.6):
        """
        :param image: The (LPI-oriented) knee ``Segmentation``.
        :param cartilage_label: Label of the femoral cartilage (``dfg_knees``: 3).
        :param outlier_ratio: Cartilage components smaller than this fraction of the total
            cartilage volume are dropped from the point cloud (see
            :func:`_remove_mask_outliers`); ``0`` disables it.
        :param bone_label: Label of the femoral bone (``dfg_knees``: 1), on whose surface the
            subregions are defined and from which the femoral frame is built.
        :param laterality: ``'left'`` / ``'right'`` knee or ``None``; only adds the
            medial/lateral names (:attr:`anatomical_names`).
        :param weightbearing_fraction: Where the weight-bearing condyle ends, as a fraction of
            the distance from the trochlear notch to the condyle's posterior end (0.6).
        """
        anatomical_femur_names(laterality)  # validate early
        if not 0.0 < weightbearing_fraction < 1.0:
            raise ValueError('weightbearing_fraction must be in (0, 1), got '
                             f'{weightbearing_fraction}')
        self.image = image
        self.cartilage_label = cartilage_label
        self.bone_label = bone_label
        self.laterality = laterality
        self.weightbearing_fraction = weightbearing_fraction
        self.outlier_ratio = outlier_ratio
        self.subdivision = None
        # FemoralFrame and the two CondyleFrames once calculate_landmarks ran.
        self.frame = None
        self.left_landmarks, self.right_landmarks = None, None
        # Region key ('iclf', ...) of every point_cloud row, raw (before fragment removal).
        self.point_labels = None
        # Distance QC of the region transfer (see calculate_landmarks), None until then.
        self.transfer_qc = None
        self.iclf, self.cclf, self.eclf = None, None, None
        self.icrf, self.ccrf, self.ecrf = None, None, None
        self.alf, self.plf, self.arf, self.prf = None, None, None, None

        cartilage = np.where(self.image.array == self.cartilage_label, 1, 0)
        cartilage = _remove_mask_outliers(cartilage, self.outlier_ratio)
        self.point_cloud = np.argwhere(cartilage).astype(float)

    @property
    def anatomical_names(self) -> Optional[dict]:
        """
        The paper's medial/lateral names of the patient-side region keys.

        :return: ``{'iclf': 'icMF', ...}`` for a right knee, ``{'iclf': 'icLF', ...}`` for a
            left knee, ``None`` without a laterality (see
            :func:`morphometry.cartilage.femur_regions.anatomical_femur_names`).
        """
        return anatomical_femur_names(self.laterality)

    def calculate_landmarks(self):
        """
        Subdivide the femoral subchondral bone and label every cartilage voxel.

        Runs :func:`morphometry.cartilage.femur_regions.subdivide_femur` (stored as
        :attr:`subdivision`; its femoral frame as :attr:`frame` and the two condyles'
        frames as :attr:`left_landmarks` / :attr:`right_landmarks`, patient side) and gives
        each ``point_cloud`` voxel the region of its nearest tAB triangle (world mm), stored
        in :attr:`point_labels`. The voxel-to-triangle distances are summarised in
        :attr:`transfer_qc` (:func:`~morphometry.cartilage.surface.transfer_qc`); a WARNING is
        logged on the ``morphometry`` logger when more than ``TRANSFER_MAX_FAR_FRACTION`` (1%)
        of the voxels are farther than ``TRANSFER_FAR_MM`` (6 mm), which flags broken
        segmentations.
        """
        self.subdivision = subdivide_femur(self.image, bone_label=self.bone_label,
                                           cartilage_label=self.cartilage_label,
                                           weightbearing_fraction=self.weightbearing_fraction,
                                           laterality=self.laterality)
        self.frame = self.subdivision.frame
        self.left_landmarks = self.subdivision.condyles['left']
        self.right_landmarks = self.subdivision.condyles['right']
        codes, distances = self.subdivision.nearest_regions(
            array_indices_to_world(self.image, self.point_cloud))
        self.point_labels = np.asarray(FEMUR_REGION_KEYS, dtype=object)[codes]
        # QC: cartilage far from every tAB triangle is not explained by the subdivision
        # (broken segmentation, cartilage label on the wrong bone). Warn, do not fail.
        self.transfer_qc = transfer_qc(distances)
        if self.transfer_qc['far_fraction'] > TRANSFER_MAX_FAR_FRACTION:
            logger.warning(
                'Femur: %.1f%% of the femoral cartilage voxels are more than %.0f mm from the '
                'subchondral bone area they were assigned to (max %.1f mm); the segmentation '
                'is probably broken and the femoral subregions unreliable.',
                100 * self.transfer_qc['far_fraction'], TRANSFER_FAR_MM, self.transfer_qc['max_mm'])

    def classify_point(self, point: np.ndarray) -> str:
        """
        Classify a point as belonging to a specific region of the femur.

        The point gets the region of the nearest tAB triangle (see
        :meth:`FemurSubdivision.region_codes_at`). For the cartilage voxels themselves use the
        precomputed :attr:`point_labels` instead of calling this per point.

        :param point: A length-3 index into ``image.array``.
        :return: The region key (``'iclf'`` ... ``'prf'``, patient side).
        """
        if self.subdivision is None:
            self.calculate_landmarks()
        code = self.subdivision.region_codes_at(array_indices_to_world(self.image, point))[0]
        return FEMUR_REGION_KEYS[code]

    def extract_subregions(self):
        """
        Extract the subregions of the femoral cartilage as voxel point clouds (``self.iclf``
        etc.).
        """
        if self.point_labels is None:
            self.calculate_landmarks()
        for key in FEMUR_REGION_KEYS:
            # Safety net kept from the voxel-based scheme: a subregion is one anatomical
            # patch, so drop small disconnected fragments (a no-op when the nearest-triangle
            # transfer gives contiguous regions, which it does on clean scans).
            setattr(self, key, _remove_subregion_fragments(
                self.point_cloud[self.point_labels == key]))

    def calculate_thickness(self, tibia: Tibia = None, method=None,
                            grid_mm: float = DEFAULT_GRID_MM,
                            frame: KneeFrame = None) -> ThicknessMaps:
        """
        Calculate the femoral cartilage thickness per subregion (Wirth & Eckstein 2008, II-C).

        The thickness is the area-weighted Euclidean distance between the femoral subchondral
        bone area and the articular cartilage surface, in both directions, as implemented by
        :func:`~morphometry.cartilage.thickness.measure_thickness`. It no longer depends on any
        image axis -- the voxel-column method it replaced had to pick a depth axis per
        subregion and keyed its values by two array axes that were not the ones it measured
        across (plan step 4, open issue 1).

        **The maps need the tibia's frame.** They are keyed in the shared in-plane frame of the
        knee, which is the tibial plateau plane (see
        :mod:`morphometry.cartilage.thickness`), so that the tibiofemoral overlap
        :func:`find_thinnest_area` searches stays over the contact at every flexion angle.
        Pass the knee's ``Tibia`` (or its :meth:`Tibia.knee_frame`). Without one, the frame is
        derived from the femur itself and marked as such; such maps are a valid femoral map but
        :func:`find_thinnest_area` refuses to overlap them with tibial maps.

        :param tibia: The knee's ``Tibia``, used **only** for the shared in-plane frame -- the
            femoral subregions are derived from the femur alone. ``None`` falls back to a
            femoral frame (see above). Anything other than ``None`` or a ``Tibia`` raises, so
            that a positional ``calculate_thickness('knn')`` is caught.
        :param method: Removed; must be ``None``. See the module-level note: there is only one
            thickness definition now, so the old ``'mesh'`` / ``'knn'`` values raise rather
            than silently meaning something else.
        :param grid_mm: Side length of one in-plane map cell in mm. Ignored when ``frame`` is
            given, and ignored when ``tibia`` is -- the grid then comes from the tibia, so the
            two bones' maps cannot end up on different grids.
        :param frame: The shared in-plane frame to key the maps in; takes precedence over
            ``tibia``.
        :return: A :class:`~morphometry.cartilage.thickness.ThicknessMaps`: a
            ``{zone: {(u, v): thickness_mm}}`` dict carrying the per-subregion ``statistics``,
            the ``frame`` and the per-triangle thicknesses. The weight-bearing condyles'
            thirds are keyed by their region key (``'iclf'`` ...), the extension zones by the
            names the femoral thickness dicts have always used for them
            (``'left_anterior_zone'`` ... ``'right_posterior_zone'``).
        :raises TypeError: If ``tibia`` is neither ``None`` nor a ``Tibia``.
        :raises ValueError: If ``method`` is given, or the subdivision / articular surface
            cannot be built.
        """
        # 'tibia' is the first positional parameter, so a caller writing
        # femur.calculate_thickness('knn') would otherwise bind the method to it.
        if tibia is not None and not isinstance(tibia, Tibia):
            raise TypeError(
                f'Femur.calculate_thickness takes the tibia (for the shared in-plane frame) '
                f'first; got {tibia!r} as the tibia.')
        if method is not None:
            raise ValueError(f'Femur.calculate_thickness: {_METHOD_REMOVED}')
        self.extract_subregions()
        if frame is None:
            # With a tibia the grid size comes from it, not from grid_mm: a femoral map on a
            # different grid would not be comparable with the tibial one, and the mismatch
            # would surface later as a frame error that blames the missing tibia.
            frame = (tibia.knee_frame() if tibia is not None
                     else knee_frame_from_femur(self.subdivision, grid_mm=grid_mm))
        # -1: the femoral articular surface faces down, towards the tibia. Without this the
        # horseshoe folds onto itself in the shared plane (see _joint_facing_sheet).
        return measure_thickness(self.image, self.subdivision, FEMUR_REGION_KEYS,
                                 bone_label=self.bone_label,
                                 cartilage_label=self.cartilage_label, frame=frame,
                                 bone='femur', map_keys=ZONE_THICKNESS_KEYS, facing_sign=-1)

    def _ensure_subregions(self, tibia: Tibia = None):
        """
        Lazily compute the femoral subregion voxel clouds needed for visualisation.

        Idempotent: does nothing once the subregions are available.

        :param tibia: Ignored (see :meth:`calculate_thickness`); kept so the plotting
            methods' signatures do not change.
        """
        if self.iclf is not None:
            return
        self.extract_subregions()

    def plot_segments(self, tibia: Tibia = None, plotter: pv.Plotter = None,
                      show: bool = None) -> pv.Plotter:
        """
        Render the raw voxel volume of every femoral subregion as coloured cube
        glyphs in a single 3D scene, one distinct colour per subregion.

        :param tibia: Ignored (the subregions no longer depend on the tibia).
        :param plotter: An existing PyVista ``Plotter`` to draw into (new one if ``None``).
        :param show: Whether to call ``plotter.show()`` (see :meth:`Tibia.plot_segments`).
        :return: The ``Plotter`` the segments were drawn into.
        """
        created = plotter is None
        if plotter is None:
            plotter = pv.Plotter()
        self._ensure_subregions()

        drawn = False
        for attr, label, color, _ in FEMUR_SUBREGIONS:
            drawn |= _add_voxel_glyphs(plotter, self.image, getattr(self, attr), color, label)

        if drawn:
            plotter.add_legend(bcolor='white')
        if show or (show is None and created):
            plotter.show()
        return plotter

    def plot_regions(self, plotter: pv.Plotter = None, show: bool = None,
                     show_bone: bool = True, show_frames: bool = True) -> pv.Plotter:
        """
        Render the subdivided femoral subchondral bone area (tAB), one colour per region.

        This shows the subdivision itself (on the bone surface, triangles cut exactly),
        unlike :meth:`plot_segments`, which shows the cartilage voxels it was transferred to.
        It is the view of the paper's Fig. 6 (the condyles seen from inferior).

        :param plotter: An existing PyVista ``Plotter`` to draw into (new one if ``None``).
        :param show: Whether to call ``plotter.show()`` (see :meth:`Tibia.plot_segments`).
        :param show_bone: Also draw the rest of the femoral bone surface, translucent grey.
        :param show_frames: Draw the femoral frame's axes at the trochlear notch, the notch
            point, and the anterior (notch) and posterior (60%) border planes.
        :return: The ``Plotter`` the regions were drawn into.
        """
        created = plotter is None
        if plotter is None:
            plotter = pv.Plotter()
        if self.subdivision is None:
            self.calculate_landmarks()
        sub = self.subdivision
        if show_bone and sub.bone_mesh is not None:
            plotter.add_mesh(sub.bone_mesh, color='lightgrey', opacity=0.25)
        codes = np.asarray(sub.mesh.cell_data['region'])
        colors = {attr: color for attr, _, color, _ in FEMUR_SUBREGIONS}
        names = sub.anatomical_names
        for code, key in enumerate(FEMUR_REGION_KEYS):
            if not (codes == code).any():
                continue
            label = key + (f' ({names[key]})' if names else '')
            part = sub.mesh.extract_cells(np.flatnonzero(codes == code))
            plotter.add_mesh(part, color=colors[key], label=label)
        if show_frames:
            frame = sub.frame
            size = 2.0 * float(np.linalg.norm(frame.posterior_points['right']
                                              - frame.posterior_points['left']))
            for direction, color in ((frame.ap_axis, 'black'), (frame.ml_axis, 'white'),
                                     (frame.proximal_axis, 'dimgrey')):
                plotter.add_mesh(pv.Arrow(frame.notch_point, direction, scale=0.4 * size),
                                 color=color)
            plotter.add_mesh(pv.Sphere(radius=1.2, center=frame.notch_point), color='black')
            # The anterior border (through the notch) and each condyle's posterior border.
            levels = [(frame.notch_ap, 'black')] + [(c.boundary_ap, 'dimgrey')
                                                    for c in sub.condyles.values()]
            centre = np.asarray(sub.mesh.points, dtype=float).mean(axis=0)
            for level, color in levels:
                origin = centre + (level - centre @ frame.ap_axis) * frame.ap_axis
                plotter.add_mesh(pv.Plane(center=origin, direction=frame.ap_axis,
                                          i_size=size, j_size=size),
                                 color=color, opacity=0.2, style='surface')
        plotter.add_legend(bcolor='white')
        if show or (show is None and created):
            plotter.show()
        return plotter

    def plot_thickness(self, thicknesses: dict, tibia: Tibia = None, plotter: pv.Plotter = None,
                       show: bool = None, cmap: str = 'viridis', clim: Tuple[float, float] = None,
                       show_scalar_bar: bool = True) -> pv.Plotter:
        """
        Render the femoral subchondral bone area (tAB) coloured by cartilage thickness.

        See :meth:`Tibia.plot_thickness`; this is the femoral half of the paper's Fig. 5(b).

        :param thicknesses: The result of :meth:`calculate_thickness`.
        :param tibia: Ignored (the subregions no longer depend on the tibia).
        :param plotter: An existing PyVista ``Plotter`` to draw into (new one if ``None``).
        :param show: Whether to call ``plotter.show()`` (see :meth:`Tibia.plot_segments`).
        :param cmap: A matplotlib colormap name used for the thickness heatmap.
        :param clim: ``(min, max)`` colour limits in mm. Derived from the data when ``None``.
        :param show_scalar_bar: Whether to draw the thickness colour bar.
        :return: The ``Plotter`` the heatmap was drawn into.
        """
        created = plotter is None
        if plotter is None:
            plotter = pv.Plotter()
        if self.subdivision is None:
            self.calculate_landmarks()
        _add_thickness_meshes(plotter, _thickness_meshes(self.subdivision, thicknesses),
                              cmap, clim, show_scalar_bar)
        if show or (show is None and created):
            plotter.show()
        return plotter


def scan_qc_flags(image: Image, tibia: Optional['Tibia'] = None,
                  femur: Optional['Femur'] = None) -> str:
    """
    Describe the known bad-scan signals of one knee scan as a QC flag string.

    Flagged scans are still measured; the flags only say that their results should not be
    trusted without looking at the scan. The format follows ``scripts/process_nmdid.py``'s
    ``qc_flags`` column: ``name:value`` items joined by ``;``, empty when nothing is wrong.
    Flags, in this order:

    * ``non_lpi_orientation:<codes>`` -- the image did not land in the orientation it was
      transformed to (:attr:`~morphometry.image_io.Image.orientation_ok` is ``False``);
      ``<codes>`` are the achieved axis codes, e.g. ``LSP``. Every knee measurement
      assumes LPI (patient sides, superior-inferior surfaces), so such a scan's
      subregions and patient-side compartments may be mislabelled.
    * ``tibia_transfer_far:<fraction>`` / ``femur_transfer_far:<fraction>`` -- more than
      ``TRANSFER_MAX_FAR_FRACTION`` of that bone's cartilage voxels lie farther than
      ``TRANSFER_FAR_MM`` from the subchondral bone area they were assigned to
      (:attr:`Tibia.transfer_qc` / :attr:`Femur.transfer_qc`, the same condition that logs
      the WARNING in their ``calculate_landmarks``); ``<fraction>`` is that share
      (0-1, three decimals).

    :param image: The loaded (and reoriented) knee ``Segmentation``.
    :param tibia: The scan's ``Tibia`` after its thickness was computed, or ``None`` when
        the tibia failed or was not computed (its transfer flag is then skipped).
    :param femur: The scan's ``Femur``, likewise.
    :return: The ``;``-separated flag string (``''`` for a clean scan).
    """
    flags = []
    # orientation_ok is None when no orientation was requested; only a definite False flags.
    if image.orientation_ok is False:
        flags.append(f"non_lpi_orientation:{''.join(image.achieved_axcodes)}")
    for name, bone in (('tibia', tibia), ('femur', femur)):
        qc = getattr(bone, 'transfer_qc', None)
        if qc is not None and qc['far_fraction'] > TRANSFER_MAX_FAR_FRACTION:
            flags.append(f"{name}_transfer_far:{qc['far_fraction']:.3f}")
    return ';'.join(flags)


def _remove_mask_outliers(mask: np.ndarray, threshold_ratio: float = 0.1) -> np.ndarray:
    """
    Remove small connected components from a binary cartilage mask.

    Segmentations may contain small mislabelled blobs or detached voxel clusters that
    corrupt the reconstructed cartilage surfaces/meshes. This keeps only the connected
    components (26-connectivity) whose size exceeds ``threshold_ratio`` of the total
    masked volume, removing such outliers while retaining the genuine cartilage — a single
    femoral horseshoe, or the two tibial plateaus, are all far larger than the threshold.
    This mirrors :meth:`Segmentation.remove_outliers` but operates on a single extracted
    cartilage mask (backend-agnostic, no mutation of the shared segmentation).

    :param mask: A binary (0/1) mask of a single cartilage label.
    :param threshold_ratio: Minimum component size as a fraction of the total masked
        volume; components at or below it are removed. ``<= 0`` disables removal.
    :return: The cleaned binary mask (same dtype as the input).
    """
    if threshold_ratio <= 0:
        return mask
    labeled, n = label(mask, structure=np.ones((3, 3, 3), dtype=bool))
    if n <= 1:
        return mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    keep = sizes > threshold_ratio * sizes.sum()
    return keep[labeled].astype(mask.dtype)


def _remove_subregion_fragments(points: np.ndarray, ratio: float = 0.2) -> np.ndarray:
    """
    Drop small disconnected fragments from a subregion's voxel point cloud.

    A subregion is a single anatomical patch, but a voxel can be assigned to one it is not
    attached to -- the nearest subchondral-bone triangle is not always on the patch the voxel
    belongs to, e.g. across the intercondylar notch.
    This keeps only the 26-connectivity connected components whose voxel count is at least
    ``ratio`` of the largest component's, removing such fragments while retaining a
    genuinely multi-lobed subregion.

    :param points: An ``Nx3`` array of the subregion's voxel indices.
    :param ratio: Minimum component size as a fraction of the largest component; smaller
        components are removed.
    :return: The cleaned points (unchanged if empty or a single connected component).
    """
    if points is None or len(points) == 0:
        return points
    p = np.asarray(points)
    idx = p.astype(int)
    local = tuple((idx - idx.min(axis=0)).T)
    grid = np.zeros(tuple(np.max(local, axis=1) + 1), dtype=bool)
    grid[local] = True
    labeled, n = label(grid, structure=np.ones((3, 3, 3), dtype=bool))
    if n <= 1:
        return points
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    keep = sizes >= ratio * sizes.max()
    return p[keep[labeled[local]]]


def _flatten_thickness(thicknesses: dict) -> dict:
    """
    Flatten a per-subregion thickness dict into a single ``{(u, v): thickness}`` map.

    Positions that appear in more than one subregion are averaged and ``NaN`` / missing values
    are dropped. Two things put a position in more than one subregion: a cell straddling a
    region cut, and -- much more often on the femur -- the surface folding onto itself in the
    projection, where the trochlea and the posterior condyle land on the same cell from
    opposite ends of the horseshoe. ``measure_thickness`` keeps only the joint-facing sheet
    (:func:`~morphometry.cartilage.thickness._joint_facing_sheet`), which removes almost all of
    the second kind before the maps are built; what is left is the first kind.

    :param thicknesses: ``{subregion: {(u, v): thickness}}`` as returned by
        ``calculate_thickness``, keyed in the shared in-plane frame.
    :return: A flat ``{(u, v): thickness}`` map over all subregions.
    """
    accumulated: dict = {}
    for subregion_map in thicknesses.values():
        for coord, value in subregion_map.items():
            if value is None or np.isnan(value):
                continue
            accumulated.setdefault(coord, []).append(float(value))
    return {coord: float(np.mean(values)) for coord, values in accumulated.items()}


def _overlap_spacing(tibia_thickness: dict, femur_thickness: dict, image: Image) -> float:
    """
    The in-plane sampling step (mm) of two thickness maps, and a check that they are comparable.

    Maps from ``calculate_thickness`` are keyed in a shared in-plane
    :class:`~morphometry.cartilage.thickness.KneeFrame`, whose ``grid_mm`` is the step. Both
    maps must be in the *same* frame -- a femoral map keyed in a femoral frame (because no
    ``Tibia`` was passed to ``Femur.calculate_thickness``) does not describe the same positions
    as a tibial one, and overlapping them would silently pair up unrelated places.

    Plain dicts without a frame (hand-built maps, e.g. in the unit tests) fall back to the
    image's in-plane voxel spacing, which is what the per-voxel-column maps used to be keyed by.

    :param tibia_thickness: The tibial per-subregion thickness dict.
    :param femur_thickness: The femoral per-subregion thickness dict.
    :param image: The segmentation image (its spacing is the fallback step).
    :return: The in-plane step in mm.
    :raises ValueError: If exactly one map carries a frame, or the two frames differ.
    """
    tibia_frame, femur_frame = thickness_frame(tibia_thickness), thickness_frame(femur_thickness)
    if tibia_frame is None and femur_frame is None:
        return float(np.mean(image.spacing[:2]))
    if tibia_frame is None or femur_frame is None:
        raise ValueError('One thickness map is keyed in a shared in-plane frame and the other '
                         'is not; they cannot be overlapped.')
    if not tibia_frame.matches(femur_frame):
        raise ValueError(
            f'The tibial ({tibia_frame.source}) and femoral ({femur_frame.source}) thickness '
            'maps are keyed in different in-plane frames, so their positions are not '
            'comparable. Pass the knee\'s Tibia (or its knee_frame()) to '
            'Femur.calculate_thickness so both are keyed in the tibial plateau frame.')
    return float(tibia_frame.grid_mm)


def _interior_cells(footprint: np.ndarray, margin_mm: float, spacing: float) -> np.ndarray:
    """
    The cells of a footprint at least ``margin_mm`` from its boundary.

    The boundary is measured with a Euclidean distance transform over a padded copy, so that a
    footprint touching the array border is still treated as ending there.

    :param footprint: ``(H, W)`` bool mask of the cells a bone's map covers.
    :param margin_mm: Required distance to the nearest cell outside the footprint, in mm.
        ``<= 0`` returns the footprint unchanged.
    :param spacing: Cell size in mm.
    :return: ``(H, W)`` bool mask of the interior cells.
    """
    if margin_mm <= 0 or not footprint.any():
        return footprint
    pad = int(np.ceil(margin_mm / spacing)) + 1
    padded = np.pad(footprint, pad, mode='constant', constant_values=False)
    inside = distance_transform_edt(padded, sampling=spacing) > margin_mm
    return inside[pad:-pad, pad:-pad]


def find_thinnest_area(tibia_thickness: dict, femur_thickness: dict, image: Image,
                       neighbourhood_mm: float = 3.0, radius_mm: Optional[float] = None,
                       edge_margin_mm: float = DEFAULT_EDGE_MARGIN_MM,
                       min_thickness_mm: float = 0.0) -> dict:
    """
    Locate the thinnest *area* of the tibiofemoral cartilage, robust to outliers.

    A plain ``argmin`` over a thickness map is sensitive to single erroneous cells, so
    this searches for the thinnest neighbourhood instead. The tibial and femoral thickness
    maps are summed over the in-plane positions present in *both* (the contact region),
    the combined map is median-smoothed to suppress single-cell outliers, the thinnest
    neighbourhood centre is found, and the result is refined to the actual thinnest cell
    within a disc around that centre.

    Positions are ``(u, v)`` integer cells of the knee's shared in-plane frame -- the tibial
    plateau plane (:class:`~morphometry.cartilage.thickness.KneeFrame`) -- so the overlap is
    the part of the plateau the femur sits over, at any flexion angle. Both maps must be keyed
    in the same frame (see :func:`_overlap_spacing`).

    .. warning::

       **Without an edge margin this search returns a cartilage-label edge, not a thin spot.**
       A cartilage map does not fade out at its rim, it stops; and the last fraction of a
       millimetre of it reads near zero anyway, because ``cartilage_coverage`` is a distance
       test and cAB reaches slightly past the cartilage onto bone that is beside rather than
       under it (see :func:`~morphometry.cartilage.surface.cartilage_coverage`). Those cells
       are the global minimum of almost every knee. Measured over the 93 clean ``dfg_knees``
       with ``edge_margin_mm=0``: the femoral thickness *at the returned point* has a median of
       0.61 mm and is below 1 mm in 77% of scans (tibial 0.74 mm, below 1 mm in 83%), against
       real cartilage of 1.3-3.5 mm; the femoral region at the point is a central
       weight-bearing third in only 3 of 279 compartment results; and the arc-length position
       of the point piles up at both ends of the condyle (87 of 285 below 0.1, 59 above 0.9).
       A flexion trend computed on that is a trend in *which end of the labelled cartilage
       wins the argmin*, not in where the cartilage is thin. The defect is inherited from the
       voxel-column maps, not introduced with the surface ones.

    Three restrictions keep the search off that rim, all of them parameters:

    * the smoothing window must lie wholly inside the overlap (always on -- see the comments
      below; it guards the *overlap* boundary, which is not the same thing as either bone's
      own cartilage edge);
    * every cell must be at least ``edge_margin_mm`` from **each bone's own** map boundary;
    * and, optionally, both bones must be at least ``min_thickness_mm`` thick there.

    If nothing survives, the whole overlap is searched and a WARNING is logged, so a scan is
    never silently dropped.

    :param tibia_thickness: Per-subregion tibial thickness dict (from
        ``Tibia.calculate_thickness``): ``{subregion: {(u, v): thickness}}``.
    :param femur_thickness: Per-subregion femoral thickness dict (from
        ``Femur.calculate_thickness``).
    :param image: The segmentation image; used only as the fallback in-plane step for maps
        that carry no frame (see :func:`_overlap_spacing`).
    :param neighbourhood_mm: Diameter (mm) of the median-smoothing neighbourhood. The
        kernel size ``k = round(neighbourhood_mm / in-plane step)`` is forced odd and
        at least 3, so the search is comparable across differently-sampled scans.
    :param radius_mm: Radius (mm) of the refinement disc around the neighbourhood centre.
        Defaults to ``neighbourhood_mm / 2`` so the disc matches the smoothing extent.
    :param edge_margin_mm: How far (mm) a candidate cell must be from each bone's own map
        boundary; ``0`` (the default, see :data:`DEFAULT_EDGE_MARGIN_MM`) is off. Anything up
        to 1.5 mm is a no-op at the default neighbourhood and grid.
    :param min_thickness_mm: Optional floor (mm) on each bone's thickness at a candidate cell,
        off by default. It attacks the same artefact from the other side -- useful on a cohort
        whose real cartilage is known to be thicker than some value, but it is an assumption
        about the anatomy where the edge margin is only an assumption about the geometry.
    :return: A dict with ``'point'`` (the actual thinnest ``(u, v)`` cell), ``'center'``
        (the smoothed-neighbourhood centre ``(u, v)``), ``'center_thickness'`` (the robust,
        median-smoothed combined thickness of the thin area, mm), ``'combined_thickness'``
        (the raw, non-outlier-suppressed tibial + femoral thickness at ``point``, mm),
        ``'tibial_thickness'`` / ``'femoral_thickness'`` (the per-bone contributions at
        ``point``), and ``'kernel_size'`` (``k``).
    :raises ValueError: If the two maps share no overlapping positions, or are keyed in
        different frames.
    """
    spacing = _overlap_spacing(tibia_thickness, femur_thickness, image)
    tibia_flat = _flatten_thickness(tibia_thickness)
    femur_flat = _flatten_thickness(femur_thickness)

    overlap = set(tibia_flat) & set(femur_flat)
    if not overlap:
        raise ValueError('Tibial and femoral thickness maps have no overlapping coordinates.')

    # The bounding box spans *both* bones' footprints, not just their overlap: a cell well
    # inside the overlap can still be at one bone's own cartilage edge, and the edge margin
    # below has to see that edge.
    coords = np.array(list(set(tibia_flat) | set(femur_flat)), dtype=int)
    x_min, y_min = coords[:, 0].min(), coords[:, 1].min()
    x_max, y_max = coords[:, 0].max(), coords[:, 1].max()

    # Rasterise the per-bone maps onto dense grids; cells outside a bone's (irregular)
    # footprint stay NaN. Indexing by grid position keeps everything downstream free of
    # float-coordinate dict lookups.
    tibia_grid = np.full((x_max - x_min + 1, y_max - y_min + 1), np.nan)
    femur_grid = np.full_like(tibia_grid, np.nan)
    for flat, target in ((tibia_flat, tibia_grid), (femur_flat, femur_grid)):
        for coord, value in flat.items():
            target[int(coord[0]) - x_min, int(coord[1]) - y_min] = value
    grid = tibia_grid + femur_grid          # combined thickness; NaN outside the overlap
    valid = ~np.isnan(grid)

    k = int(round(neighbourhood_mm / spacing))
    if k % 2 == 0:
        k += 1
    k = max(3, k)

    # Median smooth with reflection padding, ignoring NaN neighbours. A window holding no
    # measured cell at all yields NaN and warns; that is expected (the overlap footprint is
    # irregular) and such cells are excluded from the search below.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'All-NaN slice encountered', RuntimeWarning)
        smoothed = generic_filter(grid, np.nanmedian, size=k, mode='reflect')

    # Restrict the neighbourhood-centre search to cells whose whole k x k neighbourhood is
    # inside the overlap footprint. We are looking for the thinnest *area*, so the area has to
    # be fully measured: a cell on the footprint boundary has its median taken over the few
    # neighbours that exist, and the boundary is exactly where the cartilage runs out and the
    # measured thickness falls towards zero, so a partially-observed window would win the
    # argmin on nearly every knee.
    searchable = minimum_filter(valid, size=k, mode='constant', cval=False) & ~np.isnan(smoothed)
    # ... and away from *each bone's own* cartilage edge, and thick enough to be cartilage.
    # See the function docstring: this is what keeps the search off the rim.
    for bone_grid in (tibia_grid, femur_grid):
        searchable &= _interior_cells(~np.isnan(bone_grid), edge_margin_mm, spacing)
        if min_thickness_mm > 0:
            searchable &= np.nan_to_num(bone_grid, nan=-np.inf) >= min_thickness_mm
    if not searchable.any():
        logger.warning('find_thinnest_area: nothing survives the %.1f mm search neighbourhood, '
                       'the %.1f mm edge margin and the %.2f mm minimum thickness; falling back '
                       'to the whole tibiofemoral overlap.',
                       neighbourhood_mm, edge_margin_mm, min_thickness_mm)
        searchable = valid & ~np.isnan(smoothed)
    search = np.where(searchable, smoothed, np.inf)
    ci, cj = np.unravel_index(np.argmin(search), search.shape)
    center = (float(ci + x_min), float(cj + y_min))

    # Refine to the actual thinnest cell within a disc around the centre, on the raw map. The
    # disc radius defaults to half the neighbourhood so it matches the smoothing extent (k is
    # a diameter) rather than doubling it. The refinement obeys the same restrictions, or it
    # would hand back a rim cell from just inside the disc.
    radius = neighbourhood_mm / 2.0 if radius_mm is None else radius_mm
    r = max(1, int(round(radius / spacing)))
    ii, jj = np.indices(grid.shape)
    disc = (ii - ci) ** 2 + (jj - cj) ** 2 <= r ** 2
    restricted = np.where(disc & searchable, grid, np.inf)
    pi, pj = np.unravel_index(np.argmin(restricted), restricted.shape)
    point = (float(pi + x_min), float(pj + y_min))

    return {
        'point': point,
        'center': center,
        # 'center_thickness' is the robust (median-smoothed) combined thickness of the thin
        # area; 'combined_thickness' is the raw thickness at the actual thinnest voxel and
        # so is not outlier-suppressed.
        'center_thickness': float(smoothed[ci, cj]),
        'combined_thickness': float(grid[pi, pj]),
        'tibial_thickness': float(tibia_grid[pi, pj]),
        'femoral_thickness': float(femur_grid[pi, pj]),
        'kernel_size': k,
    }


def split_tibia_thickness_by_compartment(tibia_thickness: dict) -> dict:
    """
    Partition a per-subregion tibial thickness dict into the two tibiofemoral compartments.

    The compartment is identified by the subregion key suffix, matched case-insensitively
    so both attribute names (``clt``) and display labels (``cLT``) work: ``*LT`` subregions
    (cLT, iLT, eLT, aLT, pLT) belong to the patient-**left** plateau and ``*RT`` subregions
    (cRT, iRT, eRT, aRT, pRT) to the patient-**right** plateau, as named by
    :func:`morphometry.cartilage.tibia_regions.find_plates` (the plate with the smaller RAS
    world x, i.e. towards the patient's left). Which of the two is medial or lateral depends
    on whether the knee is a left or a right knee; :attr:`Tibia.anatomical_names` gives that
    mapping, but a dict re-keyed with it (``cMT`` ... ``pLT``, where ``LT`` means *lateral*)
    must not be passed here: it is refused (any ``*MT`` key raises), because its ``*LT``
    keys would otherwise be read as patient-left.

    :param tibia_thickness: ``{subregion: {(u, v): thickness}}`` as returned by
        :meth:`Tibia.calculate_thickness`.
    :return: ``{'left': {subregion: map}, 'right': {subregion: map}}`` keyed by patient
        side. The per-subregion maps are the original objects (not copied), and a
        :class:`~morphometry.cartilage.thickness.ThicknessMaps` input yields subsets that keep
        its in-plane frame (so they can still be overlapped with the femoral map); a side is
        an empty dict if the input holds none of its subregions.
    :raises ValueError: If a key belongs to neither compartment (it would otherwise be
        silently dropped from both searches).
    """
    anatomical = [key for key in tibia_thickness if key.lower().endswith('mt')]
    if anatomical:
        raise ValueError(f'Tibial subregion keys {anatomical} are medial/lateral names; '
                         'split_tibia_thickness_by_compartment needs the patient-side keys '
                         '(*LT = patient left, *RT = patient right).')
    sides = {'left': [], 'right': []}
    for key in tibia_thickness:
        suffix = key.lower()[-2:]
        if suffix == 'lt':
            sides['left'].append(key)
        elif suffix == 'rt':
            sides['right'].append(key)
        else:
            raise ValueError(f"Tibial subregion {key!r} belongs to neither compartment "
                             f"(expected a '*LT' or '*RT' key).")
    if isinstance(tibia_thickness, ThicknessMaps):
        return {side: tibia_thickness.subset(keys) for side, keys in sides.items()}
    return {side: {key: tibia_thickness[key] for key in keys} for side, keys in sides.items()}


def find_thinnest_area_by_compartment(tibia_thickness: dict, femur_thickness: dict,
                                      image: Image, neighbourhood_mm: float = 3.0,
                                      radius_mm: Optional[float] = None,
                                      edge_margin_mm: float = DEFAULT_EDGE_MARGIN_MM,
                                      min_thickness_mm: float = 0.0) -> dict:
    """
    Locate the thinnest cartilage area separately in each tibiofemoral compartment.

    The tibial thickness dict is partitioned by patient side with
    :func:`split_tibia_thickness_by_compartment`, and :func:`find_thinnest_area` is run on
    each side's tibial subset against the *full* femoral map. Because that search only
    considers positions where tibial and femoral thickness overlap, each search is
    confined to its own plateau: a globally thinner point in the other compartment cannot
    win, and the median smoothing never mixes values across the intercondylar notch (the
    notch cells are outside the overlap and stay NaN).

    The rim warning on :func:`find_thinnest_area` applies here in full, and the per-compartment
    results are the ones it bites hardest: restricting the search moves the chosen cell on
    nearly every compartment of every scan, and reorders which compartment looks most
    significant in a flexion trend.

    :param tibia_thickness: Per-subregion tibial thickness dict (from
        ``Tibia.calculate_thickness``): ``{subregion: {(u, v): thickness}}``.
    :param femur_thickness: Per-subregion femoral thickness dict (from
        ``Femur.calculate_thickness``); used whole for both compartments.
    :param image: The segmentation image; only the fallback in-plane step for maps with no
        frame (see :func:`_overlap_spacing`).
    :param neighbourhood_mm: Diameter (mm) of the median-smoothing neighbourhood, see
        :func:`find_thinnest_area`.
    :param radius_mm: Radius (mm) of the refinement disc, see :func:`find_thinnest_area`.
    :param edge_margin_mm: Distance (mm) a candidate must keep from each bone's own map
        boundary -- read the warning on :func:`find_thinnest_area` before changing it.
    :param min_thickness_mm: Optional per-bone thickness floor, see :func:`find_thinnest_area`.
    :return: ``{'left': result, 'right': result}`` keyed by **patient** side, where each
        result has exactly the fields returned by :func:`find_thinnest_area` (in-plane cells
        of the knee's shared frame).
    :raises ValueError: If a compartment has no tibial thickness values or no overlap with
        the femoral map (the message names the compartment), or if a tibial subregion key
        belongs to neither compartment.
    """
    results = {}
    for side, subset in split_tibia_thickness_by_compartment(tibia_thickness).items():
        if not any(subset.values()):
            raise ValueError(f"The patient-{side} compartment has no tibial thickness values.")
        try:
            results[side] = find_thinnest_area(subset, femur_thickness, image,
                                               neighbourhood_mm=neighbourhood_mm,
                                               radius_mm=radius_mm,
                                               edge_margin_mm=edge_margin_mm,
                                               min_thickness_mm=min_thickness_mm)
        except ValueError as exc:
            raise ValueError(f'The patient-{side} compartment: {exc}') from exc
    return results


def thinnest_point_position(tibia_thickness: dict, femur_thickness: dict,
                            point: tuple) -> np.ndarray:
    """
    Locate the 3D world position of a thinnest ``(u, v)`` cell on the contact surface.

    The thinnest point is an in-plane cell of the knee's shared frame (the thickness maps
    collapse the direction the joint is compressed along). Its depth is recovered from the two
    articular surfaces recorded for that cell
    (:attr:`~morphometry.cartilage.thickness.ThicknessMaps.positions`): the femoral and the
    tibial articular surface bracket the joint contact, so their midpoint is used. If only one
    bone recorded a surface there, that bone's surface is used; if neither did, the cell's own
    in-plane centre on the frame plane is the last resort.

    :param tibia_thickness: The tibial :class:`~morphometry.cartilage.thickness.ThicknessMaps`.
    :param femur_thickness: The femoral one.
    :param point: The thinnest ``(u, v)`` cell.
    :return: The marker's world ``(x, y, z)`` position in mm.
    :raises ValueError: If neither map carries an in-plane frame to place the cell in.
    """
    cell = (int(point[0]), int(point[1]))
    frame = thickness_frame(tibia_thickness) or thickness_frame(femur_thickness)
    if frame is None:
        raise ValueError('The thickness maps carry no in-plane frame, so the thinnest point '
                         f'{point} cannot be placed in 3D.')
    surfaces = [maps.positions[cell] for maps in (tibia_thickness, femur_thickness)
                if cell in getattr(maps, 'positions', {})]
    if not surfaces:
        return frame.cell_centre(cell)
    return np.mean(np.asarray(surfaces, dtype=float), axis=0)


# Arrow colours for labelled thinnest points (patient-side compartments and the whole knee);
# unknown labels cycle through the spare colours.
_THINNEST_COLORS = {'left': 'red', 'right': 'blue', 'both': 'black'}
_THINNEST_SPARE_COLORS = ('magenta', 'darkorange', 'cyan', 'green', 'purple')


def _add_thinnest_arrow(plotter: pv.Plotter, marker: np.ndarray, length: float,
                        color: str, label: str):
    """
    Point a labelled arrow at a thinnest-point marker from an oblique offset.

    :param plotter: The PyVista ``Plotter`` to draw into.
    :param marker: The marker's physical ``(x, y, z)`` position (the arrow tip).
    :param length: Arrow length in physical units (typically scaled to the scene size).
    :param color: Colour of the arrow, the marker point and its label.
    :param label: Text placed at the marker.
    """
    direction = np.array([1.0, 0.0, 1.0])
    direction /= np.linalg.norm(direction)
    start = marker - direction * length
    arrow = pv.Arrow(start=start, direction=direction, scale=length)
    plotter.add_mesh(arrow, color=color)
    plotter.add_point_labels(np.array([marker]), [label], point_color=color,
                             point_size=12, font_size=14, text_color=color,
                             always_visible=True)


def plot_thinnest_point(tibia: 'Tibia', femur: 'Femur', tibia_thickness: dict,
                        femur_thickness: dict, thinnest: dict, plotter: pv.Plotter = None,
                        show: bool = None, base: str = 'thickness') -> pv.Plotter:
    """
    Render the knee cartilage in 3D and mark the thinnest point(s) with arrows.

    :param tibia: The ``Tibia`` for the knee.
    :param femur: The ``Femur`` for the same knee.
    :param tibia_thickness: The tibial thickness maps from ``Tibia.calculate_thickness``; they
        carry the in-plane frame and the articular-surface position of every cell, which is
        what places the arrows in 3D (:func:`thinnest_point_position`).
    :param femur_thickness: The femoral thickness maps, likewise.
    :param thinnest: Either a single result dict from :func:`find_thinnest_area` (its
        ``'point'`` is marked with a red arrow labelled ``'thinnest'``), or a
        ``{label: result}`` dict such as the output of
        :func:`find_thinnest_area_by_compartment`, optionally extended with a ``'both'``
        whole-knee entry. Each entry then gets its own arrow in one scene, labelled
        ``'thinnest (<label>)'`` and coloured by label (``left`` red, ``right`` blue,
        ``both`` black; other labels cycle through spare colours). ``None`` entries
        (e.g. a compartment whose search failed) are skipped.
    :param plotter: An existing PyVista ``Plotter`` to draw into (a new one if ``None``).
    :param show: Whether to call ``plotter.show()``. Defaults to ``True`` only when this
        function created the plotter.
    :param base: ``'thickness'`` to render the thickness heatmap, or ``'segments'`` to
        render the raw subregion voxel volume, underneath the arrow.
    :return: The ``Plotter`` the scene was drawn into.
    """
    if base not in ('thickness', 'segments'):
        raise ValueError("base must be 'thickness' or 'segments'.")
    created = plotter is None
    if plotter is None:
        plotter = pv.Plotter()

    if base == 'segments':
        plot_knee_segments(tibia, femur, plotter=plotter, show=False)
    else:
        plot_knee_thickness(tibia, femur, tibia_thickness, femur_thickness,
                            plotter=plotter, show=False)

    # Arrow length is scaled to the cartilage scene, measured before any arrow is added.
    b = plotter.bounds
    diagonal = float(np.linalg.norm([b[1] - b[0], b[3] - b[2], b[5] - b[4]]))
    length = 0.2 * diagonal if diagonal > 0 else 1.0

    if 'point' in thinnest:                      # a single find_thinnest_area result
        markers = [('thinnest', 'red', thinnest)]
    else:                                        # {label: result}, e.g. per compartment
        spare = itertools.cycle(_THINNEST_SPARE_COLORS)
        markers = [(f'thinnest ({label})',
                    _THINNEST_COLORS[label] if label in _THINNEST_COLORS else next(spare),
                    result)
                   for label, result in thinnest.items() if result is not None]

    for label, color, result in markers:
        marker = thinnest_point_position(tibia_thickness, femur_thickness, result['point'])
        _add_thinnest_arrow(plotter, marker, length, color, label)

    if show or (show is None and created):
        plotter.show()
    return plotter


# --- Visualisation ---------------------------------------------------------
#
# Fixed, distinct colours per subregion for the "segment" view. Each entry is
# ``(attribute_name, display_label, colour)`` for the tibia and
# ``(attribute_name, display_label, colour, thickness_dict_key)`` for the femur
# (the femoral thickness dict keys do not all match the voxel attribute names).

TIBIA_SUBREGIONS = [
    ('clt', 'cLT', 'crimson'),
    ('ilt', 'iLT', 'royalblue'),
    ('elt', 'eLT', 'seagreen'),
    ('alt', 'aLT', 'darkorange'),
    ('plt', 'pLT', 'mediumpurple'),
    ('crt', 'cRT', 'gold'),
    ('irt', 'iRT', 'deepskyblue'),
    ('ert', 'eRT', 'limegreen'),
    ('art', 'aRT', 'salmon'),
    ('prt', 'pRT', 'orchid'),
]

FEMUR_SUBREGIONS = [
    ('iclf', 'iclf', 'royalblue', 'iclf'),
    ('cclf', 'cclf', 'crimson', 'cclf'),
    ('eclf', 'eclf', 'seagreen', 'eclf'),
    ('icrf', 'icrf', 'deepskyblue', 'icrf'),
    ('ccrf', 'ccrf', 'gold', 'ccrf'),
    ('ecrf', 'ecrf', 'limegreen', 'ecrf'),
    ('alf', 'alf', 'darkorange', 'left_anterior_zone'),
    ('arf', 'arf', 'salmon', 'right_anterior_zone'),
    ('plf', 'plf', 'mediumpurple', 'left_posterior_zone'),
    ('prf', 'prf', 'orchid', 'right_posterior_zone'),
]


def _indices_to_physical(image: Image, indices: np.ndarray) -> np.ndarray:
    """
    Convert an ``Nx3`` array of voxel indices to physical (world) coordinates.

    Mirrors :meth:`Image.transform_index_to_physical_point` but is vectorised for
    the nibabel backend; SimpleITK is transformed per point (its API requires an
    integer index tuple).

    :param image: The ``Image`` whose coordinate system defines the transform.
    :param indices: An ``Nx3`` array of voxel indices.
    :return: An ``Nx3`` array of physical points (empty ``(0, 3)`` array if no input).
    """
    indices = np.asarray(indices, dtype=float)
    if len(indices) == 0:
        return np.empty((0, 3))
    if image.type == 'nibabel':
        affine = image.image.affine
        M = affine[:3, :3]
        abc = affine[:3, 3]
        return indices @ M.T + abc
    return np.array([
        image.transform_index_to_physical_point([int(round(v)) for v in idx])
        for idx in indices
    ])


def _add_voxel_glyphs(plotter: pv.Plotter, image: Image, points: np.ndarray, color: str, label: str) -> bool:
    """
    Add the voxels of one subregion to a plotter as uniformly-sized cube glyphs.

    Cubes are sized to the image voxel spacing so the rendered volume matches the
    physical extent of the segmentation. No-op for an empty/absent point cloud.

    :param plotter: The PyVista ``Plotter`` to draw into.
    :param image: The ``Image`` providing the index->physical transform and spacing.
    :param points: An ``Nx3`` array of voxel indices belonging to the subregion.
    :param color: The cube colour.
    :param label: The legend label for this subregion.
    :return: ``True`` if glyphs were added, ``False`` for an empty/absent point cloud.
    """
    if points is None or len(points) == 0:
        return False
    phys = _indices_to_physical(image, np.asarray(points))
    cloud = pv.PolyData(phys)
    sx, sy, sz = (float(s) for s in image.spacing[:3])
    cube = pv.Cube(x_length=sx, y_length=sy, z_length=sz)
    glyphs = cloud.glyph(geom=cube, scale=False, orient=False)
    plotter.add_mesh(glyphs, color=color, label=label)
    return True


def _thickness_meshes(subdivision, thicknesses: dict) -> list:
    """
    The subchondral bone area as a mesh carrying the measured thickness as a cell scalar.

    This is the surface the thickness was measured on, so no reconstruction is involved: the
    subdivision's own tAB mesh is returned with the per-triangle ``cAB -> AC`` distances
    (:attr:`~morphometry.cartilage.thickness.ThicknessMaps.triangle_thickness`) attached as
    the cell array ``'thickness'``.

    Fallback for a plain per-position dict (one not produced by ``calculate_thickness``, e.g.
    a hand-built test map): the positions themselves are rendered as an unconnected point
    cloud carrying the thickness, which still colours correctly but shows no surface.

    :param subdivision: The bone's subdivision (its ``mesh`` is coloured).
    :param thicknesses: The :class:`~morphometry.cartilage.thickness.ThicknessMaps` from
        ``calculate_thickness``, or any ``{subregion: {(u, v): thickness}}`` dict.
    :return: A list of meshes carrying a ``'thickness'`` array (empty if there is nothing to
        draw), for :func:`_add_thickness_meshes`.
    """
    values = getattr(thicknesses, 'triangle_thickness', None)
    if values is not None and subdivision is not None and subdivision.mesh.n_cells == len(values):
        mesh = subdivision.mesh.copy()
        mesh.cell_data['thickness'] = np.asarray(values, dtype=float)
        return [mesh]

    frame = thickness_frame(thicknesses)
    if frame is None:
        return []
    points, scalars = [], []
    for tmap in thicknesses.values():
        for cell, value in (tmap or {}).items():
            if value is None or np.isnan(value):
                continue
            points.append(thicknesses.positions.get(tuple(cell), frame.cell_centre(cell))
                          if hasattr(thicknesses, 'positions') else frame.cell_centre(cell))
            scalars.append(float(value))
    if not points:
        return []
    cloud = pv.PolyData(np.asarray(points, dtype=float))
    cloud['thickness'] = np.asarray(scalars, dtype=float)
    return [cloud]


def _add_thickness_meshes(plotter: pv.Plotter, meshes: list, cmap: str,
                          clim: Tuple[float, float], show_scalar_bar: bool):
    """
    Add thickness heatmap surfaces to a plotter with a shared colour scale.

    A single scalar bar (in mm) is drawn for the whole set; all surfaces share the
    same colour limits so their colours are directly comparable.

    :param plotter: The PyVista ``Plotter`` to draw into.
    :param meshes: A list of ``PolyData`` surfaces each carrying a ``'thickness'`` scalar.
    :param cmap: A matplotlib colormap name.
    :param clim: ``(min, max)`` colour limits; derived from the meshes when ``None``.
    :param show_scalar_bar: Whether to draw the shared thickness colour bar.
    """
    if not meshes:
        return
    if clim is None:
        all_values = np.concatenate([m['thickness'] for m in meshes])
        clim = (float(np.nanmin(all_values)), float(np.nanmax(all_values)))

    for i, mesh in enumerate(meshes):
        plotter.add_mesh(
            mesh, scalars='thickness', cmap=cmap, clim=clim,
            show_scalar_bar=(show_scalar_bar and i == 0),
            scalar_bar_args={'title': 'Thickness (mm)'},
        )


def plot_knee_segments(tibia: Tibia, femur: Femur, plotter: pv.Plotter = None,
                       show: bool = None) -> pv.Plotter:
    """
    Render both the tibial and femoral cartilage subregions in a single 3D scene,
    each subregion in its own distinct colour.

    :param tibia: A ``Tibia`` for the knee.
    :param femur: A ``Femur`` for the same knee.
    :param plotter: An existing PyVista ``Plotter`` to draw into (new one if ``None``).
    :param show: Whether to call ``plotter.show()`` before returning. Defaults to
        ``True`` only when this function created the plotter.
    :return: The ``Plotter`` both bones were drawn into.
    """
    created = plotter is None
    if plotter is None:
        plotter = pv.Plotter()
    tibia.plot_segments(plotter=plotter, show=False)
    femur.plot_segments(tibia=tibia, plotter=plotter, show=False)
    if show or (show is None and created):
        plotter.show()
    return plotter


def plot_knee_thickness(tibia: Tibia, femur: Femur, tibia_thicknesses: dict,
                        femur_thicknesses: dict, plotter: pv.Plotter = None, show: bool = None,
                        cmap: str = 'viridis', clim: Tuple[float, float] = None) -> pv.Plotter:
    """
    Render both the tibial and femoral articular surfaces as a thickness heatmap in a
    single 3D scene, sharing one colour scale and a single colour bar.

    :param tibia: A ``Tibia`` for the knee.
    :param femur: A ``Femur`` for the same knee.
    :param tibia_thicknesses: The tibial thickness dict (from ``Tibia.calculate_thickness``).
    :param femur_thicknesses: The femoral thickness dict (from ``Femur.calculate_thickness``).
    :param plotter: An existing PyVista ``Plotter`` to draw into (new one if ``None``).
    :param show: Whether to call ``plotter.show()`` (see :func:`plot_knee_segments`).
    :param cmap: A matplotlib colormap name used for the heatmap.
    :param clim: ``(min, max)`` colour limits in mm, shared by both bones. Derived from
        both thickness dicts when ``None`` so the colours are directly comparable.
    :return: The ``Plotter`` both heatmaps were drawn into.
    """
    created = plotter is None
    if plotter is None:
        plotter = pv.Plotter()
    if clim is None:
        clim = _combined_thickness_clim(tibia_thicknesses, femur_thicknesses)
    # Only the tibia draws the shared scalar bar; the femur reuses the same clim.
    tibia.plot_thickness(tibia_thicknesses, plotter=plotter, show=False, cmap=cmap, clim=clim)
    femur.plot_thickness(femur_thicknesses, tibia=tibia, plotter=plotter, show=False,
                         cmap=cmap, clim=clim, show_scalar_bar=False)
    if show or (show is None and created):
        plotter.show()
    return plotter


def _combined_thickness_clim(*thickness_dicts: dict) -> Optional[Tuple[float, float]]:
    """
    Compute shared ``(min, max)`` thickness colour limits across one or more thickness results.

    The limits are taken from whatever the heatmap actually draws: the per-triangle
    thicknesses when a result carries them (:func:`_thickness_meshes` colours the tAB mesh
    with those), otherwise the per-position maps. Mixing the two would clip the extremes of
    the triangle values, since a map cell is an area-weighted mean of several triangles.
    ``NaN`` values are ignored.

    :param thickness_dicts: Any number of
        :class:`~morphometry.cartilage.thickness.ThicknessMaps`, or plain
        ``{subregion: {(u, v): thickness}}`` dicts.
    :return: The ``(min, max)`` thickness, or ``None`` if there are no finite values.
    """
    values = []
    for thicknesses in thickness_dicts:
        triangles = getattr(thicknesses, 'triangle_thickness', None)
        if triangles is not None and len(triangles):
            finite = np.asarray(triangles, dtype=float)
            values.extend(finite[np.isfinite(finite)].tolist())
            continue
        for tmap in thicknesses.values():
            values.extend(v for v in tmap.values() if v is not None and not np.isnan(v))
    if not values:
        return None
    return float(np.min(values)), float(np.max(values))
