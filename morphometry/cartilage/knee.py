import itertools
import logging

import pandas as pd
import pyvista as pv
import numpy as np
from typing import Optional, Tuple
from scipy.spatial import KDTree
from scipy.ndimage import label, generic_filter
from morphometry.image_io import Image
from morphometry.cartilage.surface import (
    array_indices_to_world, index_to_world_affine, transfer_qc, TRANSFER_FAR_MM,
    TRANSFER_MAX_FAR_FRACTION,
)
from morphometry.cartilage.tibia_regions import (
    subdivide_tibia, anatomical_tibia_names, tibia_display_label, TIBIA_REGION_KEYS,
)
from morphometry.cartilage.femur_regions import (
    subdivide_femur, anatomical_femur_names, FEMUR_REGION_KEYS, ZONE_THICKNESS_KEYS,
)

logger = logging.getLogger(__name__)

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
        self.superior_surface, self.inferior_surface = None, None
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
        Get the points of the superior and inferior surfaces of the tibia.
        """
        image_array = self.image.array
        cartilage = np.where(image_array == self.cartilage_label, 1, 0)
        cartilage = _remove_mask_outliers(cartilage, self.outlier_ratio)
        cartilage = np.argwhere(cartilage)
        self.point_cloud = cartilage.astype(float)
        self.superior_surface, self.inferior_surface = get_superior_and_inferior_surface_points(cartilage)

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

    def mesh_method(self, superior_surface: np.ndarray, inferior_surface: np.ndarray) -> dict:
        """
        Calculate the thickness of a cartilage using a mesh-based ray tracing method.

        :param superior_surface: The superior surface of the zone.
        :param inferior_surface: The inferior surface of the zone.
        :return: A dictionary where keys are coordinates and values are the thicknesses.
        """
        superior_mesh, inferior_mesh = build_cartilage_meshes(superior_surface, inferior_surface)

        superior_mesh = superior_mesh.compute_normals(cell_normals=False, point_normals=True, inplace=False,
                                                      auto_orient_normals=True)

        thicknesses = dict()
        for i, point in enumerate(superior_mesh.points):
            point_coords = np.hstack((point[0], point[1]))
            point_coords = tuple(point_coords)
            vec = superior_mesh['Normals'][i] * superior_mesh.length
            v0 = point - vec
            v1 = point + vec
            iv, ic = inferior_mesh.ray_trace(v0, v1, first_point=True)
            # dist = np.linalg.norm(iv - point) * self.image.get_spacing()[0]
            if len(iv) == 0:
                thicknesses[point_coords] = np.nan
                continue

            iv_world = self.image.transform_index_to_physical_point(iv)
            point_world = self.image.transform_index_to_physical_point(point)
            dist = np.linalg.norm(iv_world - point_world)
            thicknesses[point_coords] = dist  # discard axial coordinate

        return thicknesses

    def knn_method(self, superior_surface: np.ndarray, inferior_surface: np.ndarray) -> dict:
        """
        Calculate the thickness of a cartilage using a k-nearest neighbour method.

        :param superior_surface: The superior surface of the zone.
        :param inferior_surface: The inferior surface of the zone.
        :return: A dictionary where keys are coordinates and values are the thicknesses.
        """
        thicknesses = dict()
        f = lambda x: self.image.transform_index_to_physical_point(x)
        ss_world = list(superior_surface)
        ss_world = list(map(f, ss_world))
        ss_world = np.array(ss_world)
        is_world = list(inferior_surface)
        is_world = list(map(f, is_world))
        is_world = np.array(is_world)
        superior_tree = KDTree(ss_world)
        distances, indices = superior_tree.query(is_world, k=1)

        for i, distance in enumerate(distances):
            point_coords = np.hstack((inferior_surface[i][0], inferior_surface[i][1]))
            point_coords = tuple(point_coords)
            thicknesses[point_coords] = distance

        return thicknesses

    def calculate_thickness(self, method: str = 'mesh') -> dict:
        """
        Calculate the thickness of the tibial cartilage for all subregions of the tibial cartilage.

        :param method: The method used for thickness calculation. Can be either 'mesh' or 'knn'.
        :return: A dictionary where keys are zone labels and values are dictionaries, where keys are coordinates and values are thicknesses.
        """
        assert method in ['mesh', 'knn'], 'Method must be either "mesh" or "knn".'
        thicknesses = dict()
        self.get_surface_points()
        self.calculate_landmarks()
        self.extract_subregions()

        for subregion in ['clt', 'ilt', 'elt', 'alt', 'plt', 'crt', 'irt', 'ert', 'art', 'prt']:
            superior_surface, inferior_surface = get_superior_and_inferior_surface_points(getattr(self, subregion))
            if method == 'mesh':
                tmp = self.mesh_method(superior_surface, inferior_surface)
            else:
                tmp = self.knn_method(superior_surface, inferior_surface)

            thicknesses[subregion] = tmp

        return thicknesses

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
        Render the tibial articular (superior) surface coloured by measured cartilage
        thickness, as a heatmap with a shared colour scale across all subregions.

        :param thicknesses: The per-subregion thickness dict returned by
            :meth:`calculate_thickness` (``{subregion: {(x, y): thickness_mm}}``).
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
        self._ensure_subregions()

        meshes = []
        for attr, label, color in TIBIA_SUBREGIONS:
            tmap = thicknesses.get(attr)
            if not tmap:
                continue
            surf = _thickness_surface(self.image, getattr(self, attr), tmap)
            if surf is not None:
                meshes.append(surf)

        _add_thickness_meshes(plotter, meshes, cmap, clim, show_scalar_bar)
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

    def mesh_method(self, superior_surface: np.ndarray, inferior_surface: np.ndarray) -> dict:
        """
        Calculate the thickness of a cartilage using a mesh-based ray tracing method.

        :param superior_surface: The superior surface of the zone.
        :param inferior_surface: The inferior surface of the zone.
        :return: A dictionary where keys are coordinates and values are the thicknesses.
        """
        superior_mesh, inferior_mesh = build_cartilage_meshes(superior_surface, inferior_surface)

        superior_mesh = superior_mesh.compute_normals(cell_normals=False, point_normals=True, inplace=False,
                                                      auto_orient_normals=True)

        thicknesses = dict()
        # The key is always array axes 0 and 1, whatever axis the surfaces were taken along;
        # for a zone measured along axis 1 that second component is the depth coordinate and
        # the keys collide. See Femur.calculate_thickness and plan step 4.
        for i, point in enumerate(superior_mesh.points):
            vec = superior_mesh['Normals'][i] * superior_mesh.length
            v0 = point - vec
            v1 = point + vec
            iv, ic = inferior_mesh.ray_trace(v0, v1, first_point=True)
            if len(iv) == 0:
                thicknesses[(point[0], point[1])] = np.nan
                continue
            # dist = np.linalg.norm(iv - point) * self.image.get_spacing()[0]
            iv_world = self.image.transform_index_to_physical_point(iv)
            point_world = self.image.transform_index_to_physical_point(point)
            dist = np.linalg.norm(iv_world - point_world)
            thicknesses[(point[0], point[1])] = dist  # discard axial coordinate

        return thicknesses

    def knn_method(self, superior_surface: np.ndarray, inferior_surface: np.ndarray) -> dict:
        """
        Calculate the thickness of a cartilage using a k-nearest neighbour method.

        :param superior_surface: The superior surface of the zone.
        :param inferior_surface: The inferior surface of the zone.
        :return: A dictionary where keys are coordinates and values are the thicknesses.
        """
        thicknesses = dict()
        f = lambda x: self.image.transform_index_to_physical_point(x)
        ss_world = list(superior_surface)
        ss_world = list(map(f, ss_world))
        ss_world = np.array(ss_world)
        is_world = list(inferior_surface)
        is_world = list(map(f, is_world))
        is_world = np.array(is_world)
        superior_tree = KDTree(ss_world)
        distances, indices = superior_tree.query(is_world, k=1)
        # The key is always array axes 0 and 1, whatever axis the surfaces were taken along;
        # for a zone measured along axis 1 that second component is the depth coordinate and
        # the keys collide. See Femur.calculate_thickness and plan step 4.
        for i, distance in enumerate(distances):
            thicknesses[(inferior_surface[i][0], inferior_surface[i][1])] = distance

        return thicknesses

    def calculate_thickness(self, tibia: Tibia = None, method: str = 'mesh') -> dict:
        """
        Calculate the cartilage thickness for all subregions of the femoral cartilage.

        The thickness itself is still measured per voxel column: the two opposing surfaces of
        each subregion are extracted per image column and the distance between them taken
        (step 4 of ``docs/wirth_eckstein_2008_plan.md`` replaces this with the paper's
        area-weighted surface distances). The columns run along the superior-inferior image
        axis, except for the two posterior-condyle zones, whose articular surface faces
        posteriorly: theirs run along the image axis nearest the **femoral** anterior-posterior
        axis (:func:`_depth_axis`), which is the image's anterior-posterior axis in an extended
        knee but not in a flexed one.

        **The map keys do not follow the depth axis.** :meth:`mesh_method` / :meth:`knn_method`
        key every point by its array axes 0 and 1, so a zone measured along axis 1 is keyed by
        ``(x, depth)`` rather than by a position in the plane it was measured across: two
        points of such a zone that share ``(x, y)`` overwrite each other, and
        :func:`find_thinnest_area` / :func:`_thickness_surface` place its values at the wrong
        in-plane position. The defect predates this method -- the posterior zones were always
        measured across one axis and keyed by another -- but it used to apply to every scan and
        now applies only to those whose femoral A-P axis is the image's (i.e. at low flexion
        but not at 60-90 degrees), so it varies *within* a flexion series. It is not fixable
        here: the two bones would have to share one in-plane frame, which is what the
        area-weighted thickness of plan step 4 brings.

        :param tibia: Ignored, and kept only so that callers written against the old
            tibia-relative subdivision keep working. The femoral subregions are now derived
            from the femur alone. Anything other than ``None`` or a ``Tibia`` raises, so that
            ``calculate_thickness('knn')`` -- which would bind the method to this parameter and
            silently run the mesh method -- is caught.
        :param method: The method used for thickness calculation. Can be either 'mesh' or 'knn'.
        :return: A dictionary where keys are zone labels and values are dictionaries, where
            keys are coordinates and values are thicknesses. The weight-bearing condyles'
            thirds are keyed by their region key (``'iclf'`` ...), the extension zones by the
            names the femoral thickness dicts have always used for them
            (``'left_anterior_zone'`` ... ``'right_posterior_zone'``).
        :raises TypeError: If ``tibia`` is neither ``None`` nor a ``Tibia``.
        """
        # 'tibia' is ignored, but it is still the first positional parameter, so a caller
        # writing femur.calculate_thickness('knn') would silently run the mesh method.
        if tibia is not None and not isinstance(tibia, Tibia):
            raise TypeError(
                f'Femur.calculate_thickness takes the (ignored) tibia first and the method '
                f'second; got {tibia!r} as the tibia. Call it as '
                f'calculate_thickness(method={tibia!r}).')
        assert method in ['mesh', 'knn'], 'Method must be either "mesh" or "knn".'
        self.extract_subregions()

        thicknesses = dict()
        for key in FEMUR_REGION_KEYS:
            points = getattr(self, key)
            zone = ZONE_THICKNESS_KEYS.get(key, key)
            if points is None or len(points) == 0:
                thicknesses[zone] = dict()
                continue
            superior_surface, inferior_surface = _zone_surface_points(
                points, depth_axis=self.depth_axis(key))
            if method == 'mesh':
                thicknesses[zone] = self.mesh_method(superior_surface, inferior_surface)
            else:
                thicknesses[zone] = self.knn_method(superior_surface, inferior_surface)

        return thicknesses

    def depth_axis(self, key: str) -> int:
        """
        The array axis a subregion's (voxel-column) thickness is measured along.

        :param key: A region key from
            :data:`~morphometry.cartilage.femur_regions.FEMUR_REGION_KEYS`.
        :return: ``2`` (superior-inferior) for every region except the two posterior-condyle
            zones, whose surface faces posteriorly: for those, the array axis nearest the
            femoral anterior-posterior axis (see :func:`_depth_axis`).
        """
        if key not in ('plf', 'prf') or self.frame is None:
            return 2
        return _depth_axis(self.image, self.frame.ap_axis)

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
        Render the femoral articular (superior) surface coloured by measured cartilage
        thickness, as a heatmap with a shared colour scale across all subregions.

        :param thicknesses: The per-subregion thickness dict returned by
            :meth:`calculate_thickness`.
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
        self._ensure_subregions()

        meshes = []
        for attr, label, color, thickness_key in FEMUR_SUBREGIONS:
            tmap = thicknesses.get(thickness_key)
            if not tmap:
                continue
            surf = _thickness_surface(self.image, getattr(self, attr), tmap,
                                      depth_axis=self.depth_axis(attr))
            if surf is not None:
                meshes.append(surf)

        _add_thickness_meshes(plotter, meshes, cmap, clim, show_scalar_bar)
        if show or (show is None and created):
            plotter.show()
        return plotter


def _depth_axis(image: Image, direction: np.ndarray) -> int:
    """
    The axis of ``image.array`` most nearly parallel to a world direction.

    The voxel-column thickness method measures between the two extremes of a subregion along
    one array axis. Which axis that should be depends on the direction the articular surface
    faces, which is a property of the anatomy (the femoral frame), not of the image -- a
    flexed knee turns the femur's anterior-posterior axis into a different image axis.

    :param image: The ``Image`` / ``Segmentation`` whose array axes are meant (either backend;
        ``array`` is ``(i, j, k)`` for nibabel and ``(k, j, i)`` for SimpleITK).
    :param direction: A world (RAS, mm) direction.
    :return: ``0``, ``1`` or ``2``: the index into ``image.array``'s shape.
    """
    columns = index_to_world_affine(image)[:3, :3]
    if image.type != 'nibabel':
        columns = columns[:, ::-1]
    unit = columns / np.linalg.norm(columns, axis=0)
    return int(np.argmax(np.abs(np.asarray(direction, dtype=float) @ unit)))


def _zone_surface_points(points: np.ndarray, depth_axis: int = 2):
    """
    The two opposing surfaces of one subregion's voxel cloud along a given array axis.

    The voxels are grouped by their two other coordinates and the extremes along
    ``depth_axis`` taken, so for the default axis 2 (superior-inferior in an LPI image) this
    is the subregion's superior and inferior surface.

    :param points: ``Nx3`` voxel indices of the subregion (LPI index space).
    :param depth_axis: The array axis to measure the subregion's thickness along.
    :return: ``(near, far)``, each an ``Mx3`` array of voxel indices in the original axis
        order; ``near`` holds the smaller coordinate along ``depth_axis``.
    """
    pts = np.asarray(points, dtype=float)
    order = [a for a in range(3) if a != depth_axis] + [depth_axis]
    near, far = get_superior_and_inferior_surface_points(pts[:, order])
    inverse = np.argsort(order)
    return near[:, inverse], far[:, inverse]


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
    Flatten a per-subregion thickness dict into a single ``{(x, y): thickness}`` map.

    Coordinates that appear in more than one subregion (rare, at subregion boundaries)
    are averaged; ``NaN`` / missing values are dropped.

    :param thicknesses: ``{subregion: {(x, y): thickness}}`` as returned by
        ``calculate_thickness``.
    :return: A flat ``{(x, y): thickness}`` map over all subregions.
    """
    accumulated: dict = {}
    for subregion_map in thicknesses.values():
        for coord, value in subregion_map.items():
            if value is None or np.isnan(value):
                continue
            accumulated.setdefault(coord, []).append(float(value))
    return {coord: float(np.mean(values)) for coord, values in accumulated.items()}


def find_thinnest_area(tibia_thickness: dict, femur_thickness: dict, image: Image,
                       neighbourhood_mm: float = 3.0, radius_mm: Optional[float] = None) -> dict:
    """
    Locate the thinnest *area* of the tibiofemoral cartilage, robust to outliers.

    A plain ``argmin`` over a thickness map is sensitive to single erroneous voxels, so
    this searches for the thinnest neighbourhood instead. The tibial and femoral thickness
    maps are summed over the axial coordinates present in *both* (the contact region),
    the combined map is median-smoothed to suppress single-voxel outliers, the thinnest
    neighbourhood centre is found, and the result is refined to the actual thinnest voxel
    within a disc around that centre.

    :param tibia_thickness: Per-subregion tibial thickness dict (from
        ``Tibia.calculate_thickness``): ``{subregion: {(x, y): thickness}}``.
    :param femur_thickness: Per-subregion femoral thickness dict (from
        ``Femur.calculate_thickness``).
    :param image: The segmentation image; only its in-plane voxel spacing is used, to
        convert the physical neighbourhood/radius sizes into an odd kernel size.
    :param neighbourhood_mm: Diameter (mm) of the median-smoothing neighbourhood. The
        kernel size ``k = round(neighbourhood_mm / in-plane spacing)`` is forced odd and
        at least 3, so the search is comparable across differently-sampled scans.
    :param radius_mm: Radius (mm) of the refinement disc around the neighbourhood centre.
        Defaults to ``neighbourhood_mm / 2`` so the disc matches the smoothing extent.
    :return: A dict with ``'point'`` (the actual thinnest ``(x, y)`` voxel), ``'center'``
        (the smoothed-neighbourhood centre ``(x, y)``), ``'center_thickness'`` (the robust,
        median-smoothed combined thickness of the thin area, mm), ``'combined_thickness'``
        (the raw, non-outlier-suppressed tibial + femoral thickness at ``point``, mm),
        ``'tibial_thickness'`` / ``'femoral_thickness'`` (the per-bone contributions at
        ``point``), and ``'kernel_size'`` (``k``).
    :raises ValueError: If the two maps share no overlapping coordinates.
    """
    tibia_flat = _flatten_thickness(tibia_thickness)
    femur_flat = _flatten_thickness(femur_thickness)

    overlap = set(tibia_flat) & set(femur_flat)
    if not overlap:
        raise ValueError('Tibial and femoral thickness maps have no overlapping coordinates.')

    coords = np.array(list(overlap), dtype=int)
    x_min, y_min = coords[:, 0].min(), coords[:, 1].min()
    x_max, y_max = coords[:, 0].max(), coords[:, 1].max()

    # Rasterise the per-bone maps onto dense grids over the overlap bounding box; cells
    # outside the (irregular) overlap footprint stay NaN. Indexing by grid position keeps
    # everything downstream free of float-coordinate dict lookups.
    tibia_grid = np.full((x_max - x_min + 1, y_max - y_min + 1), np.nan)
    femur_grid = np.full_like(tibia_grid, np.nan)
    for coord in overlap:
        gx, gy = int(coord[0]) - x_min, int(coord[1]) - y_min
        tibia_grid[gx, gy] = tibia_flat[coord]
        femur_grid[gx, gy] = femur_flat[coord]
    grid = tibia_grid + femur_grid          # combined thickness; NaN outside the overlap
    valid = ~np.isnan(grid)

    spacing = float(np.mean(image.spacing[:2]))
    k = int(round(neighbourhood_mm / spacing))
    if k % 2 == 0:
        k += 1
    k = max(3, k)

    # Median smooth with reflection padding, ignoring NaN neighbours.
    smoothed = generic_filter(grid, np.nanmedian, size=k, mode='reflect')

    # Restrict the neighbourhood-centre search to the overlap footprint.
    search = np.where(valid & ~np.isnan(smoothed), smoothed, np.inf)
    ci, cj = np.unravel_index(np.argmin(search), search.shape)
    center = (float(ci + x_min), float(cj + y_min))

    # Refine to the actual thinnest voxel within a disc around the centre, on the raw map.
    # The disc radius defaults to half the neighbourhood so it matches the smoothing extent
    # (k is a diameter) rather than doubling it.
    radius = neighbourhood_mm / 2.0 if radius_mm is None else radius_mm
    r = max(1, int(round(radius / spacing)))
    ii, jj = np.indices(grid.shape)
    disc = (ii - ci) ** 2 + (jj - cj) ** 2 <= r ** 2
    restricted = np.where(disc & valid, grid, np.inf)
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

    :param tibia_thickness: ``{subregion: {(x, y): thickness}}`` as returned by
        :meth:`Tibia.calculate_thickness`.
    :return: ``{'left': {subregion: map}, 'right': {subregion: map}}`` keyed by patient
        side. The per-subregion maps are the original objects (not copied); a side is an
        empty dict if the input holds none of its subregions.
    :raises ValueError: If a key belongs to neither compartment (it would otherwise be
        silently dropped from both searches).
    """
    anatomical = [key for key in tibia_thickness if key.lower().endswith('mt')]
    if anatomical:
        raise ValueError(f'Tibial subregion keys {anatomical} are medial/lateral names; '
                         'split_tibia_thickness_by_compartment needs the patient-side keys '
                         '(*LT = patient left, *RT = patient right).')
    split = {'left': {}, 'right': {}}
    for key, subregion_map in tibia_thickness.items():
        suffix = key.lower()[-2:]
        if suffix == 'lt':
            split['left'][key] = subregion_map
        elif suffix == 'rt':
            split['right'][key] = subregion_map
        else:
            raise ValueError(f"Tibial subregion {key!r} belongs to neither compartment "
                             f"(expected a '*LT' or '*RT' key).")
    return split


def find_thinnest_area_by_compartment(tibia_thickness: dict, femur_thickness: dict,
                                      image: Image, neighbourhood_mm: float = 3.0,
                                      radius_mm: Optional[float] = None) -> dict:
    """
    Locate the thinnest cartilage area separately in each tibiofemoral compartment.

    The tibial thickness dict is partitioned by patient side with
    :func:`split_tibia_thickness_by_compartment`, and :func:`find_thinnest_area` is run on
    each side's tibial subset against the *full* femoral map. Because that search only
    considers coordinates where tibial and femoral thickness overlap, each search is
    confined to its own plateau: a globally thinner point in the other compartment cannot
    win, and the median smoothing never mixes values across the intercondylar notch (the
    notch cells are outside the overlap and stay NaN).

    :param tibia_thickness: Per-subregion tibial thickness dict (from
        ``Tibia.calculate_thickness``): ``{subregion: {(x, y): thickness}}``.
    :param femur_thickness: Per-subregion femoral thickness dict (from
        ``Femur.calculate_thickness``); used whole for both compartments.
    :param image: The segmentation image (in-plane spacing sets the kernel size).
    :param neighbourhood_mm: Diameter (mm) of the median-smoothing neighbourhood, see
        :func:`find_thinnest_area`.
    :param radius_mm: Radius (mm) of the refinement disc, see :func:`find_thinnest_area`.
    :return: ``{'left': result, 'right': result}`` keyed by **patient** side, where each
        result has exactly the fields returned by :func:`find_thinnest_area` (voxel
        indices in the same LPI index space).
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
                                               radius_mm=radius_mm)
        except ValueError as exc:
            raise ValueError(f'The patient-{side} compartment: {exc}') from exc
    return results


def _thinnest_point_physical(tibia: 'Tibia', femur: 'Femur', point: tuple) -> np.ndarray:
    """
    Locate the 3D physical position of a thinnest ``(x, y)`` column on the contact surface.

    The thinnest point is an axial ``(x, y)`` coordinate (the thickness maps discard the
    superior-inferior axis). Its depth is recovered from the cartilage voxels in that
    column: the femoral tibia-facing surface (largest z) and the tibial femur-facing
    surface (smallest z) bracket the joint contact, so their midpoint is used. If only one
    bone has voxels in the column, that bone's contact surface is used instead.

    :param tibia: The ``Tibia`` (its ``point_cloud`` and ``image`` are read).
    :param femur: The ``Femur`` (its ``point_cloud`` is read).
    :param point: The thinnest ``(x, y)`` voxel coordinate.
    :return: The marker's physical ``(x, y, z)`` position.
    :raises ValueError: If neither bone has cartilage voxels in the column.
    """
    if getattr(tibia, 'point_cloud', None) is None:
        tibia.get_surface_points()
    x, y = int(point[0]), int(point[1])

    def surface_z(point_cloud, extreme):
        if point_cloud is None or len(point_cloud) == 0:
            return None
        column = point_cloud[(point_cloud[:, 0] == x) & (point_cloud[:, 1] == y)]
        return None if len(column) == 0 else extreme(column[:, 2])

    femur_z = surface_z(femur.point_cloud, np.max)   # femoral tibia-facing (inferior) surface
    tibia_z = surface_z(tibia.point_cloud, np.min)   # tibial femur-facing (superior) surface
    zs = [z for z in (femur_z, tibia_z) if z is not None]
    if not zs:
        raise ValueError(f'No cartilage voxels found in column {point} to place the marker.')

    index = np.array([[x, y, float(np.mean(zs))]])
    return _indices_to_physical(tibia.image, index)[0]


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
    :param tibia_thickness: The tibial thickness dict (used when ``base='thickness'``).
    :param femur_thickness: The femoral thickness dict (used when ``base='thickness'``).
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
        marker = _thinnest_point_physical(tibia, femur, result['point'])
        _add_thinnest_arrow(plotter, marker, length, color, label)

    if show or (show is None and created):
        plotter.show()
    return plotter


def build_cartilage_meshes(superior_points: np.ndarray, inferior_points: np.ndarray) -> Tuple[pv.PolyData, pv.PolyData]:
    """
    Build superior and inferior cartilage surface meshes from superior and inferior surface points.

    :param superior_points: A Nx3 point cloud representation of superior surface points.
    :param inferior_points: A Nx3 point cloud representation of inferior surface points.
    :return: Reconstructed surface meshes of the superior and inferior cartilage.
    """
    superior_mesh = pv.PolyData(superior_points)
    inferior_mesh = pv.PolyData(inferior_points)

    superior_mesh = superior_mesh.delaunay_2d(alpha=1.0)
    inferior_mesh = inferior_mesh.delaunay_2d(alpha=1.0)

    return superior_mesh, inferior_mesh


def get_superior_and_inferior_surface_points(cartilage: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the superior and inferior surface points of a cartilage.

    :param cartilage: A Nx3 point cloud representation of the cartilage.
    :return: Two point clouds of the superior and inferior surface points.
    """
    cols = ['x', 'y', 'z']

    df = pd.DataFrame(cartilage, columns=cols)
    """
    superior_points = df.groupby(group).min().reset_index() if transversal_direction == 1 else df.groupby(group).max().reset_index()
    superior_points = superior_points[cols]
    inferior_points = df.groupby(group).max().reset_index() if transversal_direction == 1 else df.groupby(group).min().reset_index()
    inferior_points = inferior_points[cols]
    """
    inferior_points = df.groupby(['x', 'y']).max().reset_index()
    superior_points = df.groupby(['x', 'y']).min().reset_index()

    return superior_points.to_numpy(), inferior_points.to_numpy()


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


def _thickness_surface(image: Image, points: np.ndarray, thickness_map: dict,
                       depth_axis: int = 2) -> pv.PolyData:
    """
    Reconstruct a subregion's articular (superior) surface and attach thickness values.

    The superior surface points are recovered from the subregion voxels with the same
    grouping used during thickness computation, then each surface point is coloured by
    the thickness measured at its ``(x, y)`` location. Points without a thickness entry
    (or ``NaN``) are dropped.

    :param image: The ``Image`` providing the index->physical transform.
    :param points: An ``Nx3`` array of the subregion's voxel indices.
    :param thickness_map: ``{(x, y): thickness_mm}`` for this subregion.
    :param depth_axis: The array axis the thickness was measured along (see
        :meth:`Femur.depth_axis`); the surface is the subregion's extreme along it.
    :return: A triangulated ``PolyData`` surface carrying a point-scalar ``'thickness'``,
        or ``None`` if fewer than three points could be coloured.
    """
    if points is None or len(points) == 0:
        return None
    superior, _ = _zone_surface_points(points, depth_axis=depth_axis)

    # The (x, y) keys rely on integer-valued voxel indices: mesh_method stores keys as
    # float32 (from PyVista mesh points) while this reconstruction is float64, so the
    # lookup only matches because integer voxel coordinates are exact in both dtypes.
    coords, scalars = [], []
    for p in superior:
        value = thickness_map.get((p[0], p[1]))
        if value is None or np.isnan(value):
            continue
        coords.append(p)
        scalars.append(value)

    if len(coords) < 3:
        return None

    phys = _indices_to_physical(image, np.array(coords))
    mesh = pv.PolyData(phys)
    mesh['thickness'] = np.array(scalars, dtype=float)
    return mesh.delaunay_2d()


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
    Compute shared ``(min, max)`` thickness colour limits across one or more
    per-subregion thickness dicts, ignoring ``NaN`` values.

    :param thickness_dicts: Any number of ``{subregion: {(x, y): thickness}}`` dicts.
    :return: The ``(min, max)`` thickness, or ``None`` if there are no finite values.
    """
    values = []
    for thicknesses in thickness_dicts:
        for tmap in thicknesses.values():
            values.extend(v for v in tmap.values() if v is not None and not np.isnan(v))
    if not values:
        return None
    return float(np.min(values)), float(np.max(values))
