"""
Tibial cartilage subregions after Wirth & Eckstein 2008 (IEEE TMI 27(6), section II-A).

Each tibial plate is divided on its *total area of subchondral bone* (tAB, see
:mod:`morphometry.cartilage.surface`) into five subregions:

- **central** (``c``): inside an elliptical cylinder about the plate's centre of gravity S,
  along the plate normal, scaled so that it holds ``central_fraction`` (20%) of the plate's
  tAB **area**;
- four **peripheral** wedges of the remaining area, cut by two planes through S that contain
  the plate normal and lie at +-45 degrees to ``tc``, the line joining the centres of
  gravity of the two plates: **internal** (``i``, the wedge facing the other plate),
  **external** (``e``, opposite), **anterior** (``a``) and **posterior** (``p``).

Every cut splits triangles exactly (:func:`~morphometry.cartilage.surface.split_mesh_by_scalar`),
and all geometry is in world millimetres, so the shares do not depend on voxel size or on
the knee's position in the scanner.

**Naming.** The plates are named by **patient side**, as everywhere else in the cartilage
code: region keys ``clt, ilt, elt, alt, plt`` belong to the patient-*left* plate and
``crt, irt, ert, art, prt`` to the patient-*right* plate (display labels ``cLT`` / ``cRT``
etc.). This ``LT`` means *left*, not the paper's *lateral*. When the knee's laterality is
known, :func:`anatomical_tibia_names` gives the paper's medial/lateral names (``cMT`` ...
``pLT``) as a separate mapping; the patient-side keys never change meaning.

Deviations from the paper (all deliberate, see the function docstrings):

- The paper takes the moments from the mesh **vertices**; here they are the exact
  area-weighted second moments of the triangles, which do not depend on how finely or
  evenly the surface is triangulated.
- The paper orients anterior/posterior (and checks medial/lateral) with a manually placed
  fibula landmark. The segmentations here have no fibula, so anterior is the world
  anterior axis (+y of RAS) projected into the plate plane; medial/lateral come from the
  caller-supplied laterality.
- The two plates are the two largest edge-connected components of the covered bone area
  (cAB); denuded holes are filled *per plate*, so the intercondylar area can never be
  enclosed and filled even when the two cartilage plates touch.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from morphometry.image_io import Image
from morphometry.cartilage import surface as S

# Non-fatal diagnostics (see the transfer QC below). Like measurements/hip.py, this logs
# rather than warnings.warn so every affected scan of a cohort run is reported; attach a
# handler to logging.getLogger('morphometry') to collect them.
logger = logging.getLogger(__name__)


# Region letters within a plate, in code order, and the two plates by patient side.
REGION_LETTERS = ('c', 'i', 'e', 'a', 'p')
PLATE_SIDES = ('left', 'right')
# Region key per integer code: code = plate_index * 5 + letter_index.
TIBIA_REGION_KEYS = tuple(f'{letter}{side[0]}t' for side in PLATE_SIDES for letter in REGION_LETTERS)

# QC of the transfer of the regions to the cartilage voxels (Tibia.calculate_landmarks): a
# tibial cartilage voxel is at most about one cartilage thickness (<= ~4 mm in dfg_knees,
# thickest regional maxima ~3.1-3.4 mm) plus a voxel half-diagonal from the plate's bone
# surface; on the clean dfg_knees no voxel is more than 5 mm from its nearest tAB triangle.
# Voxels farther than TRANSFER_FAR_MM (6 mm, a margin over that) are cartilage the tAB does
# not explain -- a plate missing from the bone coverage, cartilage label on the wrong bone,
# a broken segmentation (K7_90_belastet: 49% of its voxels are > 4 mm away, up to 45 mm).
# More than TRANSFER_MAX_FAR_FRACTION of them triggers a WARNING; a few stray voxels do not.
TRANSFER_FAR_MM = 6.0
TRANSFER_MAX_FAR_FRACTION = 0.01

# World anterior direction. The surface meshes are in RAS (+x patient right, +y anterior,
# +z superior) for both image backends, see surface.index_to_world_affine.
_ANTERIOR = np.array([0.0, 1.0, 0.0])
_SUPERIOR = np.array([0.0, 0.0, 1.0])


def tibia_display_label(key: str) -> str:
    """
    Display label of a patient-side region key, e.g. ``'clt'`` -> ``'cLT'``.

    :param key: A key from :data:`TIBIA_REGION_KEYS`.
    :return: The display label (subregion letter lower case, plate upper case).
    """
    return key[0] + key[1:].upper()


def anatomical_tibia_names(laterality: Optional[str]) -> Optional[Dict[str, str]]:
    """
    Map patient-side tibial region keys to the paper's medial/lateral names.

    In a **right** knee the patient-left plate is the medial one (it faces the body's
    midline), in a **left** knee the patient-right plate is. So for ``laterality='right'``
    ``'clt' -> 'cMT'`` and ``'crt' -> 'cLT'``; for ``'left'`` ``'clt' -> 'cLT'`` and
    ``'crt' -> 'cMT'``. Internal/external/anterior/posterior are the same wedges under both
    names (they are defined relative to the other plate and the anterior axis, not by side).

    Note the clash: in a right knee the *anatomical* ``cLT`` (central lateral) is the
    *patient-side* ``cRT``, while the patient-side display label ``cLT`` means central
    patient-*left*. Keep the two naming schemes apart: the thickness dicts, the plotting
    code and :func:`morphometry.cartilage.knee.split_tibia_thickness_by_compartment` all use
    patient-side keys, and the latter refuses dicts holding ``*MT`` keys.

    :param laterality: ``'left'`` or ``'right'`` (the knee, i.e. the patient's leg), or
        ``None`` if unknown.
    :return: ``{patient_side_key: anatomical_name}`` for all ten regions, or ``None`` when
        ``laterality`` is ``None``.
    :raises ValueError: For any other ``laterality`` value.
    """
    if laterality is None:
        return None
    if laterality not in ('left', 'right'):
        raise ValueError(f"laterality must be 'left', 'right' or None, got {laterality!r}")
    medial_side = 'left' if laterality == 'right' else 'right'
    names = {}
    for side in PLATE_SIDES:
        plate = 'MT' if side == medial_side else 'LT'
        for letter in REGION_LETTERS:
            names[f'{letter}{side[0]}t'] = letter + plate
    return names


def rename_tibia_regions(regions: dict, laterality: str) -> dict:
    """
    Re-key a ``{patient_side_key: value}`` dict with the paper's medial/lateral names.

    Accepts attribute keys (``'clt'``) or display labels (``'cLT'``), case-insensitively.
    The result is for reporting only; do not feed it back into code that expects
    patient-side keys.

    :param regions: E.g. the output of :meth:`morphometry.cartilage.knee.Tibia.calculate_thickness`.
    :param laterality: ``'left'`` or ``'right'``.
    :return: A new dict with the same values under ``cMT`` ... ``pLT`` keys.
    :raises ValueError: If ``laterality`` is not ``'left'``/``'right'`` or a key is unknown.
    """
    names = anatomical_tibia_names(laterality)
    if names is None:
        raise ValueError("rename_tibia_regions needs laterality 'left' or 'right'")
    out = {}
    for key, value in regions.items():
        if key.lower() not in names:
            raise ValueError(f'unknown tibial region key {key!r}')
        out[names[key.lower()]] = value
    return out


@dataclass
class PlateFrame:
    """
    Geometry of one tibial plate's subdivision (all vectors unit length, world mm, RAS).

    :ivar side: Patient side of the plate, ``'left'`` or ``'right'``.
    :ivar centroid: Centre of gravity S of the plate's tAB (area-weighted), ``(3,)``.
    :ivar axes: ``(3, 3)`` rows ``b1, b2, b3``: principal axes of the tAB's area
        distribution in decreasing order of spread; ``b3`` is the plate normal, oriented
        superior, and the frame is right-handed (``b2 = b3 x b1``).
    :ivar variances: ``(3,)`` area-weighted variances (mm^2) along ``b1, b2, b3``.
    :ivar ml_direction: Unit vector in the plate plane towards the other plate's centre
        (``tc`` projected into the plate plane): the internal direction.
    :ivar anterior_direction: Unit vector in the plate plane perpendicular to
        ``ml_direction``, signed towards world anterior.
    :ivar ellipse_level: The normalised radius rho* bounding the central region.
    :ivar semi_axes: ``(2,)`` ellipse semi-axes (mm) along ``b1, b2``:
        ``rho* * sqrt(variances[:2])``.
    :ivar tab_area: The plate's tAB area (mm^2).
    :ivar cab_area: The plate's covered (cAB) area (mm^2).
    :ivar region_areas: ``{key: mm^2}`` for the plate's five regions (tAB area).
    """
    side: str
    centroid: np.ndarray
    axes: np.ndarray
    variances: np.ndarray
    ml_direction: np.ndarray
    anterior_direction: np.ndarray
    ellipse_level: float
    semi_axes: np.ndarray
    tab_area: float
    cab_area: float
    region_areas: Dict[str, float] = field(default_factory=dict)

    @property
    def region_shares(self) -> Dict[str, float]:
        """
        Region areas as fractions of the plate's tAB.

        :return: ``{key: share}``, summing to 1.
        """
        return {k: v / self.tab_area for k, v in self.region_areas.items()}


@dataclass
class TibiaSubdivision:
    """
    Result of :func:`subdivide_tibia`.

    :ivar mesh: The labelled tAB of both plates: an all-triangle ``pv.PolyData`` in world
        mm with cell data ``'region'`` (int code, see :data:`TIBIA_REGION_KEYS`), ``'plate'``
        (0 = patient left, 1 = patient right) and ``'covered'`` (True for cAB triangles,
        False for filled denuded holes).
    :ivar plates: ``{'left': PlateFrame, 'right': PlateFrame}`` (patient side).
    :ivar laterality: ``'left'``, ``'right'`` or ``None`` as passed in.
    :ivar plate_split: How the plates were found: ``'components'`` (the two largest
        covered components) or ``'two_means'`` (fallback when the two plates' cartilage is
        one connected patch, see :func:`subdivide_tibia`).
    :ivar dropped_area: Covered bone area (mm^2) that ended up in neither plate's tAB
        (small covered islands outside both plates).
    :ivar bone_mesh: The full tibial bone surface the subdivision was built on.
    """
    mesh: pv.PolyData
    plates: Dict[str, PlateFrame]
    laterality: Optional[str] = None
    plate_split: str = 'components'
    dropped_area: float = 0.0
    bone_mesh: Optional[pv.PolyData] = None
    _tree: Optional[cKDTree] = field(default=None, repr=False)

    @property
    def region_areas(self) -> Dict[str, float]:
        """
        tAB area of every region.

        :return: ``{key: mm^2}`` for all ten patient-side keys.
        """
        out = {}
        for plate in self.plates.values():
            out.update(plate.region_areas)
        return out

    @property
    def region_shares(self) -> Dict[str, float]:
        """
        Every region's share of its own plate's tAB (the quantity of the paper's Table I).

        :return: ``{key: fraction}``; the five shares of each plate sum to 1.
        """
        out = {}
        for plate in self.plates.values():
            out.update(plate.region_shares)
        return out

    @property
    def anatomical_names(self) -> Optional[Dict[str, str]]:
        """
        Medial/lateral names of the regions (see :func:`anatomical_tibia_names`).

        :return: ``{patient_side_key: anatomical_name}``, or ``None`` without a laterality.
        """
        return anatomical_tibia_names(self.laterality)

    def nearest_regions(self, points_world: np.ndarray):
        """
        Region code of, and distance to, the tAB triangle nearest to each point.

        "Nearest triangle" is approximated by the nearest triangle *centroid*; at the
        ~0.1 mm^2 triangles of a knee bone mesh the two differ only within a fraction of a
        voxel of a region boundary.

        :param points_world: ``(N, 3)`` points in world mm (RAS).
        :return: ``(codes, distances)``: ``(N,)`` int region codes (index into
            :data:`TIBIA_REGION_KEYS`) and ``(N,)`` distances in mm to that centroid.
        """
        if self._tree is None:
            self._tree = cKDTree(S.triangle_centroids(self.mesh))
        points_world = np.asarray(points_world, dtype=float).reshape(-1, 3)
        if len(points_world) == 0:
            return np.zeros(0, dtype=int), np.zeros(0)
        distances, nearest = self._tree.query(points_world, k=1)
        return np.asarray(self.mesh.cell_data['region'])[nearest], distances

    def region_codes_at(self, points_world: np.ndarray) -> np.ndarray:
        """
        Region code of the tAB triangle nearest to each point (see :meth:`nearest_regions`).

        :param points_world: ``(N, 3)`` points in world mm (RAS).
        :return: ``(N,)`` int array of region codes (index into :data:`TIBIA_REGION_KEYS`).
        """
        return self.nearest_regions(points_world)[0]


# ----------------------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------------------

def area_moments(mesh: pv.PolyData):
    """
    Exact area-weighted centroid and covariance of a triangulated surface.

    Uses the closed-form second moment of a uniform triangle with vertices ``a, b, c`` and
    area ``A``: ``integral of x x^T dA = A / 12 * (a a^T + b b^T + c c^T + s s^T)`` with
    ``s = a + b + c``. Summed over triangles and divided by the total area this gives the
    second moment of the surface itself, independent of how it is triangulated (a vertex
    average would weight regions by their vertex density).

    :param mesh: All-triangle ``pv.PolyData``.
    :return: ``(centroid, covariance, area)``: ``(3,)``, ``(3, 3)`` and the total area.
    """
    faces = S._faces_array(mesh)
    p = np.asarray(mesh.points, dtype=float)
    a, b, c = p[faces[:, 0]], p[faces[:, 1]], p[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    total = areas.sum()
    s = a + b + c
    centroid = (areas[:, None] * s).sum(axis=0) / (3.0 * total)
    # Second moments about the centroid (subtracting it first keeps the sums well-conditioned).
    a, b, c, s = a - centroid, b - centroid, c - centroid, s - 3.0 * centroid
    second = np.einsum('n,ni,nj->ij', areas, a, a) + np.einsum('n,ni,nj->ij', areas, b, b) \
        + np.einsum('n,ni,nj->ij', areas, c, c) + np.einsum('n,ni,nj->ij', areas, s, s)
    covariance = second / (12.0 * total)
    return centroid, covariance, total


def principal_frame(covariance: np.ndarray, reference: Optional[np.ndarray] = None):
    """
    Principal axes of a plate's area distribution, oriented reproducibly.

    The eigenvectors of the covariance in decreasing order of variance are ``b1, b2, b3``;
    for a plate-like surface ``b3`` (least spread) is its normal. ``b3`` is oriented
    superior (positive world z), ``b1`` towards ``reference`` (if given; otherwise towards
    positive world x), and ``b2 = b3 x b1`` makes the frame right-handed. The signs do not
    affect any region (the ellipse and the wedge planes are symmetric in them); they only
    make the reported frame deterministic.

    The paper speaks of the eigenvectors of the inertia tensor. For any mass distribution the
    inertia tensor ``tr(C) I - C`` has the same eigenvectors as the covariance ``C``, with the
    order reversed (the axis of least spread has the largest moment of inertia), so the
    frames are identical; only the ranking convention differs.

    :param covariance: ``(3, 3)`` symmetric covariance.
    :param reference: Optional ``(3,)`` direction ``b1`` should point towards.
    :return: ``(axes, variances)``: ``(3, 3)`` rows ``b1, b2, b3`` and the ``(3,)``
        variances along them.
    """
    values, vectors = np.linalg.eigh(covariance)        # ascending
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    b1, b3 = vectors[:, 0], vectors[:, 2]
    if b3 @ _SUPERIOR < 0:
        b3 = -b3
    ref = np.array([1.0, 0.0, 0.0]) if reference is None else np.asarray(reference, dtype=float)
    if b1 @ ref < 0:
        b1 = -b1
    b2 = np.cross(b3, b1)
    return np.stack([b1, b2, b3]), np.clip(values, 0.0, None)


def _two_means(points: np.ndarray, weights: np.ndarray, max_iterations: int = 100) -> np.ndarray:
    """
    Weighted 2-means clustering, deterministically initialised along the main spread.

    :param points: ``(N, 3)`` points.
    :param weights: ``(N,)`` non-negative weights (triangle areas).
    :param max_iterations: Lloyd iteration cap.
    :return: ``(N,)`` bool array, True for the second cluster.
    """
    mean = (weights[:, None] * points).sum(axis=0) / weights.sum()
    centred = points - mean
    cov = (weights[:, None, None] * np.einsum('ni,nj->nij', centred, centred)).sum(axis=0)
    axis = np.linalg.eigh(cov)[1][:, -1]
    assign = centred @ axis > 0
    for _ in range(max_iterations):
        if assign.all() or not assign.any():
            break
        c0 = (weights[~assign, None] * points[~assign]).sum(0) / weights[~assign].sum()
        c1 = (weights[assign, None] * points[assign]).sum(0) / weights[assign].sum()
        new = np.linalg.norm(points - c1, axis=1) < np.linalg.norm(points - c0, axis=1)
        if np.array_equal(new, assign):
            break
        assign = new
    return assign


def find_plates(bone_mesh: pv.PolyData, covered: np.ndarray, min_plate_ratio: float = 0.2):
    """
    Split the covered tibial bone area (cAB) into the two plates' cAB.

    The plates are the two largest edge-connected components of the covered triangles, if
    the smaller of the two has at least ``min_plate_ratio`` times the area of the larger.
    The remaining covered components ("islands") belong to neither plate; an island inside
    a plate's denuded hole still ends up in that plate's tAB through the per-plate hole
    filling, the others are dropped.

    Fallback: if the two plates' cartilage forms one connected patch (no second component
    that large), the largest component is split in two by area-weighted 2-means on its
    triangle centroids (initialised along its direction of largest spread, i.e. medial-
    lateral for two side-by-side plates), and each half keeps its largest edge-connected
    component. Any cartilage bridge between the plates is then cut at the 2-means boundary
    and half of it is counted with each plate.

    Plates are then named by patient side: the one whose cAB centroid has the smaller world
    x is the patient-left plate (RAS: +x points to the patient's right).

    :param bone_mesh: The tibial bone surface (world mm, RAS).
    :param covered: ``(n_cells,)`` bool cAB flags.
    :param min_plate_ratio: Minimum area ratio of the second-largest to the largest covered
        component for the two to be taken as the plates.
    :return: ``(plates, method)``: ``plates`` maps ``'left'`` / ``'right'`` to ``(n_cells,)``
        bool masks of each plate's cAB; ``method`` is ``'components'`` or ``'two_means'``.
    :raises ValueError: If there is no covered area or it cannot be split into two plates.
    """
    labels, _, areas = S.face_components(bone_mesh, covered)
    if len(areas) == 0:
        raise ValueError('no cartilage-covered tibial bone surface (check the bone and cartilage labels)')
    order = np.argsort(areas)[::-1]
    if len(areas) >= 2 and areas[order[1]] >= min_plate_ratio * areas[order[0]]:
        masks = [labels == order[0], labels == order[1]]
        method = 'components'
    else:
        method = 'two_means'
        faces = np.flatnonzero(labels == order[0])
        tri_areas = S.triangle_areas(bone_mesh)
        second = _two_means(S.triangle_centroids(bone_mesh)[faces], tri_areas[faces])
        masks = []
        for half in (faces[~second], faces[second]):
            mask = np.zeros(bone_mesh.n_cells, dtype=bool)
            mask[half] = True
            sub_labels, _, sub_areas = S.face_components(bone_mesh, mask)
            if len(sub_areas) == 0:
                raise ValueError('could not split the tibial cartilage into two plates')
            masks.append(sub_labels == np.argmax(sub_areas))
        small, large = sorted(tri_areas[m].sum() for m in masks)
        if small < min_plate_ratio * large:
            raise ValueError('could not split the tibial cartilage into two plates of '
                             f'comparable size ({small:.0f} vs {large:.0f} mm^2)')

    centroids = S.triangle_centroids(bone_mesh)
    tri_areas = S.triangle_areas(bone_mesh)
    mean_x = [np.average(centroids[m, 0], weights=tri_areas[m]) for m in masks]
    left, right = (masks[0], masks[1]) if mean_x[0] < mean_x[1] else (masks[1], masks[0])
    return {'left': left, 'right': right}, method


def central_level(mesh: pv.PolyData, rho: np.ndarray, fraction: float,
                  tolerance: float = 1e-5, max_iterations: int = 200) -> float:
    """
    Find the level rho* whose sub-level set ``rho <= rho*`` holds ``fraction`` of the area.

    ``rho`` is given per vertex and interpolated linearly along edges by the exact splitter,
    so the area below a level is a continuous, non-decreasing function of the level but not
    simply an area-weighted quantile of ``rho``. It is solved by bisection on the actually
    cut mesh, to ``|area(rho*) / total - fraction| <= tolerance`` (the paper stops at 0.1%
    of tAB; the default here is 0.001%).

    :param mesh: All-triangle plate tAB mesh.
    :param rho: ``(n_points,)`` normalised elliptical radius per vertex.
    :param fraction: Target share of the area, in ``(0, 1)``.
    :param tolerance: Allowed absolute deviation of the share.
    :param max_iterations: Bisection cap.
    :return: The level rho*.
    :raises ValueError: If ``fraction`` is not in ``(0, 1)``.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f'central_fraction must be in (0, 1), got {fraction}')
    total = S.triangle_areas(mesh).sum()
    lo, hi = 0.0, float(rho.max())
    level = hi
    for _ in range(max_iterations):
        level = 0.5 * (lo + hi)
        below, _ = S.split_mesh_by_scalar(mesh, rho, level)
        share = S.triangle_areas(below).sum() / total if below.n_cells else 0.0
        if abs(share - fraction) <= tolerance:
            break
        if share < fraction:
            lo = level
        else:
            hi = level
    return level


def _subdivide_plate(plate_mesh: pv.PolyData, side: str, other_centroid: np.ndarray,
                     central_fraction: float, tolerance: float):
    """
    Subdivide one plate's tAB into its five regions.

    :param plate_mesh: The plate's tAB (all-triangle, world mm), with a ``'covered'`` cell
        array.
    :param side: ``'left'`` or ``'right'`` (patient side), sets the region codes.
    :param other_centroid: Centre of gravity of the other plate's tAB.
    :param central_fraction: Share of the plate's tAB in the central region.
    :param tolerance: Bisection tolerance on the central share (see :func:`central_level`).
    :return: ``(labelled_mesh, frame)``: the plate's tAB cut into regions with
        ``'region'`` / ``'plate'`` cell data, and its :class:`PlateFrame`.
    :raises ValueError: If the plates are not side by side in the anterior-posterior sense
        (the in-plane perpendicular to ``tc`` is more than 60 degrees off world A-P).
    """
    centroid, covariance, tab_area = area_moments(plate_mesh)
    tc = np.asarray(other_centroid, dtype=float) - centroid
    axes, variances = principal_frame(covariance, reference=tc)
    b1, b2, b3 = axes

    # Internal direction: tc projected into the plate plane. Anterior: its in-plane
    # perpendicular, signed towards world anterior (the paper used a fibula landmark).
    ml = tc - (tc @ b3) * b3
    ml /= np.linalg.norm(ml)
    ap = np.cross(b3, ml)
    if abs(ap @ _ANTERIOR) < 0.5:
        raise ValueError(f'patient-{side} plate: the line joining the plates is more than 60 deg '
                         'off the medial-lateral direction; cannot tell anterior from posterior')
    if ap @ _ANTERIOR < 0:
        ap = -ap

    # Central region: normalised elliptical radius in the plate plane, semi-axes in the ratio
    # of the in-plane standard deviations (a uniform elliptical plate gives a scaled copy of
    # its own outline), measured along b3 as a cylinder.
    rel = np.asarray(plate_mesh.points, dtype=float) - centroid
    sigma = np.sqrt(variances[:2])
    rho = np.sqrt(((rel @ b1) / sigma[0]) ** 2 + ((rel @ b2) / sigma[1]) ** 2)
    level = central_level(plate_mesh, rho, central_fraction, tolerance)
    central, peripheral = S.split_mesh_by_scalar(plate_mesh, rho, level)

    # Peripheral wedges: planes through S containing b3 at +-45 deg to the internal direction.
    # With alpha = (p - S).ml and beta = (p - S).ap: d1 = alpha - beta, d2 = alpha + beta;
    # internal = both > 0, external = both <= 0, anterior = d2 > 0 >= d1, posterior = d1 > 0 >= d2.
    d1_neg, d1_pos = S.split_mesh_by_plane(peripheral, centroid, ml - ap)
    external, anterior = S.split_mesh_by_plane(d1_neg, centroid, ml + ap)
    posterior, internal = S.split_mesh_by_plane(d1_pos, centroid, ml + ap)

    plate_index = PLATE_SIDES.index(side)
    parts = []
    region_areas = {}
    for letter, part in zip(REGION_LETTERS, (central, internal, external, anterior, posterior)):
        code = plate_index * len(REGION_LETTERS) + REGION_LETTERS.index(letter)
        region_areas[TIBIA_REGION_KEYS[code]] = float(S.triangle_areas(part).sum()) if part.n_cells else 0.0
        if part.n_cells:
            part.cell_data['region'] = np.full(part.n_cells, code, dtype=int)
            part.cell_data['plate'] = np.full(part.n_cells, plate_index, dtype=int)
            parts.append(part)

    covered = np.asarray(plate_mesh.cell_data['covered'], dtype=bool)
    frame = PlateFrame(
        side=side, centroid=centroid, axes=axes, variances=variances, ml_direction=ml,
        anterior_direction=ap, ellipse_level=float(level), semi_axes=level * sigma,
        tab_area=float(tab_area),
        cab_area=float(S.triangle_areas(plate_mesh)[covered].sum()),
        region_areas=region_areas)
    return S.merge_meshes(parts), frame


# ----------------------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------------------

def subdivide_tibia(image: Image, bone_label: int = 2, cartilage_label: int = 4,
                    central_fraction: float = 0.2, laterality: Optional[str] = None,
                    coverage_tolerance_mm: Optional[float] = None,
                    max_hole_fraction: float = 0.5, min_plate_ratio: float = 0.2,
                    tolerance: float = 1e-5) -> TibiaSubdivision:
    """
    Subdivide the tibial subchondral bone area into the Wirth & Eckstein subregions.

    Steps (all in world mm):

    1. Tibial bone surface (:func:`~morphometry.cartilage.surface.bone_surface_mesh`) and its
       cartilage-covered part cAB (:func:`~morphometry.cartilage.surface.cartilage_coverage`).
    2. The two plates' cAB (:func:`find_plates`).
    3. tAB **per plate**: :func:`~morphometry.cartilage.surface.total_subchondral_area` with
       only that plate's cAB counted as covered. The other plate is then uncovered bone, so
       the intercondylar area is always connected (through it) to the open non-articular
       remainder of the bone and can never be filled, however the cartilage touches; a
       denuded hole inside one plate is still filled.
    4. Per plate: centre of gravity S and principal axes b1, b2, b3 of its tAB from the
       exact area moments (:func:`area_moments`, :func:`principal_frame`), b3 = normal.
    5. Central region: normalised elliptical radius
       ``rho = sqrt((x / sigma1)^2 + (y / sigma2)^2)`` with ``x, y`` the in-plane
       coordinates along b1, b2 and ``sigma1, sigma2`` the in-plane standard deviations of
       the tAB area. The cylinder ``rho <= rho*`` is cut out with rho* found by bisection so
       that it holds ``central_fraction`` of the plate's tAB area (:func:`central_level`).
    6. Peripheral regions: the rest cut by two planes through S containing b3, at +-45
       degrees to the projection of ``tc`` (S of this plate -> S of the other) into the
       plate plane. The wedge around +tc is internal, around -tc external, the other two
       anterior / posterior by the world anterior axis.

    **Ellipse aspect (interpretation).** The paper writes the cylinder as
    ``x^2 / lambda1^2 + y^2 / lambda2^2 = 1`` "with the parameters of the quadric surface
    reflecting the different diameters of the medial and lateral tibia in anterior-posterior
    and in medial-lateral direction", and then scales the parameters iteratively to 20%, so
    only their *ratio* matters and it is not stated how it is derived from the inertia
    tensor. Here the semi-axes are in the ratio of the in-plane standard deviations
    ``sqrt(var1) : sqrt(var2)``. For a uniform elliptical plate that is exactly the ratio of
    its diameters, so the central region is a scaled copy of the plate outline centred on
    S. (Taking the variances themselves would square the aspect; taking moments of
    inertia about b1/b2 would swap it.)

    :param image: The (LPI-oriented) knee ``Segmentation``; either image backend.
    :param bone_label: Label of the tibial bone (``dfg_knees``: 2).
    :param cartilage_label: Label of the tibial cartilage (``dfg_knees``: 4).
    :param central_fraction: Share of each plate's tAB in its central region (paper: 0.2;
        its Table IV varies it from 0.1 to 0.5).
    :param laterality: ``'left'`` / ``'right'`` knee, or ``None``. Only adds the
        medial/lateral naming (:func:`anatomical_tibia_names`); the subdivision itself
        does not depend on it.
    :param coverage_tolerance_mm: cAB distance tolerance, see
        :func:`~morphometry.cartilage.surface.cartilage_coverage` (``None`` = its default).
    :param max_hole_fraction: Largest denuded hole filled, as a fraction of the *plate's*
        cAB, see :func:`~morphometry.cartilage.surface.total_subchondral_area`.
    :param min_plate_ratio: See :func:`find_plates`.
    :param tolerance: Allowed absolute deviation of the central share from
        ``central_fraction``.
    :return: A :class:`TibiaSubdivision`.
    :raises ValueError: If the bone or cartilage label is missing, the plates cannot be
        found, or anterior cannot be determined.
    """
    anatomical_tibia_names(laterality)  # validates the value
    bone = S.bone_surface_mesh(image, bone_label)
    if bone.n_cells == 0:
        raise ValueError(f'tibial bone label {bone_label} is absent; the tibial subregions '
                         'are defined on the subchondral bone surface')
    covered = S.cartilage_coverage(bone, image, cartilage_label, coverage_tolerance_mm)
    plate_cab, method = find_plates(bone, covered, min_plate_ratio)

    tab = {side: S.total_subchondral_area(bone, plate_cab[side], max_hole_fraction)
           for side in PLATE_SIDES}
    if (tab['left'] & tab['right']).any():
        raise ValueError('the two tibial plates overlap after hole filling')
    plate_meshes = {}
    for side in PLATE_SIDES:
        sub = S.extract_faces(bone, tab[side])
        sub.cell_data['covered'] = plate_cab[side][tab[side]]
        plate_meshes[side] = sub
    # Centres of gravity first: each plate's internal direction needs the other's.
    centroids = {side: area_moments(m)[0] for side, m in plate_meshes.items()}

    parts, plates = [], {}
    for side, other in (('left', 'right'), ('right', 'left')):
        labelled, frame = _subdivide_plate(plate_meshes[side], side, centroids[other],
                                           central_fraction, tolerance)
        parts.append(labelled)
        plates[side] = frame

    areas = S.triangle_areas(bone)
    dropped = float(areas[covered & ~tab['left'] & ~tab['right']].sum())
    return TibiaSubdivision(mesh=S.merge_meshes(parts), plates=plates, laterality=laterality,
                            plate_split=method, dropped_area=dropped, bone_mesh=bone)


def transfer_qc(distances: np.ndarray, far_mm: float = TRANSFER_FAR_MM) -> Dict[str, float]:
    """
    Summarise how far cartilage voxels are from the tAB triangle they took their region from.

    :param distances: ``(N,)`` voxel-to-nearest-tAB-centroid distances in mm (from
        :meth:`TibiaSubdivision.nearest_regions`).
    :param far_mm: Distance above which a voxel counts as unexplained by the tAB (see
        :data:`TRANSFER_FAR_MM`).
    :return: ``{'n_voxels', 'median_mm', 'p99_mm', 'max_mm', 'far_fraction', 'far_mm'}``;
        the statistics are ``NaN`` (and ``far_fraction`` 0) without voxels.
    """
    distances = np.asarray(distances, dtype=float)
    if len(distances) == 0:
        return {'n_voxels': 0, 'median_mm': np.nan, 'p99_mm': np.nan, 'max_mm': np.nan,
                'far_fraction': 0.0, 'far_mm': far_mm}
    return {'n_voxels': int(len(distances)), 'median_mm': float(np.median(distances)),
            'p99_mm': float(np.percentile(distances, 99)), 'max_mm': float(distances.max()),
            'far_fraction': float(np.mean(distances > far_mm)), 'far_mm': far_mm}
