"""
Femoral cartilage subregions after Wirth & Eckstein 2008 (IEEE TMI 27(6), section II-B).

The weight-bearing part of each femoral condyle (cMF / cLF) is divided on its *total area of
subchondral bone* (tAB, see :mod:`morphometry.cartilage.surface`) into three subregions of
equal **arc length** across the condyle -- **internal** (``i``, the third facing the other
condyle), **central** (``c``) and **external** (``e``) -- which hold about a third of the
condyle's tAB area each (the paper's Table II: 33.2-33.5%).

Unlike the tibia, the paper does not derive the weight-bearing condyle's anterior-posterior
extent from the cartilage itself: it is "already limited based on anatomical landmarks"
(the paper's refs [10]-[12]). This module uses the Chondrometrics / OAI convention:

- the **anterior border** is the plane through the most posterior point of the trochlear
  notch -- the anterior apex of the intercondylar notch, where the trochlea closes over it --
  parallel to the femoral shaft (i.e. with the femoral anterior-posterior axis as its normal);
- the **posterior border** of each condyle is at ``weightbearing_fraction`` (0.6; some later
  studies use 0.75) of the distance from that plane to the condyle's own posterior end,
  measured along the femoral anterior-posterior axis.

Everything is expressed in a **femoral frame** built from the femur alone -- the distal shaft
axis and the posterior condylar line -- so it does not move with knee flexion, with the knee's
position in the scanner, or with the tibia. This is the main difference from the subdivision
this module replaced, which took the condyles' anterior-posterior window from the tibial
central region and therefore moved with every one of those.

Two zones outside the paper are kept as a documented **in-house extension**, because the
existing thickness tables report them: the **trochlea** (``alf`` / ``arf``, everything
anterior of the notch plane) and the **posterior condyle** (``plf`` / ``prf``, everything
posterior of the 60% plane), each split into the two condyles by the same notch plane that
separates them.

**Naming.** The condyles are named by **patient side**, as everywhere else in the cartilage
code: ``iclf, cclf, eclf`` (and ``alf``, ``plf``) belong to the patient-*left* condyle,
``icrf, ccrf, ecrf`` (and ``arf``, ``prf``) to the patient-*right* one. This ``lf`` means
*left femur*, not the paper's *lateral femur*. When the knee's laterality is known,
:func:`anatomical_femur_names` gives the paper's medial/lateral names (``icMF`` ... ``ecLF``)
as a separate mapping; the patient-side keys never change meaning.

Deviations from the paper (all deliberate, see the function docstrings):

- The paper walks the ordered triangle strips produced by its own surface reconstruction and
  divides each strip's anterior and posterior edge sequence into three parts of equal length.
  Here the same quantity is computed as a continuous, normalised arc-length coordinate per
  tAB vertex (:func:`arc_length_coordinate`), and the mesh is cut exactly at ``u = 1/3`` and
  ``u = 2/3`` -- the triangulation then does not have to be strip-ordered.
- The paper orients internal/external (and medial/lateral) with the contralateral condyle and
  a manually placed fibula landmark. Here internal is the side facing the other condyle in the
  femoral frame, and medial/lateral come from the caller-supplied laterality.
- The posterior border is computed per condyle, as the paper's wording implies, but the two
  come out the same: the medial-lateral axis *is* the line through the two condyles' most
  posterior points, so those points share an anterior-posterior coordinate by construction (on
  the 96 ``dfg_knees`` the two borders differ by at most 0.10 mm). Read step 4 of
  :func:`subdivide_femur` with that in mind: the condyles are cut independently, but their
  windows coincide. A frame whose medial-lateral axis came from somewhere else -- the first
  principal direction of the distal femur, say -- would separate them by a few millimetres of
  the ~30 mm window.
- The femoral shaft axis has to be estimated from the ~25-40 mm of femur that a knee field of
  view holds above the cartilage (:func:`estimate_shaft_axis`).
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyvista as pv
from scipy.ndimage import label as nd_label
from scipy.spatial import cKDTree

from morphometry.image_io import Image
from morphometry.cartilage import surface as S

# Non-fatal diagnostics (region transfer QC, clamped notch line). Like tibia_regions, this
# logs rather than warnings.warn so every affected scan of a cohort run is reported; attach a
# handler to logging.getLogger('morphometry') to collect them.
logger = logging.getLogger(__name__)


# The three thirds of a weight-bearing condyle, in code order, and the two condyles by
# patient side. Region key = letter + 'c' + side initial + 'f', e.g. 'iclf'.
REGION_LETTERS = ('i', 'c', 'e')
CONDYLE_SIDES = ('left', 'right')
CONDYLE_KEYS = tuple(f'{letter}c{side[0]}f' for side in CONDYLE_SIDES for letter in REGION_LETTERS)
# The in-house extension zones, anterior (trochlea) and posterior condyle, per patient side.
EXTENSION_KEYS = ('alf', 'plf', 'arf', 'prf')
# Region key per integer code; the condylar thirds first, then the extension zones.
FEMUR_REGION_KEYS = CONDYLE_KEYS + EXTENSION_KEYS

# The extension zones keep the key the femoral thickness dicts have always used for them
# (the condylar thirds use their own key there).
ZONE_THICKNESS_KEYS = {
    'alf': 'left_anterior_zone', 'plf': 'left_posterior_zone',
    'arf': 'right_anterior_zone', 'prf': 'right_posterior_zone',
}

# The surface meshes are in RAS (+x patient right, +y anterior, +z superior) for both image
# backends, see surface.index_to_world_affine. The only world direction the femoral frame
# needs is patient right, to tell the two condyles apart; proximal comes from the bone and
# anterior from the right-handed completion of the two.
_PATIENT_RIGHT = np.array([1.0, 0.0, 0.0])

# Share of the weight-bearing A-P window that may be discarded as the posterior condyle's
# proximal wrap without a warning. The wrap is a thin sliver of the window in a healthy
# subdivision (at most 14.9% over the 96 dfg_knees); much more than that means the window
# holds two comparable sheets and the largest-patch rule is guessing (see subdivide_femur).
MAX_WRAP_FRACTION = 0.25

# Stand-in depth (mm) for a coronal slice in which the intercondylar fossa is open, i.e. one
# with no bone at all between the condyles: deeper than any real valley, and finite so the
# deepest run can be located the same way as in a closed slice (notch_depth_profile).
_OPEN_VALLEY_MM = 1e6


def anatomical_femur_names(laterality: Optional[str]) -> Optional[Dict[str, str]]:
    """
    Map patient-side femoral region keys to the paper's medial/lateral names.

    In a **right** knee the patient-left condyle is the medial one (it faces the body's
    midline), in a **left** knee the patient-right condyle is. So for ``laterality='right'``
    ``'iclf' -> 'icMF'`` and ``'icrf' -> 'icLF'``; for ``'left'`` the other way round.
    Internal/central/external mean the same thing under both names (they are defined relative
    to the other condyle, not by side).

    The two extension zones have no name in the paper, which subdivides only the
    weight-bearing condyle. They are mapped to ``aMF`` / ``pMF`` / ``aLF`` / ``pLF``
    (anterior = trochlea, posterior = posterior condyle), following the same letter
    convention as the paper's tibial wedges -- **our names, not the paper's**.

    Note the clash: in a right knee the *anatomical* ``icLF`` (internal lateral femur) is the
    *patient-side* ``icrf``, while the patient-side key ``iclf`` means internal patient-left.
    Keep the two naming schemes apart: the thickness dicts, the plotting code and the voxel
    attributes of :class:`morphometry.cartilage.knee.Femur` all use patient-side keys.

    :param laterality: ``'left'`` or ``'right'`` (the knee, i.e. the patient's leg), or
        ``None`` if unknown.
    :return: ``{patient_side_key: anatomical_name}`` for all ten regions, additionally keyed
        by the four extension zones' thickness-dict names (``'left_anterior_zone'`` ...), or
        ``None`` when ``laterality`` is ``None``.
    :raises ValueError: For any other ``laterality`` value.
    """
    if laterality is None:
        return None
    if laterality not in ('left', 'right'):
        raise ValueError(f"laterality must be 'left', 'right' or None, got {laterality!r}")
    medial_side = 'left' if laterality == 'right' else 'right'
    names = {}
    for side in CONDYLE_SIDES:
        condyle = 'MF' if side == medial_side else 'LF'
        for letter in REGION_LETTERS:
            names[f'{letter}c{side[0]}f'] = letter + 'c' + condyle
        for letter in ('a', 'p'):
            key = f'{letter}{side[0]}f'
            names[key] = letter + condyle
            names[ZONE_THICKNESS_KEYS[key]] = letter + condyle
    return names


def rename_femur_regions(regions: dict, laterality: str) -> dict:
    """
    Re-key a ``{patient_side_key: value}`` dict with the paper's medial/lateral names.

    Accepts the voxel attribute keys (``'iclf'``, ``'alf'``) and the thickness-dict keys of
    the extension zones (``'left_anterior_zone'``), case-insensitively. The result is for
    reporting only; do not feed it back into code that expects patient-side keys.

    :param regions: E.g. the output of :meth:`morphometry.cartilage.knee.Femur.calculate_thickness`.
    :param laterality: ``'left'`` or ``'right'``.
    :return: A new dict with the same values under ``icMF`` ... ``pLF`` keys.
    :raises ValueError: If ``laterality`` is not ``'left'``/``'right'`` or a key is unknown.
    """
    names = anatomical_femur_names(laterality)
    if names is None:
        raise ValueError("rename_femur_regions needs laterality 'left' or 'right'")
    out = {}
    for key, value in regions.items():
        if key.lower() not in names:
            raise ValueError(f'unknown femoral region key {key!r}')
        out[names[key.lower()]] = value
    return out


# ----------------------------------------------------------------------------------------
# Result types
# ----------------------------------------------------------------------------------------

@dataclass
class FemoralFrame:
    """
    The femur's own coordinate frame and the landmarks the subdivision is cut with.

    The three axes are orthonormal and right-handed in the order ``(ml, ap, proximal)``:
    ``ml x ap = proximal``, which in RAS is ``+x cross +y = +z``.

    :ivar proximal_axis: Unit femoral shaft axis, pointing proximally
        (:func:`estimate_shaft_axis`).
    :ivar ml_axis: Unit medial-lateral axis along the posterior condylar line, pointing to the
        patient's **right** (:func:`posterior_condylar_axis`).
    :ivar ap_axis: Unit anterior-posterior axis, ``proximal_axis x ml_axis``, pointing
        anteriorly.
    :ivar notch_point: The trochlear-notch landmark in world mm: the floor of the trochlear
        groove in the most posterior coronal slice in which the intercondylar fossa has closed
        (see :func:`find_trochlear_notch`).
    :ivar notch_ap: The anterior border of both weight-bearing condyles along ``ap_axis``
        (the centre of that slice, within half a slice of ``notch_point @ ap_axis``).
    :ivar divider: ``(slope, intercept)`` of the line of intercondylar-fossa centres in the
        femoral frame, ``ml = slope * ap + intercept``. The plane containing the shaft axis
        through that line separates the two condyles (and the trochlea into its two halves).
    :ivar posterior_points: The two most posterior condylar points (patient left, patient
        right) that the posterior condylar line was drawn through, in world mm.
    :ivar shaft_residual_mm: Largest distance of a shaft cross-section centroid from the fitted
        shaft axis -- a straightness check on the axis fit (``NaN`` if not recorded).
    :ivar shaft_bins: Number of cross-sections the shaft axis was fitted to.
    """
    proximal_axis: np.ndarray
    ml_axis: np.ndarray
    ap_axis: np.ndarray
    notch_point: np.ndarray
    notch_ap: float
    divider: Tuple[float, float]
    posterior_points: Dict[str, np.ndarray] = field(default_factory=dict)
    shaft_residual_mm: float = float('nan')
    shaft_bins: int = 0

    def divider_scalar(self, points: np.ndarray) -> np.ndarray:
        """
        Signed distance-like coordinate across the intercondylar notch.

        Negative on the patient-left condyle, positive on the patient-right one. It is linear
        in the point coordinates (the tilted notch plane), so
        :func:`~morphometry.cartilage.surface.split_mesh_by_scalar` cuts along it exactly.

        :param points: ``(N, 3)`` world mm.
        :return: ``(N,)`` values, ``<= 0`` for the patient-left side.
        """
        slope, intercept = self.divider
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        return points @ self.ml_axis - slope * (points @ self.ap_axis) - intercept


@dataclass
class CondyleFrame:
    """
    The anterior-posterior limits and areas of one weight-bearing femoral condyle.

    :ivar side: Patient side of the condyle, ``'left'`` or ``'right'``.
    :ivar posterior_ap: Anterior-posterior coordinate (along the frame's ``ap_axis``) of this
        condyle's most posterior tAB point.
    :ivar boundary_ap: The posterior border of the weight-bearing condyle,
        ``notch_ap - weightbearing_fraction * (notch_ap - posterior_ap)``.
    :ivar internal_direction: Unit in-plane direction towards the other condyle (``+ml`` for
        the patient-left condyle, ``-ml`` for the patient-right one); arc length is measured
        from the internal end.
    :ivar tab_area: tAB area of the **weight-bearing** condyle cF (mm^2) -- the denominator of
        the paper's Table II.
    :ivar cab_area: Cartilage-covered part of that area (mm^2).
    :ivar region_areas: ``{key: mm^2}`` for this condyle's three thirds.
    :ivar anterior_area: tAB area of this side's trochlea zone (mm^2, the extension).
    :ivar posterior_area: tAB area of this side's posterior condyle zone (mm^2, the extension).
    :ivar n_slices: Number of coronal tAB cross-sections the arc-length coordinate was built
        from.
    """
    side: str
    posterior_ap: float
    boundary_ap: float
    internal_direction: np.ndarray
    tab_area: float
    cab_area: float
    region_areas: Dict[str, float] = field(default_factory=dict)
    anterior_area: float = 0.0
    posterior_area: float = 0.0
    n_slices: int = 0

    @property
    def region_shares(self) -> Dict[str, float]:
        """
        The thirds' areas as fractions of this condyle's weight-bearing tAB.

        :return: ``{key: share}``, summing to 1.
        """
        return {k: v / self.tab_area for k, v in self.region_areas.items()}


@dataclass
class FemurSubdivision:
    """
    Result of :func:`subdivide_femur`.

    :ivar mesh: The labelled femoral tAB: an all-triangle ``pv.PolyData`` in world mm with
        cell data ``'region'`` (int code, see :data:`FEMUR_REGION_KEYS`), ``'condyle'``
        (0 = patient left, 1 = patient right) and ``'covered'`` (True for cAB triangles,
        False for filled denuded holes).
    :ivar frame: The :class:`FemoralFrame` the cuts were made in.
    :ivar condyles: ``{'left': CondyleFrame, 'right': CondyleFrame}`` (patient side).
    :ivar laterality: ``'left'``, ``'right'`` or ``None`` as passed in.
    :ivar weightbearing_fraction: The fraction used for the posterior border.
    :ivar tab_area: Area (mm^2) of the whole femoral tAB (all ten regions).
    :ivar bone_mesh: The full femoral bone surface the subdivision was built on.
    """
    mesh: pv.PolyData
    frame: FemoralFrame
    condyles: Dict[str, CondyleFrame]
    laterality: Optional[str] = None
    weightbearing_fraction: float = 0.6
    tab_area: float = 0.0
    bone_mesh: Optional[pv.PolyData] = None
    _tree: Optional[cKDTree] = field(default=None, repr=False)

    @property
    def region_areas(self) -> Dict[str, float]:
        """
        tAB area of every region.

        :return: ``{key: mm^2}`` for all ten patient-side keys.
        """
        out = {}
        for condyle in self.condyles.values():
            out.update(condyle.region_areas)
            out[f'a{condyle.side[0]}f'] = condyle.anterior_area
            out[f'p{condyle.side[0]}f'] = condyle.posterior_area
        return out

    @property
    def region_shares(self) -> Dict[str, float]:
        """
        Each third's share of its own condyle's weight-bearing tAB (the paper's Table II).

        The extension zones are not included: they are not part of cF and have no share of it.

        :return: ``{key: fraction}``; the three shares of each condyle sum to 1.
        """
        out = {}
        for condyle in self.condyles.values():
            out.update(condyle.region_shares)
        return out

    @property
    def anatomical_names(self) -> Optional[Dict[str, str]]:
        """
        Medial/lateral names of the regions (see :func:`anatomical_femur_names`).

        :return: ``{patient_side_key: anatomical_name}``, or ``None`` without a laterality.
        """
        return anatomical_femur_names(self.laterality)

    def nearest_regions(self, points_world: np.ndarray):
        """
        Region code of, and distance to, the tAB triangle nearest to each point.

        "Nearest triangle" is approximated by the nearest triangle *centroid*; at the
        ~0.1 mm^2 triangles of a knee bone mesh the two differ only within a fraction of a
        voxel of a region boundary.

        :param points_world: ``(N, 3)`` points in world mm (RAS).
        :return: ``(codes, distances)``: ``(N,)`` int region codes (index into
            :data:`FEMUR_REGION_KEYS`) and ``(N,)`` distances in mm to that centroid.
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
        :return: ``(N,)`` int array of region codes (index into :data:`FEMUR_REGION_KEYS`).
        """
        return self.nearest_regions(points_world)[0]


# ----------------------------------------------------------------------------------------
# The femoral frame
# ----------------------------------------------------------------------------------------

def label_points_world(image: Image, value: int, largest_component: bool = True) -> np.ndarray:
    """
    World (RAS, mm) coordinates of the voxel centres carrying one label.

    :param image: The (LPI-oriented) ``Segmentation``.
    :param value: The label value to extract.
    :param largest_component: Keep only the largest 26-connected component, so stray
        mislabelled voxels elsewhere in the volume do not move centroids.
    :return: ``(N, 3)`` world coordinates in mm (empty ``(0, 3)`` if the label is absent).
    """
    mask = np.asarray(S.label_array_xyz(image)) == value
    if not mask.any():
        return np.empty((0, 3))
    if largest_component:
        components, n = nd_label(mask, structure=np.ones((3, 3, 3), dtype=bool))
        if n > 1:
            sizes = np.bincount(components.ravel())
            sizes[0] = 0
            mask = components == np.argmax(sizes)
    return S.indices_to_world(S.index_to_world_affine(image), np.argwhere(mask))


def articular_limit(points: np.ndarray, axis: np.ndarray, percentile: float = 99.0) -> float:
    """
    Where the articular surface ends along a direction, robustly.

    The plain extreme is one voxel, and segmentations do carry the odd stray cartilage voxel
    well away from the joint (the femoral cartilage of 9 of the 96 ``dfg_knees`` scans breaks
    into several components, the largest holding as little as 32% of it), which would move
    the shaft-fitting band or the notch search by centimetres. A high percentile moves by at
    most a millimetre or two on a clean label and is unmoved by a handful of strays.

    :param points: ``(N, 3)`` world mm (the cartilage voxels).
    :param axis: The unit direction to measure along.
    :param percentile: Percentile of the projected coordinates to return.
    :return: The projected coordinate at that percentile.
    """
    return float(np.percentile(np.asarray(points, dtype=float).reshape(-1, 3) @ axis, percentile))


def estimate_shaft_axis(bone_points: np.ndarray, distal_points: np.ndarray,
                        band_mm: Tuple[float, float] = (-5.0, 25.0), bin_mm: float = 3.0,
                        min_bin_ratio: float = 0.75, iterations: int = 3,
                        distal_percentile: float = 99.0):
    """
    Estimate the femoral shaft axis from the distal femur a knee field of view holds.

    A knee FOV holds only the distal 25-40 mm of femur above the cartilage -- metaphysis
    rather than true diaphysis -- and its proximal end is usually cut, often very obliquely,
    or fades out where the acquisition ends. A principal-component axis of the whole distal
    femur is useless (the condyles are as wide as the visible shaft is long), so the axis is
    fitted to the **centroids of cross-sections** of the bone in a band just proximal of the
    articular surface:

    1. Seed direction: from the centroid of ``distal_points`` (the femoral cartilage) to the
       centroid of the bone, which points proximally whatever the knee's pose.
    2. The band is ``band_mm`` around where the articular surface ends along the current axis
       (:func:`articular_limit`), clipped to the bone that is actually there. The default band, from 5 mm
       below to 25 mm above the top of the articular surface, is the part of the distal femur
       that is present in *every* scan of this cohort: bands reaching further proximally lose
       their proximal end to the FOV in some acquisitions of a knee but not in others, which
       is exactly the reproducibility the subregions need.

       **The default band was tuned on the 96 dfg_knees scans** by that reproducibility, so the
       figures quoted for it elsewhere are not an unbiased estimate of it, and it is not even
       the argmax of its own objective. Over the 24 flexion series (CV of the weight-bearing
       condyle's area, averaged over the two condyles, median/p90/max; CV of its
       anterior-posterior extent, median/max):
       ``(-5, 25)`` 2.41/4.40/6.66 and 1.47/2.86; ``(-5, 35)`` 2.54/3.45/4.48 and 0.14/2.22;
       ``(0, 30)`` 2.77/3.80/4.16 and 1.63/3.75; ``(5, 35)`` 2.53/3.90/5.25 and 1.61/8.57, and
       one scan it cannot fit at all. ``(-5, 35)`` is at least as defensible; the default only
       wins the median. The band barely moves the thirds' *shares* (at most 1.58 percentage
       points between any two of those four bands on one scan, median 0.24) but does move the
       axis itself (median 1.9, max 7.5 degrees between the first two) and with it the condyle's
       area and extent (median 0.3%, max 4.2%).
    3. The band is sliced into ``bin_mm`` cross-sections perpendicular to the axis and each
       one's centroid taken. Cross-sections cut by the FOV are incomplete and their centroids
       biased, so the proximal-most bins are dropped while they hold less than
       ``min_bin_ratio`` of the voxels of the next bin towards the condyles -- a complete
       metaphyseal cross-section shrinks by only ~10% per 3 mm going proximally, a clipped one
       by much more.
    4. The axis is the first principal direction of the surviving centroids, and steps 2-4 are
       repeated ``iterations`` times (the band is anchored to the cartilage, so it barely moves).

    The estimate is a *metaphyseal* axis: it is as close to the true shaft axis as the field of
    view allows, and -- which matters more for the subregions -- it is reproducible for one
    femur across acquisitions, because it is measured on the part of the bone that is always
    there. Expect a few degrees of difference from an axis fitted to a full femur.

    :param bone_points: ``(N, 3)`` world mm of the femoral bone voxels.
    :param distal_points: ``(M, 3)`` world mm of the femoral cartilage voxels: the distal
        reference, used for the seed direction and to anchor the band.
    :param band_mm: ``(start, end)`` of the fitting band in mm relative to the proximal end of
        the articular surface (negative = distal of it).
    :param bin_mm: Thickness of one cross-section in mm.
    :param min_bin_ratio: Smallest voxel-count ratio to the next bin towards the condyles at
        which a proximal bin still counts as a complete cross-section.
    :param iterations: Number of refit passes.
    :param distal_percentile: Percentile of ``distal_points`` along the axis that the band is
        measured from (see :func:`articular_limit`).
    :return: ``(axis, residual_mm, n_bins)``: the unit axis pointing proximally, the largest
        distance of a used centroid from the fitted line, and how many centroids were used.
    :raises ValueError: If the bone or the distal reference is empty, or the field of view
        holds too little femur above the cartilage to fit a line.
    """
    bone_points = np.asarray(bone_points, dtype=float).reshape(-1, 3)
    distal_points = np.asarray(distal_points, dtype=float).reshape(-1, 3)
    if len(bone_points) == 0 or len(distal_points) == 0:
        raise ValueError('estimate_shaft_axis needs both femoral bone and cartilage points')
    axis = bone_points.mean(axis=0) - distal_points.mean(axis=0)
    axis /= np.linalg.norm(axis)

    residual, used = float('nan'), 0
    for _ in range(max(1, iterations)):
        s = bone_points @ axis
        start = articular_limit(distal_points, axis, distal_percentile) + band_mm[0]
        end = min(start - band_mm[0] + band_mm[1], float(s.max()))
        if end - start < 3 * bin_mm:
            raise ValueError(
                f'only {max(0.0, end - start):.0f} mm of femur above the cartilage fits the '
                'shaft band; the femoral shaft axis cannot be estimated from this field of view')
        edges = np.arange(start, end + 1e-9, bin_mm)
        counts, centroids = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            inside = (s >= lo) & (s < hi)
            if inside.sum() >= 30:
                counts.append(int(inside.sum()))
                centroids.append(bone_points[inside].mean(axis=0))
        if len(centroids) < 3:
            raise ValueError('too few femoral cross-sections hold enough bone to fit a shaft axis')
        counts, centroids = np.asarray(counts, dtype=float), np.asarray(centroids)
        # Drop proximal bins while they are much smaller than the next bin towards the
        # condyles: those cross-sections are cut by the FOV and their centroids are biased.
        keep = len(counts)
        while keep > 3 and counts[keep - 1] < min_bin_ratio * counts[keep - 2]:
            keep -= 1
        centroids = centroids[:keep]

        offsets = centroids - centroids.mean(axis=0)
        direction = np.linalg.eigh(offsets.T @ offsets)[1][:, -1]
        if direction @ axis < 0:
            direction = -direction
        axis = direction
        residual = float(np.linalg.norm(offsets - np.outer(offsets @ axis, axis), axis=1).max())
        used = len(centroids)
    return axis, residual, used


def posterior_condylar_axis(condyle_points: np.ndarray, proximal_axis: np.ndarray,
                            iterations: int = 3):
    """
    Medial-lateral axis of the femur: the posterior condylar line.

    The line joins the most posterior point of each condyle, which is the standard femoral
    rotational reference and -- unlike anything taken from the tibia -- does not move with
    flexion. "Most posterior" needs an anterior-posterior direction, which needs this axis, so
    the two are iterated: starting from world patient-right the condyles are split at the median
    of their coordinate along the current axis, the most posterior point of each half is taken
    along ``proximal_axis x ml``, and the axis is redrawn through them. Two or
    three passes converge (the initial guess is already within a few degrees unless the leg is
    strongly rotated in the scanner).

    The axis returned is orthogonalised against ``proximal_axis``, so the frame is orthonormal;
    the raw posterior points are returned as well.

    :param condyle_points: ``(N, 3)`` world mm of the distal femur (bone surface points below
        the articular surface's proximal end).
    :param proximal_axis: The unit femoral shaft axis, pointing proximally.
    :param iterations: Number of passes.
    :return: ``(ml_axis, posterior_points)``: the unit medial-lateral axis pointing to the
        patient's right, and ``{'left': point, 'right': point}`` (patient side) in world mm.
    :raises ValueError: If one side of the condyles holds no points.
    """
    condyle_points = np.asarray(condyle_points, dtype=float).reshape(-1, 3)
    ml = _orthogonalise(_PATIENT_RIGHT, proximal_axis)
    posterior = {}
    for _ in range(max(1, iterations)):
        ap = np.cross(proximal_axis, ml)
        across = condyle_points @ ml
        split = float(np.median(across))
        halves = {'left': condyle_points[across <= split], 'right': condyle_points[across > split]}
        if min(len(h) for h in halves.values()) == 0:
            raise ValueError('the distal femur could not be split into two condyles')
        posterior = {side: half[np.argmin(half @ ap)] for side, half in halves.items()}
        ml = _orthogonalise(posterior['right'] - posterior['left'], proximal_axis)
        if ml @ _PATIENT_RIGHT < 0:
            ml = -ml
    return ml, posterior


def _orthogonalise(vector: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """
    Unit component of ``vector`` perpendicular to ``axis``.

    :param vector: Any non-zero vector.
    :param axis: A unit vector to remove from it.
    :return: The normalised perpendicular component.
    :raises ValueError: If ``vector`` is (numerically) parallel to ``axis``.
    """
    vector = np.asarray(vector, dtype=float)
    rest = vector - (vector @ axis) * axis
    norm = np.linalg.norm(rest)
    if norm < 1e-9:
        raise ValueError('cannot orthogonalise a vector parallel to the axis')
    return rest / norm


def notch_depth_profile(condyle_points: np.ndarray, ap_axis: np.ndarray, ml_axis: np.ndarray,
                        proximal_axis: np.ndarray, level_mm: float = 1.0,
                        ml_bin_mm: float = 1.0, edge_fraction: float = 0.15):
    """
    How deep the valley between the two condyles is, coronal slice by coronal slice.

    Seen from distal, the distal femur's outline has a valley along its middle: posteriorly the
    intercondylar fossa -- 20-30 mm deep, and open, so there is no bone at all between the
    condyles -- which shallows anteriorly and turns into the 3-6 mm deep trochlear groove. The
    level at which the fossa gives way to the groove is the trochlear notch
    (:func:`find_trochlear_notch`).

    The depth is measured on the **bone**, not on the cartilage: how far the trochlear
    cartilage is labelled across the groove varies between acquisitions of the same knee, the
    bone does not. Per coronal slice (``level_mm`` thick) the most distal bone coordinate is
    taken in each ``ml_bin_mm`` medial-lateral bin; the valley depth is the largest drop from
    the lower of the two flanking maxima to a bin between them, and a bin with no bone at all
    (the open fossa) counts as infinitely deep. Bins within ``edge_fraction`` of either end of
    the slice's bone are not considered, so the slope off the outer side of a condyle is never
    mistaken for the valley.

    :param condyle_points: ``(N, 3)`` world mm of the distal femur's bone voxels.
    :param ap_axis: Unit anterior-posterior axis of the femoral frame (anterior positive).
    :param ml_axis: Unit medial-lateral axis (patient right positive).
    :param proximal_axis: Unit shaft axis (proximal positive); depth is measured along it.
    :param level_mm: Thickness of one coronal slice in mm.
    :param ml_bin_mm: Width of one medial-lateral bin in mm.
    :param edge_fraction: Share of the slice's medial-lateral width at either end in which no
        valley is looked for.
    :return: ``(levels, depth, centre)``: the slices' anterior-posterior coordinates (band
        centres), their valley depth in mm (``inf`` where the fossa is open, ``NaN`` for a
        slice with too little bone) and the medial-lateral coordinate of the deepest bin.
    """
    condyle_points = np.asarray(condyle_points, dtype=float).reshape(-1, 3)
    ap = condyle_points @ ap_axis
    ml = condyle_points @ ml_axis
    distal = -(condyle_points @ proximal_axis)

    level_index = np.floor((ap - ap.min()) / level_mm).astype(int)
    ml_index = np.floor((ml - ml.min()) / ml_bin_mm).astype(int)
    n_levels, n_bins = level_index.max() + 1, ml_index.max() + 1
    profile = np.full((n_levels, n_bins), -np.inf)
    np.maximum.at(profile, (level_index, ml_index), distal)

    levels = ap.min() + (np.arange(n_levels) + 0.5) * level_mm
    depth = np.full(n_levels, np.nan)
    centre = np.full(n_levels, np.nan)
    for i, row in enumerate(profile):
        occupied = np.flatnonzero(np.isfinite(row))
        if len(occupied) < 10:
            continue
        lo, hi = occupied[0], occupied[-1]
        row = row[lo:hi + 1]
        # Running maxima of the bone outline to the left / right of each bin.
        left = np.maximum.accumulate(row)
        right = np.maximum.accumulate(row[::-1])[::-1]
        flank = np.minimum(left[:-2], right[2:])
        candidate = flank - row[1:-1]
        margin = max(1, int(round(edge_fraction * len(row))))
        inside = np.zeros(len(candidate), dtype=bool)
        inside[margin - 1:len(candidate) - margin + 1] = True
        if not inside.any():
            continue
        candidate = np.where(inside, candidate, -np.inf)
        finite = np.where(np.isposinf(candidate), _OPEN_VALLEY_MM, candidate)
        best = int(np.argmax(finite))
        if finite[best] <= 0:
            continue
        # The valley is a run of bins, not a single one: where the fossa is open every empty
        # bin is equally (infinitely) deep, and over a flat floor the deepest bin is picked by
        # noise. The notch centre is the middle of the contiguous run around the deepest bin
        # that is at least half as deep.
        start = stop = best
        while start > 0 and finite[start - 1] >= 0.5 * finite[best]:
            start -= 1
        while stop < len(finite) - 1 and finite[stop + 1] >= 0.5 * finite[best]:
            stop += 1
        depth[i] = candidate[best]
        centre[i] = ml.min() + (lo + 0.5 * (start + stop) + 1.5) * ml_bin_mm
    return levels, depth, centre


def find_trochlear_notch(condyle_points: np.ndarray, ap_axis: np.ndarray, ml_axis: np.ndarray,
                         proximal_axis: np.ndarray, anterior_limit: float,
                         level_mm: float = 1.0, min_depth_mm: float = 8.0,
                         max_slope: float = 0.5):
    """
    Locate the trochlear notch, the anterior border of the weight-bearing femoral condyles.

    The notch is read off the valley-depth profile of the distal femur
    (:func:`notch_depth_profile`): starting at the deepest slice -- unambiguously inside the
    intercondylar fossa -- the slices are walked anteriorly until the valley is shallower than
    ``min_depth_mm`` (one slice above it inside the run is tolerated, two in a row end it).
    That first shallow slice is where the fossa has become the trochlear groove: the trochlear
    notch. Its landmark point is the most distal bone point of the deepest bin there, i.e. the
    floor of the groove, and the weight-bearing condyles' anterior border is the plane through
    it with ``ap_axis`` as its normal -- a plane parallel to the femoral shaft, as the
    Chondrometrics convention requires.

    The threshold separates the fossa from the groove; it is not a sharp anatomical edge. On
    the ``dfg_knees`` femurs the depth falls by roughly 1.5 mm per mm of anterior-posterior
    travel there, so 2 mm of threshold move the border by little more than 1 mm.

    The deepest bins of the slices inside the fossa are fitted with a straight line
    ``ml = slope * ap + intercept`` (after rejecting outliers with a median-absolute-deviation
    band). The plane through that line containing the shaft axis is what separates the two
    condyles, and the trochlea into its two halves. The slope is clamped to ``max_slope`` -- a
    notch that appears to run more than ~27 degrees off the femoral frame's sagittal plane is a
    fitting artefact, not anatomy -- and a WARNING is logged when it is.

    :param condyle_points: ``(N, 3)`` world mm of the distal femur's bone voxels.
    :param ap_axis: Unit anterior-posterior axis of the femoral frame (anterior positive).
    :param ml_axis: Unit medial-lateral axis (patient right positive).
    :param proximal_axis: Unit shaft axis (proximal positive).
    :param anterior_limit: Anterior-posterior coordinate beyond which no notch is looked for
        (the anterior end of the articular surface): a trochlea that runs out of field of view
        can show a spurious valley.
    :param level_mm: Thickness of one coronal slice in mm.
    :param min_depth_mm: Valley depth at which the intercondylar fossa counts as closed.
    :param max_slope: Largest |slope| of the notch line, in mm of ml per mm of ap.
    :return: ``(notch_point, notch_ap, divider)``: the landmark in world mm, its
        anterior-posterior coordinate, and the ``(slope, intercept)`` of the notch line.
    :raises ValueError: If no slice shows an intercondylar valley at all (the distal femur is
        not recognisable as two condyles).
    """
    levels, depth, centre = notch_depth_profile(condyle_points, ap_axis, ml_axis, proximal_axis,
                                                level_mm=level_mm)
    searchable = levels <= anterior_limit
    deep = searchable & (depth >= min_depth_mm)
    if not deep.any():
        raise ValueError('the distal femur shows no intercondylar notch at all; the trochlear '
                         'notch could not be located')
    # The deepest slice is inside the fossa whatever else is going on; walk anteriorly from it.
    anchor = int(np.argmax(np.where(deep, np.nan_to_num(depth, nan=-np.inf,
                                                        posinf=_OPEN_VALLEY_MM), -np.inf)))
    last, misses = anchor, 0
    for i in range(anchor + 1, len(levels)):
        if not searchable[i]:
            break
        if deep[i]:
            last, misses = i, 0
        else:
            misses += 1
            if misses >= 2:
                break
    notch_index = min(last + 1, len(levels) - 1)

    run = np.zeros(len(levels), dtype=bool)
    run[anchor:last + 1] = deep[anchor:last + 1]
    slope, intercept = _fit_notch_line(levels[run], centre[run], max_slope)

    # The landmark: the floor of the groove in the notch slice -- the most distal bone point
    # nearest the notch line there.
    ap = condyle_points @ ap_axis
    level = levels[notch_index]
    band = np.flatnonzero(np.abs(ap - level) <= 0.5 * level_mm)
    if len(band) == 0:
        band = np.flatnonzero(np.abs(ap - level) <= 2.0 * level_mm)
    ml = condyle_points[band] @ ml_axis
    near = band[np.abs(ml - (slope * level + intercept)) <= max(1.0, level_mm)]
    if len(near) == 0:
        near = band[[int(np.argmin(np.abs(ml - (slope * level + intercept))))]]
    notch_point = condyle_points[near[int(np.argmin(condyle_points[near] @ proximal_axis))]]
    return notch_point, float(level), (slope, intercept)


def _fit_notch_line(ap_values: np.ndarray, ml_values: np.ndarray, max_slope: float):
    """
    Fit the line of intercondylar-notch centres, robustly and with a bounded slope.

    :param ap_values: Anterior-posterior coordinates of the notch-gap centres.
    :param ml_values: Their medial-lateral coordinates.
    :param max_slope: Largest |slope| accepted; a steeper fit is clamped and logged.
    :return: ``(slope, intercept)`` of ``ml = slope * ap + intercept``.
    """
    ap_values = np.asarray(ap_values, dtype=float)
    ml_values = np.asarray(ml_values, dtype=float)
    if len(ap_values) < 2:
        return 0.0, float(ml_values.mean())
    median = np.median(ml_values)
    deviation = np.median(np.abs(ml_values - median))
    if deviation > 0:
        inliers = np.abs(ml_values - median) <= 5.0 * deviation
        if inliers.sum() >= 2:
            ap_values, ml_values = ap_values[inliers], ml_values[inliers]
    slope, intercept = np.polyfit(ap_values, ml_values, 1)
    if abs(slope) > max_slope:
        logger.warning('Femur: the intercondylar notch line is tilted %.2f mm/mm against the '
                       'femoral sagittal plane; clamping it to %.2f.', slope, max_slope)
        clamped = float(np.clip(slope, -max_slope, max_slope))
        intercept += (slope - clamped) * float(np.mean(ap_values))
        slope = clamped
    return float(slope), float(intercept)


# ----------------------------------------------------------------------------------------
# Arc length across a condyle
# ----------------------------------------------------------------------------------------

def _polyline_components(points: np.ndarray, lines: np.ndarray) -> List[List[int]]:
    """
    Order the line segments of a mesh slice into point-index sequences.

    VTK's cutter returns an unordered soup of segments; this chains them by shared points.
    Each connected component becomes one sequence, walked from an open end if it has one
    (otherwise from an arbitrary point, which is what a closed loop deserves). At a branch
    point the first unvisited neighbour is taken.

    :param points: ``(N, 3)`` slice points (only their count is used here).
    :param lines: The ``pv.PolyData.lines`` connectivity array.
    :return: A list of point-index sequences, one per connected component.
    """
    neighbours: Dict[int, List[int]] = {}
    i = 0
    lines = np.asarray(lines).ravel()
    while i < len(lines):
        n = int(lines[i])
        ids = lines[i + 1:i + 1 + n]
        for a, b in zip(ids[:-1], ids[1:]):
            neighbours.setdefault(int(a), []).append(int(b))
            neighbours.setdefault(int(b), []).append(int(a))
        i += n + 1

    visited = set()
    sequences = []
    # Start at open ends first so an open curve is walked end to end in one piece.
    starts = [p for p in neighbours if len(neighbours[p]) == 1] + sorted(neighbours)
    for start in starts:
        if start in visited:
            continue
        sequence, current = [start], start
        visited.add(start)
        while True:
            nxt = next((n for n in neighbours[current] if n not in visited), None)
            if nxt is None:
                break
            visited.add(nxt)
            sequence.append(nxt)
            current = nxt
        sequences.append(sequence)
    return sequences


def _slice_arc_length(mesh: pv.PolyData, level: float, ap_axis: np.ndarray,
                      internal_axis: np.ndarray):
    """
    Parametrise one coronal cross-section of a condyle's tAB by normalised arc length.

    :param mesh: The condyle's tAB (world mm).
    :param level: The cross-section's coordinate along ``ap_axis``.
    :param ap_axis: Unit anterior-posterior axis (the slice plane's normal).
    :param internal_axis: Unit direction towards the other condyle; arc length is measured from
        the end of the contour that lies on that side.
    :return: ``(points, u)``: the ordered contour points (world mm) and their normalised arc
        length in ``[0, 1]``, or ``None`` if the cross-section has no usable contour.
    """
    cut = mesh.slice(normal=ap_axis, origin=level * ap_axis)
    if cut is None or cut.n_points < 2 or cut.n_cells == 0:
        return None
    points = np.asarray(cut.points, dtype=float)
    best, best_length = None, 0.0
    for sequence in _polyline_components(points, cut.lines):
        if len(sequence) < 2:
            continue
        ordered = points[sequence]
        length = float(np.linalg.norm(np.diff(ordered, axis=0), axis=1).sum())
        if length > best_length:
            best, best_length = ordered, length
    if best is None or best_length <= 0:
        return None
    # Measure from the internal end: the end of the contour facing the other condyle.
    edge = max(1, len(best) // 5)
    if (best[:edge] @ internal_axis).mean() < (best[-edge:] @ internal_axis).mean():
        best = best[::-1]
    steps = np.linalg.norm(np.diff(best, axis=0), axis=1)
    u = np.concatenate([[0.0], np.cumsum(steps)]) / steps.sum()
    return best, u


def arc_length_coordinate(mesh: pv.PolyData, ap_axis: np.ndarray, ml_axis: np.ndarray,
                          proximal_axis: np.ndarray, internal_axis: np.ndarray,
                          slice_mm: float = 0.5):
    """
    Normalised arc-length coordinate across a weight-bearing condyle's tAB.

    The paper divides each of its triangle strips' edge sequences into three parts of equal
    length, internal to external. The same quantity, as a continuous field: the condyle's tAB is
    cut into coronal cross-sections ``slice_mm`` apart (planes perpendicular to the femoral
    anterior-posterior axis), each cross-section's contour is parametrised by arc length from
    its internal end (0) to its external end (1), and every mesh vertex takes the value
    interpolated between the two cross-sections that bracket it -- within a cross-section by the
    nearest contour point, between them linearly in the anterior-posterior coordinate.

    Cutting the mesh at ``u = 1/3`` and ``u = 2/3`` then reproduces the paper's division; for a
    condyle whose cross-sections are straight (a cylinder along the medial-lateral axis) the
    thirds have exactly a third of the area each.

    :param mesh: The weight-bearing condyle's tAB (all-triangle ``pv.PolyData``, world mm).
    :param ap_axis: Unit anterior-posterior axis (the cross-sections' normal).
    :param ml_axis: Unit medial-lateral axis (in the cross-section plane).
    :param proximal_axis: Unit shaft axis (the other in-plane direction).
    :param internal_axis: Unit direction towards the other condyle (``+ml_axis`` for the
        patient-left condyle, ``-ml_axis`` for the patient-right one).
    :param slice_mm: Spacing of the cross-sections in mm.
    :return: ``(u, n_slices)``: an ``(n_points,)`` array in ``[0, 1]``, 0 at the internal end,
        and the number of cross-sections it was built from.
    :raises ValueError: If fewer than two cross-sections have a usable contour.
    """
    points = np.asarray(mesh.points, dtype=float)
    ap = points @ ap_axis
    # Cross-sections strictly inside the condyle's A-P extent (a plane through its very edge
    # cuts nothing).
    levels = np.arange(ap.min() + 0.5 * slice_mm, ap.max(), slice_mm)
    contours = []
    for level in levels:
        parametrised = _slice_arc_length(mesh, float(level), ap_axis, internal_axis)
        if parametrised is None:
            continue
        contour, u = parametrised
        plane = np.stack([contour @ ml_axis, contour @ proximal_axis], axis=1)
        contours.append((float(level), cKDTree(plane), u))
    if len(contours) < 2:
        raise ValueError('the weight-bearing femoral condyle has too few usable cross-sections '
                         'to measure arc length across it')

    in_plane = np.stack([points @ ml_axis, points @ proximal_axis], axis=1)
    slice_levels = np.array([c[0] for c in contours])
    # Nearest-contour-point value on every cross-section, then interpolate between the two
    # that bracket each vertex (clamped to the first / last one outside them).
    per_slice = np.empty((len(contours), len(points)))
    for i, (_, tree, u) in enumerate(contours):
        per_slice[i] = u[tree.query(in_plane, k=1)[1]]
    upper = np.clip(np.searchsorted(slice_levels, ap), 1, len(contours) - 1)
    lower = upper - 1
    span = slice_levels[upper] - slice_levels[lower]
    weight = np.clip((ap - slice_levels[lower]) / span, 0.0, 1.0)
    rows = np.arange(len(points))
    return (1 - weight) * per_slice[lower, rows] + weight * per_slice[upper, rows], len(contours)


# ----------------------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------------------

def subdivide_femur(image: Image, bone_label: int = 1, cartilage_label: int = 3,
                    weightbearing_fraction: float = 0.6, laterality: Optional[str] = None,
                    coverage_tolerance_mm: Optional[float] = None,
                    max_hole_fraction: float = 0.5, slice_mm: float = 0.5,
                    notch_depth_mm: float = 8.0,
                    shaft_band_mm: Optional[Tuple[float, float]] = None,
                    shaft_bin_mm: float = 3.0,
                    shaft_min_bin_ratio: float = 0.75) -> FemurSubdivision:
    """
    Subdivide the femoral subchondral bone area into the Wirth & Eckstein subregions.

    Steps (all in world mm):

    1. Femoral bone surface (:func:`~morphometry.cartilage.surface.bone_surface_mesh`), its
       cartilage-covered part cAB (:func:`~morphometry.cartilage.surface.cartilage_coverage`)
       and the tAB that fills cAB's enclosed denuded holes
       (:func:`~morphometry.cartilage.surface.total_subchondral_area`). The femoral cartilage
       is a horseshoe open posteriorly, so the intercondylar area is connected to the
       non-articular rest of the bone and can never be filled -- unlike the tibia, no
       per-plate hole filling is needed.
    2. The femoral frame: shaft axis (:func:`estimate_shaft_axis`), posterior condylar line
       (:func:`posterior_condylar_axis`) and the anterior-posterior axis completing them.
    3. The trochlear notch and the line of intercondylar-fossa centres
       (:func:`find_trochlear_notch`). The plane through that line containing the shaft axis
       separates the two condyles; the plane through the notch point with the
       anterior-posterior axis as its normal is the weight-bearing condyles' anterior border.
    4. Per condyle: its most posterior tAB point sets the posterior border at
       ``weightbearing_fraction`` of the way back from the notch plane. Anterior of the notch
       plane is the trochlea zone (``alf`` / ``arf``), posterior of the 60% plane the posterior
       condyle zone (``plf`` / ``prf``) -- both in-house extensions, not in the paper.
    5. The weight-bearing condyle cF is cut at a third and two thirds of the arc length across
       it (:func:`arc_length_coordinate`) into the internal, central and external subregion.

    :param image: The (LPI-oriented) knee ``Segmentation``; either image backend.
    :param bone_label: Label of the femoral bone (``dfg_knees``: 1).
    :param cartilage_label: Label of the femoral cartilage (``dfg_knees``: 3).
    :param weightbearing_fraction: Where the weight-bearing condyle ends, as a fraction of the
        distance from the notch plane to the condyle's posterior end (paper's convention 0.6;
        0.75 is used by some later studies).
    :param laterality: ``'left'`` / ``'right'`` knee, or ``None``. Only adds the medial/lateral
        naming (:func:`anatomical_femur_names`); the subdivision does not depend on it.
    :param coverage_tolerance_mm: cAB distance tolerance, see
        :func:`~morphometry.cartilage.surface.cartilage_coverage` (``None`` = its default).
    :param max_hole_fraction: Largest denuded hole filled, as a fraction of cAB, see
        :func:`~morphometry.cartilage.surface.total_subchondral_area`.
    :param slice_mm: Spacing of the coronal cross-sections used for the arc length.
    :param notch_depth_mm: Valley depth at which the intercondylar fossa counts as closed,
        see :func:`find_trochlear_notch`.
    :param shaft_band_mm: ``(start, end)`` of the band the shaft axis is fitted in, relative to
        the proximal end of the articular surface; ``None`` uses
        :func:`estimate_shaft_axis`'s default. The default was *tuned* on this cohort, so it is
        exposed here rather than hidden in a module default.
    :param shaft_bin_mm: Thickness of one shaft cross-section, see :func:`estimate_shaft_axis`.
    :param shaft_min_bin_ratio: Completeness ratio for a shaft cross-section, see
        :func:`estimate_shaft_axis`.
    :return: A :class:`FemurSubdivision`.
    :raises ValueError: If the bone or cartilage label is missing, the field of view holds too
        little femur to estimate the shaft axis, the trochlear notch cannot be found, or a
        condyle ends up empty.
    """
    anatomical_femur_names(laterality)  # validates the value
    if not 0.0 < weightbearing_fraction < 1.0:
        raise ValueError(f'weightbearing_fraction must be in (0, 1), got {weightbearing_fraction}')
    bone = S.bone_surface_mesh(image, bone_label)
    if bone.n_cells == 0:
        raise ValueError(f'femoral bone label {bone_label} is absent; the femoral subregions '
                         'are defined on the subchondral bone surface')
    covered = S.cartilage_coverage(bone, image, cartilage_label, coverage_tolerance_mm)
    if not covered.any():
        raise ValueError(f'no femoral bone surface is covered by cartilage label {cartilage_label}')
    tab = S.total_subchondral_area(bone, covered, max_hole_fraction)
    tab_mesh = S.extract_faces(bone, tab)
    tab_mesh.cell_data['covered'] = covered[tab]

    # --- the femoral frame ---------------------------------------------------------------
    bone_points = label_points_world(image, bone_label)
    # Every cartilage component counts: the femoral cartilage is one horseshoe on a clean
    # scan, but a labelling gap at the trochlea splits it in two, and dropping the smaller
    # half would move the distal reference by centimetres (see articular_limit).
    cartilage_points = label_points_world(image, cartilage_label, largest_component=False)
    shaft_kwargs = {} if shaft_band_mm is None else {'band_mm': tuple(shaft_band_mm)}
    proximal_axis, residual, n_bins = estimate_shaft_axis(
        bone_points, cartilage_points, bin_mm=shaft_bin_mm,
        min_bin_ratio=shaft_min_bin_ratio, **shaft_kwargs)
    # The condyles: the bone distal of the articular surface's proximal end.
    surface_points = np.asarray(bone.points, dtype=float)
    distal_limit = articular_limit(cartilage_points, proximal_axis)
    condyle_points = surface_points[surface_points @ proximal_axis <= distal_limit]
    ml_axis, posterior_points = posterior_condylar_axis(condyle_points, proximal_axis)
    # Anterior needs no world direction of its own: with the shaft axis pointing proximally
    # and the condylar axis to the patient's right, the right-handed completion points
    # anteriorly whatever pose the femur is in -- including a knee flexed by 90 degrees,
    # where the femoral anterior is nowhere near the world anterior axis.
    ap_axis = np.cross(proximal_axis, ml_axis)

    distal_bone = bone_points[bone_points @ proximal_axis <= distal_limit]
    notch_point, notch_ap, divider = find_trochlear_notch(
        distal_bone, ap_axis, ml_axis, proximal_axis,
        anterior_limit=articular_limit(cartilage_points, ap_axis), min_depth_mm=notch_depth_mm)
    frame = FemoralFrame(proximal_axis=proximal_axis, ml_axis=ml_axis, ap_axis=ap_axis,
                         notch_point=notch_point, notch_ap=notch_ap,
                         divider=divider, posterior_points=posterior_points,
                         shaft_residual_mm=residual, shaft_bins=n_bins)

    # --- cut the tAB ---------------------------------------------------------------------
    # Anterior of the notch plane is the trochlea, posterior of it the two condyles.
    condylar, trochlea = S.split_mesh_by_scalar(tab_mesh, tab_mesh.points @ ap_axis,
                                                frame.notch_ap)
    parts, condyles = [], {}
    for index, side in enumerate(CONDYLE_SIDES):
        condyle = _side_of(condylar, frame, side)
        if condyle.n_cells == 0:
            raise ValueError(f'the patient-{side} femoral condyle has no subchondral bone area')
        internal_axis = ml_axis if side == 'left' else -ml_axis
        posterior_ap = float((np.asarray(condyle.points) @ ap_axis).min())
        boundary_ap = frame.notch_ap - weightbearing_fraction * (frame.notch_ap - posterior_ap)
        posterior_zone, weightbearing = S.split_mesh_by_scalar(
            condyle, np.asarray(condyle.points) @ ap_axis, boundary_ap)
        if weightbearing.n_cells == 0:
            raise ValueError(f'the patient-{side} weight-bearing femoral condyle is empty')
        # The articular surface wraps proximally around the posterior condyle, so a coronal
        # plane in the posterior part of the window cuts it twice: once on the weight-bearing
        # (distal) sheet and once on that proximal wrap, which the anterior-posterior window
        # alone cannot tell apart. Within the window the two sheets are separate patches (they
        # meet only behind it, at the condyle's posterior extreme), so cF is the largest patch
        # and the rest is counted with the posterior condyle zone it continues.
        weightbearing, wrap = _largest_patch(weightbearing)
        wrap_area = float(S.triangle_areas(wrap).sum()) if wrap.n_cells else 0.0
        window_area = wrap_area + float(S.triangle_areas(weightbearing).sum())
        if window_area > 0 and wrap_area > MAX_WRAP_FRACTION * window_area:
            logger.warning(
                'Femur: %.0f%% of the patient-%s weight-bearing window was discarded as the '
                'posterior condyle\'s proximal wrap; the window holds two comparable patches '
                'and the weight-bearing condyle may be the wrong one.',
                100 * wrap_area / window_area, side)
        posterior_zone = S.merge_meshes([posterior_zone, wrap])

        u, n_slices = arc_length_coordinate(weightbearing, ap_axis, ml_axis, proximal_axis,
                                            internal_axis, slice_mm=slice_mm)
        weightbearing.point_data['arc_length'] = u
        internal, rest = S.split_mesh_by_scalar(weightbearing, 'arc_length', 1.0 / 3.0)
        central, external = S.split_mesh_by_scalar(rest, 'arc_length', 2.0 / 3.0)

        anterior_zone = _side_of(trochlea, frame, side)
        region_areas = {}
        for letter, part in zip(REGION_LETTERS, (internal, central, external)):
            code = FEMUR_REGION_KEYS.index(f'{letter}c{side[0]}f')
            region_areas[FEMUR_REGION_KEYS[code]] = _label_part(part, code, index, parts)
        anterior_area = _label_part(anterior_zone, FEMUR_REGION_KEYS.index(f'a{side[0]}f'),
                                    index, parts)
        posterior_area = _label_part(posterior_zone, FEMUR_REGION_KEYS.index(f'p{side[0]}f'),
                                     index, parts)

        covered_flags = np.asarray(weightbearing.cell_data['covered'], dtype=bool)
        condyles[side] = CondyleFrame(
            side=side, posterior_ap=posterior_ap, boundary_ap=float(boundary_ap),
            internal_direction=internal_axis,
            tab_area=float(S.triangle_areas(weightbearing).sum()),
            cab_area=float(S.triangle_areas(weightbearing)[covered_flags].sum()),
            region_areas=region_areas, anterior_area=anterior_area,
            posterior_area=posterior_area, n_slices=n_slices)

    mesh = S.merge_meshes(parts)
    return FemurSubdivision(mesh=mesh, frame=frame, condyles=condyles, laterality=laterality,
                            weightbearing_fraction=weightbearing_fraction,
                            tab_area=float(S.triangle_areas(tab_mesh).sum()), bone_mesh=bone)


def _side_of(mesh: pv.PolyData, frame: FemoralFrame, side: str) -> pv.PolyData:
    """
    The part of a mesh on one side of the intercondylar notch plane.

    :param mesh: An all-triangle ``pv.PolyData`` in world mm (empty is allowed).
    :param frame: The femoral frame holding the notch line.
    :param side: ``'left'`` or ``'right'`` (patient side).
    :return: The sub-mesh on that side (possibly empty).
    """
    if mesh.n_cells == 0:
        return mesh
    left, right = S.split_mesh_by_scalar(mesh, frame.divider_scalar(mesh.points), 0.0)
    return left if side == 'left' else right


def _largest_patch(mesh: pv.PolyData):
    """
    Split a mesh into its largest edge-connected patch and everything else.

    :param mesh: An all-triangle ``pv.PolyData`` (empty is allowed).
    :return: ``(largest, rest)``; ``rest`` is empty when the mesh is one patch.
    """
    if mesh.n_cells == 0:
        return mesh, pv.PolyData()
    labels, _, areas = S.face_components(mesh, np.ones(mesh.n_cells, dtype=bool))
    if len(areas) <= 1:
        return mesh, pv.PolyData()
    keep = labels == int(np.argmax(areas))
    return S.extract_faces(mesh, keep), S.extract_faces(mesh, ~keep)


def _label_part(part: pv.PolyData, code: int, condyle_index: int, parts: list) -> float:
    """
    Tag one region's triangles with their region code and collect them.

    :param part: The region's sub-mesh (may be empty).
    :param code: Its index into :data:`FEMUR_REGION_KEYS`.
    :param condyle_index: 0 for the patient-left condyle, 1 for the patient-right one.
    :param parts: The list the labelled sub-mesh is appended to (modified in place).
    :return: The region's tAB area in mm^2.
    """
    if part.n_cells == 0:
        return 0.0
    part.cell_data['region'] = np.full(part.n_cells, code, dtype=int)
    part.cell_data['condyle'] = np.full(part.n_cells, condyle_index, dtype=int)
    parts.append(part)
    return float(S.triangle_areas(part).sum())
