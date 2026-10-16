"""
Bone-surface geometry for area-based knee cartilage subregions (Wirth & Eckstein 2008).

Wirth & Eckstein define their subregions on the *total area of subchondral bone* (tAB): the
bone surface under the cartilage, including denuded (cartilage-free) patches inside it,
represented as a triangle mesh. Every regional share (e.g. "the central tibial region holds
20% of the plate") is a share of surface **area** in millimetres, and region boundaries cut
triangles exactly rather than assigning whole voxels.

This module provides that shared geometry and nothing knee-specific:

- :func:`bone_surface_mesh` -- a smoothed triangle mesh of one bone label in physical
  (world) millimetres, open where the bone leaves the field of view.
- :func:`cartilage_coverage` -- the covered bone area cAB: triangles within a tolerance of
  the cartilage mask.
- :func:`total_subchondral_area` -- tAB = cAB plus the enclosed denuded holes.
- :func:`triangle_areas`, :func:`split_mesh_by_scalar`, :func:`split_mesh_by_plane` --
  exact area bookkeeping, with triangles split along a plane or a scalar level set.
- :func:`face_components`, :func:`extract_faces`, :func:`merge_meshes` -- edge-connected
  patches, sub-meshes and concatenation (used to split the tibial cAB into plates).
- :func:`array_indices_to_world` -- world mm of indices into ``Image.array`` (either backend).
- :func:`transfer_qc` -- how far the cartilage voxels ended up from the bone surface region
  they were given (shared by the tibial and femoral subdivisions).

All coordinates are physical (world) mm obtained through the image's full index-to-world
transform (direction cosines, spacing and origin), so results do not depend on where or how
obliquely the knee sits in the scanner, nor on the voxel spacing beyond discretisation error.
The world frame is **RAS** (+x towards patient Right, +y Anterior, +z Superior; nibabel's
convention) for *both* image backends: SimpleITK's native LPS frame is converted (see
:func:`index_to_world_affine`), so anatomical directions in world coordinates mean the same
thing whichever backend loaded the image.
"""
import numpy as np
import pyvista as pv
import SimpleITK as sitk
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.ndimage import label as nd_label
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

from morphometry.image_io import Image


# ----------------------------------------------------------------------------------------
# Image geometry
# ----------------------------------------------------------------------------------------

def index_to_world_affine(image: Image) -> np.ndarray:
    """
    Return the 4x4 matrix mapping voxel indices ``(i, j, k, 1)`` to **RAS** world mm.

    For nibabel this is the image affine (RAS by the NIfTI convention). SimpleITK's physical
    space is LPS (``world_LPS = D @ diag(spacing) @ index + origin``, ``index`` in SimpleITK's
    ``(x, y, z)`` order, which is also the axis order of :func:`label_array_xyz`); it is
    converted to RAS by negating the first two world axes, so both backends give the same
    world coordinates for the same file.

    :param image: The ``Image`` / ``Segmentation`` whose geometry is wanted.
    :return: A ``(4, 4)`` float array; ``[:3, :3]`` holds the direction-scaled-by-spacing
        matrix and ``[:3, 3]`` the origin.
    """
    if image.type == 'nibabel':
        return np.asarray(image.image.affine, dtype=float)
    direction = np.asarray(image.image.GetDirection(), dtype=float).reshape(3, 3)
    affine = np.eye(4)
    affine[:3, :3] = direction @ np.diag(np.asarray(image.image.GetSpacing(), dtype=float))
    affine[:3, 3] = np.asarray(image.image.GetOrigin(), dtype=float)
    return np.diag([-1.0, -1.0, 1.0, 1.0]) @ affine  # LPS -> RAS


def label_array_xyz(image: Image) -> np.ndarray:
    """
    Return the label array indexed ``[i, j, k]`` consistently with :func:`index_to_world_affine`.

    nibabel arrays already are; SimpleITK's ``GetArrayFromImage`` returns ``[k, j, i]`` and is
    transposed.

    :param image: The ``Image`` / ``Segmentation`` to read.
    :return: The voxel array in ``(i, j, k)`` axis order.
    """
    if image.type == 'nibabel':
        return np.asanyarray(image.image.dataobj)
    return sitk.GetArrayFromImage(image.image).transpose(2, 1, 0)


def voxel_spacing(image: Image) -> np.ndarray:
    """
    Return the voxel edge lengths (mm) along the three index axes.

    Computed as the column norms of the index-to-world matrix, so it is correct for any
    direction matrix (and matches the header zooms of a well-formed NIfTI).

    :param image: The ``Image`` / ``Segmentation``.
    :return: A length-3 float array of spacings in mm.
    """
    return np.linalg.norm(index_to_world_affine(image)[:3, :3], axis=0)


def indices_to_world(affine: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """
    Map an ``(N, 3)`` array of (possibly fractional) voxel indices to world mm, vectorised.

    :param affine: The 4x4 index-to-world matrix (see :func:`index_to_world_affine`).
    :param indices: ``(N, 3)`` voxel indices.
    :return: ``(N, 3)`` world coordinates in mm.
    """
    indices = np.asarray(indices, dtype=float).reshape(-1, 3)
    return indices @ affine[:3, :3].T + affine[:3, 3]


def array_indices_to_world(image: Image, indices: np.ndarray) -> np.ndarray:
    """
    Map indices into ``image.array`` to RAS world mm, for either backend.

    ``Image.array`` is ``(i, j, k)``-ordered for nibabel but ``(k, j, i)`` for SimpleITK
    (``GetArrayFromImage``), so the columns are reversed for SimpleITK before applying
    :func:`index_to_world_affine`. Use this for point clouds built with
    ``np.argwhere(image.array == label)``.

    :param image: The ``Image`` / ``Segmentation`` the indices refer to.
    :param indices: ``(N, 3)`` indices into ``image.array``.
    :return: ``(N, 3)`` world coordinates in mm (RAS).
    """
    indices = np.asarray(indices, dtype=float).reshape(-1, 3)
    if image.type != 'nibabel':
        indices = indices[:, ::-1]
    return indices_to_world(index_to_world_affine(image), indices)


# ----------------------------------------------------------------------------------------
# Mesh helpers
# ----------------------------------------------------------------------------------------

def _faces_array(mesh: pv.PolyData) -> np.ndarray:
    """
    Return the ``(M, 3)`` vertex-index array of an all-triangle mesh.

    :param mesh: A ``pv.PolyData`` consisting only of triangles.
    :return: ``(M, 3)`` int array of vertex indices per face.
    :raises ValueError: If the mesh holds non-triangle cells.
    """
    if mesh.n_cells == 0:
        return np.empty((0, 3), dtype=int)
    if not mesh.is_all_triangles:
        raise ValueError('mesh must consist of triangles only (call .triangulate() first)')
    return np.asarray(mesh.faces).reshape(-1, 4)[:, 1:]


def triangle_areas(mesh: pv.PolyData) -> np.ndarray:
    """
    Exact per-triangle areas (in the mesh's units, mm² for the meshes built here).

    :param mesh: An all-triangle ``pv.PolyData``.
    :return: ``(n_cells,)`` float array of triangle areas.
    """
    faces = _faces_array(mesh)
    p = np.asarray(mesh.points, dtype=float)
    a, b, c = p[faces[:, 0]], p[faces[:, 1]], p[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def triangle_centroids(mesh: pv.PolyData) -> np.ndarray:
    """
    Per-triangle centroids (mean of the three vertices).

    :param mesh: An all-triangle ``pv.PolyData``.
    :return: ``(n_cells, 3)`` float array.
    """
    faces = _faces_array(mesh)
    return np.asarray(mesh.points, dtype=float)[faces].mean(axis=1)


def _face_adjacency(faces: np.ndarray):
    """
    Edge adjacency of a triangle mesh.

    Two faces are adjacent when they share an (undirected) edge. Edges used by exactly one
    face are the mesh's open boundary. The faces of a non-manifold edge (three or more faces)
    are chained into one connected group (consecutive pairs only, not every pair).

    :param faces: ``(M, 3)`` vertex indices.
    :return: ``(pairs, on_boundary)``: ``pairs`` is a ``(K, 2)`` array of adjacent face
        indices, ``on_boundary`` an ``(M,)`` bool array, True for faces with an open edge.
    """
    m = len(faces)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges.sort(axis=1)
    owner = np.tile(np.arange(m), 3)
    # Group identical edges: sort lexicographically, then find runs.
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, owner = edges[order], owner[order]
    new_run = np.ones(len(edges), dtype=bool)
    new_run[1:] = np.any(edges[1:] != edges[:-1], axis=1)
    run_id = np.cumsum(new_run) - 1
    run_size = np.bincount(run_id)

    on_boundary = np.zeros(m, dtype=bool)
    on_boundary[owner[run_size[run_id] == 1]] = True

    # Manifold edges: consecutive entries in a run of two.
    same = ~new_run[1:]
    pairs = np.stack([owner[:-1][same], owner[1:][same]], axis=1)
    # For a non-manifold run (>2 faces) these consecutive pairs chain all its faces together,
    # which is all a connected-components pass needs.
    return pairs, on_boundary


# ----------------------------------------------------------------------------------------
# 1. Bone surface
# ----------------------------------------------------------------------------------------

def bone_surface_mesh(image: Image, bone_label: int, largest_component: bool = True,
                      smoothing_iterations: int = 20, pass_band: float = 0.1,
                      open_at_fov: bool = True, fov_extension: int = 4) -> pv.PolyData:
    """
    Triangle mesh of one bone label's surface in physical (world) mm.

    Marching cubes (level 0.5) is run on the binary bone mask in *index* space and the vertices
    are then mapped to world mm with the full index-to-world transform, so the mesh is correct
    for anisotropic voxels and for any (oblique, reflected) direction matrix. The triangle
    winding is chosen so that normals always point out of the bone.

    **Field-of-view cuts.** A knee FOV cuts through the femoral and tibial shafts. Zero-padding
    the mask there would close the surface with a flat artificial cap, which is not bone
    surface, and smoothing would round the shaft into it. Instead, wherever the bone touches
    an array face, the mask is extended past that face by ``fov_extension`` replicated slices
    (the shaft continues straight), the closed surface is built and smoothed, and it is then
    cut *exactly* (triangles split, :func:`split_mesh_by_scalar`) at the array face, i.e. at
    the outer voxel boundary, index ``-0.5`` or ``n - 0.5``. The result is open along the FOV
    cut, with the surface up to the cut unaffected by the padding. That open boundary is what
    :func:`total_subchondral_area` uses to recognise the non-articular remainder of the bone:
    an uncovered region reaching the FOV cut is never mistaken for an enclosed denuded hole.
    Array faces the bone does not touch are zero-padded as usual (closed surface there).

    **Smoothing.** Windowed-sinc smoothing (``smooth_taubin``, in world mm) removes the voxel
    staircase without the shrinkage of Laplacian smoothing: on a 10 mm sphere it brings the
    area from +9..15% (raw marching cubes) to within 1% of analytic and moves the surface by
    ~0.02 mm. Sharp edges are rounded (a 10 mm voxel box loses ~4% of its area, 2.4% of it
    already to marching cubes' chamfer); bone has no such edges at this scale.
    ``smoothing_iterations=0`` skips it.

    :param image: The (LPI-oriented) ``Segmentation`` holding the bone label.
    :param bone_label: Label value of the bone (``dfg_knees``: 1 = femur, 2 = tibia).
    :param largest_component: Keep only the largest 26-connected component of the label, so
        stray mislabelled voxels elsewhere do not produce extra surfaces.
    :param smoothing_iterations: Windowed-sinc iterations (``smooth_taubin``'s ``n_iter``).
    :param pass_band: Windowed-sinc pass band (lower = smoother, 0 < pass_band < 2).
    :param open_at_fov: Cut the surface open at the array faces the bone touches (see above).
        With ``False`` those faces are zero-padded and the mesh is closed by a flat cap.
    :param fov_extension: Number of replicated slices beyond a touched array face; it keeps
        the smoothing and the padding cap away from the cut.
    :return: An all-triangle ``pv.PolyData`` in world mm with outward normals in
        ``cell_data['Normals']`` / ``point_data['Normals']``. Empty if the label is absent.
    """
    labels = label_array_xyz(image)
    mask = np.asarray(labels) == bone_label
    if not mask.any():
        return pv.PolyData()

    if largest_component:
        components, n = nd_label(mask, structure=np.ones((3, 3, 3), dtype=bool))
        if n > 1:
            sizes = np.bincount(components.ravel())
            sizes[0] = 0
            mask = components == np.argmax(sizes)

    # Crop to the bone's bounding box (speed). `lo` maps cropped indices back to the array.
    shape = np.asarray(mask.shape)
    nz = np.argwhere(mask)
    lo, hi = nz.min(axis=0), nz.max(axis=0) + 1
    cropped = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]

    # Which array faces the bone touches: (axis, side) with side 0 = index 0, 1 = index n-1.
    touches = [(axis, side) for axis in range(3) for side in (0, 1)
               if open_at_fov and (lo[axis] == 0 if side == 0 else hi[axis] == shape[axis])]
    extend = [[fov_extension if (axis, side) in touches else 0 for side in (0, 1)]
              for axis in range(3)]
    padded = np.pad(np.pad(cropped, extend, mode='edge'), 1).astype(np.float32)
    verts, faces, _, _ = marching_cubes(padded, level=0.5, allow_degenerate=False)
    verts = verts.astype(float) + (lo - 1 - np.array([e[0] for e in extend]))  # array indices
    faces = faces.astype(np.int64)

    affine = index_to_world_affine(image)
    # In index space skimage's winding gives right-hand-rule normals pointing *into* a mask
    # whose inside is the high value (checked on a sphere phantom), so reverse it; a
    # reflecting index-to-world transform (negative determinant) reverses it once more.
    if np.linalg.det(affine[:3, :3]) > 0:
        faces = faces[:, ::-1]
    mesh = pv.PolyData(indices_to_world(affine, verts),
                       np.hstack([np.full((len(faces), 1), 3), faces]).ravel())
    if smoothing_iterations > 0:
        mesh = mesh.smooth_taubin(n_iter=smoothing_iterations, pass_band=pass_band,
                                  boundary_smoothing=False, feature_smoothing=False,
                                  normalize_coordinates=True)

    # Cut the extension off exactly at the touched array faces (outer voxel boundary).
    inverse = np.linalg.inv(affine)
    for axis, side in touches:
        index_coord = indices_to_world(inverse, mesh.points)[:, axis]
        below, above = split_mesh_by_scalar(mesh, index_coord, -0.5 if side == 0 else shape[axis] - 0.5)
        mesh = above if side == 0 else below

    # Drop orphaned points only; keep every (possibly sliver) triangle a triangle.
    mesh = mesh.clean(point_merging=False, lines_to_points=False, polys_to_lines=False)
    mesh = mesh.compute_normals(cell_normals=True, point_normals=True, consistent_normals=False,
                                auto_orient_normals=False, split_vertices=False)
    return mesh


# ----------------------------------------------------------------------------------------
# 2. Covered area (cAB)
# ----------------------------------------------------------------------------------------

def default_coverage_tolerance(image: Image) -> float:
    """
    Default distance (mm) within which a bone triangle counts as covered by cartilage.

    A bone-surface point on the face between a bone voxel and an adjacent cartilage voxel lies
    at most half a voxel *diagonal* from that cartilage voxel's centre (half a voxel along the
    face normal plus up to half a voxel along each in-face axis). Smoothing moves the surface
    by a fraction of a voxel on top of that. The default is therefore ``0.6`` voxel diagonals
    -- the half diagonal plus a 20% margin -- which is 0.52 mm at ``dfg_knees``' 0.75 x 0.31 x
    0.31 mm. A larger value widens cAB by roughly ``tolerance - half diagonal`` along its whole
    rim (about 150 mm of rim per tibial plate), so it is kept as tight as the geometry allows.

    :param image: The ``Segmentation`` whose voxel size sets the scale.
    :return: The tolerance in mm.
    """
    return 0.6 * float(np.linalg.norm(voxel_spacing(image)))


def cartilage_coverage(bone_mesh: pv.PolyData, image: Image, cartilage_label: int,
                       tolerance_mm: float = None) -> np.ndarray:
    """
    Per-triangle cartilage coverage of a bone surface (the cAB of Wirth & Eckstein).

    A triangle is covered when its centroid lies within ``tolerance_mm`` of the centre of a
    voxel carrying ``cartilage_label``, measured in world mm (KD-tree on the cartilage voxel
    centres). Only the given cartilage label is consulted, so tibial cartilage (a different
    label) can never cover the femur and vice versa.

    Limitation: the test is purely distance-based. Cartilage of the *same* label that is not
    attached to this bone but lies within the tolerance of it would count; across the
    tibiofemoral joint this would need the opposing bone surface within ~0.5 mm of this
    cartilage, i.e. the other bone's cartilage fully worn through, which a distance rule
    cannot tell apart from attached cartilage. Cartilage separated from the bone by more than
    the tolerance (a gap of two through-plane voxels already exceeds it) is not counted.

    The default tolerance is an empirical knee point rather than a derived optimum. On
    K1_0_belastet the tibial cAB is 1681 / 1931 / 2000 / 2024 / 2099 mm² at 0.40 / 0.44 /
    0.48 / 0.52 / 0.75 mm: below ~0.48 mm the covered area breaks up into pinholes (which
    :func:`total_subchondral_area` then has to fill), above it the cAB keeps growing along
    its rim. Because a distance test also reaches *sideways* past the cartilage edge, it
    gives a ~1-1.6% larger tAB than a normal probe -- covered if the label 0.4 mm out along
    the outward normal is cartilage (K1_0_belastet tibia 2027 vs 1995 mm², femur 4707 vs
    4660 mm²). The normal probe is the alternative if that rim bias matters; it depends on
    the normals and misses cartilage overhanging a sharp bone edge.

    :param bone_mesh: Bone surface from :func:`bone_surface_mesh` (world mm, triangles).
    :param image: The ``Segmentation`` holding the cartilage label (same geometry as the one
        the mesh was built from).
    :param cartilage_label: Label of the cartilage on this bone (``dfg_knees``: 3 = femoral,
        4 = tibial).
    :param tolerance_mm: Coverage distance in mm; ``None`` uses
        :func:`default_coverage_tolerance`.
    :return: ``(n_cells,)`` bool array, True for covered triangles. All False if the label is
        absent.
    """
    if tolerance_mm is None:
        tolerance_mm = default_coverage_tolerance(image)
    covered = np.zeros(bone_mesh.n_cells, dtype=bool)
    cartilage_idx = np.argwhere(np.asarray(label_array_xyz(image)) == cartilage_label)
    if len(cartilage_idx) == 0 or bone_mesh.n_cells == 0:
        return covered
    cartilage_xyz = indices_to_world(index_to_world_affine(image), cartilage_idx)
    tree = cKDTree(cartilage_xyz)
    distances, _ = tree.query(triangle_centroids(bone_mesh), k=1,
                              distance_upper_bound=tolerance_mm)
    return np.isfinite(distances)


# ----------------------------------------------------------------------------------------
# 3. Total subchondral bone area (tAB)
# ----------------------------------------------------------------------------------------

def face_components(mesh: pv.PolyData, selected: np.ndarray):
    """
    Edge-connected components of a subset of a mesh's triangles.

    Two selected triangles belong to the same component when a chain of selected triangles,
    each sharing an edge with the next, joins them (sharing only a vertex is not enough).

    :param mesh: All-triangle mesh.
    :param selected: ``(n_cells,)`` bool array, True for the triangles to group.
    :return: ``(labels, touches_boundary, areas)``: ``labels`` is an ``(n_cells,)`` int array
        (component id ``0..K-1`` per selected face, ``-1`` for unselected faces);
        ``touches_boundary[c]`` is True if component ``c`` has a face on the mesh's open
        (e.g. FOV-cut) boundary; ``areas[c]`` is the component's area.
    """
    selected = np.asarray(selected, dtype=bool)
    faces = _faces_array(mesh)
    m = len(faces)
    labels = np.full(m, -1)
    if not selected.any():
        return labels, np.zeros(0, dtype=bool), np.zeros(0)
    pairs, on_boundary = _face_adjacency(faces)
    keep = selected[pairs[:, 0]] & selected[pairs[:, 1]]
    p = pairs[keep]
    graph = sparse.coo_matrix((np.ones(len(p)), (p[:, 0], p[:, 1])), shape=(m, m))
    _, comp = connected_components(graph, directed=False)
    # Renumber the components of the selected faces 0..K-1.
    _, labels[selected] = np.unique(comp[selected], return_inverse=True)
    n = labels.max() + 1
    areas = np.bincount(labels[selected], weights=triangle_areas(mesh)[selected], minlength=n)
    touches = np.bincount(labels[selected], weights=on_boundary[selected], minlength=n) > 0
    return labels, touches, areas


def uncovered_components(bone_mesh: pv.PolyData, covered: np.ndarray):
    """
    Edge-connected components of the uncovered triangles.

    :param bone_mesh: All-triangle bone mesh.
    :param covered: ``(n_cells,)`` bool coverage (see :func:`cartilage_coverage`).
    :return: ``(labels, touches_boundary, areas)``: ``labels`` is an ``(n_cells,)`` int array
        (component id per uncovered face, ``-1`` for covered faces); ``touches_boundary[c]``
        is True if component ``c`` has a face on the mesh's open (FOV-cut) boundary;
        ``areas[c]`` is the component's area.
    """
    return face_components(bone_mesh, ~np.asarray(covered, dtype=bool))


def total_subchondral_area(bone_mesh: pv.PolyData, covered: np.ndarray,
                           max_hole_fraction: float = 0.5) -> np.ndarray:
    """
    Total area of subchondral bone (tAB): the covered area plus enclosed denuded holes.

    The uncovered triangles are split into edge-connected components. Every such component is
    bounded by covered triangles, by the mesh's open boundary, or both. A component is an
    enclosed hole -- denuded bone inside the articular surface -- when it

    - does not reach the open boundary (the FOV cut; see :func:`bone_surface_mesh`), and
    - has an area below ``max_hole_fraction`` times the covered area.

    The second rule separates holes from the non-articular remainder of the bone, which is
    one large uncovered component that also borders cAB; on a closed mesh (no FOV cut) only
    its size gives it away. It is many times the cAB area on a knee FOV, so the default 0.5
    leaves a wide margin on both sides while still accepting a denuded patch half the size of
    the remaining cartilage. Denuded bone at the *rim* of the articular surface is, by
    construction, not enclosed and not recovered -- the same limitation as any hole filling.

    Small, isolated covered islands far from the articular surface (e.g. a stray cartilage
    voxel touching non-articular bone) are kept here; deciding which covered patches belong to
    which articular surface is left to the per-bone subdivision.

    The rule knows nothing about plates or condyles: any uncovered region ringed by cartilage
    is filled. If the medial and lateral tibial cartilage touched both anteriorly and
    posteriorly, the intercondylar area between them would be enclosed and (being well below
    half the cAB) filled into tAB. The tibial subdivision therefore calls this once *per
    plate*, with only that plate's cAB as ``covered``
    (:func:`morphometry.cartilage.tibia_regions.subdivide_tibia`).

    :param bone_mesh: All-triangle bone mesh from :func:`bone_surface_mesh`.
    :param covered: ``(n_cells,)`` bool cAB flags from :func:`cartilage_coverage`.
    :param max_hole_fraction: Largest hole, as a fraction of the covered area, that is filled.
    :return: ``(n_cells,)`` bool array, True for triangles in tAB.
    """
    covered = np.asarray(covered, dtype=bool)
    if not covered.any():
        return covered.copy()
    labels, touches, areas = uncovered_components(bone_mesh, covered)
    covered_area = triangle_areas(bone_mesh)[covered].sum()
    is_hole = (~touches) & (areas < max_hole_fraction * covered_area)
    total = covered.copy()
    free = labels >= 0
    total[free] = is_hole[labels[free]]
    return total


# ----------------------------------------------------------------------------------------
# 4. Exact splitting
# ----------------------------------------------------------------------------------------

def split_mesh_by_scalar(mesh: pv.PolyData, scalars, value: float):
    """
    Split a triangle mesh exactly along the level set ``scalars == value``.

    The scalar field is given per point and interpolated linearly along every edge, so a
    triangle straddling the level is cut along the straight iso-line through it: the side
    holding a single vertex gets one triangle, the other side the remaining quadrilateral as
    two triangles. Where the level passes exactly through a vertex the cut reuses that vertex
    and the resulting zero-area pieces are dropped, so no duplicate points or spurious open
    edges appear. The two outputs therefore tile the input exactly -- total area is conserved
    to double precision -- and every piece keeps its parent's orientation (winding).

    Implemented in numpy rather than with VTK's clip filters, which interpolate the cut
    position in single precision (area errors ~1e-8 relative). Each cut point is computed once
    per mesh edge, so neighbouring pieces share it exactly and the parts stay connected.

    Cell data (e.g. coverage flags) are copied from the parent triangle to its pieces; point
    data are interpolated linearly to the new cut points (as floats).

    :param mesh: All-triangle ``pv.PolyData``.
    :param scalars: Either the name of a ``point_data`` array of ``mesh`` or an
        ``(n_points,)`` array.
    :param value: The level at which to cut.
    :return: ``(below, above)``: the parts where the scalar is ``<= value`` and ``> value``
        (an empty ``pv.PolyData`` for a side with nothing on it).
    """
    s = np.asarray(mesh.point_data[scalars] if isinstance(scalars, str) else scalars,
                   dtype=float) - value
    if s.shape != (mesh.n_points,):
        raise ValueError('scalars must have one value per mesh point')
    if mesh.n_cells == 0:
        return pv.PolyData(), pv.PolyData()
    faces = _faces_array(mesh)
    points = np.asarray(mesh.points, dtype=float)

    up = s > 0
    face_up = up[faces]
    n_up = face_up.sum(axis=1)
    mixed = np.flatnonzero((n_up == 1) | (n_up == 2))

    # Rotate each straddling triangle (cyclically, keeping its winding) so that its lone
    # vertex -- the one on its own side of the level -- comes first: (a, b, c).
    lone_up = n_up[mixed] == 1
    lone = np.where(lone_up, np.argmax(face_up[mixed], axis=1), np.argmin(face_up[mixed], axis=1))
    order = (lone[:, None] + np.arange(3)) % 3
    a, b, c = faces[mixed[:, None], order].T

    # One cut point per crossed edge, shared by both triangles on that edge.
    crossed = np.sort(np.concatenate([np.stack([a, b], 1), np.stack([a, c], 1)]), axis=1)
    edges, inverse = np.unique(crossed, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    u, v = edges[:, 0], edges[:, 1]
    # One end is > 0, the other <= 0, so the denominator never vanishes.
    t = s[u] / (s[u] - s[v])
    cut_points = points[u] + t[:, None] * (points[v] - points[u])
    # A level through a vertex (s == 0 there) cuts at that vertex itself: reuse it rather
    # than adding a coincident point, which would leave the parts with spurious open edges.
    cut_index = len(points) + np.arange(len(edges))
    cut_index = np.where(s[u] == 0, u, np.where(s[v] == 0, v, cut_index))
    p_ab = cut_index[inverse[:len(mixed)]]
    p_ac = cut_index[inverse[len(mixed):]]

    lone_tri = np.stack([a, p_ab, p_ac], axis=1)
    quad_tris = np.concatenate([np.stack([p_ab, b, c], 1), np.stack([p_ab, c, p_ac], 1)])
    quad_src = np.concatenate([mixed, mixed])
    quad_up = np.concatenate([~lone_up, ~lone_up])

    all_points = np.concatenate([points, cut_points])
    point_data = {}
    for name in mesh.point_data.keys():
        values = np.asarray(mesh.point_data[name], dtype=float)
        vu, vv = values[u], values[v]
        tt = t.reshape((-1,) + (1,) * (values.ndim - 1))
        point_data[name] = np.concatenate([values, vu + tt * (vv - vu)])

    parts = []
    for side_up in (False, True):
        whole = np.flatnonzero(n_up == (3 if side_up else 0))
        tris = np.concatenate([faces[whole], lone_tri[lone_up == side_up], quad_tris[quad_up == side_up]])
        src = np.concatenate([whole, mixed[lone_up == side_up], quad_src[quad_up == side_up]])
        # Pieces with a repeated vertex (from a cut through a vertex) have zero area: drop them.
        proper = (tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2])
        tris, src = tris[proper], src[proper]
        parts.append(_submesh(mesh, all_points, point_data, tris, src))
    return parts[0], parts[1]


def _submesh(mesh: pv.PolyData, points: np.ndarray, point_data: dict, tris: np.ndarray,
             source_faces: np.ndarray) -> pv.PolyData:
    """
    Build a compact triangle mesh from a subset of (possibly new) triangles.

    :param mesh: The parent mesh, whose cell data are copied.
    :param points: All candidate points (parent points followed by new ones).
    :param point_data: ``{name: array}`` point data aligned with ``points``.
    :param tris: ``(K, 3)`` vertex indices into ``points``.
    :param source_faces: ``(K,)`` index of the parent triangle of each output triangle.
    :return: A ``pv.PolyData`` holding only the referenced points (empty if ``K == 0``).
    """
    if len(tris) == 0:
        return pv.PolyData()
    used, local = np.unique(tris, return_inverse=True)
    local = local.reshape(-1, 3)
    out = pv.PolyData(points[used], np.hstack([np.full((len(local), 1), 3), local]).ravel())
    for name, values in point_data.items():
        out.point_data[name] = values[used]
    for name in mesh.cell_data.keys():
        out.cell_data[name] = np.asarray(mesh.cell_data[name])[source_faces]
    return out


def extract_faces(mesh: pv.PolyData, selected) -> pv.PolyData:
    """
    Compact sub-mesh of the selected triangles (cell and point data carried over).

    Unlike VTK's ``extract_cells`` this returns ``PolyData`` and keeps the selected triangles
    in their original order.

    :param mesh: All-triangle ``pv.PolyData``.
    :param selected: ``(n_cells,)`` bool mask or an integer index array of the faces to keep.
    :return: A ``pv.PolyData`` holding only the selected triangles and the points they use
        (empty if nothing is selected).
    """
    index = np.asarray(selected)
    index = np.flatnonzero(index) if index.dtype == bool else index.astype(int)
    point_data = {name: np.asarray(mesh.point_data[name]) for name in mesh.point_data.keys()}
    return _submesh(mesh, np.asarray(mesh.points, dtype=float), point_data,
                    _faces_array(mesh)[index], index)


def merge_meshes(meshes) -> pv.PolyData:
    """
    Concatenate triangle meshes without merging points, keeping their common cell data.

    The cells of the result are the cells of ``meshes[0]``, then of ``meshes[1]``, etc., in
    order; points are not deduplicated, so the parts stay disjoint. Only cell-data arrays
    present in every non-empty input are kept.

    :param meshes: An iterable of all-triangle ``pv.PolyData`` (empty ones are skipped).
    :return: The concatenated ``pv.PolyData`` (empty if every input is empty).
    """
    meshes = [m for m in meshes if m.n_cells > 0]
    if not meshes:
        return pv.PolyData()
    points, faces, offset = [], [], 0
    for m in meshes:
        points.append(np.asarray(m.points, dtype=float))
        faces.append(_faces_array(m) + offset)
        offset += m.n_points
    faces = np.concatenate(faces)
    out = pv.PolyData(np.concatenate(points), np.hstack([np.full((len(faces), 1), 3), faces]).ravel())
    names = set(meshes[0].cell_data.keys())
    for m in meshes[1:]:
        names &= set(m.cell_data.keys())
    for name in sorted(names):
        out.cell_data[name] = np.concatenate([np.asarray(m.cell_data[name]) for m in meshes])
    return out


def split_mesh_by_plane(mesh: pv.PolyData, origin, normal):
    """
    Split a triangle mesh exactly along a plane.

    Implemented as :func:`split_mesh_by_scalar` on the signed distance to the plane, which is
    linear, so the cut is exact.

    :param mesh: All-triangle ``pv.PolyData``.
    :param origin: A point on the plane (world mm).
    :param normal: The plane normal (need not be unit length).
    :return: ``(negative, positive)``: the parts on the side opposite to ``normal`` and on the
        side ``normal`` points to.
    """
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    signed = (np.asarray(mesh.points, dtype=float) - np.asarray(origin, dtype=float)) @ normal
    return split_mesh_by_scalar(mesh, signed, 0.0)


# ----------------------------------------------------------------------------------------
# 5. Region-transfer QC
# ----------------------------------------------------------------------------------------

# QC of the transfer of bone-surface regions to the cartilage voxels (the ``Tibia`` /
# ``Femur`` classes): a cartilage voxel is at most about one cartilage thickness (<= ~4 mm in
# dfg_knees) plus a voxel half-diagonal from the bone surface it belongs to; on the clean
# dfg_knees no tibial voxel is more than 5 mm from its nearest tAB triangle. Voxels farther
# than TRANSFER_FAR_MM (6 mm, a margin over that) are cartilage the tAB does not explain -- a
# plate or condyle missing from the bone coverage, cartilage label on the wrong bone, a broken
# segmentation (K7_90_belastet: 49% of its tibial voxels are > 6 mm away, up to 45 mm). More
# than TRANSFER_MAX_FAR_FRACTION of them triggers a WARNING; a few stray voxels do not.
TRANSFER_FAR_MM = 6.0
TRANSFER_MAX_FAR_FRACTION = 0.01


def transfer_qc(distances: np.ndarray, far_mm: float = TRANSFER_FAR_MM) -> dict:
    """
    Summarise how far cartilage voxels are from the tAB triangle they took their region from.

    :param distances: ``(N,)`` voxel-to-nearest-tAB-centroid distances in mm (from
        ``TibiaSubdivision.nearest_regions`` / ``FemurSubdivision.nearest_regions``).
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
