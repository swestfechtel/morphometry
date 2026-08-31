import pandas as pd
import pyvista as pv
import numpy as np
from typing import Optional, Tuple
from scipy.spatial import KDTree
from scipy.ndimage import label, generic_filter
from sklearn.cluster import KMeans
from morphometry.image_io import Image


class Tibia:
    def __init__(self, image: Image, cartilage_label: int, outlier_ratio: float = 0.1):
        self.image = image
        self.cartilage_label = cartilage_label
        self.outlier_ratio = outlier_ratio
        self.left_landmarks, self.right_landmarks = None, None
        self.center = None
        self.point_cloud = None
        self.superior_surface, self.inferior_surface = None, None
        self.clt, self.ilt, self.elt, self.alt, self.plt = None, None, None, None, None
        self.crt, self.irt, self.prt, self.ert, self.art = None, None, None, None, None

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
        Calculate the landmarks that define the different regions of the tibia.
        """
        # --- Connected Component Analysis to split into two parts ---
        kdtree = KDTree(self.point_cloud)
        n_points = self.point_cloud.shape[0]
        visited = np.zeros(n_points, dtype=bool)
        labels = np.full(n_points, -1, dtype=int)
        component_id = 0
        neighbor_radius = 2.5  # adjust as needed for your data's scale

        for idx in range(n_points):
            if not visited[idx]:
                queue = [idx]
                visited[idx] = True
                labels[idx] = component_id
                while queue:
                    current = queue.pop(0)
                    neighbors = kdtree.query_ball_point(self.point_cloud[current], r=neighbor_radius)
                    for nb in neighbors:
                        if not visited[nb]:
                            visited[nb] = True
                            labels[nb] = component_id
                            queue.append(nb)
                component_id += 1

        # Find sizes of all components
        unique, counts = np.unique(labels, return_counts=True)
        if len(unique) < 2:
            raise ValueError(f"Expected at least 2 connected components, found {len(unique)}")
        # Keep only the two largest components
        largest_two = unique[np.argsort(counts)[-2:]]
        mask = np.isin(labels, largest_two)
        filtered_points = self.point_cloud[mask]
        filtered_labels = labels[mask]
        # Remap labels to 0 and 1
        new_label_map = {old: new for new, old in enumerate(largest_two)}
        filtered_labels = np.vectorize(new_label_map.get)(filtered_labels)

        left_plate = filtered_points[filtered_labels == 0]
        right_plate = filtered_points[filtered_labels == 1]

        # Ensure left/right assignment is consistent with image orientation
        if left_plate[:, 0].mean() < right_plate[:, 0].mean():
            left_plate, right_plate = right_plate, left_plate

        left_plate_center = KMeans(n_clusters=1).fit(left_plate).cluster_centers_[0]
        right_plate_center = KMeans(n_clusters=1).fit(right_plate).cluster_centers_[0]

        left_ellipse = calculate_ellipse(left_plate, left_plate_center)
        right_ellipse = calculate_ellipse(right_plate, right_plate_center)

        left_plate_corners = get_plate_corners(left_plate)
        left_plate_corners = {'upper_right': left_plate_corners[0], 'lower_right': left_plate_corners[1], 'upper_left': left_plate_corners[2], 'lower_left': left_plate_corners[3]}
        right_plate_corners = get_plate_corners(right_plate)
        right_plate_corners = {'upper_right': right_plate_corners[0], 'lower_right': right_plate_corners[1], 'upper_left': right_plate_corners[2], 'lower_left': right_plate_corners[3]}

        self.left_landmarks = {'center': left_plate_center, 'ellipse': left_ellipse, 'corners': left_plate_corners}
        self.right_landmarks = {'center': right_plate_center, 'ellipse': right_ellipse, 'corners': right_plate_corners}
        self.center = (left_plate_center + right_plate_center) / 2

    def classify_point(self, point: np.ndarray) -> str:
        """
        Classify a point as belonging to a specific region of the tibia.
        :param point: A 1x3 point.
        :return: The label of the region the point belongs to.
        """
        if point[0] < self.center[0]:  # left side of the image, i.e. right patient side
            if np.linalg.norm(point - self.right_landmarks['center']) < self.right_landmarks['ellipse']:
                return 'cRT'

            upper_right = self.right_landmarks['corners']['upper_right']
            lower_right = self.right_landmarks['corners']['lower_right']
            upper_left = self.right_landmarks['corners']['upper_left']
            lower_left = self.right_landmarks['corners']['lower_left']

            ll_ur = upper_right - lower_left  # ac
            ul_lr = lower_right - upper_left  # db
            p_ur = upper_right - np.hstack((point[0], point[1]))  # xc
            p_lr = lower_right - np.hstack((point[0], point[1]))  # xb

            p_cross_ll_ur = np.cross(p_ur, ll_ur)  # xac
            p_cross_ul_lr = np.cross(p_lr, ul_lr)  # xdb

            if p_cross_ll_ur > 0:
                if p_cross_ul_lr > 0:
                    return 'pRT'
                else:
                    return 'eRT'
            else:
                if p_cross_ul_lr > 0:
                    return 'iRT'
                else:
                    return 'aRT'

        else:  # right side of the image, i.e. left patient side
            if np.linalg.norm(point - self.left_landmarks['center']) < self.left_landmarks['ellipse']:
                return 'cLT'

            upper_right = self.left_landmarks['corners']['upper_right']
            lower_right = self.left_landmarks['corners']['lower_right']
            upper_left = self.left_landmarks['corners']['upper_left']
            lower_left = self.left_landmarks['corners']['lower_left']

            ll_ur = upper_right - lower_left  # ac
            ul_lr = lower_right - upper_left  # db
            p_ur = upper_right - np.hstack((point[0], point[1]))  # xc
            p_lr = lower_right - np.hstack((point[0], point[1]))  # xb

            p_cross_ll_ur = np.cross(p_ur, ll_ur)  # xac
            p_cross_ul_lr = np.cross(p_lr, ul_lr)  # xdb

            if p_cross_ll_ur > 0:
                if p_cross_ul_lr > 0:
                    return 'pLT'
                else:
                    return 'iLT'
            else:
                if p_cross_ul_lr > 0:
                    return 'eLT'
                else:
                    return 'aLT'

    def extract_subregions(self):
        """
        Extract the subregions of the tibia.
        """
        clt, ilt, elt, alt, plt = list(), list(), list(), list(), list()
        crt, irt, ert, art, prt = list(), list(), list(), list(), list()

        for point in self.point_cloud:
            label = self.classify_point(point)
            if label == 'cLT':
                clt.append(point)
            elif label == 'iLT':
                ilt.append(point)
            elif label == 'eLT':
                elt.append(point)
            elif label == 'aLT':
                alt.append(point)
            elif label == 'pLT':
                plt.append(point)
            elif label == 'cRT':
                crt.append(point)
            elif label == 'iRT':
                irt.append(point)
            elif label == 'eRT':
                ert.append(point)
            elif label == 'aRT':
                art.append(point)
            elif label == 'pRT':
                prt.append(point)

        self.clt = np.array(clt)
        self.ilt = np.array(ilt)
        self.elt = np.array(elt)
        self.alt = np.array(alt)
        self.plt = np.array(plt)
        self.crt = np.array(crt)
        self.irt = np.array(irt)
        self.ert = np.array(ert)
        self.art = np.array(art)
        self.prt = np.array(prt)

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
    def __init__(self, image: Image, cartilage_label: int, outlier_ratio: float = 0.1):
        self.image = image
        self.cartilage_label = cartilage_label
        self.outlier_ratio = outlier_ratio
        self.dividing_line = None
        self.left_part, self.right_part = None, None
        self.left_cwbz, self.right_cwbz = None, None
        self.left_anterior_zone, self.right_anterior_zone = None, None
        self.left_posterior_zone, self.right_posterior_zone = None, None
        self.eclf, self.iclf, self.cclf, self.ecrf, self.icrf, self.ccrf = None, None, None, None, None, None
        self.alf, self.arf, self.plf, self.prf = None, None, None, None

        cartilage = np.where(self.image.array == self.cartilage_label, 1, 0)
        cartilage = _remove_mask_outliers(cartilage, self.outlier_ratio)
        self.point_cloud = np.argwhere(cartilage).astype(float)

        # Split the cartilage into its two condyles along the intercondylar notch. The
        # dividing axis (``x = slope * y + intercept``, x = left-right, y = anterior-
        # posterior, in voxel-index space) is fitted to the notch-gap centres across all
        # coronal slices where the cartilage separates into two condyles. This is robust
        # to the trochlea being continuous anteriorly, unlike the previous approach of
        # subtracting two independently-estimated notch points.
        slope, intercept = _femoral_condyle_dividing_line(cartilage)
        self.dividing_line = (slope, intercept)

        # A high left-right index is the image-left side, so voxels above the dividing
        # axis form the left part and the rest the right part.
        line_lr = slope * self.point_cloud[:, 1] + intercept
        self.left_part = self.point_cloud[self.point_cloud[:, 0] > line_lr]
        self.right_part = self.point_cloud[self.point_cloud[:, 0] <= line_lr]

    def _si_gap(self) -> float:
        """
        The S-I gap (in voxels) that separates the folded trochlea from the contact surface.

        Fixed at ~1.5 mm of superior-inferior clearance, converted to voxels via the S-I
        voxel spacing, so :func:`_restrict_to_contact_cluster` splits fold-from-contact
        consistently across differently-sampled scans.

        :return: The gap threshold in voxels (at least 2).
        """
        return max(2.0, 1.5 / float(self.image.spacing[2]))

    def extract_central_weightbearing_zone(self, tibia: Tibia, side: str = 'left'):
        """
        Extract the central weight-bearing zone of the cartilage.

        The zone is the femoral cartilage inside the anterior-posterior / left-right window
        of the tibial central plateau, further restricted to the superior-inferior cluster
        in contact with the tibia (see :func:`_restrict_to_contact_cluster`) so a flexed
        knee's anterior trochlea is not mistaken for weight-bearing cartilage.
        :param tibia: A Tibia object.
        :param side: The side of the tibia (patient side) to extract the central weight-bearing zone from. Can be either 'left' or 'right'.
        """
        central_tibia = list()
        internal_external_tibia = list()
        for point in tibia.point_cloud:
            label = tibia.classify_point(point)
            if label in (['iLT', 'eLT'] if side == 'left' else ['iRT', 'eRT']):
                internal_external_tibia.append(point)
            if label == ('cLT' if side == 'left' else 'cRT'):
                central_tibia.append(point)

        central_tibia = np.array(central_tibia)
        internal_external_tibia = np.array(internal_external_tibia)
        max_anterior = central_tibia[:, 1].min()  # min because anterior - posterior is low to high
        max_posterior = central_tibia[:, 1].max()
        max_left = internal_external_tibia[:, 0].min()  # refers to image side
        max_right = internal_external_tibia[:, 0].max()

        central_weightbearing_zone = self.point_cloud[self.point_cloud[:, 0] >= max_left]
        central_weightbearing_zone = central_weightbearing_zone[central_weightbearing_zone[:, 0] <= max_right]
        central_weightbearing_zone = central_weightbearing_zone[central_weightbearing_zone[:, 1] >= max_anterior]
        central_weightbearing_zone = central_weightbearing_zone[central_weightbearing_zone[:, 1] <= max_posterior]

        # The A-P/L-R window alone captures the anterior trochlea when the knee is flexed
        # (it folds into the same A-P range as the weight-bearing surface). Keep only the
        # superior-inferior cluster in contact with the tibia to exclude that fold.
        central_weightbearing_zone = _restrict_to_contact_cluster(
            central_weightbearing_zone, central_tibia[:, 2].min(), max_gap=self._si_gap())

        if side == 'left':
            self.left_cwbz = central_weightbearing_zone
        else:
            self.right_cwbz = central_weightbearing_zone

    def get_femoral_thirds(self, side: str = 'left') -> Tuple[int, int]:
        """
        Divide the central weight-bearing zone of the femoral cartilage into three subregions along the sagittal axis,
        each comprising 33% of the total volume.

        :param side: The side (patient side) of the cartilage.
        :return: The sagittal coordinates of the split points.
        """
        plate = self.left_cwbz if side == 'left' else self.right_cwbz
        lr_min = np.min(plate[:, 0])
        lr_max = np.max(plate[:, 0])
        lr_range = lr_max - lr_min
        first_split = lr_min + int(lr_range / 3)
        second_split = lr_min + 2 * int(lr_range / 3)

        points_in_first_third = list()
        points_in_second_third = list()
        num_it = 0

        while not (abs(len(points_in_first_third) / len(plate) - .33) < .02):
            if num_it > 30:
                break

            points_in_first_third = list()
            for point in plate:
                if point[0] < first_split:
                    points_in_first_third.append(point)

            if len(points_in_first_third) / len(plate) > 0.33:
                first_split -= 1
            else:
                first_split += 1

            num_it += 1

        num_it = 0

        while not (abs(len(points_in_second_third) / len(plate) - .33) < .02):
            if num_it > 30:
                break

            points_in_second_third = list()
            for point in plate:
                if first_split <= point[0] < second_split:
                    points_in_second_third.append(point)

            if len(points_in_second_third) / len(plate) > 0.33:
                second_split -= 1
            else:
                second_split += 1

            num_it += 1

        return first_split, second_split

    def extract_anterior_posterior_zones(self, side: str = 'left'):
        """
        Extract the anterior and posterior zones of the cartilage.

        The posterior zone is restricted to the superior-inferior cluster contiguous with
        the weight-bearing surface (see :func:`_restrict_to_contact_cluster`) so a flexed
        knee's folded trochlea — which falls posterior to the CWBZ in image coordinates — is
        excluded; the anterior zone genuinely is the trochlea and is left unrestricted.
        :param side: The side (patient) of the cartilage.
        """
        # split_axis = np.median(self.point_cloud[:, 0])
        # cartilage = self.point_cloud[self.point_cloud[:, 0] < split_axis] if side == 'right' else self.point_cloud[self.point_cloud[:, 0] > split_axis]
        cartilage = self.left_part if side == 'left' else self.right_part

        cwbz = self.left_cwbz if side == 'left' else self.right_cwbz
        cwbz_most_anterior = cwbz[:, 1].min()
        cwbz_most_posterior = cwbz[:, 1].max()

        anterior_zone = cartilage[cartilage[:, 1] < cwbz_most_anterior]
        posterior_zone = cartilage[cartilage[:, 1] > cwbz_most_posterior]

        # At high flexion the folded anterior trochlea also falls posterior to the CWBZ in
        # image A-P coordinates, so the posterior zone picks it up as a separate superior
        # S-I cluster. Keep only the cluster contiguous with the weight-bearing surface (the
        # anterior zone genuinely is the trochlea, so it is left untouched).
        if len(posterior_zone) > 0:
            posterior_zone = _restrict_to_contact_cluster(
                posterior_zone, float(cwbz[:, 2].mean()), max_gap=self._si_gap())

        if side == 'left':
            self.left_anterior_zone = anterior_zone
            self.left_posterior_zone = posterior_zone
        else:
            self.right_anterior_zone = anterior_zone
            self.right_posterior_zone = posterior_zone

    def extract_subregions(self):
        """
        Extract the subregions of the femoral cartilage.
        """
        alf, arf, plf, prf = list(), list(), list(), list()
        eclf, iclf, cclf, ecrf, icrf, ccrf = list(), list(), list(), list(), list(), list()

        for point in self.left_anterior_zone:
            alf.append(point)

        for point in self.right_anterior_zone:
            arf.append(point)

        for point in self.left_posterior_zone:
            plf.append(point)

        for point in self.right_posterior_zone:
            prf.append(point)

        first_split, second_split = self.get_femoral_thirds(side='left')
        for point in self.left_cwbz:
            if point[0] < first_split:
                iclf.append(point)
            elif first_split <= point[0] < second_split:
                cclf.append(point)
            else:
                eclf.append(point)

        first_split, second_split = self.get_femoral_thirds(side='right')
        for point in self.right_cwbz:
            if point[0] < first_split:
                ecrf.append(point)
            elif first_split <= point[0] < second_split:
                ccrf.append(point)
            else:
                icrf.append(point)

        self.alf = np.array(alf)
        self.arf = np.array(arf)
        self.plf = np.array(plf)
        self.prf = np.array(prf)
        self.eclf = np.array(eclf)
        self.iclf = np.array(iclf)
        self.cclf = np.array(cclf)
        self.ecrf = np.array(ecrf)
        self.icrf = np.array(icrf)
        self.ccrf = np.array(ccrf)


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
        for i, distance in enumerate(distances):
            thicknesses[(inferior_surface[i][0], inferior_surface[i][1])] = distance

        return thicknesses

    def calculate_thickness(self, tibia: Tibia, method: str = 'mesh') -> dict:
        """
        Calculate the cartilage thickness for all zones (subregions).

        :param tibia: A Tibia object.
        :param method: The method used for thickness calculation. Can be either 'mesh' or 'knn'.
        :return: A dictionary where keys are zone labels and values are dictionaries, where keys are coordinates and values are thicknesses.
        """
        assert method in ['mesh', 'knn'], 'Method must be either "mesh" or "knn".'
        self.extract_central_weightbearing_zone(tibia, side='left')
        self.extract_central_weightbearing_zone(tibia, side='right')
        self.extract_anterior_posterior_zones(side='left')
        self.extract_anterior_posterior_zones(side='right')

        thicknesses = dict()
        for zone in ['left_cwbz', 'right_cwbz', 'left_posterior_zone', 'right_posterior_zone', 'left_anterior_zone', 'right_anterior_zone']:
            if zone in ['left_posterior_zone', 'right_posterior_zone']:
                tmp = getattr(self, zone).copy()
                tmp[:, 1], tmp[:, 2] = tmp[:, 2], tmp[:, 1].copy()  # rotate to allow extraction of anterior and posterior surface
                superior_surface, inferior_surface = get_superior_and_inferior_surface_points(tmp)
                superior_surface[:, 1], superior_surface[:, 2] = superior_surface[:, 2], superior_surface[:, 1].copy()  # rotate back
                inferior_surface[:, 1], inferior_surface[:, 2] = inferior_surface[:, 2], inferior_surface[:, 1].copy()
            else:
                superior_surface, inferior_surface = get_superior_and_inferior_surface_points(getattr(self, zone))

            if zone == 'left_cwbz':  # remember: left & right = patient side
                first_split, second_split = self.get_femoral_thirds(side='left')
                iclf_superior = superior_surface[superior_surface[:, 0] < first_split]
                iclf_inferior = inferior_surface[inferior_surface[:, 0] < first_split]
                cclf_superior = superior_surface[superior_surface[:, 0] >= first_split]
                cclf_superior = cclf_superior[cclf_superior[:, 0] < second_split]
                cclf_inferior = inferior_surface[inferior_surface[:, 0] >= first_split]
                cclf_inferior = cclf_inferior[cclf_inferior[:, 0] < second_split]
                eclf_superior = superior_surface[superior_surface[:, 0] >= second_split]
                eclf_inferior = inferior_surface[inferior_surface[:, 0] >= second_split]

                if method == 'mesh':
                    thicknesses['iclf'] = self.mesh_method(iclf_superior, iclf_inferior)
                    thicknesses['cclf'] = self.mesh_method(cclf_superior, cclf_inferior)
                    thicknesses['eclf'] = self.mesh_method(eclf_superior, eclf_inferior)
                else:
                    thicknesses['iclf'] = self.knn_method(iclf_superior, iclf_inferior)
                    thicknesses['cclf'] = self.knn_method(cclf_superior, cclf_inferior)
                    thicknesses['eclf'] = self.knn_method(eclf_superior, eclf_inferior)
            elif zone == 'right_cwbz':
                first_split, second_split = self.get_femoral_thirds(side='right')
                icrf_superior = superior_surface[superior_surface[:, 0] >= second_split]
                icrf_inferior = inferior_surface[inferior_surface[:, 0] >= second_split]
                ccrf_superior = superior_surface[superior_surface[:, 0] >= first_split]
                ccrf_superior = ccrf_superior[ccrf_superior[:, 0] < second_split]
                ccrf_inferior = inferior_surface[inferior_surface[:, 0] >= first_split]
                ccrf_inferior = ccrf_inferior[ccrf_inferior[:, 0] < second_split]
                ecrf_superior = superior_surface[superior_surface[:, 0] < first_split]
                ecrf_inferior = inferior_surface[inferior_surface[:, 0] < first_split]

                if method == 'mesh':
                    thicknesses['icrf'] = self.mesh_method(icrf_superior, icrf_inferior)
                    thicknesses['ccrf'] = self.mesh_method(ccrf_superior, ccrf_inferior)
                    thicknesses['ecrf'] = self.mesh_method(ecrf_superior, ecrf_inferior)
                else:
                    thicknesses['icrf'] = self.knn_method(icrf_superior, icrf_inferior)
                    thicknesses['ccrf'] = self.knn_method(ccrf_superior, ccrf_inferior)
                    thicknesses['ecrf'] = self.knn_method(ecrf_superior, ecrf_inferior)
            else:
                if method == 'mesh':
                    thicknesses[zone] = self.mesh_method(superior_surface, inferior_surface)
                else:
                    thicknesses[zone] = self.knn_method(superior_surface, inferior_surface)

        return thicknesses

    def _ensure_subregions(self, tibia: Tibia = None):
        """
        Lazily compute the femoral subregion voxel clouds needed for visualisation.

        The femoral subregions are defined relative to the tibial plateau, so a
        ``Tibia`` is required the first time they are computed (its landmarks are
        computed on demand if missing). Idempotent once the subregions exist.

        :param tibia: A ``Tibia`` instance for the same knee. Required only if the
            femoral subregions have not been extracted yet.
        :raises ValueError: If the subregions are not yet available and no ``tibia``
            is provided.
        """
        if self.alf is not None:
            return
        if tibia is None:
            raise ValueError(
                'Femoral subregions require a Tibia (they are defined relative to the '
                'tibial plateau); pass tibia=... to the plotting method.'
            )
        if tibia.point_cloud is None:
            tibia.get_surface_points()
        if tibia.left_landmarks is None:
            tibia.calculate_landmarks()
        self.extract_central_weightbearing_zone(tibia, side='left')
        self.extract_central_weightbearing_zone(tibia, side='right')
        self.extract_anterior_posterior_zones(side='left')
        self.extract_anterior_posterior_zones(side='right')
        self.extract_subregions()

    def plot_segments(self, tibia: Tibia = None, plotter: pv.Plotter = None,
                      show: bool = None) -> pv.Plotter:
        """
        Render the raw voxel volume of every femoral subregion as coloured cube
        glyphs in a single 3D scene, one distinct colour per subregion.

        :param tibia: A ``Tibia`` for the same knee (needed to derive the subregions
            unless they have already been extracted).
        :param plotter: An existing PyVista ``Plotter`` to draw into (new one if ``None``).
        :param show: Whether to call ``plotter.show()`` (see :meth:`Tibia.plot_segments`).
        :return: The ``Plotter`` the segments were drawn into.
        """
        created = plotter is None
        if plotter is None:
            plotter = pv.Plotter()
        self._ensure_subregions(tibia)

        drawn = False
        for attr, label, color, _ in FEMUR_SUBREGIONS:
            drawn |= _add_voxel_glyphs(plotter, self.image, getattr(self, attr), color, label)

        if drawn:
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
        :param tibia: A ``Tibia`` for the same knee (needed to derive the subregions
            unless they have already been extracted).
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
        self._ensure_subregions(tibia)

        meshes = []
        for attr, label, color, thickness_key in FEMUR_SUBREGIONS:
            tmap = thicknesses.get(thickness_key)
            if not tmap:
                continue
            swap_yz = thickness_key in ('left_posterior_zone', 'right_posterior_zone')
            surf = _thickness_surface(self.image, getattr(self, attr), tmap, swap_yz=swap_yz)
            if surf is not None:
                meshes.append(surf)

        _add_thickness_meshes(plotter, meshes, cmap, clim, show_scalar_bar)
        if show or (show is None and created):
            plotter.show()
        return plotter


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


def _femoral_condyle_dividing_line(cartilage: np.ndarray, min_size: int = 40) -> Tuple[float, float]:
    """
    Fit the anterior-posterior axis that separates the two femoral condyles.

    The femoral cartilage is a horseshoe: the two condyles are separated posteriorly by
    the intercondylar notch but join anteriorly at the trochlea. For every coronal slice
    (fixed anterior-posterior index) that splits into exactly two connected components
    (the two condyles), the left-right centre of the gap between them is recorded. A
    least-squares line is fitted through these notch centres, giving the dividing axis in
    voxel-index space as ``x = slope * y + intercept`` where ``x`` is the left-right axis
    (axis 0) and ``y`` is the anterior-posterior axis (axis 1).

    This is robust because it aggregates evidence over many slices and relies only on the
    posterior region where the condyles are genuinely separate; it does not depend on a
    (non-existent) anterior notch. Spurious notch centres (e.g. from a stray anterior
    two-component slice) are rejected with a median-absolute-deviation band before fitting.

    :param cartilage: A binary 3-D mask of the femoral cartilage in LPI index convention
        (axis 0 = left-right, axis 1 = anterior-posterior, axis 2 = superior-inferior).
    :param min_size: Minimum voxel count for a connected component in a slice to count as
        a condyle (filters small speckle).
    :return: ``(slope, intercept)`` of the dividing axis ``x = slope * y + intercept``.
        Falls back to a vertical (``slope = 0``) mid-sagittal axis when no slice separates
        into two condyles.
    """
    centers_y, centers_x = [], []
    for c in range(cartilage.shape[1]):
        labeled, n = label(cartilage[:, c, :])
        if n < 2:
            continue
        sizes = np.bincount(labeled.ravel())
        kept = [lab for lab in range(1, n + 1) if sizes[lab] >= min_size]
        if len(kept) != 2:
            continue
        components = [np.argwhere(labeled == lab) for lab in kept]
        components.sort(key=lambda comp: comp[:, 0].mean())
        gap_center = (components[0][:, 0].max() + components[1][:, 0].min()) / 2.0
        centers_y.append(float(c))
        centers_x.append(float(gap_center))

    if len(centers_x) == 0:
        # No coronal slice separates into two condyles; fall back to the mid-sagittal
        # plane through the cartilage's left-right median.
        return 0.0, float(np.median(np.argwhere(cartilage)[:, 0]))

    centers_y = np.asarray(centers_y)
    centers_x = np.asarray(centers_x)

    # Reject outlier notch centres before fitting so a stray slice cannot tilt the axis.
    median = np.median(centers_x)
    mad = np.median(np.abs(centers_x - median))
    if mad > 0:
        inliers = np.abs(centers_x - median) <= 5.0 * mad
        if inliers.sum() >= 2:
            centers_y, centers_x = centers_y[inliers], centers_x[inliers]

    if len(centers_x) < 2:
        return 0.0, float(centers_x[0])

    slope, intercept = np.polyfit(centers_y, centers_x, 1)
    return float(slope), float(intercept)


def _restrict_to_contact_cluster(points: np.ndarray, reference_si: float,
                                 max_gap: float = 3.0) -> np.ndarray:
    """
    Keep only the superior-inferior cluster of points nearest a reference S-I level.

    The femoral central weight-bearing / posterior zones are selected by an anterior-
    posterior (and left-right) window. At high knee flexion the curved femoral cartilage
    folds back on itself, so that window intersects the cartilage at two separate
    superior-inferior levels: the true weight-bearing surface (in contact with the tibia)
    and the anterior trochlea, which has rotated into the same A-P range. This helper
    removes the fold by clustering the candidate points along the S-I axis (axis 2) —
    splitting wherever there is an S-I gap larger than ``max_gap`` voxels — and keeping only
    the cluster closest to the tibial contact level. At low flexion the candidates form a
    single cluster and are returned unchanged.

    ``max_gap`` is an *absolute* voxel gap (the caller scales it by the S-I voxel spacing):
    the two surfaces are separated by an anatomical clearance of a few voxels, whereas a
    genuine single contact band is S-I-contiguous. A span-relative threshold would fail
    here, because a *distant* fold inflates the span and pushes the threshold above the
    real (small) valley gap between the fold and the contact surface. This assumes the true
    zone is itself S-I-contiguous within ``max_gap``; a genuine internal S-I gap larger than
    that would trim the zone to the reference-side sub-cluster.

    :param points: An ``Nx3`` array of candidate voxels (LPI index space).
    :param reference_si: The tibial contact S-I level (axis 2) the true zone sits against,
        e.g. the most-superior tibial central-zone voxel, or the CWBZ's mean S-I level.
    :param max_gap: The maximum S-I gap (in voxels) within one cluster; larger gaps split.
    :return: The subset of ``points`` belonging to the S-I cluster nearest ``reference_si``.
    """
    if len(points) == 0:
        return points
    si_sorted = np.sort(points[:, 2])

    tolerance = max(2.0, max_gap)
    labels = np.zeros(len(si_sorted), dtype=int)
    for boundary in np.where(np.diff(si_sorted) > tolerance)[0]:
        labels[boundary + 1:] += 1

    best_label, best_distance = 0, np.inf
    for label_id in np.unique(labels):
        distance = np.min(np.abs(si_sorted[labels == label_id] - reference_si))
        if distance < best_distance:
            best_distance, best_label = distance, label_id

    kept = si_sorted[labels == best_label]
    return points[(points[:, 2] >= kept.min()) & (points[:, 2] <= kept.max())]


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


def plot_thinnest_point(tibia: 'Tibia', femur: 'Femur', tibia_thickness: dict,
                        femur_thickness: dict, thinnest: dict, plotter: pv.Plotter = None,
                        show: bool = None, base: str = 'thickness') -> pv.Plotter:
    """
    Render the knee cartilage in 3D and mark the thinnest point with an arrow.

    :param tibia: The ``Tibia`` for the knee.
    :param femur: The ``Femur`` for the same knee.
    :param tibia_thickness: The tibial thickness dict (used when ``base='thickness'``).
    :param femur_thickness: The femoral thickness dict (used when ``base='thickness'``).
    :param thinnest: The result dict from :func:`find_thinnest_area` (its ``'point'`` is marked).
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

    marker = _thinnest_point_physical(tibia, femur, thinnest['point'])

    # Point an arrow at the marker from an oblique offset scaled to the scene size.
    b = plotter.bounds
    diagonal = float(np.linalg.norm([b[1] - b[0], b[3] - b[2], b[5] - b[4]]))
    length = 0.2 * diagonal if diagonal > 0 else 1.0
    direction = np.array([1.0, 0.0, 1.0])
    direction /= np.linalg.norm(direction)
    start = marker - direction * length
    arrow = pv.Arrow(start=start, direction=direction, scale=length)
    plotter.add_mesh(arrow, color='red')
    plotter.add_point_labels(np.array([marker]), ['thinnest'], point_color='red',
                             point_size=12, font_size=14, text_color='red',
                             always_visible=True)

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


def calculate_ellipse(points: np.ndarray, center: float) -> float:
    """
    Calculate an ellipse around a center point that covers ~20% of the points.

    :param points: A point cloud.
    :param center: The center of mass of the point cloud.
    :return: The radius of the ellipse.
    """
    r = 20.  # initial guess
    max_iter = 100  # stop condition
    num_points = len(points)
    quintile = int(num_points * 0.2)

    points_in_ellipse = np.array([])
    i = 0
    while (len(points_in_ellipse) < quintile) and (i < max_iter):
        points_in_ellipse = points[np.linalg.norm(points - center, axis=1) < r]

        if len(points_in_ellipse) < quintile:
            r += .5
        else:
            r /= 2.

        i += 1

    return r


def get_plate_corners(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the corners (2D) of a plate defined by a point cloud.

    :param points: A point cloud.
    :return: The corners (image orientation: axial view; left image side = right patient side,
    upper image half = posterior) of the plate.
    """
    left_image_boundary = points[:, 0].min()  # right patient side
    right_image_boundary = points[:, 0].max()  # left patient side
    upper_image_boundary = points[:, 1].max()  # posterior
    lower_image_boundary = points[:, 1].min()  # anterior

    upper_right = np.array([right_image_boundary, upper_image_boundary])
    lower_right = np.array([right_image_boundary, lower_image_boundary])
    upper_left = np.array([left_image_boundary, upper_image_boundary])
    lower_left = np.array([left_image_boundary, lower_image_boundary])

    return upper_right, lower_right, upper_left, lower_left


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
                       swap_yz: bool = False) -> pv.PolyData:
    """
    Reconstruct a subregion's articular (superior) surface and attach thickness values.

    The superior surface points are recovered from the subregion voxels with the same
    grouping used during thickness computation, then each surface point is coloured by
    the thickness measured at its ``(x, y)`` location. Points without a thickness entry
    (or ``NaN``) are dropped.

    :param image: The ``Image`` providing the index->physical transform.
    :param points: An ``Nx3`` array of the subregion's voxel indices.
    :param thickness_map: ``{(x, y): thickness_mm}`` for this subregion.
    :param swap_yz: If ``True``, swap the y/z axes before extracting the superior
        surface (matching the rotated extraction used for the femoral posterior zones,
        whose articular surface faces posteriorly rather than superiorly).
    :return: A triangulated ``PolyData`` surface carrying a point-scalar ``'thickness'``,
        or ``None`` if fewer than three points could be coloured.
    """
    if points is None or len(points) == 0:
        return None
    pts = np.asarray(points, dtype=float)
    if swap_yz:
        pts = pts.copy()
        pts[:, [1, 2]] = pts[:, [2, 1]]
    superior, _ = get_superior_and_inferior_surface_points(pts)
    if swap_yz:
        superior = superior.copy()
        superior[:, [1, 2]] = superior[:, [2, 1]]

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
