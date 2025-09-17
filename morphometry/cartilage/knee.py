import pandas as pd
import pyvista as pv
import numpy as np
from typing import Tuple
from scipy.spatial import KDTree
from scipy.ndimage import center_of_mass
from sklearn.cluster import KMeans, DBSCAN
from morphometry.image_io import Image
from morphometry.utils import num_connected_components
from matplotlib import pyplot as plt


class Tibia:
    def __init__(self, image: Image, cartilage_label: int):
        self.image = image
        self.cartilage_label = cartilage_label
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


class Femur:
    def __init__(self, image: Image, cartilage_label: int):
        self.image = image
        self.cartilage_label = cartilage_label
        self.left_part, right_part = None, None
        self.left_cwbz, self.right_cwbz = None, None
        self.left_anterior_zone, self.right_anterior_zone = None, None
        self.left_posterior_zone, self.right_posterior_zone = None, None
        self.eclf, self.iclf, self.cclf, self.ecrf, self.icrf, self.ccrf = None, None, None, None, None, None
        self.alf, self.arf, self.plf, self.prf = None, None, None, None

        cartilage = np.where(self.image.array == self.cartilage_label, 1, 0)
        self.point_cloud = np.argwhere(cartilage).astype(float)

        c = np.max(self.point_cloud[:, 1])  # max posterior extent
        c = int(c)
        n = num_connected_components(cartilage[:, c, :], min_size=40)
        while n != 2:
            c -= 1
            n = num_connected_components(cartilage[:, c, :], min_size=40)

        while n != 1:
            c -= 1
            n = num_connected_components(cartilage[:, c, :], min_size=40)

        # fig, ax = plt.subplots(ncols=2, figsize=(20, 10))
        # ax[0].imshow(cartilage[:, c, :].T)

        pts = np.argwhere(cartilage[:, c, :])
        mean_s = int(np.mean(pts[:, 0]))
        tmp = pts[pts[:, 0] == mean_s]
        min_t = min(tmp[:, 0])
        notch_p = np.array([mean_s, c, min_t])

        c = np.min(self.point_cloud[:, 1])  # max anterior extent
        c = int(c)
        n = num_connected_components(cartilage[:, c, :], min_size=40)
        while n != 2:
            c += 1
            n = num_connected_components(cartilage[:, c, :], min_size=40)

        while n != 1:
            c += 1
            n = num_connected_components(cartilage[:, c, :], min_size=40)

        """
        ax[0].imshow(cartilage[:, c - 1, :].T)
        ax[1].imshow(cartilage[:, c, :].T)
        fig.show()
        plt.close(fig)
        """

        pts = np.argwhere(cartilage[:, c, :])
        mean_s = int(np.mean(pts[:, 0]))
        tmp = pts[pts[:, 0] == mean_s]
        min_t = min(tmp[:, 0])

        # notch_a = center_of_mass(cartilage[:, c, :])
        # notch_a = np.array([notch_a[0], c, notch_a[1]])
        notch_a = np.array([mean_s + 10, c, min_t])

        dividing_vector = notch_a - notch_p
        print(notch_a, notch_p, dividing_vector)

        self.left_part = list()
        self.right_part = list()

        for point in self.point_cloud:
            tmp = notch_a[:2] - point[:2]
            tmp = np.cross(tmp, dividing_vector[:2])
            if tmp > 0:
                self.left_part.append(point)
            else:
                self.right_part.append(point)

        self.left_part = np.array(self.left_part)
        self.right_part = np.array(self.right_part)

    def extract_central_weightbearing_zone(self, tibia: Tibia, side: str = 'left'):
        """
        Extract the central weight-bearing zone of the cartilage.
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
        :param side: The side (patient) of the cartilage.
        """
        # split_axis = np.median(self.point_cloud[:, 0])
        # cartilage = self.point_cloud[self.point_cloud[:, 0] < split_axis] if side == 'right' else self.point_cloud[self.point_cloud[:, 0] > split_axis]
        cartilage = self.left_part if side == 'left' else self.right_part

        cwbz = self.left_cwbz if side == 'left' else self.right_cwbz
        cwbz_most_anterior = cwbz[:, 1].min()
        cwbz_most_posterior = cwbz[:, 1].max()

        if side == 'left':
            self.left_anterior_zone = cartilage[cartilage[:, 1] < cwbz_most_anterior]
            self.left_posterior_zone = cartilage[cartilage[:, 1] > cwbz_most_posterior]
        else:
            self.right_anterior_zone = cartilage[cartilage[:, 1] < cwbz_most_anterior]
            self.right_posterior_zone = cartilage[cartilage[:, 1] > cwbz_most_posterior]

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
