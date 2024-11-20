import SimpleITK as sitk
import pandas as pd
import pyvista as pv
import numpy as np
from typing import Tuple
from scipy.spatial import KDTree
from sklearn.cluster import KMeans


class Tibia:
    def __init__(self, image: sitk.Image, cartilage_label: int):
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
        image_array = sitk.GetArrayFromImage(self.image)
        image_array = np.swapaxes(image_array, 0, 1)  # assuming original ordering is sagittal, axial, coronal
        cartilage = np.where(image_array == self.cartilage_label, 1, 0)
        cartilage = np.argwhere(cartilage)
        self.point_cloud = cartilage.astype(float)
        self.superior_surface, self.inferior_surface = get_superior_and_inferior_surface_points(cartilage)

    def calculate_landmarks(self):
        """
        Calculate the landmarks that define the different regions of the tibia.
        """
        cluster = KMeans(n_clusters=1).fit(self.point_cloud)
        self.center = cluster.cluster_centers_[0]

        left_plate = self.point_cloud[self.point_cloud[:, 1] > self.center[1]]  # left side = left patient side, i.e. right side of image
        right_plate = self.point_cloud[self.point_cloud[:, 1] <= self.center[1]]  # right side = right patient side, i.e. left side of image

        left_plate_center = KMeans(n_clusters=1).fit(left_plate).cluster_centers_[0]
        right_plate_center = KMeans(n_clusters=1).fit(right_plate).cluster_centers_[0]

        left_ellipse = calculate_ellipse(left_plate, left_plate_center)
        right_ellipse = calculate_ellipse(right_plate, right_plate_center)

        left_plate_corners = get_plate_corners(left_plate)  # upper right (LP), lower right (LA), upper left (RP), lower left (RA)
        left_plate_corners = {'upper_right': left_plate_corners[0], 'lower_right': left_plate_corners[1], 'upper_left': left_plate_corners[2], 'lower_left': left_plate_corners[3]}
        right_plate_corners = get_plate_corners(right_plate)
        right_plate_corners = {'upper_right': right_plate_corners[0], 'lower_right': right_plate_corners[1], 'upper_left': right_plate_corners[2], 'lower_left': right_plate_corners[3]}

        self.left_landmarks = {'center': left_plate_center, 'ellipse': left_ellipse, 'corners': left_plate_corners}
        self.right_landmarks = {'center': right_plate_center, 'ellipse': right_ellipse, 'corners': right_plate_corners}

    def classify_point(self, point) -> str:
        """
        Classify a point as belonging to a specific region of the tibia.
        :param point: A 1x3 point.
        :return: The label of the region the point belongs to.
        """
        if point[1] < self.center[1]:  # left side of the image, i.e. right patient side
            if np.linalg.norm(point - self.right_landmarks['center']) < self.right_landmarks['ellipse']:
                return 'cRT'

            upper_right = self.right_landmarks['corners']['upper_right']
            lower_right = self.right_landmarks['corners']['lower_right']
            upper_left = self.right_landmarks['corners']['upper_left']
            lower_left = self.right_landmarks['corners']['lower_left']

            ll_ur = upper_right - lower_left  # ac
            ul_lr = lower_right - upper_left  # db
            p_ur = upper_right - point[1:]  # xc
            p_lr = lower_right - point[1:]  # xb

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
            p_ur = upper_right - point[1:]  # xc
            p_lr = lower_right - point[1:]  # xb

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

    def mesh_method(self, superior_surface, inferior_surface) -> dict:
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
            dist = np.linalg.norm(iv - point) * self.image.GetSpacing()[
                1]  # spacing[1] because image orientation is still sagittal, axial, coronal
            thicknesses[(point[1], point[2])] = dist  # discard axial coordinate

        return thicknesses

    def knn_method(self, superior_surface, inferior_surface) -> dict:
        """
        Calculate the thickness of a cartilage using a k-nearest neighbour method.

        :param superior_surface: The superior surface of the zone.
        :param inferior_surface: The inferior surface of the zone.
        :return: A dictionary where keys are coordinates and values are the thicknesses.
        """
        thicknesses = dict()
        superior_tree = KDTree(superior_surface)
        distances, indices = superior_tree.query(inferior_surface, k=1)
        for i, distance in enumerate(distances):
            thicknesses[(inferior_surface[i][1], inferior_surface[i][2])] = distance * self.image.GetSpacing()[
                1]  # spacing[1] because image orientation is still sagittal, axial, coronal

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
    def __init__(self, image: sitk.Image, cartilage_label: int):
        self.image = image
        self.cartilage_label = cartilage_label
        self.left_cwbz, self.right_cwbz = None, None
        self.left_anterior_zone, self.right_anterior_zone = None, None
        self.left_posterior_zone, self.right_posterior_zone = None, None

        image_array = sitk.GetArrayFromImage(self.image)
        image_array = np.swapaxes(image_array, 0, 1)  # assuming original ordering is sagittal, axial, coronal
        cartilage = np.where(image_array == self.cartilage_label, 1, 0)
        cartilage = np.argwhere(cartilage)
        self.point_cloud = cartilage.astype(float)

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
        max_anterior = central_tibia[:, 2].min()  # min because anterior - posterior is low to high
        max_posterior = central_tibia[:, 2].max()
        max_left = internal_external_tibia[:, 1].min()  # refers to image side
        max_right = internal_external_tibia[:, 1].max()

        central_weightbearing_zone = self.point_cloud[self.point_cloud[:, 1] >= max_left]
        central_weightbearing_zone = central_weightbearing_zone[central_weightbearing_zone[:, 1] <= max_right]
        central_weightbearing_zone = central_weightbearing_zone[central_weightbearing_zone[:, 2] >= max_anterior]
        central_weightbearing_zone = central_weightbearing_zone[central_weightbearing_zone[:, 2] <= max_posterior]

        if side == 'left':
            self.left_cwbz = central_weightbearing_zone
        else:
            self.right_cwbz = central_weightbearing_zone

    def get_femoral_thirds(self, side: str = 'left') -> [int, int]:
        """
        Divide the central weight-bearing zone of the femoral cartilage into three subregions along the sagittal axis,
        each comprising 33% of the total volume.

        :param side: The side (patient side) of the cartilage.
        :return: The sagittal coordinates of the split points.
        """
        plate = self.left_cwbz if side == 'left' else self.right_cwbz
        ymin = np.min(plate[:, 1])
        ymax = np.max(plate[:, 1])
        yrange = ymax - ymin
        first_split = ymin + int(yrange / 3)
        second_split = ymin + 2 * int(yrange / 3)

        points_in_first_third = list()
        points_in_second_third = list()
        num_it = 0

        while not (abs(len(points_in_first_third) / len(plate) - .33) < .02):
            if num_it > 30:
                break

            points_in_first_third = list()
            for point in plate:
                if point[1] < first_split:
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
                if first_split <= point[1] < second_split:
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
        split_axis = np.median(self.point_cloud[:, 1])
        cartilage = self.point_cloud[self.point_cloud[:, 1] < split_axis] if side == 'right' else self.point_cloud[self.point_cloud[:, 1] > split_axis]
        cwbz = self.left_cwbz if side == 'left' else self.right_cwbz
        cwbz_most_anterior = cwbz[:, 2].min()
        cwbz_most_posterior = cwbz[:, 2].max()

        if side == 'left':
            self.left_anterior_zone = cartilage[cartilage[:, 2] < cwbz_most_anterior]
            self.left_posterior_zone = cartilage[cartilage[:, 2] > cwbz_most_posterior]
        else:
            self.right_anterior_zone = cartilage[cartilage[:, 2] < cwbz_most_anterior]
            self.right_posterior_zone = cartilage[cartilage[:, 2] > cwbz_most_posterior]

    def mesh_method(self, superior_surface: np.array, inferior_surface: np.array) -> dict:
        """
        Calculate the thickness of a zone using a mesh-based ray tracing method.
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
            dist = np.linalg.norm(iv - point) * self.image.GetSpacing()[
                1]  # spacing[1] because image orientation is still sagittal, axial, coronal
            thicknesses[(point[1], point[2])] = dist  # discard axial coordinate

        return thicknesses

    def calculate_thickness(self, tibia: Tibia) -> dict:
        """
        Calculate the cartilage thickness for all zones (subregions).

        :param tibia: A Tibia object.
        :return: A dictionary where keys are zone labels and values are dictionaries, where keys are coordinates and values are thicknesses.
        """
        self.extract_central_weightbearing_zone(tibia, side='left')
        self.extract_central_weightbearing_zone(tibia, side='right')
        self.extract_anterior_posterior_zones(side='left')
        self.extract_anterior_posterior_zones(side='right')

        thicknesses = dict()
        for zone in ['left_cwbz', 'right_cwbz', 'left_posterior_zone', 'right_posterior_zone', 'left_anterior_zone', 'right_anterior_zone']:
            if zone in ['left_posterior_zone', 'right_posterior_zone']:
                tmp = getattr(self, zone).copy()
                tmp[:, 2], tmp[:, 0] = tmp[:, 0], tmp[:, 2].copy()  # rotate to allow extraction of anterior and posterior surface
                superior_surface, inferior_surface = get_superior_and_inferior_surface_points(tmp)
                superior_surface[:, 2], superior_surface[:, 0] = superior_surface[:, 0], superior_surface[:, 2].copy()  # rotate back
                inferior_surface[:, 2], inferior_surface[:, 0] = inferior_surface[:, 0], inferior_surface[:, 2].copy()
            else:
                superior_surface, inferior_surface = get_superior_and_inferior_surface_points(getattr(self, zone))

            if zone == 'left_cwbz':  # remember: left & right = patient side
                first_split, second_split = self.get_femoral_thirds(side='left')
                iclf_superior = superior_surface[superior_surface[:, 1] < first_split]
                iclf_inferior = inferior_surface[inferior_surface[:, 1] < first_split]
                cclf_superior = superior_surface[superior_surface[:, 1] >= first_split]
                cclf_superior = cclf_superior[cclf_superior[:, 1] < second_split]
                cclf_inferior = inferior_surface[inferior_surface[:, 1] >= first_split]
                cclf_inferior = cclf_inferior[cclf_inferior[:, 1] < second_split]
                eclf_superior = superior_surface[superior_surface[:, 1] >= second_split]
                eclf_inferior = inferior_surface[inferior_surface[:, 1] >= second_split]

                thicknesses['iclf'] = self.mesh_method(iclf_superior, iclf_inferior)
                thicknesses['cclf'] = self.mesh_method(cclf_superior, cclf_inferior)
                thicknesses['eclf'] = self.mesh_method(eclf_superior, eclf_inferior)
            elif zone == 'right_cwbz':
                first_split, second_split = self.get_femoral_thirds(side='right')
                icrf_superior = superior_surface[superior_surface[:, 1] >= second_split]
                icrf_inferior = inferior_surface[inferior_surface[:, 1] >= second_split]
                ccrf_superior = superior_surface[superior_surface[:, 1] >= first_split]
                ccrf_superior = ccrf_superior[ccrf_superior[:, 1] < second_split]
                ccrf_inferior = inferior_surface[inferior_surface[:, 1] >= first_split]
                ccrf_inferior = ccrf_inferior[ccrf_inferior[:, 1] < second_split]
                ecrf_superior = superior_surface[superior_surface[:, 1] < first_split]
                ecrf_inferior = inferior_surface[inferior_surface[:, 1] < first_split]

                thicknesses['icrf'] = self.mesh_method(icrf_superior, icrf_inferior)
                thicknesses['ccrf'] = self.mesh_method(ccrf_superior, ccrf_inferior)
                thicknesses['ecrf'] = self.mesh_method(ecrf_superior, ecrf_inferior)
            else:
                thicknesses[zone] = self.mesh_method(superior_surface, inferior_surface)

        return thicknesses


def build_cartilage_meshes(superior_points: np.array, inferior_points: np.array) -> Tuple[pv.PolyData, pv.PolyData]:
    """
    Build superior and inferior cartilage surface meshes from superior and inferior surface points.
    Points (x, y, z) are expected to be x = axial, y = sagittal, z = coronal.

    :param superior_points: A Nx3 point cloud representation of superior surface points.
    :param inferior_points: A Nx3 point cloud representation of inferior surface points.
    :return: Reconstructed surface meshes of the superior and inferior cartilage.
    """
    superior_mesh = pv.PolyData(superior_points)
    inferior_mesh = pv.PolyData(inferior_points)

    superior_mesh = superior_mesh.delaunay_2d(alpha=1.0)
    inferior_mesh = inferior_mesh.delaunay_2d(alpha=1.0)

    return superior_mesh, inferior_mesh


def get_superior_and_inferior_surface_points(cartilage: np.array) -> Tuple[np.array, np.array]:
    """
    Get the superior and inferior surface points of a cartilage. Points (x, y, z) are expected to be
    x = axial, y = sagittal, z = coronal.

    :param cartilage: A Nx3 point cloud representation of the cartilage, where the first component is the axial
    coordinate, the second component the sagittal coordinate, and the third component the coronal coordinate.
    :return: Two point clouds of the superior and inferior surface points.
    """
    df = pd.DataFrame(cartilage, columns=['axial', 'sagittal', 'coronal'])
    superior_points = df.groupby(['sagittal', 'coronal']).min().reset_index()  # remember: min is superior
    superior_points = superior_points[['axial', 'sagittal', 'coronal']]  # need to re-order the columns
    inferior_points = df.groupby(['sagittal', 'coronal']).max().reset_index()
    inferior_points = inferior_points[['axial', 'sagittal', 'coronal']]

    return superior_points.to_numpy(), inferior_points.to_numpy()


def calculate_ellipse(points, center) -> float:
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


def get_plate_corners(points) -> Tuple[np.array, np.array, np.array, np.array]:
    """
    Get the corners (2D) of a plate defined by a point cloud.

    :param points: A point cloud.
    :return: The corners (image orientation: axial view; left image side = right patient side,
    upper image half = posterior) of the plate.
    """
    min_sagittal = points[:, 1].min()  # right patient side
    max_sagittal = points[:, 1].max()  # left patient side
    min_coronal = points[:, 2].min()  # posterior
    max_coronal = points[:, 2].max()  # anterior

    upper_right = np.array([max_sagittal, max_coronal])  # upper right image corner, i.e. left patient side, posterior
    lower_right = np.array([max_sagittal, min_coronal])  # lower right image corner, i.e. left patient side, anterior
    upper_left = np.array([min_sagittal, max_coronal])  # upper left image corner, i.e. right patient side, posterior
    lower_left = np.array([min_sagittal, min_coronal])  # lower left image corner, i.e. right patient side, anterior

    return upper_right, lower_right, upper_left, lower_left
