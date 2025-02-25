from morphometry.cartilage import knee
from morphometry.image_io import Image
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path
from tqdm import tqdm
from typing import Tuple
import numpy as np
import pandas as pd

ul = 7
tibia_label = 4
femur_label = 3
data_path = '/home/simon/Data'
# data_path = 'E:/Data/Daten UKA'


def process_patient(row_identifier: str, unloaded_image: Image, med_to_lat_image: Image, lat_to_med_image: Image, dataframe_unloaded: pd.DataFrame, dataframe_med_to_lat: pd.DataFrame, dataframe_lat_to_med: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tibia = knee.Tibia(unloaded_image, cartilage_label=tibia_label)
    tibia_results = tibia.calculate_thickness()

    femur = knee.Femur(unloaded_image, cartilage_label=femur_label)
    femur_results = femur.calculate_thickness(tibia, 'knn')

    map = {'clf': 'left_cwbz', 'plf': 'left_posterior_zone', 'alf': 'left_anterior_zone', 'crf': 'right_cwbz',
           'prf': 'right_posterior_zone', 'arf': 'right_anterior_zone',
           'iclf': 'iclf', 'cclf': 'cclf', 'eclf': 'eclf', 'icrf': 'icrf', 'ccrf': 'ccrf', 'ecrf': 'ecrf'}

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    for subregion in tibia_results.keys():
        scatter = plot_heatmap(tibia_results[subregion], ax[0, 0], ul=ul)
    plt.colorbar(scatter, ax=ax[0, 0])

    for subregion in femur_results.keys():
        scatter = plot_heatmap(femur_results[subregion], ax[1, 0], ul=ul)
    plt.colorbar(scatter, ax=ax[1, 0])

    row = dict()
    row['Patient'] = row_identifier
    for region in ['iclf', 'cclf', 'eclf', 'plf', 'alf', 'icrf', 'ccrf', 'ecrf', 'prf', 'arf']:
        array = np.array(list(femur_results[map[region]].values()))
        row[region] = f'{np.nanmean(array):.2f}'
        row[f'{region}.std'] = f'{np.nanstd(array):.2f}'

    for region in ['clt', 'ilt', 'elt', 'alt', 'plt', 'crt', 'irt', 'ert', 'art', 'prt']:
        array = np.array(list(tibia_results[region].values()))
        row[region] = f'{np.nanmean(array):.2f}'
        row[f'{region}.std'] = f'{np.nanstd(array):.2f}'

    dataframe_unloaded = pd.concat([dataframe_unloaded, pd.DataFrame([row])])

    tibia = knee.Tibia(med_to_lat_image, cartilage_label=tibia_label)
    tibia_results = tibia.calculate_thickness()

    femur = knee.Femur(med_to_lat_image, cartilage_label=femur_label)
    femur_results = femur.calculate_thickness(tibia, 'knn')

    for subregion in tibia_results.keys():
        scatter = plot_heatmap(tibia_results[subregion], ax[0, 1], ul=ul)
    plt.colorbar(scatter, ax=ax[0, 1])
    for subregion in femur_results.keys():
        scatter = plot_heatmap(femur_results[subregion], ax[1, 1], ul=ul)
    plt.colorbar(scatter, ax=ax[1, 1])

    row = dict()
    row['Patient'] = row_identifier
    for region in ['iclf', 'cclf', 'eclf', 'plf', 'alf', 'icrf', 'ccrf', 'ecrf', 'prf', 'arf']:
        array = np.array(list(femur_results[map[region]].values()))
        row[region] = f'{np.nanmean(array):.2f}'
        row[f'{region}.std'] = f'{np.nanstd(array):.2f}'

    for region in ['clt', 'ilt', 'elt', 'alt', 'plt', 'crt', 'irt', 'ert', 'art', 'prt']:
        array = np.array(list(tibia_results[region].values()))
        row[region] = f'{np.nanmean(array):.2f}'
        row[f'{region}.std'] = f'{np.nanstd(array):.2f}'

    dataframe_med_to_lat = pd.concat([dataframe_med_to_lat, pd.DataFrame([row])])

    tibia = knee.Tibia(lat_to_med_image, cartilage_label=tibia_label)
    tibia_results = tibia.calculate_thickness()

    femur = knee.Femur(lat_to_med_image, cartilage_label=femur_label)
    femur_results = femur.calculate_thickness(tibia, 'knn')

    for subregion in tibia_results.keys():
        scatter = plot_heatmap(tibia_results[subregion], ax[0, 2], ul=ul)
    plt.colorbar(scatter, ax=ax[0, 2])
    for subregion in femur_results.keys():
        scatter = plot_heatmap(femur_results[subregion], ax[1, 2], ul=ul)
    plt.colorbar(scatter, ax=ax[1, 2])

    row = dict()
    row['Patient'] = row_identifier
    for region in ['iclf', 'cclf', 'eclf', 'plf', 'alf', 'icrf', 'ccrf', 'ecrf', 'prf', 'arf']:
        array = np.array(list(femur_results[map[region]].values()))
        row[region] = f'{np.nanmean(array):.2f}'
        row[f'{region}.std'] = f'{np.nanstd(array):.2f}'

    for region in ['clt', 'ilt', 'elt', 'alt', 'plt', 'crt', 'irt', 'ert', 'art', 'prt']:
        array = np.array(list(tibia_results[region].values()))
        row[region] = f'{np.nanmean(array):.2f}'
        row[f'{region}.std'] = f'{np.nanstd(array):.2f}'

    dataframe_lat_to_med = pd.concat([dataframe_lat_to_med, pd.DataFrame([row])])

    plt.show()
    return dataframe_unloaded, dataframe_med_to_lat, dataframe_lat_to_med


def plot_heatmap(distances: dict, axis: plt.Axes, ul: int = 15):
    """
    Plot a heatmap of any distances.
    :param distances: A dictionary where keys are coordinates and values are distances.
    :param axis: The axis to plot the heatmap on.
    :param ul: The upper limit of the colormap.
    :return:
    """
    coords = list(distances.keys())
    dists = list(distances.values())

    normaliser = Normalize(0, ul)
    scatter = axis.scatter([x[1] for x in coords], [x[0] for x in coords], c=dists, cmap='viridis', norm=normaliser)
    axis.set_xlabel('Sagittal')
    axis.set_ylabel('Coronal')

    return scatter


if __name__ == '__main__':
    unloaded_df = pd.DataFrame(columns=['Patient', 'iclf', 'cclf', 'eclf', 'plf', 'alf', 'icrf', 'ccrf', 'ecrf', 'prf', 'arf', 'clt', 'ilt', 'elt', 'alt', 'plt', 'crt', 'irt', 'ert', 'art', 'prt',
                                        'iclf.std', 'cclf.std', 'eclf.std', 'plf.std', 'alf.std', 'icrf.std', 'ccrf.std', 'ecrf.std', 'prf.std', 'arf.std', 'clt.std', 'ilt.std', 'elt.std', 'alt.std', 'plt.std', 'crt.std', 'irt.std', 'ert.std', 'art.std', 'prt.std'])
    med_to_lat_df = unloaded_df.copy()
    lat_to_med_df = unloaded_df.copy()
    for path in tqdm(Path(f'{data_path}/Duesseldorf/T1rho/').iterdir()):
        if str(path.name).__contains__('.csv'):
            continue

        unloaded = Image('nibabel')
        unloaded.read_image(f'{data_path}/Duesseldorf/T1rho/{path.name}/1Relaxed/1Relaxed.nii.gz')
        unloaded.transform_coordinate_system()
        med_to_lat = Image('nibabel')
        med_to_lat.read_image(f'{data_path}/Duesseldorf/T1rho/{path.name}/2MedToLat/2MedToLat.nii.gz')
        med_to_lat.transform_coordinate_system()
        lat_to_med = Image('nibabel')
        lat_to_med.read_image(f'{data_path}/Duesseldorf/T1rho/{path.name}/3LatToMed/3LatToMed.nii.gz')
        lat_to_med.transform_coordinate_system()

        unloaded_df, med_to_lat_df, lat_to_med_df = process_patient(path.name, unloaded, med_to_lat, lat_to_med, unloaded_df, med_to_lat_df, lat_to_med_df)

        if str(path.name) == 'P04':
            unloaded_new = Image('nibabel')
            unloaded_new.read_image(f'{data_path}/Duesseldorf/T1rho/{path.name}/1Relaxed/1Relaxednew.nii.gz')
            unloaded_new.transform_coordinate_system()
            med_to_lat_new = Image('nibabel')
            med_to_lat_new.read_image(f'{data_path}/Duesseldorf/T1rho/{path.name}/2MedToLat/2MedToLatnew.nii.gz')
            med_to_lat_new.transform_coordinate_system()
            lat_to_med_new = Image('nibabel')
            lat_to_med_new.read_image(f'{data_path}/Duesseldorf/T1rho/{path.name}/3LatToMed/3LatToMednew.nii.gz')
            lat_to_med_new.transform_coordinate_system()

            unloaded_df, med_to_lat_df, lat_to_med_df = process_patient('P04_new', unloaded_new, med_to_lat_new, lat_to_med_new, unloaded_df, med_to_lat_df, lat_to_med_df)

        if str(path.name) == 'P10':
            unloaded_new = Image('nibabel')
            unloaded_new.read_image(f'{data_path}/Duesseldorf/T1rho/{path.name}/1Relaxed/1RelaxedNEW.nii.gz')
            unloaded_new.transform_coordinate_system()
            med_to_lat_new = Image('nibabel')
            med_to_lat_new.read_image(f'{data_path}/Duesseldorf/T1rho/{path.name}/2MedToLat/2MedToLatNEW.nii.gz')
            med_to_lat_new.transform_coordinate_system()
            lat_to_med_new = Image('nibabel')
            lat_to_med_new.read_image(f'{data_path}/Duesseldorf/T1rho/{path.name}/3LatToMed/3LatToMedNEW.nii.gz')
            lat_to_med_new.transform_coordinate_system()

            unloaded_df, med_to_lat_df, lat_to_med_df = process_patient('P10_new', unloaded_new, med_to_lat_new,
                                                                        lat_to_med_new, unloaded_df, med_to_lat_df,
                                                                        lat_to_med_df)

    unloaded_df.to_excel('../unloaded_df.xlsx')
    med_to_lat_df.to_excel('../med_to_lat_df.xlsx')
    lat_to_med_df.to_excel('../lat_to_med_df.xlsx')
