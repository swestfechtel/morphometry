from morphometry.cartilage import knee
from morphometry.utils import read_nifti
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm
from typing import Tuple
import SimpleITK as sitk
import numpy as np
import pandas as pd

ul = 7
tibia_label = 4
femur_label = 3
data_path = '/home/simon/Data'
# data_path = 'E:/Data/Daten UKA'


def process_patient(row_identifier: str, unloaded_image: sitk.Image, med_to_lat_image: sitk.Image, lat_to_med_image: sitk.Image, dataframe_unloaded: pd.DataFrame, dataframe_med_to_lat: pd.DataFrame, dataframe_lat_to_med: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tibia = knee.Tibia(unloaded_image, cartilage_label=tibia_label)
    tibia_results = tibia.calculate_thickness()

    femur = knee.Femur(unloaded_image, cartilage_label=femur_label)
    femur_results = femur.calculate_thickness(tibia)

    map = {'clf': 'left_cwbz', 'plf': 'left_posterior_zone', 'alf': 'left_anterior_zone', 'crf': 'right_cwbz',
           'prf': 'right_posterior_zone', 'arf': 'right_anterior_zone',
           'iclf': 'iclf', 'cclf': 'cclf', 'eclf': 'eclf', 'icrf': 'icrf', 'ccrf': 'ccrf', 'ecrf': 'ecrf'}
    row = dict()
    row['Patient'] = row_identifier
    for region in ['iclf', 'cclf', 'eclf', 'plf', 'alf', 'icrf', 'ccrf', 'ecrf', 'prf', 'arf']:
        array = np.array(list(femur_results[map[region]].values()))
        row[region] = f'{array.mean():.2f}'
        row[f'{region}.std'] = f'{array.std():.2f}'

    for region in ['clt', 'ilt', 'elt', 'alt', 'plt', 'crt', 'irt', 'ert', 'art', 'prt']:
        array = np.array(list(tibia_results[region].values()))
        row[region] = f'{array.mean():.2f}'
        row[f'{region}.std'] = f'{array.std():.2f}'

    dataframe_unloaded = pd.concat([dataframe_unloaded, pd.DataFrame([row])])

    tibia = knee.Tibia(med_to_lat_image, cartilage_label=tibia_label)
    tibia_results = tibia.calculate_thickness()

    femur = knee.Femur(med_to_lat_image, cartilage_label=femur_label)
    femur_results = femur.calculate_thickness(tibia)

    row = dict()
    row['Patient'] = row_identifier
    for region in ['iclf', 'cclf', 'eclf', 'plf', 'alf', 'icrf', 'ccrf', 'ecrf', 'prf', 'arf']:
        array = np.array(list(femur_results[map[region]].values()))
        row[region] = f'{array.mean():.2f}'
        row[f'{region}.std'] = f'{array.std():.2f}'

    for region in ['clt', 'ilt', 'elt', 'alt', 'plt', 'crt', 'irt', 'ert', 'art', 'prt']:
        array = np.array(list(tibia_results[region].values()))
        row[region] = f'{array.mean():.2f}'
        row[f'{region}.std'] = f'{array.std():.2f}'

    dataframe_med_to_lat = pd.concat([dataframe_med_to_lat, pd.DataFrame([row])])

    tibia = knee.Tibia(lat_to_med_image, cartilage_label=tibia_label)
    tibia_results = tibia.calculate_thickness()

    femur = knee.Femur(lat_to_med_image, cartilage_label=femur_label)
    femur_results = femur.calculate_thickness(tibia)

    row = dict()
    row['Patient'] = row_identifier
    for region in ['iclf', 'cclf', 'eclf', 'plf', 'alf', 'icrf', 'ccrf', 'ecrf', 'prf', 'arf']:
        array = np.array(list(femur_results[map[region]].values()))
        row[region] = f'{array.mean():.2f}'
        row[f'{region}.std'] = f'{array.std():.2f}'

    for region in ['clt', 'ilt', 'elt', 'alt', 'plt', 'crt', 'irt', 'ert', 'art', 'prt']:
        array = np.array(list(tibia_results[region].values()))
        row[region] = f'{array.mean():.2f}'
        row[f'{region}.std'] = f'{array.std():.2f}'

    dataframe_lat_to_med = pd.concat([dataframe_lat_to_med, pd.DataFrame([row])])
    return dataframe_unloaded, dataframe_med_to_lat, dataframe_lat_to_med


if __name__ == '__main__':
    unloaded_df = pd.DataFrame(columns=['Patient', 'iclf', 'cclf', 'eclf', 'plf', 'alf', 'icrf', 'ccrf', 'ecrf', 'prf', 'arf', 'clt', 'ilt', 'elt', 'alt', 'plt', 'crt', 'irt', 'ert', 'art', 'prt',
                                        'iclf.std', 'cclf.std', 'eclf.std', 'plf.std', 'alf.std', 'icrf.std', 'ccrf.std', 'ecrf.std', 'prf.std', 'arf.std', 'clt.std', 'ilt.std', 'elt.std', 'alt.std', 'plt.std', 'crt.std', 'irt.std', 'ert.std', 'art.std', 'prt.std'])
    med_to_lat_df = unloaded_df.copy()
    lat_to_med_df = unloaded_df.copy()
    for path in tqdm(Path(f'{data_path}/Duesseldorf/T1rho/').iterdir()):
        if str(path.name).__contains__('.csv'):
            continue

        # unloaded = sitk.ReadImage(f'{data_path}/Duesseldorf/T1rho/{path.name}/1Relaxed/1Relaxed.nii.gz')
        # med_to_lat = sitk.ReadImage(f'{data_path}/Duesseldorf/T1rho/{path.name}/2MedToLat/2MedToLat.nii.gz')
        # lat_to_med = sitk.ReadImage(f'{data_path}/Duesseldorf/T1rho/{path.name}/3LatToMed/3LatToMed.nii.gz')
        unloaded = read_nifti(f'{data_path}/Duesseldorf/T1rho/{path.name}/1Relaxed/1Relaxed.nii.gz')
        med_to_lat = read_nifti(f'{data_path}/Duesseldorf/T1rho/{path.name}/2MedToLat/2MedToLat.nii.gz')
        lat_to_med = read_nifti(f'{data_path}/Duesseldorf/T1rho/{path.name}/3LatToMed/3LatToMed.nii.gz')

        unloaded_df, med_to_lat_df, lat_to_med_df = process_patient(path.name, unloaded, med_to_lat, lat_to_med, unloaded_df, med_to_lat_df, lat_to_med_df)

        if str(path.name) == 'P04':
            # lat_to_med_new = sitk.ReadImage(f'{data_path}/Duesseldorf/T1rho/P04/3LatToMed/3LatToMednew.nii.gz')
            # med_to_lat_new = sitk.ReadImage(f'{data_path}/Duesseldorf/T1rho/P04/2MedToLat/2MedToLatnew.nii.gz')
            # unloaded_new = sitk.ReadImage(f'{data_path}/Duesseldorf/T1rho/P04/1Relaxed/1Relaxednew.nii.gz')
            lat_to_med_new = read_nifti(f'{data_path}/Duesseldorf/T1rho/P04/3LatToMed/3LatToMednew.nii.gz')
            med_to_lat_new = read_nifti(f'{data_path}/Duesseldorf/T1rho/P04/2MedToLat/2MedToLatnew.nii.gz')
            unloaded_new = read_nifti(f'{data_path}/Duesseldorf/T1rho/P04/1Relaxed/1Relaxednew.nii.gz')

            unloaded_df, med_to_lat_df, lat_to_med_df = process_patient('P04_new', unloaded_new, med_to_lat_new, lat_to_med_new, unloaded_df, med_to_lat_df, lat_to_med_df)

        if str(path.name) == 'P10':
            # lat_to_med_new = sitk.ReadImage(f'{data_path}/Duesseldorf/T1rho/P10/3LatToMed/3LatToMedNEW.nii.gz')
            # med_to_lat_new = sitk.ReadImage(f'{data_path}/Duesseldorf/T1rho/P10/2MedToLat/2MedToLatNEW.nii.gz')
            # unloaded_new = sitk.ReadImage(f'{data_path}/Duesseldorf/T1rho/P10/1Relaxed/1RelaxedNEW.nii.gz')
            lat_to_med_new = read_nifti(f'{data_path}/Duesseldorf/T1rho/P10/3LatToMed/3LatToMedNEW.nii.gz')
            med_to_lat_new = read_nifti(f'{data_path}/Duesseldorf/T1rho/P10/2MedToLat/2MedToLatNEW.nii.gz')
            unloaded_new = read_nifti(f'{data_path}/Duesseldorf/T1rho/P10/1Relaxed/1RelaxedNEW.nii.gz')

            unloaded_df, med_to_lat_df, lat_to_med_df = process_patient('P10_new', unloaded_new, med_to_lat_new,
                                                                        lat_to_med_new, unloaded_df, med_to_lat_df,
                                                                        lat_to_med_df)


    unloaded_df.to_excel('../unloaded_df.xlsx')
    med_to_lat_df.to_excel('../med_to_lat_df.xlsx')
    lat_to_med_df.to_excel('../lat_to_med_df.xlsx')
    """"
    fig, ax = plt.subplots()
    for _, value in thicknesses.items():
        scatter = utils.plot_heatmap(value, ax, ul=ul)

    plt.colorbar(scatter, ax=ax)
    plt.show()
    """
