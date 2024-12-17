from surface_distance.metrics import compute_surface_distances, compute_average_surface_distance, compute_dice_coefficient, compute_robust_hausdorff
from pathlib import Path
import SimpleITK as sitk
from typing import Tuple
import numpy as np
import pandas as pd


def compute_metrics(prediction_path: str, gt_path: str) -> pd.DataFrame:
    """
    Compute the metrics for a dataset.
    :param prediction_path: The path to the directory containing the predicted masks.
    :param gt_path: The path to the directory containing the ground truth masks.
    :return: A pandas DataFrame containing the metrics.
    """
    filenames = [x.name for x in Path(
        prediction_path).iterdir()
                        if x.suffix != '.json']
    df = pd.DataFrame(columns = ['asd', 'hd', 'dice'], index=filenames, dtype=float)
    for filename in filenames:
        prediction = sitk.ReadImage(str(Path(prediction_path) / filename))
        reference = sitk.ReadImage(str(Path(gt_path) / filename))
        pred_numpy = sitk.GetArrayFromImage(prediction).astype(bool)
        ref_numpy = sitk.GetArrayFromImage(reference).astype(bool)
        spacing = prediction.GetSpacing()
        spacing = [spacing[2], spacing[1], spacing[0]]
        surface_distances = compute_surface_distances(ref_numpy, pred_numpy, spacing)
        hausdorff_distance = compute_robust_hausdorff(surface_distances, 95)
        hausdorff_distance = 0 if np.isinf(hausdorff_distance) else hausdorff_distance
        avg_surface_distance = compute_average_surface_distance(surface_distances)
        avg_surface_distance = (avg_surface_distance[0] + avg_surface_distance[1]) / 2
        dice_coefficient = compute_dice_coefficient(ref_numpy, pred_numpy)
        df.loc[filename] = [avg_surface_distance, hausdorff_distance, dice_coefficient]

    return df


if __name__ == '__main__':
    # hip
    prediction_path = '/home/simon/Data/nnUnet_raw/Dataset001_AugsburgHip/labelsTs'
    gt_path = '/home/simon/Data/nnUnet_raw/Dataset001_AugsburgHip/labelsTr'
    df = compute_metrics(prediction_path, gt_path)
    df.to_csv('/home/simon/Data/Augsburg_large/proxy_metrics_hip.csv')
    print(df.describe())

    # knee
    prediction_path = '/home/simon/Data/nnUnet_raw/Dataset002_AugsburgKnee/labelsTs'
    gt_path = '/home/simon/Data/nnUnet_raw/Dataset002_AugsburgKnee/labelsTr'
    df = compute_metrics(prediction_path, gt_path)
    df.to_csv('/home/simon/Data/Augsburg_large/proxy_metrics_knee.csv')
    print(df.describe())

    # ankle
    prediction_path = '/home/simon/Data/nnUnet_raw/Dataset003_AugsburgAnkle/labelsTs'
    gt_path = '/home/simon/Data/nnUnet_raw/Dataset003_AugsburgAnkle/labelsTr'
    df = compute_metrics(prediction_path, gt_path)
    df.to_csv('/home/simon/Data/Augsburg_large/proxy_metrics_ankle.csv')
    print(df.describe())
