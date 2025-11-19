from surface_distance.metrics import compute_surface_distances, compute_average_surface_distance, compute_dice_coefficient, compute_robust_hausdorff
from pathlib import Path
from tqdm import tqdm
import SimpleITK as sitk
from typing import Tuple
import numpy as np
import pandas as pd
import argparse


def compute_metrics(prediction_path: str, gt_path: str, labels: list) -> pd.DataFrame:
    """
    Compute the metrics for a dataset.
    :param prediction_path: The path to the directory containing the predicted masks.
    :param gt_path: The path to the directory containing the ground truth masks.
    :param labels: A list of labels to compute the metrics for.
    :return: A pandas DataFrame containing the metrics.
    """
    filenames = [x.name for x in Path(
        prediction_path).iterdir()
                        if x.suffix != '.json']
    index = pd.MultiIndex.from_product([filenames, labels], names=['filename', 'label'])
    df = pd.DataFrame(columns = ['asd', 'hd', 'dice'], index=index, dtype=float)
    for filename in tqdm(filenames):
        prediction = sitk.ReadImage(str(Path(prediction_path) / filename))
        reference = sitk.ReadImage(str(Path(gt_path) / filename))
        pred_numpy = sitk.GetArrayFromImage(prediction)
        ref_numpy = sitk.GetArrayFromImage(reference)
        spacing = prediction.GetSpacing()
        spacing = [spacing[2], spacing[1], spacing[0]]

        for label in labels:
            pred_numpy_ = np.where(pred_numpy == label, pred_numpy, 0).astype(bool)
            ref_numpy_ = np.where(ref_numpy == label, ref_numpy, 0).astype(bool)

            try:

                surface_distances = compute_surface_distances(ref_numpy_, pred_numpy_, spacing)
                hausdorff_distance = compute_robust_hausdorff(surface_distances, 95)
                hausdorff_distance = 0 if np.isinf(hausdorff_distance) else hausdorff_distance
                avg_surface_distance = compute_average_surface_distance(surface_distances)
                avg_surface_distance = (avg_surface_distance[0] + avg_surface_distance[1]) / 2
                dice_coefficient = compute_dice_coefficient(ref_numpy_, pred_numpy_)
                df.loc[(filename, label)] = [avg_surface_distance, hausdorff_distance, dice_coefficient]
            except ValueError as e:
                print(filename, e)
                print(ref_numpy.shape, pred_numpy.shape)

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute surface distances for a dataset.')
    parser.add_argument('--prediction_path', type=str, required=True, help='Path to the directory containing the predicted masks.')
    parser.add_argument('--gt_path', type=str, required=True, help='Path to the directory containing the ground truth masks.')
    parser.add_argument('--labels', nargs='+', type=int, required=True, help='List of labels to compute the metrics for.')
    parser.add_argument('-o', '--output_path', type=str, required=False, help='Path to save the computed metrics.')
    args = parser.parse_args()

    df = compute_metrics(args.prediction_path, args.gt_path, args.labels)

    print(df)
    print(df.groupby(level='label').describe().T)
    if args.output_path:
        # df.to_csv(args.output_path, index=True)
        df.groupby(level='label').describe().T.to_excel(args.output_path)
