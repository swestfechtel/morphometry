from surface_distance.metrics import compute_surface_distances, compute_average_surface_distance
from pathlib import Path
import SimpleITK as sitk
import numpy as np


def asd(prediction_path: str, gt_path: str) -> float:
    """
    Calculate the average surface distance metric for a dataset.
    :param prediction_path: The path to the directory containing the predicted masks.
    :param gt_path: The path to the directory containing the ground truth masks.
    :return: The average surface distance for the dataset.
    """
    prediction_files = [x.name for x in Path(
        prediction_path).iterdir()
                        if x.suffix != '.json']
    predictions = [sitk.ReadImage(str(Path(
        prediction_path) / x))
                   for x in prediction_files]
    references = [sitk.ReadImage(str(Path(gt_path) / x)) for x in
                  prediction_files]

    average_distances = np.empty(len(predictions))
    for i, pred in enumerate(predictions):
        pred_numpy = sitk.GetArrayFromImage(pred).astype(bool)
        ref_numpy = sitk.GetArrayFromImage(references[i]).astype(bool)
        spacing = pred.GetSpacing()
        spacing = [spacing[2], spacing[1], spacing[0]]
        surface_distances = compute_surface_distances(ref_numpy, pred_numpy, spacing)
        avg_surface_distance = compute_average_surface_distance(surface_distances)
        average_distances[i] = (avg_surface_distance[0] + avg_surface_distance[1]) / 2

    return average_distances.mean()


if __name__ == '__main__':
    # hip
    prediction_path = '/home/simon/Data/nnUnet_results/Dataset001_AugsburgHip/nnUNetTrainer__pretraining_plans__3d_fullres/fold_0/validation'
    gt_path = '/home/simon/Data/nnUnet_raw/Dataset001_AugsburgHip/labelsTr'
    print(asd(prediction_path, gt_path))

    # knee
    prediction_path = '/home/simon/Data/nnUnet_results/Dataset002_AugsburgKnee/nnUNetTrainer__pretraining_plans__3d_fullres/fold_0/validation'
    gt_path = '/home/simon/Data/nnUnet_raw/Dataset002_AugsburgKnee/labelsTr'
    print(asd(prediction_path, gt_path))

    # ankle
    prediction_path = '/home/simon/Data/nnUnet_results/Dataset003_AugsburgAnkle/nnUNetTrainer__pretraining_plans__3d_fullres/fold_0/validation'
    gt_path = '/home/simon/Data/nnUnet_raw/Dataset003_AugsburgAnkle/labelsTr'
    print(asd(prediction_path, gt_path))