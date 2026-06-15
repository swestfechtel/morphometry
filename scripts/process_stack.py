# FIXME (measurements refactor): this batch script calls the pre-refactor API and
# needs updating to the new single-return measurements API:
#   - calculate_femoral_torsion / calculate_tibial_torsion / calculate_knee_rotation_angle
#     now return only the angle (no (angle, fig)); pass a sequence of Axes to `plot=` to
#     draw, and use get_femoral_torsion_landmarks / get_tibial_torsion_landmarks for landmarks.
#   - calculate_mechanical_axis_deviation now takes a single whole-leg CT Segmentation
#     (femur=1, tibia=2, ...), not separate hip/knee/ankle images.
# Imports have been repointed to morphometry.measurements; call sites are not yet updated.
import sys
sys.path.append('/home/simon/Work/morphometry')
import SimpleITK as sitk
import nibabel as nib
import numpy as np
from morphometry.measurements.femur import calculate_femoral_torsion
from morphometry.measurements.tibia import calculate_tibial_torsion
from morphometry.image_io import Image
from argparse import ArgumentParser
from matplotlib import pyplot as plt


if __name__ == '__main__':
    parser = ArgumentParser(description='Calculate the femoral and tibial torsion.')
    parser.add_argument('--hip_mask', type=str, help='Path to the hip segmentation mask.')
    parser.add_argument('--knee_mask', type=str, help='Path to the knee segmentation mask.')
    parser.add_argument('--ankle_mask', type=str, help='Path to the ankle segmentation mask.')
    parser.add_argument('-p', '--plot', action='store_true', help='Whether to plot the results.')
    parser.add_argument('-o', '--output', type=str, help='Path to save the results to.')

    args = parser.parse_args()

    hip = Image('nibabel')
    hip.read_image(args.hip_mask)
    hip.transform_coordinate_system()
    knee = Image('nibabel')
    knee.read_image(args.knee_mask)
    knee.transform_coordinate_system()
    ankle = Image('nibabel')
    ankle.read_image(args.ankle_mask)
    ankle.transform_coordinate_system()

    x_ratio = abs(hip.spacing[2]) / 2 * abs(hip.spacing[0])

    hip_mask = hip.array
    knee_mask = knee.array
    ankle_mask = ankle.array

    left_hip = hip_mask[:hip_mask.shape[0] // 2]  # wtf, the sagittal axis seems to be flipped with the new coordinate system??
    right_hip = hip_mask[hip_mask.shape[0] // 2:]
    left_knee = knee_mask[:knee_mask.shape[0] // 2]
    right_knee = knee_mask[knee_mask.shape[0] // 2:]
    left_ankle = ankle_mask[:ankle_mask.shape[0] // 2]
    right_ankle = ankle_mask[ankle_mask.shape[0] // 2:]

    left_hip = nib.Nifti1Image(left_hip, hip.affine, hip.header)
    left_hip = Image.from_nibabel(left_hip)
    right_hip = nib.Nifti1Image(right_hip, hip.affine, hip.header)
    right_hip = Image.from_nibabel(right_hip)

    if args.plot:
        try:
            femoral_torsion_left, fig = calculate_femoral_torsion(left_hip, left_knee, side='left', x_ratio=x_ratio, plot=args.plot)
            fig.savefig(f'{args.output}/ft_right.png')  # patient side <-> image side
            plt.close(fig)
        except (RuntimeError, AssertionError, ValueError) as e:
            print(f'Failed to calculate femoral torsion for the left side.', e)
            femoral_torsion_left = np.nan

        try:
            femoral_torsion_right, fig = calculate_femoral_torsion(right_hip, right_knee, side='right', x_ratio=x_ratio, plot=args.plot)
            fig.savefig(f'{args.output}/ft_left.png')
            plt.close(fig)
        except (RuntimeError, AssertionError, ValueError) as e:
            print(f'Failed to calculate femoral torsion for the right side.', e)
            femoral_torsion_right = np.nan

        try:
            tibial_torsion_left, fig = calculate_tibial_torsion(left_knee, left_ankle, tibia_label_knee=2, tibia_label_ankle=1, fibula_label=2, side='left', plot=args.plot)
            fig.savefig(f'{args.output}/tt_right.png')
            plt.close(fig)
        except (RuntimeError, AssertionError, ValueError) as e:
            print(f'Failed to calculate tibial torsion for the left side.', e)
            tibial_torsion_left = np.nan

        try:
            tibial_torsion_right, fig = calculate_tibial_torsion(right_knee, right_ankle, tibia_label_knee=2, tibia_label_ankle=1, fibula_label=2, side='right', plot=args.plot)
            fig.savefig(f'{args.output}/tt_left.png')
            plt.close(fig)
        except (RuntimeError, AssertionError, ValueError) as e:
            print(f'Failed to calculate tibial torsion for the right side.', e)
            tibial_torsion_right = np.nan
    else:
        try:
            femoral_torsion_left = calculate_femoral_torsion(left_hip, left_knee, side='left', x_ratio=x_ratio, plot=args.plot)
        except (RuntimeError, AssertionError, ValueError) as e:
            print(f'Failed to calculate femoral torsion for the left side.', e)
            femoral_torsion_left = np.nan

        try:
            femoral_torsion_right = calculate_femoral_torsion(right_hip, right_knee, side='right', x_ratio=x_ratio, plot=args.plot)
        except (RuntimeError, AssertionError, ValueError) as e:
            print(f'Failed to calculate femoral torsion for the right side.', e)
            femoral_torsion_right = np.nan

        try:
            tibial_torsion_left = calculate_tibial_torsion(left_knee, left_ankle, tibia_label_knee=2, tibia_label_ankle=1, fibula_label=2, side='left', plot=args.plot)
        except (RuntimeError, AssertionError, ValueError) as e:
            print(f'Failed to calculate tibial torsion for the left side.', e)
            tibial_torsion_left = np.nan

        try:
            tibial_torsion_right = calculate_tibial_torsion(right_knee, right_ankle, tibia_label_knee=2, tibia_label_ankle=1, fibula_label=2, side='right', plot=args.plot)
        except (RuntimeError, AssertionError, ValueError) as e:
            print(f'Failed to calculate tibial torsion for the right side.', e)
            tibial_torsion_right = np.nan

    print(f'Femoral torsion (right patient side): {femoral_torsion_left:.2f}°')
    print(f'Femoral torsion (left patient side): {femoral_torsion_right:.2f}°')
    print(f'Tibial torsion (right patient side): {tibial_torsion_left:.2f}°')
    print(f'Tibial torsion (left patient side): {tibial_torsion_right:.2f}°')
