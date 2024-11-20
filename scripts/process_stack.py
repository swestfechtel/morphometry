import sys
sys.path.append('/home/simon/Work/morpohmetry')
import SimpleITK as sitk
from morphometry.femur import calculate_femoral_torsion
from morphometry.tibia import calculate_tibial_torsion
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser(description='Calculate the femoral and tibial torsion.')
    parser.add_argument('--hip_mask', type=str, help='Path to the hip segmentation mask.')
    parser.add_argument('--knee_mask', type=str, help='Path to the knee segmentation mask.')
    parser.add_argument('--ankle_mask', type=str, help='Path to the ankle segmentation mask.')
    parser.add_argument('--flip_x', action='store_true', help='Whether the x-axis needs to be flipped.')
    parser.add_argument('--plot', action='store_true', help='Whether to plot the results.')

    args = parser.parse_args()

    hip = sitk.ReadImage(args.hip_mask)
    x_ratio = abs(hip.GetSpacing()[2]) / 2 * abs(hip.GetSpacing()[0])
    knee = sitk.ReadImage(args.knee_mask)
    ankle = sitk.ReadImage(args.ankle_mask)

    hip_mask = sitk.GetArrayFromImage(hip)
    knee_mask = sitk.GetArrayFromImage(knee)
    ankle_mask = sitk.GetArrayFromImage(ankle)

    if args.flip_x:
        hip_mask = hip_mask[::-1]
        knee_mask = knee_mask[::-1]
        ankle_mask = ankle_mask[::-1]

    left_hip = hip_mask[:, :, :hip_mask.shape[2] // 2]
    right_hip = hip_mask[:, :, hip_mask.shape[2] // 2:]
    left_knee = knee_mask[:, :, :knee_mask.shape[2] // 2]
    right_knee = knee_mask[:, :, knee_mask.shape[2] // 2:]
    left_ankle = ankle_mask[:, :, :ankle_mask.shape[2] // 2]
    right_ankle = ankle_mask[:, :, ankle_mask.shape[2] // 2:]

    femoral_torsion_left = calculate_femoral_torsion(left_hip, left_knee, side='left', x_ratio=x_ratio, plot=args.plot)
    femoral_torsion_right = calculate_femoral_torsion(right_hip, right_knee, side='right', x_ratio=x_ratio, plot=args.plot)

    tibial_torsion_left = calculate_tibial_torsion(left_knee, left_ankle, tibia_label_knee=2, tibia_label_ankle=1, fibula_label=2, side='left', plot=args.plot)
    tibial_torsion_right = calculate_tibial_torsion(right_knee, right_ankle, tibia_label_knee=2, tibia_label_ankle=1, fibula_label=2, side='right', plot=args.plot)

    print(f'Femoral torsion (right patient side): {femoral_torsion_left:.2f}°')
    print(f'Femoral torsion (left patient side): {femoral_torsion_right:.2f}°')
    print(f'Tibial torsion (right patient side): {tibial_torsion_left:.2f}°')
    print(f'Tibial torsion (left patient side): {tibial_torsion_right:.2f}°')