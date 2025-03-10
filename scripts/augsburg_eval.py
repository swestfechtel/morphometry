import re
import SimpleITK as sitk
import pandas as pd
import numpy as np
from pathlib import Path
from morphometry.femur import calculate_femoral_torsion
from morphometry.tibia import calculate_tibial_torsion
from morphometry.knee import calculate_knee_rotation_angle
from morphometry.whole_leg import calculate_mikulicz_deviation
from morphometry.ankle import calculate_pma_angle
from morphometry.hip import calculate_ccd
from morphometry.utils import correct_axis_ordering, write_mask, combine_masks, read_nifti
from matplotlib import pyplot as plt


if __name__ == '__main__':
    plot = True

    df = pd.read_excel('/home/simon/Downloads/Augsburg Messungen 1.xlsx', index_col=[0, 1], header=1)
    df = df.dropna(axis=1)

    devs = pd.DataFrame(columns=['CCD', 'AT', 'TT', 'KRA'], index=df.index)
    vals = pd.DataFrame(columns=['CCD', 'AT', 'AT (Murphy)', 'AT (Tomczak)', 'TT', 'KRA'], index=df.index)
    r = re.compile(r'PA\d+')
    for file in Path('/home/simon/Downloads/Augsburg/labels/huefte').iterdir():
        patient = r.search(file.name)[0]
        print(f'Processing patient {patient}...')
        try:
            # hip = sitk.ReadImage(f'/home/simon/Downloads/Augsburg/labels/huefte/t1_tse_tra_Huften_bds_{patient}.nii.gz')
            # knee = sitk.ReadImage(f'/home/simon/Downloads/Augsburg/labels/knie/t1_tse_tra_Knie_{patient}.nii.gz')
            # ankle = sitk.ReadImage(f'/home/simon/Downloads/Augsburg/labels/osg/t1_tse_tra_OSG_{patient}.nii.gz')
            hip = read_nifti(f'/home/simon/Downloads/Augsburg/labels/huefte/t1_tse_tra_Huften_bds_{patient}.nii.gz')
            knee = read_nifti(f'/home/simon/Downloads/Augsburg/labels/knie/t1_tse_tra_Knie_{patient}.nii.gz')
            ankle = read_nifti(f'/home/simon/Downloads/Augsburg/labels/osg/t1_tse_tra_OSG_{patient}.nii.gz')
        except RuntimeError as e:
            print(f'Patient {patient} could not be found.', e)
            continue

        # hip = correct_axis_ordering(hip)
        # knee = correct_axis_ordering(knee)
        # ankle = correct_axis_ordering(ankle)

        x_ratio = abs(hip.GetSpacing()[2]) / 2 * abs(hip.GetSpacing()[0])

        hip_mask = sitk.GetArrayFromImage(hip)
        knee_mask = sitk.GetArrayFromImage(knee)
        ankle_mask = sitk.GetArrayFromImage(ankle)

        left_hip = hip_mask[:, :, :hip_mask.shape[2] // 2]
        right_hip = hip_mask[:, :, hip_mask.shape[2] // 2:]
        left_knee = knee_mask[:, :, :knee_mask.shape[2] // 2]
        right_knee = knee_mask[:, :, knee_mask.shape[2] // 2:]
        left_ankle = ankle_mask[:, :, :ankle_mask.shape[2] // 2]
        right_ankle = ankle_mask[:, :, ankle_mask.shape[2] // 2:]

        try:
            if plot:
                # lee
                tmp_l = left_hip.copy()
                tmp_r = right_hip.copy()
                femoral_torsion_left, fig, tmp_l, _ = calculate_femoral_torsion(tmp_l, left_knee, side='left', x_ratio=x_ratio, plot=plot, mark_mask=True)
                fig.savefig(f'/home/simon/Downloads/Augsburg/figures/{patient}_at_left.png')
                plt.close(fig)

                femoral_torsion_right, fig, tmp_r, _ = calculate_femoral_torsion(tmp_r, right_knee, side='right',
                                                                       x_ratio=x_ratio,
                                                                       plot=plot, mark_mask=True)
                fig.savefig(f'/home/simon/Downloads/Augsburg/figures/{patient}_at_right.png')
                plt.close(fig)

                comb = combine_masks(tmp_l, tmp_r)
                write_mask(comb, hip, f'/home/simon/Downloads/Augsburg/marked_masks/{patient}_at_lee.nii.gz')

                # murphy
                tmp_l = left_hip.copy()
                tmp_r = right_hip.copy()
                print('Murphy - left image side')
                femoral_torsion_left_murphy, fig, tmp_l, _ = calculate_femoral_torsion(tmp_l, left_knee, side='left', method='murphy', x_ratio=x_ratio, plot=plot, hip_image=hip, mark_mask=True)
                fig.savefig(f'/home/simon/Downloads/Augsburg/figures/{patient}_at_left_murphy.png')
                plt.close(fig)

                print('Murphy - right image side')
                femoral_torsion_right_murphy, fig, tmp_r, _ = calculate_femoral_torsion(tmp_r, right_knee, side='right',
                                                                              method='murphy', x_ratio=x_ratio,
                                                                              plot=plot, hip_image=hip, mark_mask=True)
                fig.savefig(f'/home/simon/Downloads/Augsburg/figures/{patient}_at_right_murphy.png')
                plt.close(fig)

                comb = combine_masks(tmp_l, tmp_r)
                write_mask(comb, hip, f'/home/simon/Downloads/Augsburg/marked_masks/{patient}_at_murphy.nii.gz')

                # tomczak
                """
                tmp_l = left_hip.copy()
                tmp_r = right_hip.copy()
                femoral_torsion_left_tomczak, fig, tmp_l, _ = calculate_femoral_torsion(tmp_l, left_knee, side='left', method='tomczak', x_ratio=x_ratio, plot=plot, hip_image=hip, mark_mask=True)
                fig.savefig(f'/home/simon/Downloads/Augsburg/figures/{patient}_at_left_tomczak.png')
                plt.close(fig)

                femoral_torsion_right_tomczak, fig, tmp_r, _ = calculate_femoral_torsion(tmp_r, right_knee, side='right', method='tomczak', x_ratio=x_ratio, plot=plot, hip_image=hip, mark_mask=True)
                fig.savefig(f'/home/simon/Downloads/Augsburg/figures/{patient}_at_right_tomczak.png')
                plt.close(fig)

                comb = combine_masks(tmp_l, tmp_r)
                write_mask(comb, hip, f'/home/simon/Downloads/Augsburg/marked_masks/{patient}_at_tomczak.nii.gz')
                """
                # tibia
                tibial_torsion_left, fig = calculate_tibial_torsion(left_knee, left_ankle, tibia_label_knee=2, tibia_label_ankle=1,
                                                               fibula_label=2, side='left', plot=plot)
                fig.savefig(f'/home/simon/Downloads/Augsburg/figures/{patient}_tt_left.png')
                plt.close(fig)

                tibial_torsion_right, fig = calculate_tibial_torsion(right_knee, right_ankle, tibia_label_knee=2,
                                                                tibia_label_ankle=1, fibula_label=2, side='right', plot=plot)
                fig.savefig(f'/home/simon/Downloads/Augsburg/figures/{patient}_tt_right.png')
                plt.close(fig)
            else:
                femoral_torsion_left = calculate_femoral_torsion(left_hip, left_knee, side='left', x_ratio=x_ratio,
                                                                 plot=plot)
                femoral_torsion_left_murphy = calculate_femoral_torsion(left_hip, left_knee, side='left', method='murphy', x_ratio=x_ratio, plot=plot)
                femoral_torsion_left_tomczak = calculate_femoral_torsion(left_hip, left_knee, side='left', method='tomczak', x_ratio=x_ratio, plot=plot)

                femoral_torsion_right = calculate_femoral_torsion(right_hip, right_knee, side='right', x_ratio=x_ratio,
                                                                  plot=plot)
                femoral_torsion_right_murphy = calculate_femoral_torsion(right_hip, right_knee, side='right', method='murphy', x_ratio=x_ratio, plot=plot)
                femoral_torsion_right_tomczak = calculate_femoral_torsion(right_hip, right_knee, side='right', method='tomczak', x_ratio=x_ratio, plot=plot)

                tibial_torsion_left = calculate_tibial_torsion(left_knee, left_ankle, tibia_label_knee=2,
                                                               tibia_label_ankle=1,
                                                               fibula_label=2, side='left', plot=plot)
                tibial_torsion_right = calculate_tibial_torsion(right_knee, right_ankle, tibia_label_knee=2,
                                                                tibia_label_ankle=1, fibula_label=2, side='right',
                                                                plot=plot)
            ccd_left = calculate_ccd(left_hip, None,'left', 1, False, x_ratio)
            ccd_right = calculate_ccd(right_hip, None,'right', 1, False, x_ratio)

            kra_left = calculate_knee_rotation_angle(left_knee, 1, 2, False)
            kra_right = calculate_knee_rotation_angle(right_knee, 1, 2, False)
        except (ValueError,  RuntimeError) as e:
            print(f'Patient {patient} could not be processed.', e)
            continue

        """
        print(f'Patient {patient}:')
        print(f'Femoral torsion (right patient side): {femoral_torsion_left}°')
        print(f'Femoral torsion (left patient side): {femoral_torsion_right}°')
        print(f'Tibial torsion (right patient side): {tibial_torsion_left}°')
        print(f'Tibial torsion (left patient side): {tibial_torsion_right}°')
        print('--------------------------------------------------')
        """
        patient_nr = int(patient[2:])
        row = df.loc[patient_nr]
        devs.loc[patient_nr, 'rechts']['AT'] = round(abs(row.loc['rechts']['AT (Lee)'] - femoral_torsion_left), 1)  # image side <-> patient side
        devs.loc[patient_nr, 'links']['AT'] = round(abs(row.loc['links']['AT (Lee)'] - femoral_torsion_right), 1)
        devs.loc[patient_nr, 'rechts']['TT'] = round(abs(row.loc['rechts']['TT'] - tibial_torsion_left), 1)
        devs.loc[patient_nr, 'links']['TT'] = round(abs(row.loc['links']['TT'] - tibial_torsion_right), 1)
        devs.loc[patient_nr, 'rechts']['CCD'] = round(abs(row.loc['rechts']['CCD'] - ccd_left[1]), 1)
        devs.loc[patient_nr, 'links']['CCD'] = round(abs(row.loc['links']['CCD'] - ccd_right[1]), 1)
        devs.loc[patient_nr, 'rechts']['KRA'] = round(abs(row.loc['rechts']['Knee Rotation'] - kra_left), 1)
        devs.loc[patient_nr, 'links']['KRA'] = round(abs(row.loc['links']['Knee Rotation'] - kra_right), 1)
        vals.loc[patient_nr, 'rechts']['AT'] = femoral_torsion_left
        vals.loc[patient_nr, 'links']['AT'] = femoral_torsion_right
        vals.loc[patient_nr, 'rechts']['AT (Murphy)'] = femoral_torsion_left_murphy
        vals.loc[patient_nr, 'links']['AT (Murphy)'] = femoral_torsion_right_murphy
        # vals.loc[patient_nr, 'rechts']['AT (Tomczak)'] = femoral_torsion_left_tomczak
        # vals.loc[patient_nr, 'links']['AT (Tomczak)'] = femoral_torsion_right_tomczak
        vals.loc[patient_nr, 'rechts']['TT'] = tibial_torsion_left
        vals.loc[patient_nr, 'links']['TT'] = tibial_torsion_right
        vals.loc[patient_nr, 'rechts']['CCD'] = ccd_left[1]
        vals.loc[patient_nr, 'links']['CCD'] = ccd_right[1]
        vals.loc[patient_nr, 'rechts']['KRA'] = kra_left
        vals.loc[patient_nr, 'links']['KRA'] = kra_right

    print(devs)
    devs.to_excel('/home/simon/Downloads/Augsburg/Augsburg_devs_2.xlsx')
    vals.to_excel('/home/simon/Downloads/Augsburg/Augsburg_vals_2.xlsx')
