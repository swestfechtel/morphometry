import pandas as pd
import pingouin as pg
import numpy as np


if __name__ == '__main__':
    auto = pd.read_excel('/home/simon/Data/NaKo_sample/eval.xlsx', index_col=[0, 1])
    manual_tim = pd.read_excel('/home/simon/Data/NaKo_sample/Referenzmessungen_TH.xlsx', index_col=0)
    manual_marc = pd.read_excel('/home/simon/Data/NaKo_sample/Referenzmessungen_MH.xlsx', index_col=0, skiprows=1)

    irm_df = pd.DataFrame(index=manual_tim.index, columns=['CCD_right', 'CCD_left', 'AT_murphy_right', 'AT_murphy_left', 'AA_right', 'AA_left', 'AAV_right', 'AAV_left', 'CE_right', 'CE_left', 'Offset_right', 'Offset_left'])
    ira_long_df = pd.DataFrame(columns=['Case', 'Metric', 'Rater', 'Score'])
    irm_auto_long_df = pd.DataFrame(columns=['Case', 'Metric', 'Rater', 'Score'])

    for index, row in manual_tim.iterrows():
        marc = manual_marc.loc[str(index).split('_')[0] + '_30']

        irm_df.loc[index, 'CCD_right'] = np.mean([row['CCD rechts'], marc['CCD']])
        irm_df.loc[index, 'CCD_left'] = np.mean([row['CCD links'], marc['CCD.1']])
        irm_df.loc[index, 'AT_murphy_right'] = np.mean([row['Antetorsion rechts'], marc['Femoral Torsion (Murphy)']])
        irm_df.loc[index, 'AT_murphy_left'] = np.mean([row['Antetorsion links'], marc['Femoral Torsion (Murphy).1']])
        irm_df.loc[index, 'AA_right'] = np.mean([row['Alpha Winkel rechts anterior'], marc['Alpha (anterior)']])
        irm_df.loc[index, 'AA_left'] = np.mean([row['Alpha Winkel links anterior'], marc['Alpha (anterior).1']])
        irm_df.loc[index, 'AAV_right'] = np.mean([row['Acetabular Anteversion rechts'], marc['Acetabuläre Anteversion']])
        irm_df.loc[index, 'AAV_left'] = np.mean([row['Acetabular Anteversion links'], marc['Acetabuläre Anteversion.1']])
        irm_df.loc[index, 'CE_right'] = np.mean([row['Center-Edge Angle rechts'], marc['CE']])
        irm_df.loc[index, 'CE_left'] = np.mean([row['Center-Edge Angle links'], marc['CE.1']])
        irm_df.loc[index, 'Offset_right'] = np.mean([row['Offset rechts'], marc['Femoral Offset (mm)']])
        irm_df.loc[index, 'Offset_left'] = np.mean([row['Offset links'], marc['Femoral Offset (mm).1']])

    irm_df.to_excel('/home/simon/Data/NaKo_sample/dataframes/irm_df.xlsx')

    for index, row in manual_tim.iterrows():
        marc = manual_marc.loc[str(index).split('_')[0] + '_30']

        ira_long_df = pd.concat([ira_long_df, pd.Series({'Case': index, 'Metric': 'CCD_right', 'Rater': 'Tim', 'Score': row['CCD rechts']}).to_frame().T], ignore_index=True)
        ira_long_df = pd.concat([ira_long_df, pd.Series({'Case': index, 'Metric': 'CCD_left', 'Rater': 'Tim', 'Score': row['CCD links']}).to_frame().T], ignore_index=True)
        ira_long_df = pd.concat([ira_long_df, pd.Series({'Case': index, 'Metric': 'CCD_right', 'Rater': 'Marc', 'Score': marc['CCD']}).to_frame().T], ignore_index=True)
        ira_long_df = pd.concat([ira_long_df, pd.Series({'Case': index, 'Metric': 'CCD_left', 'Rater': 'Marc', 'Score': marc['CCD.1']}).to_frame().T], ignore_index=True)

        ira_long_df = pd.concat([ira_long_df, pd.Series({'Case': index, 'Metric': 'AT_murphy_right', 'Rater': 'Tim', 'Score': row['Antetorsion rechts']}).to_frame().T], ignore_index=True)
        ira_long_df = pd.concat([ira_long_df, pd.Series({'Case': index, 'Metric': 'AT_murphy_left', 'Rater': 'Tim', 'Score': row['Antetorsion links']}).to_frame().T], ignore_index=True)
        ira_long_df = pd.concat([ira_long_df, pd.Series({'Case': index, 'Metric': 'AT_murphy_right', 'Rater': 'Marc', 'Score': marc['Femoral Torsion (Murphy)']}).to_frame().T], ignore_index=True)
        ira_long_df = pd.concat([ira_long_df, pd.Series({'Case': index, 'Metric': 'AT_murphy_left', 'Rater': 'Marc', 'Score': marc['Femoral Torsion (Murphy).1']}).to_frame().T], ignore_index=True)

        ira_long_df = pd.concat([ira_long_df, pd.Series({'Case': index, 'Metric': 'AA_right', 'Rater': 'Tim', 'Score': row['Alpha Winkel rechts anterior']}).to_frame().T], ignore_index=True)
        ira_long_df = pd.concat([ira_long_df, pd.Series({'Case': index, 'Metric': 'AA_left', 'Rater': 'Tim', 'Score': row['Alpha Winkel links anterior']}).to_frame().T], ignore_index=True)
        ira_long_df = pd.concat([ira_long_df, pd.Series({'Case': index, 'Metric': 'AA_right', 'Rater': 'Marc', 'Score': marc['Alpha (anterior)']}).to_frame().T], ignore_index=True)
        ira_long_df = pd.concat([ira_long_df, pd.Series({'Case': index, 'Metric': 'AA_left', 'Rater': 'Marc', 'Score': marc['Alpha (anterior).1']}).to_frame().T], ignore_index=True)

        ira_long_df = pd.concat([ira_long_df, pd.Series({'Case': index, 'Metric': 'AAV_right', 'Rater': 'Tim', 'Score': row['Acetabular Anteversion rechts']}).to_frame().T], ignore_index=True)
        ira_long_df = pd.concat([ira_long_df, pd.Series({'Case': index, 'Metric': 'AAV_left', 'Rater': 'Tim', 'Score': row['Acetabular Anteversion links']}).to_frame().T], ignore_index=True)
        ira_long_df = pd.concat([ira_long_df, pd.Series({'Case': index, 'Metric': 'AAV_right', 'Rater': 'Marc', 'Score': marc['Acetabuläre Anteversion']}).to_frame().T], ignore_index=True)
        ira_long_df = pd.concat([ira_long_df, pd.Series({'Case': index, 'Metric': 'AAV_left', 'Rater': 'Marc', 'Score': marc['Acetabuläre Anteversion.1']}).to_frame().T], ignore_index=True)

        ira_long_df = pd.concat([ira_long_df, pd.Series({'Case': index, 'Metric': 'CEA_right', 'Rater': 'Tim', 'Score': row['Center-Edge Angle rechts']}).to_frame().T], ignore_index=True)
        ira_long_df = pd.concat([ira_long_df, pd.Series({'Case': index, 'Metric': 'CEA_left', 'Rater': 'Tim', 'Score': row['Center-Edge Angle links']}).to_frame().T], ignore_index=True)
        ira_long_df = pd.concat([ira_long_df, pd.Series({'Case': index, 'Metric': 'CEA_right', 'Rater': 'Marc', 'Score': marc['CE']}).to_frame().T], ignore_index=True)
        ira_long_df = pd.concat([ira_long_df, pd.Series({'Case': index, 'Metric': 'CEA_left', 'Rater': 'Marc', 'Score': marc['CE.1']}).to_frame().T], ignore_index=True)

        ira_long_df = pd.concat([ira_long_df, pd.Series({'Case': index, 'Metric': 'Offset_right', 'Rater': 'Tim', 'Score': row['Offset rechts']}).to_frame().T], ignore_index=True)
        ira_long_df = pd.concat([ira_long_df, pd.Series({'Case': index, 'Metric': 'Offset_left', 'Rater': 'Tim', 'Score': row['Offset links']}).to_frame().T], ignore_index=True)
        ira_long_df = pd.concat([ira_long_df, pd.Series({'Case': index, 'Metric': 'Offset_right', 'Rater': 'Marc', 'Score': marc['Femoral Offset (mm)']}).to_frame().T], ignore_index=True)
        ira_long_df = pd.concat([ira_long_df, pd.Series({'Case': index, 'Metric': 'Offset_left', 'Rater': 'Marc', 'Score': marc['Femoral Offset (mm).1']}).to_frame().T], ignore_index=True)

    ira_long_df['Score'] = ira_long_df['Score'].astype(float)
    ira_long_df.to_excel('/home/simon/Data/NaKo_sample/dataframes/ira_long_df.xlsx', index=False)

    for index, row in irm_df.iterrows():
        auto_right = auto.loc[(index, 'right')]
        auto_left = auto.loc[(index, 'left')]

        irm_auto_long_df = pd.concat([irm_auto_long_df, pd.Series({'Case': index, 'Metric': 'CCD_right', 'Rater': 'Auto', 'Score': auto_right['CCD']}).to_frame().T], ignore_index=True)
        irm_auto_long_df = pd.concat([irm_auto_long_df, pd.Series({'Case': index, 'Metric': 'CCD_left', 'Rater': 'Auto', 'Score': auto_left['CCD']}).to_frame().T], ignore_index=True)
        irm_auto_long_df = pd.concat([irm_auto_long_df, pd.Series({'Case': index, 'Metric': 'CCD_right', 'Rater': 'IRM', 'Score': row['CCD_right']}).to_frame().T], ignore_index=True)
        irm_auto_long_df = pd.concat([irm_auto_long_df, pd.Series({'Case': index, 'Metric': 'CCD_left', 'Rater': 'IRM', 'Score': row['CCD_left']}).to_frame().T], ignore_index=True)

        irm_auto_long_df = pd.concat([irm_auto_long_df, pd.Series({'Case': index, 'Metric': 'AT_murphy_right', 'Rater': 'Auto', 'Score': auto_right['AT_murphy']}).to_frame().T], ignore_index=True)
        irm_auto_long_df = pd.concat([irm_auto_long_df, pd.Series({'Case': index, 'Metric': 'AT_murphy_left', 'Rater': 'Auto', 'Score': auto_left['AT_murphy']}).to_frame().T], ignore_index=True)
        irm_auto_long_df = pd.concat([irm_auto_long_df, pd.Series({'Case': index, 'Metric': 'AT_murphy_right', 'Rater': 'IRM', 'Score': row['AT_murphy_right']}).to_frame().T], ignore_index=True)
        irm_auto_long_df = pd.concat([irm_auto_long_df, pd.Series({'Case': index, 'Metric': 'AT_murphy_left', 'Rater': 'IRM', 'Score': row['AT_murphy_left']}).to_frame().T], ignore_index=True)

        irm_auto_long_df = pd.concat([irm_auto_long_df, pd.Series({'Case': index, 'Metric': 'AA_right', 'Rater': 'Auto', 'Score': auto_right['AA_anterior']}).to_frame().T], ignore_index=True)
        irm_auto_long_df = pd.concat([irm_auto_long_df, pd.Series({'Case': index, 'Metric': 'AA_left', 'Rater': 'Auto', 'Score': auto_left['AA_anterior']}).to_frame().T], ignore_index=True)
        irm_auto_long_df = pd.concat([irm_auto_long_df, pd.Series({'Case': index, 'Metric': 'AA_right', 'Rater': 'IRM', 'Score': row['AA_right']}).to_frame().T], ignore_index=True)
        irm_auto_long_df = pd.concat([irm_auto_long_df, pd.Series({'Case': index, 'Metric': 'AA_left', 'Rater': 'IRM', 'Score': row['AA_left']}).to_frame().T], ignore_index=True)

        irm_auto_long_df = pd.concat([irm_auto_long_df, pd.Series({'Case': index, 'Metric': 'AAV_right', 'Rater': 'Auto', 'Score': auto_right['AAV']}).to_frame().T], ignore_index=True)
        irm_auto_long_df = pd.concat([irm_auto_long_df, pd.Series({'Case': index, 'Metric': 'AAV_left', 'Rater': 'Auto', 'Score': auto_left['AAV']}).to_frame().T], ignore_index=True)
        irm_auto_long_df = pd.concat([irm_auto_long_df, pd.Series({'Case': index, 'Metric': 'AAV_right', 'Rater': 'IRM', 'Score': row['AAV_right']}).to_frame().T], ignore_index=True)
        irm_auto_long_df = pd.concat([irm_auto_long_df, pd.Series({'Case': index, 'Metric': 'AAV_left', 'Rater': 'IRM', 'Score': row['AAV_left']}).to_frame().T], ignore_index=True)

        irm_auto_long_df = pd.concat([irm_auto_long_df, pd.Series({'Case': index, 'Metric': 'CEA_right', 'Rater': 'Auto', 'Score': auto_right['CE']}).to_frame().T], ignore_index=True)
        irm_auto_long_df = pd.concat([irm_auto_long_df, pd.Series({'Case': index, 'Metric': 'CEA_left', 'Rater': 'Auto', 'Score': auto_left['CE']}).to_frame().T], ignore_index=True)
        irm_auto_long_df = pd.concat([irm_auto_long_df, pd.Series({'Case': index, 'Metric': 'CEA_right', 'Rater': 'IRM', 'Score': row['CE_right']}).to_frame().T], ignore_index=True)
        irm_auto_long_df = pd.concat([irm_auto_long_df, pd.Series({'Case': index, 'Metric': 'CEA_left', 'Rater': 'IRM', 'Score': row['CE_left']}).to_frame().T], ignore_index=True)

        irm_auto_long_df = pd.concat([irm_auto_long_df, pd.Series({'Case': index, 'Metric': 'Offset_right', 'Rater': 'Auto', 'Score': auto_right['Offset']}).to_frame().T], ignore_index=True)
        irm_auto_long_df = pd.concat([irm_auto_long_df, pd.Series({'Case': index, 'Metric': 'Offset_left', 'Rater': 'Auto', 'Score': auto_left['Offset']}).to_frame().T], ignore_index=True)
        irm_auto_long_df = pd.concat([irm_auto_long_df, pd.Series({'Case': index, 'Metric': 'Offset_right', 'Rater': 'IRM', 'Score': row['Offset_right']}).to_frame().T], ignore_index=True)
        irm_auto_long_df = pd.concat([irm_auto_long_df, pd.Series({'Case': index, 'Metric': 'Offset_left', 'Rater': 'IRM', 'Score': row['Offset_left']}).to_frame().T], ignore_index=True)

    irm_auto_long_df['Score'] = irm_auto_long_df['Score'].astype(float)
    irm_auto_long_df.to_excel('/home/simon/Data/NaKo_sample/dataframes/irm_auto_long_df.xlsx', index=False)

    # Remove outliers: more than 3 std from mean, per Metric and Rater
    def remove_outliers(df):
        return df[abs(df['Score'] - df['Score'].mean()) <= 2 * df['Score'].std()]

    # long_df = long_df.groupby(['Metric', 'Rater'], group_keys=False).apply(remove_outliers)

    # print(f'Number of rows after outlier removal: {len(long_df)}')

    metrics = ['CCD_right', 'CCD_left', 'AT_murphy_right', 'AT_murphy_left', 'AA_right', 'AA_left', 'AAV_right', 'AAV_left', 'CEA_right', 'CEA_left', 'Offset_right', 'Offset_left']

    print('\n\n\n Inter-reader agreement (IRA) for Tim and Marc:')

    for metric in metrics:
        metric_df = ira_long_df[ira_long_df['Metric'] == metric]
        metric_df = metric_df.dropna()
        metric_df = metric_df.drop(columns=['Metric'])
        print(metric)

        try:
            icc = pg.intraclass_corr(data=metric_df, targets='Case', raters='Rater', ratings='Score', nan_policy='omit')
        except AssertionError as e:
            print('Not enough data for ICC calculation', e)
            continue

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(icc)

        wide_df = metric_df.pivot(index='Case', columns='Rater', values='Score')
        dev = wide_df['Tim'] - wide_df['Marc']
        dev = dev.abs()
        print(f'Deviation for {metric}:')
        print(dev.describe())
        # dev.to_excel(f'/home/simon/Data/NaKo_sample/dataframes/{metric}_deviation.xlsx')

    print('\n\n\n Inter-reader agreement (IRA) for Auto and IRM:')

    for metric in metrics:
        metric_df = irm_auto_long_df[irm_auto_long_df['Metric'] == metric]
        metric_df = metric_df.dropna()
        metric_df = metric_df.drop(columns=['Metric'])
        print(metric)

        try:
            icc = pg.intraclass_corr(data=metric_df, targets='Case', raters='Rater', ratings='Score', nan_policy='omit')
        except AssertionError as e:
            print('Not enough data for ICC calculation', e)
            continue

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(icc)

        wide_df = metric_df.pivot(index='Case', columns='Rater', values='Score')
        dev = wide_df['Auto'] - wide_df['IRM']
        dev = dev.abs()
        print(f'Deviation for {metric}:')
        print(dev.describe())
        # dev.to_excel(f'/home/simon/Data/NaKo_sample/dataframes/{metric}_deviation.xlsx')
