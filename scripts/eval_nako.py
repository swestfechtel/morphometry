import pandas as pd
import pingouin as pg


if __name__ == '__main__':
    auto = pd.read_excel('/home/simon/Data/NaKo_sample/eval.xlsx', index_col=[0, 1])
    manual_tim = pd.read_excel('/home/simon/Data/NaKo_sample/Referenzmessungen_TH.xlsx', index_col=0)

    long_df = pd.DataFrame(columns=['Case', 'Metric', 'Rater', 'Score'])

    for index, row in manual_tim.iterrows():
        auto_right = auto.loc[(index, 'right')]
        auto_left = auto.loc[(index, 'left')]

        long_df = pd.concat([long_df, pd.Series({'Case': index, 'Metric': 'CCD_right', 'Rater': 'Tim', 'Score': row['CCD rechts']}).to_frame().T], ignore_index=True)
        long_df = pd.concat([long_df, pd.Series({'Case': index, 'Metric': 'CCD_left', 'Rater': 'Tim', 'Score': row['CCD links']}).to_frame().T], ignore_index=True)
        long_df = pd.concat([long_df, pd.Series({'Case': index, 'Metric': 'CCD_right', 'Rater': 'Auto', 'Score': auto_right['CCD']}).to_frame().T], ignore_index=True)
        long_df = pd.concat([long_df, pd.Series({'Case': index, 'Metric': 'CCD_left', 'Rater': 'Auto', 'Score': auto_left['CCD']}).to_frame().T], ignore_index=True)

        long_df = pd.concat([long_df, pd.Series({'Case': index, 'Metric': 'FAT_right', 'Rater': 'Tim', 'Score': row['Antetorsion rechts']}).to_frame().T], ignore_index=True)
        long_df = pd.concat([long_df, pd.Series({'Case': index, 'Metric': 'FAT_left', 'Rater': 'Tim', 'Score': row['Antetorsion links']}).to_frame().T], ignore_index=True)
        long_df = pd.concat([long_df, pd.Series({'Case': index, 'Metric': 'FAT_right', 'Rater': 'Auto', 'Score': auto_right['FAT']}).to_frame().T], ignore_index=True)
        long_df = pd.concat([long_df, pd.Series({'Case': index, 'Metric': 'FAT_left', 'Rater': 'Auto', 'Score': auto_left['FAT']}).to_frame().T], ignore_index=True)

        long_df = pd.concat([long_df, pd.Series({'Case': index, 'Metric': 'AA_right', 'Rater': 'Tim', 'Score': row['Alpha Winkel rechts anterior']}).to_frame().T], ignore_index=True)
        long_df = pd.concat([long_df, pd.Series({'Case': index, 'Metric': 'AA_left', 'Rater': 'Tim', 'Score': row['Alpha Winkel links anterior']}).to_frame().T], ignore_index=True)
        long_df = pd.concat([long_df, pd.Series({'Case': index, 'Metric': 'AA_right', 'Rater': 'Auto', 'Score': auto_right['AA']}).to_frame().T], ignore_index=True)
        long_df = pd.concat([long_df, pd.Series({'Case': index, 'Metric': 'AA_left', 'Rater': 'Auto', 'Score': auto_left['AA']}).to_frame().T], ignore_index=True)

        long_df = pd.concat([long_df, pd.Series({'Case': index, 'Metric': 'AAV_right', 'Rater': 'Tim', 'Score': row['Acetabular Anteversion rechts']}).to_frame().T], ignore_index=True)
        long_df = pd.concat([long_df, pd.Series({'Case': index, 'Metric': 'AAV_left', 'Rater': 'Tim', 'Score': row['Acetabular Anteversion links']}).to_frame().T], ignore_index=True)
        long_df = pd.concat([long_df, pd.Series({'Case': index, 'Metric': 'AAV_right', 'Rater': 'Auto', 'Score': auto_right['AAV']}).to_frame().T], ignore_index=True)
        long_df = pd.concat([long_df, pd.Series({'Case': index, 'Metric': 'AAV_left', 'Rater': 'Auto', 'Score': auto_left['AAV']}).to_frame().T], ignore_index=True)

        long_df = pd.concat([long_df, pd.Series({'Case': index, 'Metric': 'CEA_right', 'Rater': 'Tim', 'Score': row['Center-Edge Angle rechts']}).to_frame().T], ignore_index=True)
        long_df = pd.concat([long_df, pd.Series({'Case': index, 'Metric': 'CEA_left', 'Rater': 'Tim', 'Score': row['Center-Edge Angle links']}).to_frame().T], ignore_index=True)
        long_df = pd.concat([long_df, pd.Series({'Case': index, 'Metric': 'CEA_right', 'Rater': 'Auto', 'Score': auto_right['CEA']}).to_frame().T], ignore_index=True)
        long_df = pd.concat([long_df, pd.Series({'Case': index, 'Metric': 'CEA_left', 'Rater': 'Auto', 'Score': auto_left['CEA']}).to_frame().T], ignore_index=True)

    long_df['Score'] = long_df['Score'].astype(float)
    long_df.to_excel('/home/simon/Data/NaKo_sample/dataframes/long_df.xlsx', index=False)

    print(f'Number of rows: {len(long_df)}')

    # Remove outliers: more than 3 std from mean, per Metric and Rater
    def remove_outliers(df):
        return df[abs(df['Score'] - df['Score'].mean()) <= 2 * df['Score'].std()]

    # long_df = long_df.groupby(['Metric', 'Rater'], group_keys=False).apply(remove_outliers)

    print(f'Number of rows after outlier removal: {len(long_df)}')

    metrics = ['CCD_right', 'CCD_left', 'FAT_right', 'FAT_left', 'AA_right', 'AA_left', 'AAV_right', 'AAV_left', 'CEA_right', 'CEA_left']

    for metric in metrics:
        metric_df = long_df[long_df['Metric'] == metric]
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
        dev = wide_df['Tim'] - wide_df['Auto']
        dev = dev.abs()
        print(f'Deviation for {metric}:')
        print(dev.describe())
        dev.to_excel(f'/home/simon/Data/NaKo_sample/dataframes/{metric}_deviation.xlsx')
