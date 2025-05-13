import pandas as pd
import pingouin as pg


if __name__ == '__main__':
    auto = pd.read_excel('/home/simon/Data/Augsburg_large/results_.xlsx', index_col=[0, 1])
    manual_felix = pd.read_excel('/home/simon/Data/Augsburg_large/reference_measurements_felix.xlsx')

    long_df = pd.DataFrame(columns=['Case', 'Metric', 'Rater', 'Score'])

    for _, row in manual_felix.iterrows():
        identifier = row['Unnamed: 0']
        auto_right = auto.loc[(identifier, 'right')]
        auto_left = auto.loc[(identifier, 'left')]

        long_df = pd.concat([long_df, pd.Series({'Case': identifier, 'Metric': 'CCD_right', 'Rater': 'Felix', 'Score': row['CCD rechts']}).to_frame().T], ignore_index=True)
        long_df = pd.concat([long_df, pd.Series({'Case': identifier, 'Metric': 'CCD_left', 'Rater': 'Felix', 'Score': row['CCD links']}).to_frame().T], ignore_index=True)
        long_df = pd.concat([long_df, pd.Series({'Case': identifier, 'Metric': 'CCD_right', 'Rater': 'Auto', 'Score': auto_right['CCD (actual)']}).to_frame().T], ignore_index=True)
        long_df = pd.concat([long_df, pd.Series({'Case': identifier, 'Metric': 'CCD_left', 'Rater': 'Auto', 'Score': auto_left['CCD (actual)']}).to_frame().T], ignore_index=True)

        long_df = pd.concat([long_df, pd.Series({'Case': identifier, 'Metric': 'FT_lee_right', 'Rater': 'Felix', 'Score': row['AT (Lee) rechts']}).to_frame().T], ignore_index=True)
        long_df = pd.concat([long_df, pd.Series({'Case': identifier, 'Metric': 'FT_lee_left', 'Rater': 'Felix', 'Score': row['AT (Lee) links']}).to_frame().T], ignore_index=True)
        long_df = pd.concat([long_df, pd.Series({'Case': identifier, 'Metric': 'FT_lee_right', 'Rater': 'Auto', 'Score': auto_right['AT (Lee)']}).to_frame().T], ignore_index=True)
        long_df = pd.concat([long_df, pd.Series({'Case': identifier, 'Metric': 'FT_lee_left', 'Rater': 'Auto', 'Score': auto_left['AT (Lee)']}).to_frame().T], ignore_index=True)

        long_df = pd.concat([long_df, pd.Series({'Case': identifier, 'Metric': 'FT_murphy_right', 'Rater': 'Felix', 'Score': row['AT (Murphy) rechts']}).to_frame().T], ignore_index=True)
        long_df = pd.concat([long_df, pd.Series({'Case': identifier, 'Metric': 'FT_murphy_left', 'Rater': 'Felix', 'Score': row['AT (Murphy) links']}).to_frame().T], ignore_index=True)
        long_df = pd.concat([long_df, pd.Series({'Case': identifier, 'Metric': 'FT_murphy_right', 'Rater': 'Auto', 'Score': auto_right['AT (Murphy)']}).to_frame().T], ignore_index=True)
        long_df = pd.concat([long_df, pd.Series({'Case': identifier, 'Metric': 'FT_murphy_left', 'Rater': 'Auto', 'Score': auto_left['AT (Murphy)']}).to_frame().T], ignore_index=True)

        long_df = pd.concat([long_df, pd.Series({'Case': identifier, 'Metric': 'TT_right', 'Rater': 'Felix', 'Score': row['Tibiatorsion rechts']}).to_frame().T], ignore_index=True)
        long_df = pd.concat([long_df, pd.Series({'Case': identifier, 'Metric': 'TT_left', 'Rater': 'Felix', 'Score': row['Tibiatorsion links']}).to_frame().T], ignore_index=True)
        long_df = pd.concat([long_df, pd.Series({'Case': identifier, 'Metric': 'TT_right', 'Rater': 'Auto', 'Score': auto_right['TT']}).to_frame().T], ignore_index=True)
        long_df = pd.concat([long_df, pd.Series({'Case': identifier, 'Metric': 'TT_left', 'Rater': 'Auto', 'Score': auto_left['TT']}).to_frame().T], ignore_index=True)

        long_df = pd.concat([long_df, pd.Series({'Case': identifier, 'Metric': 'KRA_right', 'Rater': 'Felix', 'Score': row['Knie rotation rechts']}).to_frame().T], ignore_index=True)
        long_df = pd.concat([long_df, pd.Series({'Case': identifier, 'Metric': 'KRA_left', 'Rater': 'Felix', 'Score': row['Knie rotation links']}).to_frame().T], ignore_index=True)
        long_df = pd.concat([long_df, pd.Series({'Case': identifier, 'Metric': 'KRA_right', 'Rater': 'Auto', 'Score': auto_right['KRA']}).to_frame().T], ignore_index=True)
        long_df = pd.concat([long_df, pd.Series({'Case': identifier, 'Metric': 'KRA_left', 'Rater': 'Auto', 'Score': auto_left['KRA']}).to_frame().T], ignore_index=True)

    long_df['Score'] = long_df['Score'].astype(float)

    for metric in long_df['Metric'].unique():
        metric_df = long_df[long_df['Metric'] == metric]
        metric_df = metric_df.dropna()
        metric_df = metric_df.drop(columns=['Metric'])
        print(metric)
        # print(metric_df)

        try:
            icc = pg.intraclass_corr(data=metric_df, targets='Case', raters='Rater', ratings='Score', nan_policy='omit')
        except AssertionError as e:
            print('Not enough data for ICC calculation', e)
            continue

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(icc)

        wide_df = metric_df.pivot(index='Case', columns='Rater', values='Score')
        dev = wide_df['Felix'] - wide_df['Auto']
        dev = dev.abs()
        print(dev.describe())
        print('------------------')
