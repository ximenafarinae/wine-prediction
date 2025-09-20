def convert_columns_values_to_numeric(pd, df, columns):
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def remove_outliers_percentile(df, columns, lower_percentile=0.01, upper_percentile=0.95):
    for column in columns:
        lower_bound = df[column].quantile(lower_percentile)
        upper_bound = df[column].quantile(upper_percentile)
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

def normalize_input(df):
    df_stats = df.describe().T
    df_norm = (df - df_stats['mean']) / df_stats['std']
    return df_norm

def remove_outliers_iqr(_df, _column, factor=1.5):
    Q1 = _df[_column].quantile(0.25)
    Q3 = _df[_column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR

    filter_df = _df[(_df[_column] >= lower_bound) & (_df[_column] <= upper_bound)]
    return filter_df
