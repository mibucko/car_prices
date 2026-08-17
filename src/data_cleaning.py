# The first option is to delete only rows containing > 1 missing values (pipeline clean_1).
# The second option is to dellete all rows witn missing values (pipeline clean_2).
# The third option is to leave the data without deleting extreme values or correcting the engine volume.
# All options will be evaluated.

import pandas as pd

# Standardization of text columns
def _strip_string_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    text_columns = ['make', 'model', 'condition', 'fuel_type', 'color', 'transmission', 'drive_unit', 'segment']
    for col in text_columns:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
    return df

# Conversion of text columns to category
def _convert_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    text_columns = ['make', 'model', 'condition', 'fuel_type', 'color', 'transmission', 'drive_unit', 'segment']
    for col in text_columns:
        df[col] = df[col].astype('category')
    return df

# first case: dropping only rows with > 1 missing values
def _remove_rows_with_more_than_1_nan(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df.isna().sum(axis=1) <= 1]
    return df

# second case: dropping all rows with  missing values
def _remove_rows_with_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna()
    return df

# Dropping extreme values
def _remove_extreme_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[
        (df['priceUSD'] > 150) &
        (df['priceUSD'] < 140000) &
        (df['year'] >= 1971) &
        (df['mileage(kilometers)'] < 900000) &
        ~((df['mileage(kilometers)'] == 0) & (df['year'] < 2015))]
    return df

# Correction of engine volume
def _correct_volume(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mask = df['volume(cm3)'] >= 10000
    df.loc[mask, 'volume(cm3)'] = df.loc[mask, 'volume(cm3)'] / 10
    return df

# Creating first pipeline
def clean_1(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = (df
        .pipe(_strip_string_values)
        .pipe(_convert_category)
        .pipe(_remove_rows_with_more_than_1_nan)
        .pipe(_remove_extreme_values)
        .pipe(_correct_volume))
    return df_clean
# Creating second pipeline
def clean_2(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = (df
        .pipe(_strip_string_values)
        .pipe(_convert_category)
        .pipe(_remove_rows_with_missing_values)
        .pipe(_remove_extreme_values)
        .pipe(_correct_volume))
    return df_clean
# Creating third pipeline
def clean_3(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = (df
        .pipe(_strip_string_values)
        .pipe(_convert_category)
        .pipe(_remove_rows_with_missing_values))
    return df_clean

# Defining first pipeline
def main_1(input_file: str, output_file: str) -> None:
    df_raw = pd.read_csv(input_file)
    df_cleaned = clean_1(df_raw)
    df_cleaned.to_csv(output_file, index=False)
    print(f"Cleaned dataset saved to: {output_file}")
# Defining second pipeline
def main_2(input_file: str, output_file: str) -> None:
    df_raw = pd.read_csv(input_file)
    df_cleaned = clean_2(df_raw)
    df_cleaned.to_csv(output_file, index=False)
    print(f"Cleaned dataset saved to: {output_file}")
# Defining third pipeline
def main_3(input_file: str, output_file: str) -> None:
    df_raw = pd.read_csv(input_file)
    df_cleaned = clean_3(df_raw)
    df_cleaned.to_csv(output_file, index=False)
    print(f"Cleaned dataset saved to: {output_file}")

# Pipeline execution
main_1("cars.csv", "cars_clean_1.csv")
main_2("cars.csv", "cars_clean_2.csv")
main_3("cars.csv", "cars_clean_3.csv")