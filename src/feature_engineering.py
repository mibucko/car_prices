import numpy as np
import pandas as pd

# Creating feature car_age
def _car_age(df: pd.DataFrame) -> pd.DataFrame: 
    df = df.copy() 
    df["car_age"] = (2019 - df["year"].astype(int))
    df = df.drop(columns=["year"])
    return df 

# Creating feature mileage_per_year
def _mileage_per_year(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mileage_per_year"] = np.where(
        df["car_age"] > 0,
        df["mileage(kilometers)"] / df["car_age"].astype(float),
        df["mileage(kilometers)"])
    return df

# Creating pipeline
def engineering(df: pd.DataFrame) -> pd.DataFrame:
    df_featured = (df
        .pipe(_car_age)
        .pipe(_mileage_per_year))
    return df_featured

# Defining pipeline
def main(input_file: str, output_file: str) -> None:
    df_cleaned = pd.read_csv(input_file)
    df_feature = engineering(df_cleaned)
    df_feature.to_csv(output_file, index=False)
    print(f"Featured dataset saved to: {output_file}")

# Pipeline execution
main("cars_clean_1.csv", "cars_features_1.csv")
main("cars_clean_2.csv", "cars_features_2.csv")
main("cars_clean_3.csv", "cars_features_3.csv")

