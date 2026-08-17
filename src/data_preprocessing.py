import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Target and input columns
TARGET_COLUMN = "priceUSD"
NUMERIC_FEATURES = ["car_age",
    "mileage(kilometers)",
    "mileage_per_year",
    "volume(cm3)",]
CATEGORICAL_FEATURES = ["make",
    "model",
    "fuel_type",
    "color",
    "transmission",
    "drive_unit",
    "segment",]
ORDINAL_FEATURES = ["condition",]
def get_all_feature_columns() -> list[str]:
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES + ORDINAL_FEATURES

# feature splitting
def split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[get_all_feature_columns()].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y

# Preprocessing numerical columns
def _build_numeric_transformer() -> Pipeline:
    numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(missing_values=pd.NA, strategy="median")),
            ("scaler", StandardScaler()),])
    return numeric_transformer

# Preprocessing nominal category columns
def _build_categorical_transformer() -> Pipeline:
    categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(missing_values=pd.NA, strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),])
    return categorical_transformer

# Preprocessing ordinal category columns
def _build_ordinal_transformer() -> Pipeline:
    ordinal_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(missing_values=pd.NA, strategy="most_frequent")),
            ("encoder", OrdinalEncoder(categories=[['with mileage', 'with damage', 'for parts']])),])
    return ordinal_transformer

# Merging with ColumnTransformera
def build_preprocessor() -> ColumnTransformer:
    preprocessor = ColumnTransformer(transformers=[
            ("num", _build_numeric_transformer(), NUMERIC_FEATURES),
            ("cat", _build_categorical_transformer(), CATEGORICAL_FEATURES),
            ("ord", _build_ordinal_transformer(), ORDINAL_FEATURES),], remainder="drop")
    return preprocessor

print('Functions ready')