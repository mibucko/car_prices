# We read 3 data files, and train 4 models. In total, 12 training operations.

import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from data_preprocessing import (TARGET_COLUMN, split_features_and_target, build_preprocessor)

# Data
df1 = pd.read_csv('cars_features_1.csv')
df2 = pd.read_csv('cars_features_2.csv')
df3 = pd.read_csv('cars_features_3.csv')
datasets = [df1, df2, df3]

# Models
models = {"Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),}

# Features splitting, train/test split, model training and saving: we create 12 models.
for i, df in enumerate(datasets, start=1):
    X, y = split_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for model_name, regressor in models.items():
        model = Pipeline(steps=[("preprocessor", build_preprocessor()), ("regressor", regressor),])
        model.fit(X_train, y_train)
        filename = f"{model_name.lower().replace(' ', '_')}_{i}.joblib"
        joblib.dump(model, filename)
        print(f"Model saved to: {filename}")

