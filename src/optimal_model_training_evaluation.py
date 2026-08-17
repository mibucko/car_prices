# You may use this file to create and evaluate the model with the best performance

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from data_preprocessing import (TARGET_COLUMN, split_features_and_target, build_preprocessor)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('cars_features_3.csv')

X, y = split_features_and_target(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("regressor", RandomForestRegressor(random_state=42, n_jobs=-1))])

model.fit(X_train, y_train)

joblib.dump(model, 'optimal_model.joblib')
print("Model saved to: optimal_model.joblib")

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

metrics = pd.DataFrame({"metric": ["MAE", "MSE", "RMSE", "R2"],
    "value": [mae, mse, rmse, r2],})
print("\nRegression metrics:")
print(metrics)