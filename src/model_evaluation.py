# We evaluate all 12 models. We show metrics, and evaluate models with several random values.

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

# Model names
model_names = ["Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]

# Loading 12 trained models; making prediction; calculating metrics; saving evaluation results to a new df
results = []
for i, df in enumerate(datasets, start=1):
    X, y = split_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    for model_name in model_names:
        filename = f"{model_name.lower().replace(' ', '_')}_{i}.joblib"
        model = joblib.load(filename)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, y_pred)
        metrics = pd.DataFrame({"metric": ["MAE", "MSE", "RMSE", "R2"],
                                "value": [mae, mse, rmse, r2],})
        print(f"\nDataset {i}, model {model_name}")
        print("\nRegression metrics:")
        print(metrics)

        # Table for comparison random real and predicted values
        prediction_analysis = pd.DataFrame({"actual_price": y_test, "predicted_price": y_pred})
        prediction_analysis["error"] = (prediction_analysis["actual_price"] - prediction_analysis["predicted_price"])
        prediction_analysis["absolute_error"] = (prediction_analysis["error"].abs())
        print("\nPrediction examples:")
        print(prediction_analysis.sample(5, random_state=42))

        # Highest model mistakes
        print("\nLargest prediction errors:")
        print(prediction_analysis.sort_values("absolute_error", ascending=False).head(10))

        # Saving metrics and error values into df
        largest_errors = (prediction_analysis["absolute_error"].sort_values(ascending=False).head(3).tolist())
        results.append({"dataset": i,
                        "model": model_name,
                        "MAE": mae,
                        "MSE": mse,
                        "RMSE": rmse,
                        "R2": r2,
                        "largest_error_1": largest_errors[0],
                        "largest_error_2": largest_errors[1],
                        "largest_error_3": largest_errors[2],})

results_df = pd.DataFrame(results)
results_df.to_csv("model_evaluation_results.csv", index=False)
print("\nModel evaluation results saved")

       

