Car price prediction

A model is developed to predict a car price.

We started with Colab notebook, 01_eda.ipynb.
Here we load the file 'cars.csv', analyze missing values and decide to create two options: a file where only rows with more than one missing value are deleted, and the other where all rows with missing values are deleted.
We also decided to correct the illogical values for engine volume. Namely, for volumes higher than 10000, we divide these values by ten.

The next stage is data_cleaning.py.
In this process, we create three files, as explained above: the first option is to delete only rows containing > 1 missing values (pipeline clean_1).
The second option is to delete all rows with missing values (pipeline clean_2). The third option is to leave the data without deleting extreme values or correcting the engine volume.
As a result, 3 files are created: cars_clean_1, cars_clean_2, cars_clean_3

Next stage was feature_engineering.py, where two new features (car_age and mileage_per_year) were created.
As a result, 3 files are created: cars_features_1, cars_features_2, cars_features_3

Next stage was data preprocessing.py, where data were prepared for model training. The only ordinal category is 'condition'.

Next stage was model_evaluation.py, where we used a "for loop" to evaluate 12 combinations: dataset cars_clean_1 with 4 algorithms, cars_clean_2 with 4 algorithms and cars_clean_3 with 4 algorithms. Nine of them are uploaded to GitHub, but 'random forest' models are too big for uploading.

Final stage is Colab notebook 'model_comparison.ipynb'. Using diagrams, we may observe that the optimal combination of dataset and algorithm is dataset_3 + random forest.

This trained model may not be found on GiHub as joblib file, since it is too big. To create the model with the best performance, one should follow the path:
cars.csv - data_cleaning.py - cars_clean_1.csv - feature_engineering.py - cars_features_3.csv - preprocessing.py - optimal_model_training_evaluation.py

Since we now know which model is the best, we created a new python file (optimal_model_training_evaluation.py), specifically for creating and evaluating the optimal model. This file demands significantly lower time than the model_training.py file, that trains 12 combinations.
