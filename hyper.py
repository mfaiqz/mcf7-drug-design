import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import QuantileTransformer,StandardScaler,Normalizer
from sklearn.model_selection import KFold,cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


df = pd.read_csv("balanced.csv",sep=";")

X = df.drop(columns=["pIC50","Smiles","categories"])
Y = df["pIC50"]
print("X and Y defined\n.\n.\n.")

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=df["categories"],
    )

print("X and Y divided")
print(
f"""Data Split Details
- train = {len(X_train)}
- test = {len(X_test)}\n.\n.\n.
"""
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("X standardized\n.\n.\n.")

transformer= QuantileTransformer(
    random_state=42,
    output_distribution="normal"
)
X_train = transformer.fit_transform(X_train)
X_test =transformer.transform(X_test)
print("X transformed\n.\n.\n.")

selection = VarianceThreshold(threshold=(.8 * (1 - .8)))
X_train = selection.fit_transform(X_train)
X_test =selection.transform(X_test)
print("Low variance X removed\n.\n.\n.")

print("Model is Learning\n.\n.\n.")


param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

rf = RandomForestRegressor()

grid_search = GridSearchCV(
    rf, 
    param_grid, 
    cv=5, 
    scoring='neg_mean_squared_error'
    )

grid_search.fit(X_train, Y_train)

best_params = grid_search.best_params_

# Step 7: Evaluate the model with the best hyperparameters on the test set
final_model = grid_search.best_estimator_
Y_pred = final_model.predict(X_test)

# Step 8: Calculate evaluation metrics
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

# Step 9: Print the results
print(f"Best Hyperparameters: {best_params}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")