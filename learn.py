import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer,StandardScaler,Normalizer,PowerTransformer
from sklearn.model_selection import KFold,cross_val_score

# from sklearn import svm
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, VotingRegressor, AdaBoostRegressor, StackingRegressor 

df = pd.read_csv("balanced.csv",sep=";")
kf = KFold(n_splits=10, shuffle=True, random_state=42)

X = df.drop(columns=["pIC50","Smiles","categories"])
Y = df["pIC50"]
print("X and Y defined\n.\n.\n.")

# normalizer = Normalizer()
# X = normalizer.transform(X)
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

transformer = PowerTransformer(
    # method="box-cox",
)

X_train = transformer.fit_transform(X_train)
X_test =transformer.transform(X_test)
print("X transformed\n.\n.\n.")

selection = VarianceThreshold(threshold=(.8 * (1 - .8)))
X_train = selection.fit_transform(X_train)
X_test =selection.transform(X_test)
print("Low variance X removed\n.\n.\n.")

print("Model is Learning\n.\n.\n.")
model = RandomForestRegressor(
    random_state=42,
    n_estimators=500,
    max_depth=None

    )

model.fit(X_train, Y_train)
Y_pred=model.predict(X_test)
rsquare = model.score(X_test, Y_test)
print("Model Learned")
print(Y_pred)
print(rsquare)

import matplotlib.pyplot as plt


# Plotting
plt.plot(Y_pred, Y_test , 'o', label='Prediction vs True')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Prediction vs True Values')
plt.legend()

# Save the plot to a file (e.g., PNG, PDF, etc.)
plt.savefig('prediction_vs_true_plot.png')


# cv_results = cross_val_score(model, X_train, Y_train, cv=kf, scoring='neg_mean_squared_error')
# print("Cross-Validation Results (Mean Squared Error):", cv_results)
# print("Mean MSE: {:.2f}".format(cv_results.mean()))
# print("Standard Deviation: {:.2f}".format(cv_results.std()))
