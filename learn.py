import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer,StandardScaler,Normalizer,PowerTransformer, MinMaxScaler
from sklearn.model_selection import KFold,cross_val_score
from sklearn.externals import joblib
import pickle

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
scaler = MinMaxScaler()
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
print(X_train.shape)
print("Low variance X removed\n.\n.\n.")


# transformer = PowerTransformer(
#     # method="box-cox",
# )



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

pd.DataFrame({
    "Y_pred":Y_pred,
    "Y_test":Y_test
}).to_csv("result.csv",index=False)


# Export the trained model to a file
with open('model_filename.pkl', 'wb') as file:
    pickle.dump(model, file)

# Export the trained model to a file
joblib.dump(model, 'model_filename.joblib')
