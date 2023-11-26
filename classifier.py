import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer,StandardScaler,Normalizer,PowerTransformer
from sklearn.model_selection import KFold,cross_val_score

from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("balanced.csv",sep=";")
X = df.drop(columns=["pIC50","Smiles","categories"])
Y = df["categories"]
print("X and Y defined\n.\n.\n.")

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=Y,
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

model = RandomForestClassifier(
    random_state=42,
    n_estimators=500,
    # max_depth=None,
    
    )
print("Model is Learning\n.\n.\n.")
model.fit(X_train, Y_train)
print("Model Learned\n.\n.\n.")
Y_pred=model.predict(X_test)
print("Model is Predicting\n.\n.\n.")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Assuming y_true and y_pred are your true labels and predicted labels, respectively
accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred, average='macro')
recall = recall_score(Y_test, Y_pred, average='macro')
f1 = f1_score(Y_test, Y_pred, average='macro')   

print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")