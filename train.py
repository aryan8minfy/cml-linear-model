import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv("NewExp.csv")
X = df[["x"]]
y = df["y"]

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "model.pkl")

print(f"Model Coef: {model.coef_[0]}")
