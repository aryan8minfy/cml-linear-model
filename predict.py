import joblib
import pandas as pd

model = joblib.load("model.pkl")
test = pd.DataFrame({"x": [5, 6, 7]})
predictions = model.predict(test)

for i, pred in enumerate(predictions):
    print(f"x={test.iloc[i,0]} => y={pred}")
