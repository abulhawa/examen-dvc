import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


with open("models/model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

X_test = pd.read_csv("data/processed_data/X_test_scaled.csv")
y_test = pd.read_csv("data/processed_data/y_test.csv")["silica_concentrate"]
predictions = model.predict(X_test)

pd.DataFrame({"silica_concentrate_predicted": predictions}).to_csv(
    "data/predictions.csv", index=False
)

scores = {
    "mse": float(mean_squared_error(y_test, predictions)),
    "r2": float(r2_score(y_test, predictions)),
}
Path("metrics").mkdir(parents=True, exist_ok=True)
with open("metrics/scores.json", "w", encoding="utf-8") as scores_file:
    json.dump(scores, scores_file, indent=2, allow_nan=False)
    scores_file.write("\n")

print(json.dumps(scores, indent=2))
