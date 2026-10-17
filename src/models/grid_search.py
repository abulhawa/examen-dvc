import pickle

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold


X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv")["silica_concentrate"]

search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42, n_jobs=1),
    param_grid={
        "n_estimators": [100, 200],
        "max_depth": [5, 10, None],
    },
    scoring="neg_mean_squared_error",
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1,
    refit=False,
)
search.fit(X_train, y_train)

with open("models/best_params.pkl", "wb") as output:
    pickle.dump(search.best_params_, output)

print(f"Best parameters: {search.best_params_}")
