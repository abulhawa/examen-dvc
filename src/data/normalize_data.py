import pandas as pd
from sklearn.preprocessing import StandardScaler


X_train = pd.read_csv('data/processed_data/X_train.csv')
X_test = pd.read_csv('data/processed_data/X_test.csv')
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(data=X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(data=X_test_scaled, columns=X_test.columns)

X_train_scaled.to_csv('data/processed_data/X_train_scaled.csv', index=False)
X_test_scaled.to_csv('data/processed_data/X_test_scaled.csv', index=False)
