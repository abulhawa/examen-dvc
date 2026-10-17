import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    df = pd.read_csv("data/raw_data/raw.csv")

    X = df.drop(columns=["date", "silica_concentrate"])
    y = df[["silica_concentrate"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    for name, data in {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }.items():
        data.to_csv("data/processed_data/" + f"{name}.csv", index=False)
        print(f"{name}: {data.shape}, dtypes: {set(data.dtypes)}")


if __name__ == "__main__":
    main()
