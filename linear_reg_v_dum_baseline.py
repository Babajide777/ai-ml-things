import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_SEED = 42
TEST_SIZE = 0.2


def load_data():
    california = fetch_california_housing()
    X = california.data
    y = california.target
    return X, y


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    return X_train, X_test, y_train, y_test


def make_baseline():
    return DummyRegressor(strategy="mean")


def make_model():
    return make_pipeline(StandardScaler(), LinearRegression())


def regression_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    baseline = make_baseline()
    baseline.fit(X_train, y_train)

    model = make_model()
    model.fit(X_train, y_train)

    results = pd.DataFrame({
        "DummyRegressor (mean)": regression_metrics(y_test, baseline.predict(X_test)),
        "LinearRegression (scaled)": regression_metrics(y_test, model.predict(X_test)),
    }).T
    print("Test-set metrics")
    print(results.round(4).to_string())

    print()
    print("LinearRegression train vs test R2")
    print("(large train-test gap => overfitting; both low => underfitting)")
    print(f"  train R2: {model.score(X_train, y_train):.4f}")
    print(f"  test  R2: {model.score(X_test, y_test):.4f}")


if __name__ == "__main__":
    main()
