import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


RANDOM_SEED = 42
TEST_SIZE = 0.2
TARGET_UNIT_USD = 100_000  # California target is median house value in $100,000s
MAX_DEPTHS = [1, 2, 3, 5, 8, 12, 20, None]


def load_data():
    california = fetch_california_housing()
    return california.data, california.target


def split_data(X, y):
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)


def make_tree(max_depth):
    return DecisionTreeRegressor(max_depth=max_depth, random_state=RANDOM_SEED)


def depth_complexity(depth):
    # None = unrestricted = most complex
    return float("inf") if depth is None else depth


def sweep_depths(X_train, y_train, depths=MAX_DEPTHS):
    """Train R² and 5-fold CV R² (mean ± std) for each max_depth."""
    rows = []
    for depth in depths:
        model = make_tree(depth)
        model.fit(X_train, y_train)
        train_r2 = model.score(X_train, y_train)

        cv_scores = cross_val_score(
            make_tree(depth), X_train, y_train, cv=5, scoring="r2"
        )
        rows.append({
            "max_depth": depth,
            "train_r2": train_r2,
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
        })
    return pd.DataFrame(rows)


def pick_depth(sweep):
    """
    Choose depth by CV mean. If another depth's CV mean is within one std
    of the best mean, prefer the simpler (shallower) tree.
    """
    best_mean = sweep["cv_mean"].max()
    best_std = sweep.loc[sweep["cv_mean"].idxmax(), "cv_std"]
    within = sweep[sweep["cv_mean"] >= best_mean - best_std].copy()
    within["_complexity"] = within["max_depth"].map(depth_complexity)
    chosen = within.loc[within["_complexity"].idxmin(), "max_depth"]
    # pandas can upcast ints to float when None is in the column
    if pd.isna(chosen):
        return None
    return int(chosen)


def plot_depth_curves(sweep):
    labels = [str(d) for d in sweep["max_depth"]]
    x = np.arange(len(labels))

    plt.figure()
    plt.plot(x, sweep["train_r2"], marker="o", label="Train R2")
    plt.errorbar(
        x,
        sweep["cv_mean"],
        yerr=sweep["cv_std"],
        marker="o",
        capsize=4,
        label="CV R2 (mean ± std)",
    )
    plt.xticks(x, labels)
    plt.xlabel("max_depth")
    plt.ylabel("R2")
    plt.title("Decision tree: train vs CV R2 by depth")
    plt.legend()
    plt.tight_layout()
    plt.show()


def regression_metrics(y_true, y_pred):
    return {
        "MAE ($)": mean_absolute_error(y_true, y_pred) * TARGET_UNIT_USD,
        "RMSE ($)": root_mean_squared_error(y_true, y_pred) * TARGET_UNIT_USD,
        "R2": r2_score(y_true, y_pred),
    }


def format_results(results):
    formatted = results.copy().astype(object)
    for col in ["MAE ($)", "RMSE ($)"]:
        formatted[col] = results[col].map(lambda v: f"${v:,.0f}")
    formatted["R2"] = results["R2"].map(lambda v: f"{v:.4f}")
    return formatted


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    sweep = sweep_depths(X_train, y_train)
    print("Depth sweep (train R2 and 5-fold CV R2)")
    print(
        sweep.assign(
            cv=lambda df: df.apply(
                lambda r: f"{r.cv_mean:.4f} ± {r.cv_std:.4f}", axis=1
            )
        )[["max_depth", "train_r2", "cv"]]
        .to_string(index=False)
    )

    plot_depth_curves(sweep)

    chosen_depth = pick_depth(sweep)
    print(f"\nChosen max_depth by CV (prefer simpler within 1 std): {chosen_depth}")

    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)

    linear = make_pipeline(StandardScaler(), LinearRegression())
    linear.fit(X_train, y_train)

    tree = make_tree(chosen_depth)
    tree.fit(X_train, y_train)

    results = pd.DataFrame({
        "DummyRegressor (mean)": regression_metrics(y_test, baseline.predict(X_test)),
        "LinearRegression (scaled)": regression_metrics(y_test, linear.predict(X_test)),
        f"DecisionTree (depth={chosen_depth})": regression_metrics(
            y_test, tree.predict(X_test)
        ),
    }).T
    print("\nTest-set metrics (errors in US dollars)")
    print(format_results(results).to_string())


if __name__ == "__main__":
    main()
