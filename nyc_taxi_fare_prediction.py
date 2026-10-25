import pandas as pd
from seaborn import load_dataset
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from ml_utils import format_results, regression_metrics


RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
TARGET = "fare"
MAX_DEPTHS = [1, 2, 3, 5, 8, 12, 20, None]

NUMERIC_FEATURES = ["distance", "passengers", "pickup_hour", "pickup_dayofweek"]
CATEGORICAL_FEATURES = ["pickup_borough", "dropoff_borough"]


def load_data():
    return load_dataset("taxis")


def preprocess_data(df):
    df = df.copy()
    df = df.drop(columns=["tip", "tolls", "total"])

    pickup = pd.to_datetime(df["pickup"])
    df["pickup_hour"] = pickup.dt.hour
    df["pickup_dayofweek"] = pickup.dt.dayofweek  # 0=Mon ... 6=Sun

    missing_borough = df["pickup_borough"].isna() | df["dropoff_borough"].isna()
    n_dropped_borough = int(missing_borough.sum())
    df = df.loc[~missing_borough].copy()
    print(f"Dropped {n_dropped_borough} rows with missing pickup/dropoff borough")

    keep = [TARGET, *NUMERIC_FEATURES, *CATEGORICAL_FEATURES]
    df = df[keep].dropna().drop_duplicates()
    return df


def split_xy(df):
    y = df[TARGET]
    X = df.drop(columns=[TARGET])
    return X, y


def split_data(X, y):
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )


def make_preprocessor(scale_numeric):
    numeric = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def make_pipeline(estimator, scale_numeric=False):
    return Pipeline(
        steps=[
            ("prep", make_preprocessor(scale_numeric=scale_numeric)),
            ("model", estimator),
        ]
    )


def make_dummy():
    return make_pipeline(DummyRegressor(strategy="mean"), scale_numeric=False)


def make_linear():
    return make_pipeline(LinearRegression(), scale_numeric=True)


def make_tree(max_depth):
    return make_pipeline(
        DecisionTreeRegressor(max_depth=max_depth, random_state=RANDOM_SEED),
        scale_numeric=False,
    )


def make_forest():
    return make_pipeline(
        RandomForestRegressor(n_estimators=300, random_state=RANDOM_SEED),
        scale_numeric=False,
    )


def make_hgb():
    return make_pipeline(
        HistGradientBoostingRegressor(random_state=RANDOM_SEED),
        scale_numeric=False,
    )


def make_cv():
    return KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)


def cv_mae(model, X, y):
    """Return (mean, std) of CV MAE using shuffled KFold."""
    out = cross_validate(
        model,
        X,
        y,
        cv=make_cv(),
        scoring="neg_mean_absolute_error",
    )
    mae = -out["test_score"]
    return float(mae.mean()), float(mae.std())


def depth_complexity(depth):
    return float("inf") if depth is None else depth


def depth_label(depth):
    return "None" if depth is None else str(depth)


def sweep_tree_depths(X_train, y_train, depths=MAX_DEPTHS):
    rows = []
    for depth in depths:
        mean, std = cv_mae(make_tree(depth), X_train, y_train)
        rows.append({
            "depth_label": depth_label(depth),
            "max_depth": depth,
            "cv_mae_mean": mean,
            "cv_mae_std": std,
        })
    return pd.DataFrame(rows)


def pick_depth(sweep):
    """
    Lowest CV MAE wins. If another depth is within 1 std of the best mean,
    prefer the simpler (shallower) tree.
    """
    best_mean = sweep["cv_mae_mean"].min()
    best_std = sweep.loc[sweep["cv_mae_mean"].idxmin(), "cv_mae_std"]
    within = sweep[sweep["cv_mae_mean"] <= best_mean + best_std].copy()
    within["_complexity"] = within["max_depth"].map(depth_complexity)
    chosen = within.loc[within["_complexity"].idxmin(), "max_depth"]
    if pd.isna(chosen):
        return None
    return int(chosen)


def pick_best_by_cv_mae(cv_table):
    """Same tie rule across models: within 1 std of best MAE, prefer listed earlier (= simpler)."""
    best_mean = cv_table["cv_mae_mean"].min()
    best_std = cv_table.loc[cv_table["cv_mae_mean"].idxmin(), "cv_mae_std"]
    within = cv_table[cv_table["cv_mae_mean"] <= best_mean + best_std]
    # keep original order as complexity preference among tied models
    return within.iloc[0]


def main():
    df = preprocess_data(load_data())
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # --- Tune tree depth on train CV MAE ---
    tree_sweep = sweep_tree_depths(X_train, y_train)
    print("\nDecision tree depth sweep (CV MAE)")
    print(
        tree_sweep.assign(
            cv_mae=lambda d: d.apply(
                lambda r: f"{r.cv_mae_mean:.4f} ± {r.cv_mae_std:.4f}", axis=1
            )
        )[["depth_label", "cv_mae"]].to_string(index=False)
    )
    chosen_depth = pick_depth(tree_sweep)
    print(f"Chosen tree max_depth: {depth_label(chosen_depth)}")

    models = {
        "DummyRegressor (mean)": make_dummy(),
        "LinearRegression": make_linear(),
        f"DecisionTree (depth={depth_label(chosen_depth)})": make_tree(chosen_depth),
        "RandomForest (n=300)": make_forest(),
        "HistGradientBoosting": make_hgb(),
    }

    # --- CV MAE on training data only ---
    cv_rows = []
    for name, model in models.items():
        mean, std = cv_mae(model, X_train, y_train)
        cv_rows.append({"model": name, "cv_mae_mean": mean, "cv_mae_std": std})
    cv_table = pd.DataFrame(cv_rows)
    print("\nCV MAE on training set (shuffled KFold)")
    print(
        cv_table.assign(
            cv_mae=lambda d: d.apply(
                lambda r: f"{r.cv_mae_mean:.4f} ± {r.cv_mae_std:.4f}", axis=1
            )
        )[["model", "cv_mae"]].to_string(index=False)
    )

    winner = pick_best_by_cv_mae(cv_table)
    best = cv_table.loc[cv_table["cv_mae_mean"].idxmin()]
    lead = best["cv_mae_mean"]  # lower is better; compare gap to next
    others = cv_table[cv_table["model"] != best["model"]]
    next_best = others["cv_mae_mean"].min()
    lead_over_next = next_best - best["cv_mae_mean"]
    lead_bigger_than_std = lead_over_next > best["cv_mae_std"]

    tree_row = cv_table[cv_table["model"].str.startswith("DecisionTree")].iloc[0]
    forest_row = cv_table[cv_table["model"].str.startswith("RandomForest")].iloc[0]
    hgb_row = cv_table[cv_table["model"].str.startswith("HistGradientBoosting")].iloc[0]

    # --- Final fit on full train; test metrics once ---
    test_rows = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        test_rows[name] = regression_metrics(
            y_test, model.predict(X_test), target_unit_usd=1.0
        )
    test_table = pd.DataFrame(test_rows).T
    print("\nTest-set metrics (already in US dollars)")
    print(format_results(test_table).to_string())

    print("\n--- Answers ---")
    print(
        f"CV MAE winner (tie rule): {winner['model']} "
        f"({winner['cv_mae_mean']:.4f} ± {winner['cv_mae_std']:.4f}). "
        f"Raw best: {best['model']}; lead over next={lead_over_next:.4f}, "
        f"best std={best['cv_mae_std']:.4f}, "
        f"lead > std? {lead_bigger_than_std}."
    )
    print(
        f"Forest/boosting CV std vs single tree: "
        f"tree={tree_row['cv_mae_std']:.4f}, "
        f"forest={forest_row['cv_mae_std']:.4f}, "
        f"hgb={hgb_row['cv_mae_std']:.4f}. "
        f"Forest smaller than tree? {forest_row['cv_mae_std'] < tree_row['cv_mae_std']}. "
        f"HGB smaller than tree? {hgb_row['cv_mae_std'] < tree_row['cv_mae_std']}."
    )
    print(
        "tip/tolls/total excluded because they are known only after the ride "
        "(or are derived from fare); using them would leak future information "
        "into a fare prediction made at trip start."
    )


if __name__ == "__main__":
    main()
