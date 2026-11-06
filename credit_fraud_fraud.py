import numpy as np
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

file_path = "creditcard.csv"
FEATURES = [
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount", "Time",
]
TARGET = "Class"
TEST_SIZE = 0.2
VAL_SIZE = 0.2  # fraction of the full dataset held out as validation
RANDOM_STATE = 42
FP_COST = 5
FN_COST = 500
THRESHOLDS = np.arange(0.01, 1.00, 0.01) # thresholds to test


def load_data():
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "mlg-ulb/creditcardfraud",
        file_path,
        pandas_kwargs={"compression": "zip"},
    )
    return df


def split_data(df):
    X = df[FEATURES]
    y = df[TARGET]

    # Hold out test, then split remaining into train / validation.
    # Val_size / (1 - Test_size) keeps overall validation ≈ Val_size of all data.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=VAL_SIZE / (1 - TEST_SIZE),
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def make_scaled_logistic_regression_pipeline(class_weight=None):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=RANDOM_STATE, class_weight=class_weight),
    )


def fit_model(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)
    return pipeline


def predict_at_threshold(y_proba, threshold):
    return (y_proba >= threshold).astype(int)


def total_cost(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fp * FP_COST + fn * FN_COST


def evaluate_at_half(name, pipeline, X_val, y_val):
    y_pred = pipeline.predict(X_val)
    y_proba = pipeline.predict_proba(X_val)[:, 1]

    print(f"{name} (validation)")
    print("Confusion matrix:")
    print(confusion_matrix(y_val, y_pred))
    print(f"Precision @ 0.5: {precision_score(y_val, y_pred):.4f}")
    print(f"Recall @ 0.5: {recall_score(y_val, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_val, y_proba):.4f}")
    print(f"Average precision: {average_precision_score(y_val, y_proba):.4f}")
    print()
    return y_proba


def choose_threshold(y_true, y_proba):
    best_threshold = None
    best_cost = float("inf")
    for threshold in THRESHOLDS:
        y_pred = predict_at_threshold(y_proba, threshold)
        cost = total_cost(y_true, y_pred)
        if cost < best_cost:
            best_cost = cost
            best_threshold = float(threshold)
    return best_threshold, best_cost


def main():
    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    models = {
        "Baseline": make_scaled_logistic_regression_pipeline(),
        "Weighted": make_scaled_logistic_regression_pipeline(class_weight="balanced"),
    }

    best_name = None
    best_model = None
    best_threshold = None
    best_val_cost = float("inf")

    for name, pipeline in models.items():
        pipeline = fit_model(pipeline, X_train, y_train)
        y_val_proba = evaluate_at_half(name, pipeline, X_val, y_val)
        threshold, cost = choose_threshold(y_val, y_val_proba)
        print(f"{name} chosen threshold={threshold:.2f}, validation cost=GBP {cost}")
        print()

        if cost < best_val_cost:
            best_val_cost = cost
            best_name = name
            best_model = pipeline
            best_threshold = threshold

    y_test_proba = best_model.predict_proba(X_test)[:, 1]
    y_test_pred = predict_at_threshold(y_test_proba, best_threshold)
    test_cost = total_cost(y_test, y_test_pred)

    print(f"Final check: {best_name} @ threshold={best_threshold:.2f}")
    print(f"Test cost: GBP {test_cost}")
    print(f"Test precision: {precision_score(y_test, y_test_pred):.4f}")
    print(f"Test recall: {recall_score(y_test, y_test_pred):.4f}")



if __name__ == "__main__":
    main()
