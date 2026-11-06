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
Features = [
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount", "Time",
]
Target = "Class"
Test_size = 0.2
Val_size = 0.2  # fraction of the full dataset held out as validation
Random_state = 42


def load_data():
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "mlg-ulb/creditcardfraud",
        file_path,
        pandas_kwargs={"compression": "zip"},
    )
    return df


def split_data(df):
    X = df[Features]
    y = df[Target]

    # Hold out test, then split remaining into train / validation.
    # Val_size / (1 - Test_size) keeps overall validation ≈ Val_size of all data.
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=Test_size,
        random_state=Random_state,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=Val_size / (1 - Test_size),
        random_state=Random_state,
        stratify=y_train_val,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def make_scaled_logistic_regression_pipeline():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=Random_state),
    )


def fit_model(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_baseline(pipeline, X_val, y_val):
    y_pred = pipeline.predict(X_val)
    y_proba = pipeline.predict_proba(X_val)[:, 1]

    print("Baseline LogisticRegression (validation)")
    print("Confusion matrix:")
    print(confusion_matrix(y_val, y_pred))
    print(f"Precision @ 0.5: {precision_score(y_val, y_pred):.4f}")
    print(f"Recall @ 0.5: {recall_score(y_val, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_val, y_proba):.4f}")
    print(f"Average precision: {average_precision_score(y_val, y_proba):.4f}")


def main():
    df = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    pipeline = make_scaled_logistic_regression_pipeline()
    pipeline = fit_model(pipeline, X_train, y_train)
    evaluate_baseline(pipeline, X_val, y_val)


if __name__ == "__main__":
    main()
