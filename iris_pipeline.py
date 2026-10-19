from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y


def make_model(n_neighbors):
    # Scale inside the pipeline so each CV fold fits the scaler on training folds only
    return Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=n_neighbors)),
    ])


def main():
    X, y = load_data()
    for n_neighbors in range(1, 10, 2):
        model = make_model(n_neighbors)
        scores = cross_val_score(model, X, y, cv=5)
        print(
            f"k={n_neighbors}: mean={scores.mean():.4f}, "
            f"std={scores.std():.4f}"
        )


if __name__ == "__main__":
    main()
