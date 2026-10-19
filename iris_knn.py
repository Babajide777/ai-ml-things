from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, n_neighbors):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    for n_neighbors in range(1, 10, 2):        
        print(f"Training model with {n_neighbors} neighbors")
        model = train_model(X_train, y_train, n_neighbors)
        accuracy = evaluate_model(model, X_test, y_test)
        print(f"Accuracy for {n_neighbors} neighbors: {accuracy}")


if __name__ == "__main__":
    main()
