import sklearn.datasets as datasets
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import  r2_score
from sklearn.model_selection import cross_val_score



RANDOM_SEED = 42
TEST_SIZE = 0.2

def load_data():
    california = datasets.fetch_california_housing()
    X = california.data
    y = california.target
    return X, y

def model_train(X_train, y_train, max_depth):
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    return model

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    return X_train, X_test, y_train, y_test

def model_evaluate(model, X_test, y_test):
    return model.score(X_test, y_test)

def model_predict(model, X_test):
    return model.predict(X_test)

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    max_depth = [1, 2, 3, 5, 8, 12, 20, None]
    for depth in max_depth:
        model = model_train(X_train, y_train, depth)
        print(f"Max depth: {depth}, R^2 score: {model_evaluate(model, X_test, y_test)}")
        print(f"Predictions: {model_predict(model, X_test)}")
        
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Cross-validation mean score: {cv_scores.mean()}")
        print(f"Cross-validation std score: {cv_scores.std()}")
        
        
        r2 = r2_score(y_test, model_predict(model, X_test))
        print(f"R^2 score: {r2}")
        

if __name__ == "__main__":
    main()