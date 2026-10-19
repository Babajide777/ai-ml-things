# AI / ML Things

A growing collection of small, self-contained machine learning experiments. Each mini-project lives in its own file (or folder) and can be run on its own.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Some projects pull data from [Kaggle](https://www.kaggle.com/) via `kagglehub`. Authenticate once if prompted when you first run them.

## Mini-projects

| Project | File | Topics |
| --- | --- | --- |
| [Mall Customer Segmentation](#mall-customer-segmentation) | `mall_customer_segmentation.py` | K-Means, elbow method, feature scaling |
| [Iris KNN](#iris-knn) | `iris_knn.py` | KNN, train/test split, StandardScaler |
| [Iris Pipeline + CV](#iris-pipeline--cv) | `iris_pipeline.py` | Pipeline, cross-validation, KNN |

---

### Mall Customer Segmentation

Clusters mall shoppers by **annual income** and **spending score** using K-Means.

**What it does**
1. Loads the [Mall Customers](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) dataset
2. Standardizes the two features
3. Plots an elbow curve to help choose `k`
4. Fits K-Means (`k=5` by default), plots segments + centroids
5. Labels each cluster with low / average / high income and spending bands

**Run**

```bash
python mall_customer_segmentation.py
```

---

### Iris KNN

Classifies Iris flowers with K-Nearest Neighbors, comparing odd values of `k`.

**What it does**
1. Loads the built-in Iris dataset
2. Stratified train/test split
3. Standardizes features (fit on train, transform test)
4. Trains KNN for `k = 1, 3, 5, 7, 9` and prints test accuracy

**Run**

```bash
python iris_knn.py
```

---

### Iris Pipeline + CV

Same Iris + KNN setup, but scaling and the classifier are wrapped in a `Pipeline`, evaluated with 5-fold cross-validation.

**What it does**
1. Loads the Iris dataset
2. Builds a `Pipeline(StandardScaler → KNeighborsClassifier)` for each odd `k`
3. Runs `cross_val_score(..., cv=5)` so the scaler is fit only on each fold’s training data
4. Prints mean and standard deviation of the five fold scores

**Run**

```bash
python iris_pipeline.py
```

---

<!--
Template for the next mini-project — copy under Mini-projects table + a section below:

| [Project Name](#project-name) | `your_file.py` | topic1, topic2 |

### Project Name

Short description of what the script explores.

**What it does**
1. Step one
2. Step two

**Run**

```bash
python your_file.py
```
-->

## License

Personal learning repo — use and adapt freely.
