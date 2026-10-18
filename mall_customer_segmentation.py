import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def load_dataset():
    # Set the path to the file you'd like to load
    file_path = "Mall_Customers.csv"

    # Load the dataset
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "vjchoudhary7/customer-segmentation-tutorial-in-python",
        file_path,
    )
    return df

def data_preprocessing(df):
    # Drop the CustomerID, Gender, and Age columns
    df = df.drop(columns=['CustomerID', 'Gender', 'Age'])

    return df

def standardize_data(df):
    df_raw = df.copy()  # original income & spending score
    scaler = StandardScaler()
    df_scaled = df.copy()
    cols = ['Annual Income (k$)', 'Spending Score (1-100)']
    df_scaled[cols] = scaler.fit_transform(df[cols])
    return df_raw, df_scaled, scaler

def fit_kmeans(X, k):
    model = KMeans(n_clusters=k, random_state=42, n_init="auto")
    model.fit(X) 
    return model


def kmeans_elbow(X, max_k=10):
    inertias = []
    for k in range(1, max_k + 1):
        model = fit_kmeans(X, k)
        inertias.append(model.inertia_)
    return inertias


def plot_kmeans_elbow(X):
    inertias = kmeans_elbow(X)
    ks = range(1, len(inertias) + 1)
    plt.scatter(ks, inertias)
    plt.plot(ks, inertias)  # line helps spot the elbow
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('KMeans Elbow')
    plt.show()

def main():
    df = load_dataset()
    df = data_preprocessing(df)
    df_raw, df_scaled, scaler = standardize_data(df)
    plot_kmeans_elbow(df_scaled)


if __name__ == "__main__":
    main()