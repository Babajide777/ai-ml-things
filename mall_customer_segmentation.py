import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# FIX 1: name the features once, at the top. Every function uses this list,
# so changing the features means editing one line, not hunting through the file.
INCOME = 'Annual Income (k$)'
SPEND = 'Spending Score (1-100)'
FEATURES = [INCOME, SPEND]

RANDOM_STATE = 42  # FIX 2: one named seed instead of a magic number inside a function


def load_dataset():
    file_path = "Mall_Customers.csv"
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "vjchoudhary7/customer-segmentation-tutorial-in-python",
        file_path,
    )
    return df


def data_preprocessing(df):
    # FIX 3: KEEP the columns you want instead of DROPPING the ones you don't.
    # If the CSV gains a new column, drop() would silently feed it to the model;
    # selecting by name can't. .copy() avoids pandas' SettingWithCopy warning later.
    return df[FEATURES].copy()


def standardize_data(df):
    # FIX 4: df is already just the two features (FIX 3), so scale it directly.
    # Only one copy is needed: df itself stays as the raw, unscaled data.
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[FEATURES] = scaler.fit_transform(df[FEATURES])
    return df, df_scaled, scaler


def fit_kmeans(X, k):
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
    model.fit(X)
    return model


def kmeans_elbow(X, max_k=10):
    return [fit_kmeans(X, k).inertia_ for k in range(1, max_k + 1)]


def plot_kmeans_elbow(X, max_k=10):
    inertias = kmeans_elbow(X, max_k)
    ks = range(1, max_k + 1)
    plt.figure()  # FIX 5: start a fresh figure so plots never draw on top of each other
    plt.plot(ks, inertias, marker='o')  # FIX 6: one call does points + line
    plt.xticks(ks)  # whole numbers only on the x-axis: there is no "2.5 clusters"
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('KMeans Elbow')
    plt.show()


def plot_clusters(df_raw, labels, centres, k):
    plt.figure()
    # FIX 7: clusters are CATEGORIES, so give each one its own legend entry.
    # Your colorbar was (a) attached to the red centroid scatter, hence the
    # 0.0-1.0 scale, and (b) continuous, implying a "cluster 2.5" exists.
    for cluster_id in range(k):
        members = df_raw[labels == cluster_id]
        plt.scatter(members[INCOME], members[SPEND], label=f'Cluster {cluster_id}')
    plt.scatter(
        centres[:, 0], centres[:, 1],
        marker='X', s=200, c='black', label='Centroids',
    )
    plt.xlabel(INCOME)
    plt.ylabel(SPEND)
    plt.title(f'Customer Segments (k={k})')  # FIX 8: title follows k, not a hard-coded 5
    plt.legend()
    plt.show()


def band(value, low_cut, high_cut):
    # FIX 9: THREE bands, not two. A median split only has 4 possible labels
    # (high/low x high/low) for 5 clusters, so the middle cluster got
    # mislabelled. Cut points come from the data's own thirds (tertiles).
    if value < low_cut:
        return 'low'
    if value > high_cut:
        return 'high'
    return 'average'


def describe_clusters(df_raw, labels):
    df = df_raw.copy()
    df['Cluster'] = labels

    income_lo, income_hi = df[INCOME].quantile([1 / 3, 2 / 3])
    spend_lo, spend_hi = df[SPEND].quantile([1 / 3, 2 / 3])

    # FIX 10: one groupby gives the averages AND the head-count per cluster.
    summary = df.groupby('Cluster').agg(
        avg_income=(INCOME, 'mean'),
        avg_spend=(SPEND, 'mean'),
        customers=(INCOME, 'size'),
    )

    for cluster_id, row in summary.iterrows():
        income_label = band(row.avg_income, income_lo, income_hi)
        spend_label = band(row.avg_spend, spend_lo, spend_hi)
        print(
            f"Cluster {cluster_id}: {income_label} income, {spend_label} spending, "
            f"{int(row.customers)} customers "
            f"(avg income={row.avg_income:.1f}k$, avg score={row.avg_spend:.1f})"
        )
    return summary  # FIX 11: return the table so it can be tested or reused, not only printed


def main():
    df = load_dataset()
    df = data_preprocessing(df)
    df_raw, df_scaled, scaler = standardize_data(df)

    plot_kmeans_elbow(df_scaled)

    chosen_k = 5  # read from the elbow plot
    model = fit_kmeans(df_scaled, chosen_k)
    labels = model.labels_
    centres = scaler.inverse_transform(model.cluster_centers_)  # your best line, kept

    plot_clusters(df_raw, labels, centres, chosen_k)
    describe_clusters(df_raw, labels)


if __name__ == "__main__":
    main()