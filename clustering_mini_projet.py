import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

# PART 1

df = pd.read_csv("heart.csv")

df.head()

df.info()

df = df.drop(columns=["target"])

df.describe()

df.hist(figsize=(15, 15))
plt.suptitle("Feature distributions")
plt.show()

plt.figure(figsize=(15, 8))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.title("features box plots")
plt.show()

plt.scatter(df["age"], df["chol"])
plt.xlabel("age")
plt.ylabel("cholesterol")
plt.title("age vs cholesterol")
plt.show()

df.isnull().sum()

df = df.dropna()

#Dataset has no patient ID column nothing removed

df.duplicated().sum()

df = df.drop_duplicates()

categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

df_encoded = pd.get_dummies(df, columns=categorical_cols)

X = df_encoded.copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PART 2

X_train, X_test = train_test_split(
    X_scaled, test_size=0.2, random_state=42
)

X_train.shape, X_test.shape

inertia = []

K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_train)
    inertia.append(kmeans.inertia_)

plt.plot(K_range, inertia, marker='x')
plt.xlabel("Number of clusters -k-")
plt.ylabel("inertia")
plt.title("Elbow method")
plt.show()

# elbow appears around k = 3 or 4 so we choose k=3

k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans.fit(X_train)

train_labels = kmeans.predict(X_train)
test_labels = kmeans.predict(X_test)

cluster_centers = kmeans.cluster_centers_
cluster_centers

#PART 3

inertia_value = kmeans.inertia_
silhouette = silhouette_score(X_train, train_labels)
davies_bouldin = davies_bouldin_score(X_train, train_labels)

inertia_value, silhouette, davies_bouldin

results = []

for k in range(2, 7):

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_train)

    results.append({
        "k": k,

        "inertia": km.inertia_,
        "silhouette": silhouette_score(X_train, labels),
        "davies-Bouldin": davies_bouldin_score(X_train, labels)
                                })

results_df = pd.DataFrame(results)
results_df

# k = 3 offers a good balance

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=train_labels, cmap="viridis", alpha=0.6)
plt.title("Kmeans clustering visualization(PCA)" )
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label="cluster")
plt.show()