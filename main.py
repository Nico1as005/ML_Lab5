import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import re

def load_data():
    with open('features.txt', 'r') as f:
        features = []
        for line in f:
            parts = line.strip().split(' ')
            feature_name = ' '.join(parts[1:])
            features.append(feature_name)

    unique_features = []
    feature_count = {}
    for feature in features:
        if feature in feature_count:
            feature_count[feature] += 1
            unique_features.append(f"{feature}_{feature_count[feature]}")
        else:
            feature_count[feature] = 1
            unique_features.append(feature)

    X_train = pd.read_csv('X_train.txt', sep=r'\s+', header=None, names=unique_features, engine='python')
    y_train = pd.read_csv('y_train.txt', sep=r'\s+', header=None, names=['activity'], engine='python')

    activity_labels = {}
    with open('activity_labels.txt', 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            activity_id = int(parts[0])
            activity_name = ' '.join(parts[1:])
            activity_labels[activity_id] = activity_name

    return X_train, y_train, activity_labels

X_train, y_train, activity_labels = load_data()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train['activity'], cmap='tab10', s=20, alpha=0.7)
plt.colorbar(scatter, label='Activity')
plt.title("Визуализация признаков (PCA)")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")

unique_activities = sorted(y_train['activity'].unique())
legend_labels = [f"{act}: {activity_labels[act]}" for act in unique_activities]
handles, _ = scatter.legend_elements()
plt.legend(handles, legend_labels, title="Activities", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.show()

def evaluate_clustering(model, data):
    labels = model.fit_predict(data)
    if len(set(labels)) > 1:
        score = silhouette_score(data, labels)
    else:
        score = -1
    return labels, score

print("=" * 60)
print("KMeans кластеризация")
print("=" * 60)

kmeans_params = [2, 3, 4, 5, 6]
best_score_kmeans = -1
best_kmeans = None
best_labels_kmeans = None
best_k = None
labels_for_k = []

for k in kmeans_params:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels, score = evaluate_clustering(kmeans, X_scaled)
    print(f'KMeans с k={k}, Силуэтный коэффициент: {score:.3f}')
    labels_for_k.append(labels)
    if score > best_score_kmeans:
        best_score_kmeans = score
        best_kmeans = kmeans
        best_labels_kmeans = labels
        best_k = k

print(f'Лучшее число кластеров для KMeans: {best_k} с коэффициентом: {best_score_kmeans:.3f}')

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for i, k in enumerate(kmeans_params):
    labels = labels_for_k[i]
    scatter = axes[i].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', s=20, alpha=0.7)
    axes[i].set_title(f'KMeans k={k}\nSilhouette: {silhouette_score(X_scaled, labels):.3f}')
    axes[i].set_xlabel('PC1')
    axes[i].set_ylabel('PC2')

for i in range(len(kmeans_params), 6):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("Agglomerative Clustering")
print("=" * 60)

agg_params = [2, 3, 4, 5, 6]
best_score_agg = -1
best_labels_agg = []
best_n_agg = None
labels_list_agg = []

for n in agg_params:
    agg = AgglomerativeClustering(n_clusters=n)
    labels, score = evaluate_clustering(agg, X_scaled)
    print(f'Agglomerative с n_clusters={n}, Силуэтный коэффициент: {score:.3f}')
    labels_list_agg.append(labels)
    if score > best_score_agg:
        best_score_agg = score
        best_labels_agg = labels
        best_n_agg = n

print(f'Лучшее число кластеров для Agglomerative: {best_n_agg} с коэффициентом: {best_score_agg:.3f}')

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for i, n in enumerate(agg_params):
    labels = labels_list_agg[i]
    scatter = axes[i].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', s=20, alpha=0.7)
    axes[i].set_title(f'Agglomerative n={n}\nSilhouette: {silhouette_score(X_scaled, labels):.3f}')
    axes[i].set_xlabel('PC1')
    axes[i].set_ylabel('PC2')

for i in range(len(agg_params), 6):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("Spectral Clustering")
print("=" * 60)

spectral_params = [2, 3, 4, 5, 6]
best_score_spectral = -1
best_labels_spectral = []
best_n_spectral = None
labels_list_spectral = []

for n in spectral_params:
    spectral = SpectralClustering(n_clusters=n, affinity='nearest_neighbors', random_state=42, n_init=10)
    labels, score = evaluate_clustering(spectral, X_scaled)
    print(f'SpectralClustering с n_clusters={n}, Силуэтный коэффициент: {score:.3f}')
    labels_list_spectral.append(labels)
    if score > best_score_spectral:
        best_score_spectral = score
        best_labels_spectral = labels
        best_n_spectral = n

print(f'Лучшее число кластеров для SpectralClustering: {best_n_spectral} с коэффициентом: {best_score_spectral:.3f}')

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for i, n in enumerate(spectral_params):
    labels = labels_list_spectral[i]
    scatter = axes[i].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', s=20, alpha=0.7)
    axes[i].set_title(f'Spectral n={n}\nSilhouette: {silhouette_score(X_scaled, labels):.3f}')
    axes[i].set_xlabel('PC1')
    axes[i].set_ylabel('PC2')

for i in range(len(spectral_params), 6):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=best_labels_kmeans, cmap='tab10', s=20, alpha=0.7)
plt.title(f'KMeans (k={best_k})\nSilhouette: {best_score_kmeans:.3f}')
plt.xlabel('PC1')
plt.ylabel('PC2')

plt.subplot(1, 3, 2)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=best_labels_agg, cmap='tab10', s=20, alpha=0.7)
plt.title(f'Agglomerative (n={best_n_agg})\nSilhouette: {best_score_agg:.3f}')
plt.xlabel('PC1')
plt.ylabel('PC2')

plt.subplot(1, 3, 3)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=best_labels_spectral, cmap='tab10', s=20, alpha=0.7)
plt.title(f'Spectral (n={best_n_spectral})\nSilhouette: {best_score_spectral:.3f}')
plt.xlabel('PC1')
plt.ylabel('PC2')

plt.tight_layout()
plt.show()

scores = {
    'KMeans': best_score_kmeans,
    'Agglomerative': best_score_agg,
    'SpectralClustering': best_score_spectral
}

print("\n" + "=" * 60)
print("СРАВНЕНИЕ МЕТОДОВ")
print("=" * 60)
for method, score in scores.items():
    print(f"{method}: {score:.3f}")

best_method = max(scores, key=scores.get)
print(f'\nЛучший метод кластеризации: {best_method} с коэффициентом {scores[best_method]:.3f}')

if best_method == 'KMeans':
    best_labels = best_labels_kmeans
elif best_method == 'Agglomerative':
    best_labels = best_labels_agg
else:
    best_labels = best_labels_spectral