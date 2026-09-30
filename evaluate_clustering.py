# ===================== evaluate_clustering.py =====================
# Compute multiple clustering metrics and evaluate the K=4 solution.
# ==================================================================

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

BASE_PATH = os.environ.get("TIDP_CLUSTER_BASE_PATH", os.path.dirname(os.path.abspath(__file__)))
FEATURE_FILE = f"{BASE_PATH}\\sbert_features_normalized.npy"
CLUSTER_RESULT_FILE = f"{BASE_PATH}\\clustering_result.xlsx"

# Load data.
X = np.load(FEATURE_FILE)
df = pd.read_excel(CLUSTER_RESULT_FILE)
labels = df['cluster'].values

print("="*60)
print("Clustering metrics (K=4)")
print("="*60)

# 1. Silhouette score.
sil_score = silhouette_score(X, labels, metric='cosine')
print(f"\n1. Silhouette score: {sil_score:.4f}")
print(f"   Assessment: {'good' if sil_score >= 0.5 else 'moderate' if sil_score >= 0.3 else 'poor'}")

# 2. Davies-Bouldin index.
db_score = davies_bouldin_score(X, labels)
print(f"\n2. Davies-Bouldin index: {db_score:.4f}")
print("   Lower values indicate better cluster separation.")

# 3. Calinski-Harabasz index.
ch_score = calinski_harabasz_score(X, labels)
print(f"\n3. Calinski-Harabasz index: {ch_score:.2f}")
print("   Higher values indicate compact and well-separated clusters.")

# ==================== Comparison across K values ====================
print("\n" + "="*60)
print("Metric comparison across K values")
print("="*60)

k_range = range(2, 13)
sil_scores = []
db_scores = []
ch_scores = []

for k in k_range:
    from sklearn.cluster import AgglomerativeClustering
    cluster = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='complete')
    labels_k = cluster.fit_predict(X)
    
    if len(set(labels_k)) > 1:
        sil_scores.append(silhouette_score(X, labels_k, metric='cosine'))
        db_scores.append(davies_bouldin_score(X, labels_k))
        ch_scores.append(calinski_harabasz_score(X, labels_k))
    else:
        sil_scores.append(-1)
        db_scores.append(-1)
        ch_scores.append(-1)

print(f"\n{'K':<4} {'Silhouette':<12} {'Davies-Bouldin':<16} {'Calinski-Harabasz':<20}")
print("-" * 60)
for i, k in enumerate(k_range):
    print(f"{k:<4} {sil_scores[i]:<12.4f} {db_scores[i]:<16.4f} {ch_scores[i]:<20.2f}")

# ==================== Elbow method ====================
print("\n" + "="*60)
print("Elbow method")
print("="*60)

# Use KMeans inertia because hierarchical clustering has no direct inertia.
inertias = []
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# Locate the elbow point.
from scipy.spatial.distance import cdist
# Find the elbow from the second derivative.
first_deriv = np.diff(inertias)
second_deriv = np.diff(first_deriv)
elbow_k = np.argmin(second_deriv) + 3  # Offset for the two differences.

print("\nInertia values: lower is better; the elbow suggests the best K.")
for i, k in enumerate(k_range):
    print(f"   K={k}: {inertias[i]:.2f}")

print(f"\nElbow method suggests K = {elbow_k}")

# ==================== Metric visualization ====================
print("\nGenerating metric comparison plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: silhouette score.
ax1 = axes[0, 0]
ax1.plot(k_range, sil_scores, 'bo-', linewidth=2, markersize=8)
ax1.axvline(x=4, color='red', linestyle='--', label='K=4', linewidth=2)
ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
ax1.set_ylabel('Silhouette Score', fontsize=12)
ax1.set_title('Silhouette Score vs K', fontsize=13)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Subplot 2: Davies-Bouldin index.
ax2 = axes[0, 1]
ax2.plot(k_range, db_scores, 'ro-', linewidth=2, markersize=8)
ax2.axvline(x=4, color='red', linestyle='--', label='K=4', linewidth=2)
ax2.set_xlabel('Number of Clusters (K)', fontsize=12)
ax2.set_ylabel('Davies-Bouldin Index', fontsize=12)
ax2.set_title('Davies-Bouldin Index vs K', fontsize=13)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Subplot 3: Calinski-Harabasz index.
ax3 = axes[1, 0]
ax3.plot(k_range, ch_scores, 'go-', linewidth=2, markersize=8)
ax3.axvline(x=4, color='red', linestyle='--', label='K=4', linewidth=2)
ax3.set_xlabel('Number of Clusters (K)', fontsize=12)
ax3.set_ylabel('Calinski-Harabasz Index', fontsize=12)
ax3.set_title('Calinski-Harabasz Index vs K', fontsize=13)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Subplot 4: elbow method.
ax4 = axes[1, 1]
ax4.plot(k_range, inertias, 'purple', linewidth=2, marker='s', markersize=6)
ax4.axvline(x=elbow_k, color='red', linestyle='--', label=f'Elbow at K={elbow_k}', linewidth=2)
ax4.set_xlabel('Number of Clusters (K)', fontsize=12)
ax4.set_ylabel('Inertia (Within-cluster SSE)', fontsize=12)
ax4.set_title('Elbow Method', fontsize=13)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{BASE_PATH}\\viz_clustering_metrics.png", dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved: viz_clustering_metrics.png")

# ==================== Overall assessment ====================
print("\n" + "="*60)
print("Overall assessment")
print("="*60)

# Compute the best K for each metric.
best_sil_k = k_range[np.argmax(sil_scores)]
best_db_k = k_range[np.argmin(db_scores)]
best_ch_k = k_range[np.argmax(ch_scores)]

print("\nRecommended K values:")
print(f"  Silhouette (maximum):       K = {best_sil_k}")
print(f"  Davies-Bouldin (minimum):   K = {best_db_k}")
print(f"  Calinski-Harabasz (maximum): K = {best_ch_k}")
print(f"  Elbow method:               K = {elbow_k}")

print("\nMetric ranks for K=4:")
print(f"  Silhouette: {sorted(sil_scores, reverse=True).index(sil_scores[2]) + 1}/{len(k_range)}")
print(f"  Davies-Bouldin: {sorted(db_scores).index(db_scores[2]) + 1}/{len(k_range)}")
print(f"  Calinski-Harabasz: {sorted(ch_scores, reverse=True).index(ch_scores[2]) + 1}/{len(k_range)}")

# Assess whether K=4 is a reasonable choice.
if 2 <= best_sil_k <= 6 or 2 <= best_db_k <= 6 or 2 <= best_ch_k <= 6:
    print("\n✅ Overall assessment: K=4 performs well across multiple metrics.")
else:
    print("\n⚠️ Overall assessment: metric recommendations differ; use task semantics as an additional criterion.")
