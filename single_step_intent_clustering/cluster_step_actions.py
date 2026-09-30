# ===================== cluster_actions.py =====================
# Cluster SBERT features for single-step web operations.
# ====================================================================

import numpy as np
import pandas as pd
import os
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# ==================== Configuration ====================
BASE_PATH = os.environ.get("TIDP_CLUSTER_BASE_PATH", os.path.dirname(os.path.abspath(__file__)))
FEATURE_FILE = f"{BASE_PATH}\\sbert_features_normalized.npy"
DATA_FILE = f"{BASE_PATH}\\step_actions_simplified.xlsx"

# ==================== 1. Load data ====================
print("="*60)
print("Single-step operation clustering")
print("="*60)

X = np.load(FEATURE_FILE)
df = pd.read_excel(DATA_FILE)

print(f"Feature matrix shape: {X.shape}")
print(f"Number of operations: {len(df)}")
print(f"Unique operation types: {df['unified_simplified'].nunique()}")

# ==================== 2. Search for the best K ====================
print("\nSearching for the best number of clusters K...")
print("-"*40)

results = []
for k in range(2, 13):
    cluster = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="complete")
    labels = cluster.fit_predict(X)
    
    if len(set(labels)) > 1:
        score = silhouette_score(X, labels, metric="cosine")
        results.append({'k': k, 'silhouette': score})
        print(f"K={k:2d}: silhouette score = {score:.4f}")
    else:
        results.append({'k': k, 'silhouette': -1})
        print(f"K={k:2d}: invalid")

# Best K.
best = max(results, key=lambda x: x['silhouette'])
print(f"\n🏆 Best K = {best['k']}, silhouette score = {best['silhouette']:.4f}")

# ==================== 3. Cluster with the best K ====================
print("\nRunning final clustering...")
cluster = AgglomerativeClustering(n_clusters=best['k'], metric="cosine", linkage="complete")
labels = cluster.fit_predict(X)
df['cluster'] = labels

# ==================== 4. Analyze clusters ====================
print("\nCluster size distribution:")
for cid in sorted(df['cluster'].unique()):
    size = (df['cluster'] == cid).sum()
    pct = 100 * size / len(df)
    print(f"   Cluster {cid:2d}: {size:5d} ({pct:5.1f}%)")

print("\nRepresentative operations in each cluster:")
for cid in sorted(df['cluster'].unique()):
    mask = df['cluster'] == cid
    size = mask.sum()
    top_actions = df[mask]['unified_simplified'].value_counts().head(8)
    print(f"\n📁 Cluster {cid} (n={size}):")
    for action, count in top_actions.items():
        print(f"      {action}: {count}")

# ==================== 5. Save results ====================
output_file = f"{BASE_PATH}\\clustering_result.xlsx"
df.to_excel(output_file, index=False)
print(f"\n✅ Results saved to: {output_file}")

# ==================== 6. Final assessment ====================
print("\n" + "="*60)
print("Clustering quality evaluation")
print("="*60)

final_score = best['silhouette']
if final_score >= 0.5:
    print(f"✅ Clustering quality: good (silhouette score = {final_score:.4f})")
elif final_score >= 0.3:
    print(f"⚠️ Clustering quality: moderate (silhouette score = {final_score:.4f})")
else:
    print(f"❌ Clustering quality: poor (silhouette score = {final_score:.4f})")
