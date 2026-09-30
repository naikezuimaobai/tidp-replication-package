# ===================== visualize_clustering_all_optimized.py =====================
# Generate a complete visualization of the clustering results.
# ====================================================================

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Use a font stack that also supports symbols.
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE_PATH = os.environ.get("TIDP_CLUSTER_BASE_PATH", os.path.dirname(os.path.abspath(__file__)))
FEATURE_FILE = f"{BASE_PATH}\\sbert_features_normalized.npy"
CLUSTER_RESULT_FILE = f"{BASE_PATH}\\clustering_result.xlsx"
OUTPUT_DIR = BASE_PATH

# ==================== Load data ====================
print("="*60)
print("Clustering result visualization")
print("="*60)

X = np.load(FEATURE_FILE)
df = pd.read_excel(CLUSTER_RESULT_FILE)
cluster_labels = df['cluster'].values
n_clusters = 4

# Saturated, high-contrast colors.
colors = ['#E69F00', '#56B4E9', '#009E73', '#D55E00']

cluster_names = {
    0: 'Form selection',
    1: 'Content browsing',
    2: 'Information input',
    3: 'Button interaction'
}

# Add cluster proportions to the legend labels.
cluster_sizes = df['cluster'].value_counts()
for i in range(n_clusters):
    pct = 100 * cluster_sizes[i] / len(df)
    cluster_names[i] = f'{cluster_names[i]} ({pct:.1f}%)'

print(f"Number of samples: {len(X)}")
print(f"Feature dimension: {X.shape[1]}")
print(f"Number of clusters: {n_clusters}")

# ==================== PCA projection ====================
print("\n1. Computing PCA...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print(f"   PC1: {pca.explained_variance_ratio_[0]*100:.2f}%")
print(f"   PC2: {pca.explained_variance_ratio_[1]*100:.2f}%")

# ==================== t-SNE projection ====================
print("\n2. Computing t-SNE (2D)...")
sample_size = min(5000, len(X))
np.random.seed(42)
sample_idx = np.random.choice(len(X), sample_size, replace=False)
X_sample = X[sample_idx]
labels_sample = cluster_labels[sample_idx]

pca_50 = PCA(n_components=50, random_state=42)
X_pca_50 = pca_50.fit_transform(X_sample)

tsne_2d = TSNE(n_components=2, perplexity=30, random_state=42, 
                init='pca', n_iter=1000, learning_rate='auto')
X_tsne_2d = tsne_2d.fit_transform(X_pca_50)
print("   ✅ t-SNE 2D completed")

print("\n3. Computing t-SNE (3D)...")
tsne_3d = TSNE(n_components=3, perplexity=30, random_state=42,
                init='pca', n_iter=1000, learning_rate='auto')
X_tsne_3d = tsne_3d.fit_transform(X_pca_50)
print("   ✅ t-SNE 3D completed")

# ==================== Figure 1: PCA 2D scatter plot ====================
print("\n4. Generating the PCA 2D scatter plot...")
fig, ax = plt.subplots(figsize=(12, 10))

for i in range(n_clusters):
    mask = cluster_labels == i
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=colors[i], s=30, alpha=0.7,
               label=cluster_names[i],
               edgecolors='black', linewidth=0.3)

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=13)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=13)
ax.set_title('PCA Projection of Clustered Web Actions', fontsize=15, fontweight='bold')
ax.legend(loc='upper right', fontsize=11, markerscale=1.2)
ax.grid(True, alpha=0.2, linestyle='--')
ax.set_facecolor('white')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}\\viz_pca_2d.png", dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved: viz_pca_2d.png")

# ==================== Figure 2: t-SNE 2D scatter plot ====================
print("\n5. Generating the t-SNE 2D scatter plot...")
fig, ax = plt.subplots(figsize=(14, 11))

for i in range(n_clusters):
    mask = labels_sample == i
    if mask.sum() > 0:
        ax.scatter(X_tsne_2d[mask, 0], X_tsne_2d[mask, 1],
                   c=colors[i], s=50, alpha=0.65,
                   label=cluster_names[i],
                   edgecolors='black', linewidth=0.3)

ax.set_xlabel('t-SNE Dimension 1', fontsize=14)
ax.set_ylabel('t-SNE Dimension 2', fontsize=14)
ax.set_title('t-SNE Visualization of Web Action Clusters', fontsize=16, fontweight='bold')
ax.legend(loc='upper right', fontsize=11, markerscale=1.2)
ax.grid(True, alpha=0.2, linestyle='--')
ax.set_facecolor('white')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}\\viz_tsne_2d.png", dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved: viz_tsne_2d.png")

# ==================== Figure 3: t-SNE 3D scatter plots ====================
print("\n6. Generating t-SNE 3D scatter plots...")

angles = [
    (25, 45, 'front'),
    (25, 135, 'back'),
    (45, 90, 'side'),
    (90, 0, 'top'),
]

for elev, azim, name in angles:
    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111, projection='3d')
    
    for i in range(n_clusters):
        mask = labels_sample == i
        if mask.sum() > 0:
            ax.scatter(X_tsne_3d[mask, 0], X_tsne_3d[mask, 1], X_tsne_3d[mask, 2],
                       c=colors[i], s=25, alpha=0.6,
                       label=cluster_names[i],
                       edgecolors='black', linewidth=0.2)
    
    ax.set_xlabel('t-SNE Dim 1', fontsize=11, labelpad=10)
    ax.set_ylabel('t-SNE Dim 2', fontsize=11, labelpad=10)
    ax.set_zlabel('t-SNE Dim 3', fontsize=11, labelpad=10)
    ax.set_title(f'3D t-SNE Visualization (elev={elev}, azim={azim})', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, alpha=0.15)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}\\viz_tsne_3d_{name}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: viz_tsne_3d_{name}.png")

# ==================== Figure 4: cluster subplots ====================
print("\n7. Generating cluster subplots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

for i in range(n_clusters):
    ax = axes[i]
    mask = cluster_labels == i
    points = X_pca[mask]
    
    ax.scatter(points[:, 0], points[:, 1], 
               c=colors[i], s=40, alpha=0.8, 
               edgecolors='black', linewidth=0.3)
    
    other_mask = ~mask
    other_points = X_pca[other_mask]
    other_sample = np.random.choice(len(other_points), min(3000, len(other_points)), replace=False)
    ax.scatter(other_points[other_sample, 0], other_points[other_sample, 1],
               c='#D3D3D3', s=10, alpha=0.4, edgecolors='none')
    
    ax.set_title(f'{cluster_names[i]}', fontsize=14, fontweight='bold')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_facecolor('white')

plt.suptitle('Individual Cluster Distribution in PCA Space', fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}\\viz_cluster_subplots.png", dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved: viz_cluster_subplots.png")

# ==================== Figure 5: combined figure ====================
print("\n8. Generating the combined figure...")

fig = plt.figure(figsize=(22, 12))

# Subplot 1: PCA 2D.
ax1 = fig.add_subplot(2, 3, 1)
for i in range(n_clusters):
    mask = cluster_labels == i
    ax1.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=colors[i], s=20, alpha=0.6,
                label=cluster_names[i],
                edgecolors='black', linewidth=0.2)
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
ax1.set_title('PCA Projection', fontsize=13, fontweight='bold')
ax1.legend(loc='upper right', fontsize=8)
ax1.grid(True, alpha=0.2, linestyle='--')

# Subplot 2: t-SNE 2D.
ax2 = fig.add_subplot(2, 3, 2)
for i in range(n_clusters):
    mask = labels_sample == i
    if mask.sum() > 0:
        ax2.scatter(X_tsne_2d[mask, 0], X_tsne_2d[mask, 1],
                    c=colors[i], s=25, alpha=0.6,
                    edgecolors='black', linewidth=0.2)
ax2.set_xlabel('t-SNE Dim 1', fontsize=11)
ax2.set_ylabel('t-SNE Dim 2', fontsize=11)
ax2.set_title('t-SNE Projection', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.2, linestyle='--')

# Subplot 3: t-SNE 3D, front view.
ax3 = fig.add_subplot(2, 3, 3, projection='3d')
for i in range(n_clusters):
    mask = labels_sample == i
    if mask.sum() > 0:
        ax3.scatter(X_tsne_3d[mask, 0], X_tsne_3d[mask, 1], X_tsne_3d[mask, 2],
                    c=colors[i], s=12, alpha=0.5, edgecolors='black', linewidth=0.1)
ax3.set_xlabel('Dim 1', fontsize=9)
ax3.set_ylabel('Dim 2', fontsize=9)
ax3.set_zlabel('Dim 3', fontsize=9)
ax3.set_title('3D t-SNE (Front)', fontsize=12, fontweight='bold')
ax3.view_init(elev=25, azim=45)
ax3.grid(True, alpha=0.1)

# Subplot 4: t-SNE 3D, top view.
ax4 = fig.add_subplot(2, 3, 4, projection='3d')
for i in range(n_clusters):
    mask = labels_sample == i
    if mask.sum() > 0:
        ax4.scatter(X_tsne_3d[mask, 0], X_tsne_3d[mask, 1], X_tsne_3d[mask, 2],
                    c=colors[i], s=12, alpha=0.5, edgecolors='black', linewidth=0.1)
ax4.set_xlabel('Dim 1', fontsize=9)
ax4.set_ylabel('Dim 2', fontsize=9)
ax4.set_zlabel('Dim 3', fontsize=9)
ax4.set_title('3D t-SNE (Top)', fontsize=12, fontweight='bold')
ax4.view_init(elev=90, azim=0)
ax4.grid(True, alpha=0.1)

# Subplot 5: t-SNE 3D, side view.
ax5 = fig.add_subplot(2, 3, 5, projection='3d')
for i in range(n_clusters):
    mask = labels_sample == i
    if mask.sum() > 0:
        ax5.scatter(X_tsne_3d[mask, 0], X_tsne_3d[mask, 1], X_tsne_3d[mask, 2],
                    c=colors[i], s=12, alpha=0.5, edgecolors='black', linewidth=0.1)
ax5.set_xlabel('Dim 1', fontsize=9)
ax5.set_ylabel('Dim 2', fontsize=9)
ax5.set_zlabel('Dim 3', fontsize=9)
ax5.set_title('3D t-SNE (Side)', fontsize=12, fontweight='bold')
ax5.view_init(elev=20, azim=90)
ax5.grid(True, alpha=0.1)

# Subplot 6: cluster-size bar chart.
ax6 = fig.add_subplot(2, 3, 6)
sizes = df['cluster'].value_counts().sort_index()
bars = ax6.bar(range(n_clusters), sizes.values, color=colors, edgecolor='black', linewidth=2)
for bar, size in zip(bars, sizes.values):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
             f'{size}\n({100*size/len(df):.1f}%)',
             ha='center', va='bottom', fontsize=10, fontweight='bold')
ax6.set_xticks(range(n_clusters))
ax6.set_xticklabels([f'Cluster {i}' for i in range(n_clusters)])
ax6.set_ylabel('Number of Actions', fontsize=11)
ax6.set_title('Cluster Size Distribution', fontsize=13, fontweight='bold')
ax6.set_ylim(0, max(sizes.values) * 1.1)
ax6.grid(True, alpha=0.2, linestyle='--', axis='y')

plt.suptitle('Web Action Clustering Results', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}\\viz_combo.png", dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ Saved: viz_combo.png")

# ==================== Summary ====================
print("\n" + "="*60)
print("Visualization completed.")
print("="*60)
print(f"""
Parameters:
  - Marker sizes: PCA 30, t-SNE 50, 3D 25
  - Opacity: 0.6-0.7
  - Edges: black borders
  - Grid: light dashed lines

Generated files:
  - viz_pca_2d.png
  - viz_tsne_2d.png
  - viz_tsne_3d_front.png / back.png / side.png / top.png
  - viz_cluster_subplots.png
  - viz_combo.png
""")


# # ===================== visualize_clustering_all.py =====================
# Implementation note.
# Implementation note.
# # ====================================================================

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# import warnings
# warnings.filterwarnings('ignore')

# Implementation note.
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False

# Set TIDP_CLUSTER_BASE_PATH before running this script to select the output directory.
# FEATURE_FILE = f"{BASE_PATH}\\sbert_features_normalized.npy"
# CLUSTER_RESULT_FILE = f"{BASE_PATH}\\clustering_result.xlsx"
# OUTPUT_DIR = BASE_PATH

# Implementation note.
# print("="*60)
# Implementation note.
# print("="*60)

# X = np.load(FEATURE_FILE)
# df = pd.read_excel(CLUSTER_RESULT_FILE)
# cluster_labels = df['cluster'].values
# n_clusters = 4

# Implementation note.
# colors = ['#E69F00', '#56B4E9', '#009E73', '#D55E00']
# cluster_names = {
# Implementation note.
# Implementation note.
# Implementation note.
# Implementation note.
# }

# Implementation note.
# cluster_sizes = df['cluster'].value_counts()
# for i in range(n_clusters):
#     pct = 100 * cluster_sizes[i] / len(df)
#     cluster_names[i] = f'{cluster_names[i]} ({pct:.1f}%)'

# Implementation note.
# Implementation note.
# Implementation note.

# Implementation note.
# Implementation note.
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X)
# print(f"   PC1: {pca.explained_variance_ratio_[0]*100:.2f}%")
# print(f"   PC2: {pca.explained_variance_ratio_[1]*100:.2f}%")

# Implementation note.
# Implementation note.
# sample_size = min(5000, len(X))
# np.random.seed(42)
# sample_idx = np.random.choice(len(X), sample_size, replace=False)
# X_sample = X[sample_idx]
# labels_sample = cluster_labels[sample_idx]

# Implementation note.
# pca_50 = PCA(n_components=50, random_state=42)
# X_pca_50 = pca_50.fit_transform(X_sample)

# tsne_2d = TSNE(n_components=2, perplexity=30, random_state=42, 
#                 init='pca', n_iter=1000, learning_rate='auto')
# X_tsne_2d = tsne_2d.fit_transform(X_pca_50)
# Implementation note.

# Implementation note.
# tsne_3d = TSNE(n_components=3, perplexity=30, random_state=42,
#                 init='pca', n_iter=1000, learning_rate='auto')
# X_tsne_3d = tsne_3d.fit_transform(X_pca_50)
# Implementation note.

# Implementation note.
# Implementation note.
# fig, ax = plt.subplots(figsize=(12, 10))

# for i in range(n_clusters):
#     mask = cluster_labels == i
#     ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
#                c=colors[i], s=12, alpha=0.5,
#                label=cluster_names[i],
#                edgecolors='white', linewidth=0.2)

# ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
# ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
# ax.set_title('PCA Projection of Clustered Web Actions', fontsize=14, fontweight='bold')
# ax.legend(loc='upper right', fontsize=10)
# ax.grid(True, alpha=0.15)
# plt.tight_layout()
# plt.savefig(f"{OUTPUT_DIR}\\viz_pca_2d.png", dpi=300, bbox_inches='tight')
# plt.close()
# Implementation note.

# Implementation note.
# Implementation note.
# fig, ax = plt.subplots(figsize=(14, 11))

# for i in range(n_clusters):
#     mask = labels_sample == i
#     if mask.sum() > 0:
#         ax.scatter(X_tsne_2d[mask, 0], X_tsne_2d[mask, 1],
#                    c=colors[i], s=25, alpha=0.6,
#                    label=cluster_names[i],
#                    edgecolors='white', linewidth=0.3)

# ax.set_xlabel('t-SNE Dimension 1', fontsize=13)
# ax.set_ylabel('t-SNE Dimension 2', fontsize=13)
# ax.set_title('t-SNE Visualization of Web Action Clusters', fontsize=15, fontweight='bold')
# ax.legend(loc='upper right', fontsize=10)
# ax.grid(True, alpha=0.15)
# plt.tight_layout()
# plt.savefig(f"{OUTPUT_DIR}\\viz_tsne_2d.png", dpi=300, bbox_inches='tight')
# plt.close()
# Implementation note.

# Implementation note.
# Implementation note.

# Implementation note.
# angles = [
#     (25, 45, 'front_right'),
#     (25, 135, 'back_right'),
#     (25, 225, 'back_left'),
#     (25, 315, 'front_left'),
#     (45, 90, 'side'),
#     (90, 0, 'top'),
#     (15, 60, 'angle1'),
#     (35, 120, 'angle2'),
# ]

# for elev, azim, name in angles:
#     fig = plt.figure(figsize=(14, 11))
#     ax = fig.add_subplot(111, projection='3d')
    
#     for i in range(n_clusters):
#         mask = labels_sample == i
#         if mask.sum() > 0:
#             ax.scatter(X_tsne_3d[mask, 0], X_tsne_3d[mask, 1], X_tsne_3d[mask, 2],
#                        c=colors[i], s=15, alpha=0.5,
#                        label=cluster_names[i],
#                        edgecolors='white', linewidth=0.2)
    
#     ax.set_xlabel('t-SNE Dim 1', fontsize=10, labelpad=8)
#     ax.set_ylabel('t-SNE Dim 2', fontsize=10, labelpad=8)
#     ax.set_zlabel('t-SNE Dim 3', fontsize=10, labelpad=8)
#     ax.set_title(f'3D t-SNE Visualization (elev={elev}, azim={azim})', fontsize=13, fontweight='bold')
#     ax.legend(loc='upper right', fontsize=9)
#     ax.view_init(elev=elev, azim=azim)
#     ax.grid(True, alpha=0.1)
    
#     plt.tight_layout()
#     plt.savefig(f"{OUTPUT_DIR}\\viz_tsne_3d_{name}.png", dpi=300, bbox_inches='tight')
#     plt.close()
# Implementation note.

# Implementation note.
# Implementation note.

# fig, axes = plt.subplots(2, 2, figsize=(16, 14))
# axes = axes.flatten()

# for i in range(n_clusters):
#     ax = axes[i]
#     mask = cluster_labels == i
#     points = X_pca[mask]
    
# Implementation note.
#     ax.scatter(points[:, 0], points[:, 1], 
#                c=colors[i], s=20, alpha=0.6, edgecolors='white', linewidth=0.3)
    
# Implementation note.
#     other_mask = ~mask
#     other_points = X_pca[other_mask]
#     other_sample = np.random.choice(len(other_points), min(3000, len(other_points)), replace=False)
#     ax.scatter(other_points[other_sample, 0], other_points[other_sample, 1],
#                c='lightgray', s=8, alpha=0.3, edgecolors='none')
    
#     ax.set_title(f'{cluster_names[i]}', fontsize=13, fontweight='bold')
#     ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
#     ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
#     ax.grid(True, alpha=0.15)

# plt.suptitle('Individual Cluster Distribution in PCA Space', fontsize=16, fontweight='bold', y=0.98)
# plt.tight_layout()
# plt.savefig(f"{OUTPUT_DIR}\\viz_cluster_subplots.png", dpi=300, bbox_inches='tight')
# plt.close()
# Implementation note.

# Implementation note.
# Implementation note.

# fig = plt.figure(figsize=(20, 10))

# Implementation note.
# ax1 = fig.add_subplot(2, 3, 1)
# for i in range(n_clusters):
#     mask = cluster_labels == i
#     ax1.scatter(X_pca[mask, 0], X_pca[mask, 1],
#                 c=colors[i], s=8, alpha=0.4, label=cluster_names[i],
#                 edgecolors='white', linewidth=0.2)
# ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=10)
# ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=10)
# ax1.set_title('PCA Projection', fontsize=12, fontweight='bold')
# ax1.legend(loc='upper right', fontsize=7)
# ax1.grid(True, alpha=0.15)

# Implementation note.
# ax2 = fig.add_subplot(2, 3, 2)
# for i in range(n_clusters):
#     mask = labels_sample == i
#     if mask.sum() > 0:
#         ax2.scatter(X_tsne_2d[mask, 0], X_tsne_2d[mask, 1],
#                     c=colors[i], s=12, alpha=0.5,
#                     edgecolors='white', linewidth=0.2)
# ax2.set_xlabel('t-SNE Dim 1', fontsize=10)
# ax2.set_ylabel('t-SNE Dim 2', fontsize=10)
# ax2.set_title('t-SNE Projection', fontsize=12, fontweight='bold')
# ax2.grid(True, alpha=0.15)

# Implementation note.
# ax3 = fig.add_subplot(2, 3, 3, projection='3d')
# for i in range(n_clusters):
#     mask = labels_sample == i
#     if mask.sum() > 0:
#         ax3.scatter(X_tsne_3d[mask, 0], X_tsne_3d[mask, 1], X_tsne_3d[mask, 2],
#                     c=colors[i], s=6, alpha=0.4, edgecolors='white', linewidth=0.1)
# ax3.set_xlabel('Dim 1', fontsize=8)
# ax3.set_ylabel('Dim 2', fontsize=8)
# ax3.set_zlabel('Dim 3', fontsize=8)
# ax3.set_title('3D t-SNE (front)', fontsize=12, fontweight='bold')
# ax3.view_init(elev=25, azim=45)
# ax3.grid(True, alpha=0.1)

# Implementation note.
# ax4 = fig.add_subplot(2, 3, 4, projection='3d')
# for i in range(n_clusters):
#     mask = labels_sample == i
#     if mask.sum() > 0:
#         ax4.scatter(X_tsne_3d[mask, 0], X_tsne_3d[mask, 1], X_tsne_3d[mask, 2],
#                     c=colors[i], s=6, alpha=0.4, edgecolors='white', linewidth=0.1)
# ax4.set_xlabel('Dim 1', fontsize=8)
# ax4.set_ylabel('Dim 2', fontsize=8)
# ax4.set_zlabel('Dim 3', fontsize=8)
# ax4.set_title('3D t-SNE (top)', fontsize=12, fontweight='bold')
# ax4.view_init(elev=90, azim=0)
# ax4.grid(True, alpha=0.1)

# Implementation note.
# ax5 = fig.add_subplot(2, 3, 5, projection='3d')
# for i in range(n_clusters):
#     mask = labels_sample == i
#     if mask.sum() > 0:
#         ax5.scatter(X_tsne_3d[mask, 0], X_tsne_3d[mask, 1], X_tsne_3d[mask, 2],
#                     c=colors[i], s=6, alpha=0.4, edgecolors='white', linewidth=0.1)
# ax5.set_xlabel('Dim 1', fontsize=8)
# ax5.set_ylabel('Dim 2', fontsize=8)
# ax5.set_zlabel('Dim 3', fontsize=8)
# ax5.set_title('3D t-SNE (side)', fontsize=12, fontweight='bold')
# ax5.view_init(elev=20, azim=90)
# ax5.grid(True, alpha=0.1)

# Implementation note.
# ax6 = fig.add_subplot(2, 3, 6)
# sizes = df['cluster'].value_counts().sort_index()
# bars = ax6.bar(range(n_clusters), sizes.values, color=colors, edgecolor='black', linewidth=1.5)
# for bar, size in zip(bars, sizes.values):
#     ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
#              f'{size}\n({100*size/len(df):.1f}%)',
#              ha='center', va='bottom', fontsize=9)
# ax6.set_xticks(range(n_clusters))
# ax6.set_xticklabels([f'Cluster {i}' for i in range(n_clusters)])
# ax6.set_ylabel('Number of Actions', fontsize=10)
# ax6.set_title('Cluster Size Distribution', fontsize=12, fontweight='bold')
# ax6.set_ylim(0, max(sizes.values) * 1.1)

# plt.suptitle('Web Action Clustering Results', fontsize=16, fontweight='bold', y=1.02)
# plt.tight_layout()
# plt.savefig(f"{OUTPUT_DIR}\\viz_combo.png", dpi=300, bbox_inches='tight')
# plt.close()
# Implementation note.

# Implementation note.
# print("\n" + "="*60)
# Implementation note.
# print("="*60)
# print(f"""
# Implementation note.
#   PCA 2D:
#     - viz_pca_2d.png

#   t-SNE 2D:
#     - viz_tsne_2d.png

# Implementation note.
#     - viz_tsne_3d_front_right.png
#     - viz_tsne_3d_back_right.png
#     - viz_tsne_3d_back_left.png
#     - viz_tsne_3d_front_left.png
#     - viz_tsne_3d_side.png
#     - viz_tsne_3d_top.png
#     - viz_tsne_3d_angle1.png
#     - viz_tsne_3d_angle2.png

# Implementation note.
#     - viz_cluster_subplots.png

# Implementation note.
#     - viz_combo.png

# Implementation note.
# Implementation note.
# Implementation note.
# Implementation note.
# Implementation note.
# """)
