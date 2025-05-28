import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Set the style for better visualizations
sns.set(style="white")

# Load the ping latency data
df = pd.read_csv('data/ping_latency.csv', index_col=0)

# Convert the DataFrame to a NumPy array
X = df.values

# Standardize the data (important for clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use HDBSCAN for automatic cluster detection
# min_cluster_size determines the minimum size of clusters
# min_samples affects the conservativeness of clustering
clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=1, 
                          metric='euclidean', gen_min_span_tree=True)
cluster_labels = clusterer.fit_predict(X_scaled)

# Handle noise points (labeled as -1) by assigning them to a separate cluster
if -1 in cluster_labels:
    cluster_labels = cluster_labels + 1  # Shift all labels up by 1
    cluster_labels[cluster_labels == 0] = -1  # Set original -1 back to a distinct value

# Get the number of clusters found
n_clusters = len(np.unique(cluster_labels))
print(f"HDBSCAN automatically found {n_clusters} clusters")

# Create a DataFrame with cluster labels
results_df = pd.DataFrame({
    'Cluster': cluster_labels,
    'Label': df.index  # Original row labels
})

# Use MDS (Multidimensional Scaling) for visualization only
# This doesn't affect the clustering, just helps us visualize the results
from sklearn.manifold import MDS

# Apply MDS to visualize the data in 2D
mds = MDS(n_components=2, random_state=42)
X_mds = mds.fit_transform(X_scaled)

# Add MDS coordinates to the results DataFrame
results_df['MDS1'] = X_mds[:, 0]
results_df['MDS2'] = X_mds[:, 1]

# 1. Create a scatter plot colored by clusters
plt.figure(figsize=(12, 10))

# Scatter plot
sns.scatterplot(
    x='MDS1', 
    y='MDS2',
    hue='Cluster',
    palette='nipy_spectral',
    s=100,  # Point size
    data=results_df
)

# Add labels to points if needed (comment out if too many points)
for i, txt in enumerate(results_df['Label']):
    plt.annotate(txt, (results_df['MDS1'][i], results_df['MDS2'][i]), fontsize=8)

plt.title('HDBSCAN Clustering of Ping Latency Data', fontsize=15)
plt.xlabel('MDS Dimension 1', fontsize=12)
plt.ylabel('MDS Dimension 2', fontsize=12)
plt.tight_layout()
plt.savefig('visualization_scatter.png')

# 2. Create a clustered heatmap
plt.figure(figsize=(14, 12))

# Reorder the dataframe based on clustering
row_linkage = sns.clustermap(
    df, 
    figsize=(1, 1),  # Temporary small figure just to get linkage
    col_cluster=True,
    row_cluster=True
).dendrogram_row.linkage

# Create the final clustermap with all customizations
cluster_map = sns.clustermap(
    df,
    figsize=(14, 12),
    cmap="YlGnBu",
    row_linkage=row_linkage,  # Use pre-computed linkage
    col_cluster=True,
    yticklabels=True,
    xticklabels=True,
    annot_kws={"size": 8}
)

# Adjust the plot
plt.title('Clustered Heatmap of Ping Latency Data', fontsize=15)
plt.tight_layout()
plt.savefig('visualization_heatmap.png')

# Save cluster information to file
results_df.to_csv('visualization_clusters.csv')

plt.show()
