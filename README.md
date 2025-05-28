# Ping Latency Clustering Visualization

This project visualizes the clustering of ping latency data between compute nodes using automatic cluster detection with HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise).

## Project Structure

- `data/`: Contains the raw ping latency data
- `requirements.txt`: Lists the required Python packages
- `visualize.py`: Python script to generate the cluster visualizations
- `setup_env.sh`: Bash script to set up a conda environment with all dependencies

## Setup

### Option 1: Using Conda (Recommended)

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/ping-latency-clustering.git
   cd ping-latency-clustering
   ```

2. Run the setup script to create a conda environment with all dependencies:

   ```bash
   source setup_env.sh
   ```

   This will:
   - Create a conda environment named `ping-latency-env` with Python 3.8
   - Install all required dependencies including numpy, pandas, matplotlib, seaborn, scikit-learn, and hdbscan
   - Activate the environment

3. For future use, activate the environment with:

   ```bash
   conda activate ping-latency-env
   ```

### Option 2: Using pip

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/ping-latency-clustering.git
   cd ping-latency-clustering
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Running the Visualization

To generate the visualizations, run:

```bash
python visualize.py
```

This will produce:
- `visualization_scatter.png`: A scatter plot showing the clusters in 2D space
- `visualization_heatmap.png`: A clustered heatmap of the ping latency data
- `visualization_clusters.csv`: A CSV file containing the cluster assignments

## Visualization Examples

### Network Node Clustering

The scatter plot below shows how the network nodes are automatically clustered based on their ping latency relationships. Nodes that are closer together in this visualization have similar latency profiles to other nodes in the network.

![Network Node Clustering](/assets/visualization_scatter.png)

### Latency Heatmap with Hierarchical Clustering

This heatmap visualizes the raw ping latency data between nodes, with hierarchical clustering applied to group similar nodes together. The color intensity represents the latency in milliseconds between each pair of nodes.

![Latency Heatmap](/assets/visualization_heatmap.png)

## Sample Data

The `ping_latency.csv` file contains the following data:

```
,A,B,C,D,E,F,G
A,0,12,15,20,18,25,22
B,12,0,14,19,17,24,21
C,15,14,0,16,13,20,18
D,20,19,16,0,11,18,16
E,18,17,13,11,0,15,14
F,25,24,20,18,15,0,12
G,22,21,18,16,14,12,0
```

Each cell represents the ping latency (in milliseconds) between two compute nodes.

## License

This project is licensed under the MIT License.
