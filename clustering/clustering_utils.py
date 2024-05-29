import matplotlib.pyplot as plt
import numpy as np
import random
import operator
import wandb

from typing import List, Tuple
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from collections import Counter
from Logic.core.clustering.clustering_metrics import *
import plotly.figure_factory as ff


class ClusteringUtils:

    def __init__(self, max_iter: int = 100, tol: float = 1e-4):
        self.max_iter = max_iter
        self.tol = tol

    def cluster_kmeans(self, emb_vecs: List, n_clusters: int, max_iter: int = 100) -> Tuple[List, List]:
        """
        Clusters input vectors using the K-means method.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.
        n_clusters: int
            The number of clusters to form.

        Returns
        --------
        Tuple[List, List]
            Two lists:
            1. A list containing the cluster centers.
            2. A list containing the cluster index for each input vector.
        """
        # Convert the input list of vectors to a NumPy array
        emb_vecs = np.array(emb_vecs)
        n_samples, n_features = emb_vecs.shape

        # Randomly initialize cluster centers
        np.random.seed(42)
        initial_indices = np.random.choice(n_samples, n_clusters, replace=False)
        cluster_centers = emb_vecs[initial_indices]

        for iteration in range(max_iter):
            # Assign each point to the nearest cluster
            cluster_indices = np.array([self.closest_center(point, cluster_centers) for point in emb_vecs])

            # Calculate new cluster centers
            new_centers = np.array([emb_vecs[cluster_indices == i].mean(axis=0) for i in range(n_clusters)])

            # Check for convergence (if centers do not change)
            if np.allclose(cluster_centers, new_centers):
                break

            cluster_centers = new_centers

        # Convert the cluster centers and cluster indices to lists
        cluster_centers_list = cluster_centers.tolist()
        cluster_indices_list = cluster_indices.tolist()

        return cluster_centers_list, cluster_indices_list

    def closest_center(self, point, centers):
        """
        Finds the index of the closest cluster center to a given point.

        Parameters
        ----------
        point : array-like
            The data point to compare.
        centers : array-like
            The cluster centers.

        Returns
        -------
        int
            The index of the closest cluster center.
        """
        distances = np.linalg.norm(point - centers, axis=1)
        return np.argmin(distances)

    def get_most_frequent_words(self, documents: List[str], top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Finds the most frequent words in a list of documents.

        Parameters
        -----------
        documents: List[str]
            A list of documents, where each document is a string representing a list of words.
        top_n: int, optional
            The number of most frequent words to return. Default is 10.

        Returns
        --------
        List[Tuple[str, int]]
            A list of tuples, where each tuple contains a word and its frequency, sorted in descending order of frequency.
        """
        words = [word for doc in documents for word in doc.split()]
        word_freq = Counter(words)
        return word_freq.most_common(top_n)

    def compute_wcss(self, emb_vecs: np.ndarray, cluster_indices: np.ndarray, cluster_centers: np.ndarray) -> float:
        wcss = 0.0
        for i, center in enumerate(cluster_centers):
            cluster_points = emb_vecs[cluster_indices == i]
            wcss += np.sum((cluster_points - center) ** 2)
        return wcss

    def initialize_centroids(self, emb_vecs: np.ndarray, k: int) -> np.ndarray:
        np.random.seed(42)
        initial_indices = np.random.choice(len(emb_vecs), k, replace=False)
        return emb_vecs[initial_indices]

    def cluster_kmeans_WCSS(self, emb_vecs: List[List[float]], n_clusters: int) -> Tuple[List[List[float]], List[int], float]:
        """
        This function performs K-means clustering on a list of input vectors and calculates the Within-Cluster Sum of Squares (WCSS) for the resulting clusters.

        Parameters
        -----------
        emb_vecs: List[List[float]]
            A list of vectors to be clustered.
        n_clusters: int
            The number of clusters to form.

        Returns
        --------
        Tuple[List[List[float]], List[int], float]
            Three elements:
            1) A list containing the cluster centers.
            2) A list containing the cluster index for each input vector.
            3) The Within-Cluster Sum of Squares (WCSS) value for the clustering.
        """
        emb_vecs = np.array(emb_vecs)

        # Initialize cluster centers
        cluster_centers = self.initialize_centroids(emb_vecs, n_clusters)

        for iteration in range(self.max_iter):
            # Assign each point to the nearest cluster
            cluster_indices = np.array([self.closest_center(point, cluster_centers) for point in emb_vecs])

            # Calculate new cluster centers
            new_centers = np.array([emb_vecs[cluster_indices == i].mean(axis=0) for i in range(n_clusters)])

            # Check for convergence
            if np.allclose(cluster_centers, new_centers, rtol=self.tol):
                break

            cluster_centers = new_centers

        # Compute WCSS
        wcss = self.compute_wcss(emb_vecs, cluster_indices, cluster_centers)

        # Convert the cluster centers and cluster indices to lists
        cluster_centers_list = cluster_centers.tolist()
        cluster_indices_list = cluster_indices.tolist()

        return cluster_centers_list, cluster_indices_list, wcss

    def cluster_hierarchical_single(self, emb_vecs: List, n_clusters) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with single linkage.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        emb_vecs = np.array(emb_vecs)
        cluster_indices = AgglomerativeClustering(n_clusters=n_clusters, linkage='single').fit_predict(
            emb_vecs)
        return cluster_indices

    def cluster_hierarchical_complete(self, emb_vecs: List, n_clusters) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with complete linkage.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        emb_vecs = np.array(emb_vecs)
        cluster_indices = AgglomerativeClustering(n_clusters=n_clusters, linkage='complete').fit_predict(emb_vecs)
        return cluster_indices

    def cluster_hierarchical_average(self, emb_vecs: List, n_clusters) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with average linkage.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        emb_vecs = np.array(emb_vecs)
        cluster_indices = AgglomerativeClustering(n_clusters=n_clusters, linkage='average').fit_predict(emb_vecs)
        return cluster_indices

    def cluster_hierarchical_ward(self, emb_vecs: List, n_clusters) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with Ward's method.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        emb_vecs = np.array(emb_vecs)
        cluster_indices = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit_predict(emb_vecs)
        return cluster_indices

    def visualize_kmeans_clustering_wandb(self, data, n_clusters, project_name, run_name):
        """ This function performs K-means clustering on the input data and visualizes the resulting clusters by logging a scatter plot to Weights & Biases (wandb).

        This function applies the K-means algorithm to the input data and generates a scatter plot where each data point is colored according to its assigned cluster.
        For visualization use convert_to_2d_tsne to make your scatter plot 2d and visualizable.
        The function performs the following steps:
        1. Initialize a new wandb run with the provided project and run names.
        2. Perform K-means clustering on the input data with the specified number of clusters.
        3. Obtain the cluster labels for each data point from the K-means model.
        4. Create a scatter plot of the data, coloring each point according to its cluster label.
        5. Log the scatter plot as an image to the wandb run, allowing visualization of the clustering results.
        6. Close the plot display window to conserve system resources (optional).

        Parameters
        -----------
        data: np.ndarray
            The input data to perform K-means clustering on.
        n_clusters: int
            The number of clusters to form during the K-means clustering process.
        project_name: str
            The name of the wandb project to log the clustering visualization.
        run_name: str
            The name of the wandb run to log the clustering visualization.

        Returns
        --------
        None
        """
        wandb.init(project=project_name, name=run_name)
        kmeans = KMeans(n_clusters=n_clusters)
        cluster_indices = kmeans.fit_predict(data)
        plt.scatter(data[:, 0], data[:, 1], c=cluster_indices, cmap='viridis')
        plt.title("K-means Clustering")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        wandb.log({"K-means Clustering": plt})
        plt.close()

    def wandb_plot_hierarchical_clustering_dendrogram(self, data, project_name, linkage_method, run_name):
        """ This function performs hierarchical clustering on the provided data and generates a dendrogram plot, which is then logged to Weights & Biases (wandb).

        The dendrogram is a tree-like diagram that visualizes the hierarchical clustering process. It shows how the data points (or clusters) are progressively merged into larger clusters based on their similarity or distance.

        The function performs the following steps:
        1. Initialize a new wandb run with the provided project and run names.
        2. Perform hierarchical clustering on the input data using the specified linkage method.
        3. Create a linkage matrix, which represents the merging of clusters at each step of the hierarchical clustering process.
        4. Generate a dendrogram plot using the linkage matrix.
        5. Log the dendrogram plot as an image to the wandb run.
        6. Close the plot display window to conserve system resources.

        Parameters
        -----------
        data: np.ndarray
            The input data to perform hierarchical clustering on.
        linkage_method: str
            The linkage method for hierarchical clustering. It can be one of the following: "average", "ward", "complete", or "single".
        project_name: str
            The name of the wandb project to log the dendrogram plot.
        run_name: str
            The name of the wandb run to log the dendrogram plot.

        Returns
        --------
        None
        """
        wandb.init(project=project_name, name=run_name)
        linkage_matrix = linkage(data, method=linkage_method)
        # plt.figure(figsize=(10, 7))
        # dendrogram(linkage_matrix)
        # plt.title(f'Hierarchical Clustering Dendrogram ({linkage_method} linkage)')
        # plt.xlabel('sample index')
        # plt.ylabel('distance')
        fig = ff.create_dendrogram(linkage_matrix, orientation='left')

        # Update the layout to make it more readable
        fig.update_layout(title='Hierarchical Clustering Dendrogram',
                          xaxis_title='Distance',
                          yaxis_title='Sample Index')

        # plot_filename = "dendrogram.png"
        # plt.savefig(plot_filename)
        # plt.close()

        # wandb.log({"Hierarchical Clustering Dendrogram": wandb.Image(plot_filename)})
        wandb.log({"dendrogram": fig})

    def plot_kmeans_cluster_scores(self, embeddings: List, true_labels: List, k_values: List[int], project_name=None,
                                   run_name=None):
        """ This function, using implemented metrics in clustering_metrics, calculates and plots both purity scores and silhouette scores for various numbers of clusters.
        Then using wandb plots the respective scores (each with a different color) for each k value.

        Parameters
        -----------
        embeddings : List
            A list of vectors representing the data points.
        true_labels : List
            A list of ground truth labels for each data point.
        k_values : List[int]
            A list containing the various values of 'k' (number of clusters) for which the scores will be calculated.
            Default is range(2, 9), which means it will calculate scores for k values from 2 to 8.
        project_name : str
            Your wandb project name. If None, the plot will not be logged to wandb. Default is None.
        run_name : str
            Your wandb run name. If None, the plot will not be logged to wandb. Default is None.

        Returns
        --------
        None
        """
        cm = ClusteringMetrics()
        silhouette_scores = []
        purity_scores = []
        # Calculating Silhouette Scores and Purity Scores for different values of k
        for k in k_values:
            kmeans = KMeans(n_clusters=k)
            cluster_indices = kmeans.fit_predict(embeddings)
            silhouette_scores.append(silhouette_score(embeddings, cluster_indices))
            purity_scores.append(cm.purity_score(true_labels, cluster_indices))
            # Using implemented metrics in clustering_metrics, get the score for each k in k-means clustering
            # and visualize it.

        # Plotting the scores
        plt.plot(k_values, silhouette_scores, label='Silhouette Score')
        plt.plot(k_values, purity_scores, label='Purity Score')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Score')
        plt.title('K-means Cluster Scores')
        plt.legend()

        # Logging the plot to wandb
        if project_name and run_name:
            import wandb
            run = wandb.init(project=project_name, name=run_name)
            wandb.log({"Cluster Scores": plt})
        plt.show()

    def visualize_elbow_method_wcss(self, embeddings: List, k_values: List[int], project_name: str, run_name: str):
        """ This function implements the elbow method to determine the optimal number of clusters for K-means clustering based on the Within-Cluster Sum of Squares (WCSS).

        The elbow method is a heuristic used to determine the optimal number of clusters in K-means clustering. It involves plotting the WCSS values for different values of K (number of clusters) and finding the "elbow" point in the curve, where the marginal improvement in WCSS starts to diminish. This point is considered as the optimal number of clusters.

        The function performs the following steps:
        1. Iterate over the specified range of K values.
        2. For each K value, perform K-means clustering using the `cluster_kmeans_WCSS` function and store the resulting WCSS value.
        3. Create a line plot of WCSS values against the number of clusters (K).
        4. Log the plot to Weights & Biases (wandb) for visualization and tracking.

        Parameters
        -----------
        embeddings: List
            A list of vectors representing the data points to be clustered.
        k_values: List[int]
            A list of K values (number of clusters) to explore for the elbow method.
        project_name: str
            The name of the wandb project to log the elbow method plot.
        run_name: str
            The name of the wandb run to log the elbow method plot.

        Returns
        --------
        None
        """
        # Initialize wandb
        wandb.init(project=project_name, name=run_name)

        # Compute WCSS values for different K values
        wcss_values = []
        for k in k_values:
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(embeddings)
            wcss_values.append(kmeans.inertia_)

        # Plot the elbow method
        plt.plot(k_values, wcss_values)
        plt.title('Elbow Method for Optimal k')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('WCSS')

        # Log the plot to wandb
        wandb.log({"Elbow Method": plt})
        plt.close()

    def plot_clusters(self, emb_vecs: List[List[float]], cluster_indices: List[int],
                      cluster_centers: List[List[float]]):
        emb_vecs = np.array(emb_vecs)
        cluster_centers = np.array(cluster_centers)
        unique_clusters = np.unique(cluster_indices)

        for cluster in unique_clusters:
            points = emb_vecs[np.where(cluster_indices == cluster)]
            plt.scatter(points[:, 0], points[:, 1], label=f'Cluster {cluster}')

        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=300, c='red', marker='X', label='Centroids')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('K-Means Clustering')
        plt.legend()
        plt.show()
