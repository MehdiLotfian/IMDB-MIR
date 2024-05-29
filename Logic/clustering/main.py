import numpy as np
import time
from tqdm import tqdm

from Logic.core.word_embedding.fasttext_data_loader import FastTextDataLoader
from Logic.core.word_embedding.fasttext_data_loader import preprocess_text
from sklearn.preprocessing import LabelEncoder
from Logic.core.word_embedding.fasttext_model import FastText
from Logic.core.clustering.dimension_reduction import DimensionReduction
from Logic.core.clustering.clustering_metrics import ClusteringMetrics
from Logic.core.clustering.clustering_utils import ClusteringUtils
from collections import Counter

import sys
sys.setrecursionlimit(20000)
tqdm.pandas()

if __name__ == '__main__':
    # Main Function: Clustering Tasks

    # 0. Embedding Extraction
    fasttext_dl = FastTextDataLoader('C:/Users/mehdi/IMDB-MIR/Logic/core/IMDB_crawled.json')
    df = fasttext_dl.read_data_to_df()
    columns_to_keep = ["summaries", "genres"]
    df = df.dropna(subset=columns_to_keep)

    df['text'] = df.progress_apply(lambda row: ' '.join(row['summaries']), axis=1)
    df['genre'] = df.progress_apply(lambda row: row['genres'][0] if len(row['genres']) > 0 else None, axis=1)
    df = df.dropna(subset=['genre'])

    X = df['text'].progress_apply(lambda x: preprocess_text(x))

    le = LabelEncoder()
    y = le.fit_transform(df['genre'])

    fasttext = FastText()
    fasttext.prepare(None, 'load', path='C:/Users/mehdi/IMDB-MIR/Logic/core/word_embedding/FastText_model.bin')
    X = [fasttext.model.get_sentence_vector(i) for i in X]
    X_list = X.copy()
    X = np.array(X)

    # 1. Dimension Reduction Perform Principal Component Analysis (PCA): - Reduce the dimensionality of features using
    # PCA. (you can use the reduced feature afterward or use to the whole embeddings) - Find the Singular Values and use
    # the explained_variance_ratio_ attribute to determine the percentage of variance explained by each principal
    # component. - Draw plots to visualize the results.

    dimension_reduction = DimensionReduction()
    X_reduced = dimension_reduction.pca_reduce_dimension(X, n_components=5)

    singular_values = dimension_reduction.pca.singular_values_
    explained_variance_ratio = dimension_reduction.pca.explained_variance_ratio_

    print("Singular Values:", singular_values)
    print("Explained Variance Ratio:", explained_variance_ratio)

    dimension_reduction.wandb_plot_explained_variance_by_components(X, 'MIR_P2', 'plot_explained_variance_by_components')

    # Implement t-SNE (t-Distributed Stochastic Neighbor Embedding): - Create the convert_to_2d_tsne function,
    # which takes a list of embedding vectors as input and reduces the dimensionality to two dimensions using the t-SNE
    # method. - Use the output vectors from this step to draw the diagram.

    X_2d = dimension_reduction.convert_to_2d_tsne(X)
    dimension_reduction.wandb_plot_2d_tsne(X, 'MIR_P2', 'plot_2d_tsne')

    # 2. Clustering
    ## K-Means Clustering
    # Implement the K-means clustering algorithm from scratch.
    #  Create document clusters using K-Means.
    # Run the algorithm with several different values of k.
    # For each run:
    #     - Determine the genre of each cluster based on the number of documents in each cluster.
    #     - Draw the resulting clustering using the two-dimensional vectors from the previous section.
    #     - Check the implementation and efficiency of the algorithm in clustering similar documents.
    # Draw the silhouette score graph for different values of k and perform silhouette analysis to choose the appropriate k.
    # Plot the purity value for k using the labeled data and report the purity value for the final k. (Use the provided functions in utilities)
    n_cluster = len(np.unique(y))
    clustering_utils = ClusteringUtils()
    range_k = [10, 15, 20, 25, 30, 35]
    clustering_metrics = ClusteringMetrics()
    for k in range_k:
        start_time = time.time()
        cluster_centers, cluster_indices, wcss = clustering_utils.cluster_kmeans_WCSS(X_list, k)
        end_time = time.time()
        duration = end_time - start_time
        print(f"duration for k={k}: {duration}")

    #    Determine the genre of each cluster
        genre_counts = Counter(cluster_indices)
        print(f"Genre counts for k={k}: {genre_counts}")

    # Plot clusters
        clustering_utils.plot_clusters(X_list, cluster_indices, cluster_centers)

    #     Calculate and print purity
        print(f"adjusted rand score for k={k}")
        print(clustering_metrics.adjusted_rand_score(y, cluster_indices))
        print(f"purity score for k={k}")
        print(clustering_metrics.purity_score(y, cluster_indices))
        print(f"silhouette score score for k={k}")
        print(clustering_metrics.silhouette_score(X_list, cluster_indices))
    clustering_utils.visualize_elbow_method_wcss(X_list, range_k, 'MIR_P2', 'visualize_elbow_method_wcss')


    ## Hierarchical Clustering
    # Perform hierarchical clustering with all different linkage methods.
    # Visualize the results.
    indices = np.random.choice([i for i in range(len(X))], 20)
    X_samples = X[indices]
    X_list_samples = X_samples.tolist()

    # Full data
    y_single = clustering_utils.cluster_hierarchical_single(X_list, n_cluster)
    y_complete = clustering_utils.cluster_hierarchical_complete(X_list, n_cluster)
    y_average = clustering_utils.cluster_hierarchical_average(X_list, n_cluster)
    y_ward = clustering_utils.cluster_hierarchical_ward(X_list, n_cluster)
    clustering_utils.wandb_plot_hierarchical_clustering_dendrogram(X, 'MIR_P2', 'single', 'plot_hierarchical_single')
    clustering_utils.wandb_plot_hierarchical_clustering_dendrogram(X, 'MIR_P2', 'complete', 'plot_hierarchical_complete')
    clustering_utils.wandb_plot_hierarchical_clustering_dendrogram(X, 'MIR_P2', 'average', 'plot_hierarchical_average')
    clustering_utils.wandb_plot_hierarchical_clustering_dendrogram(X, 'MIR_P2', 'ward', 'plot_hierarchical_ward')

    # Sample
    #y_single = clustering_utils.cluster_hierarchical_single(X_list_samples, n_cluster)
    #y_complete = clustering_utils.cluster_hierarchical_complete(X_list_samples, n_cluster)
    #y_average = clustering_utils.cluster_hierarchical_average(X_list_samples, n_cluster)
    #y_ward = clustering_utils.cluster_hierarchical_ward(X_list_samples, n_cluster)
    clustering_utils.wandb_plot_hierarchical_clustering_dendrogram(X_samples, 'MIR_P2', 'single', 'plot_hierarchical_single')
    clustering_utils.wandb_plot_hierarchical_clustering_dendrogram(X_samples, 'MIR_P2', 'complete', 'plot_hierarchical_complete')
    clustering_utils.wandb_plot_hierarchical_clustering_dendrogram(X_samples, 'MIR_P2', 'average', 'plot_hierarchical_average')
    clustering_utils.wandb_plot_hierarchical_clustering_dendrogram(X_samples, 'MIR_P2', 'ward', 'plot_hierarchical_ward')

    # 3. Evaluation
    # Using clustering metrics, evaluate how well your clustering method is performing.
    clustering_metrics = ClusteringMetrics()
    print('SINGLE method')
    print('adjusted rand score')
    print(clustering_metrics.adjusted_rand_score(y, y_single))
    print('purity score')
    print(clustering_metrics.purity_score(y, y_single))
    print('silhouette score')
    print(clustering_metrics.silhouette_score(X_list, y_single))

    print('COMPLETE method')
    print('adjusted rand score')
    print(clustering_metrics.adjusted_rand_score(y, y_complete))
    print('purity score')
    print(clustering_metrics.purity_score(y, y_complete))
    print('silhouette score')
    print(clustering_metrics.silhouette_score(X_list, y_complete))

    print('AVERAGE method')
    print('adjusted rand score')
    print(clustering_metrics.adjusted_rand_score(y, y_average))
    print('purity score')
    print(clustering_metrics.purity_score(y, y_average))
    print('silhouette score')
    print(clustering_metrics.silhouette_score(X_list, y_average))

    print('WARD method')
    print('adjusted rand score')
    print(clustering_metrics.adjusted_rand_score(y, y_ward))
    print('purity score')
    print(clustering_metrics.purity_score(y, y_ward))
    print('silhouette score')
    print(clustering_metrics.silhouette_score(X_list, y_ward))
