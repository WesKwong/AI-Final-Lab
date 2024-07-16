import sys

import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from loguru import logger
logger.remove()
logger.add(sys.stdout, level="INFO")

def my_rbf_ker(X1, X2, gamma=None):
    # X1: [n_samples1, n_features]
    # X2: [n_samples2, n_features]
    # return: [n_samples1, n_samples2]
    if gamma is None:
        gamma = 1.0 / X1.shape[1]
    sq_dists = np.sum(X1 ** 2, axis=1, keepdims=True) + \
               np.sum(X2 ** 2, axis=1) - 2 * np.dot(X1, X2.T)
    K = np.exp(-gamma * sq_dists)
    return K

def get_kernel_function(kernel:str):
    if kernel == "rbf":
        return my_rbf_ker
    else:
        raise ValueError("Kernel not supported: {}".format(kernel))

class PCA:
    def __init__(self, n_components:int=2, kernel:str="rbf") -> None:
        self.n_components = n_components
        self.kernel_f = get_kernel_function(kernel)
        # ...

    def fit(self, X:np.ndarray):
        # X: [n_samples, n_features]
        # TODO: implement PCA algorithm
        self.X_mean = np.mean(X, axis=0)
        X_centered = X - self.X_mean
        self.X_centered = X_centered

        if self.kernel_f is not None:
            # kernel PCA
            gram_matrix = self.kernel_f(X_centered, X_centered)
            n_samples = X.shape[0]
            np.fill_diagonal(gram_matrix, 0)
            eigenvalues, eigenvectors = np.linalg.eigh(gram_matrix)
            idx = eigenvalues.argsort()[::-1]
            top_eigenvalues = eigenvalues[idx[:self.n_components]]
            top_eigenvectors = eigenvectors[:, idx[:self.n_components]]
            self.explained_variance_ = top_eigenvalues / (n_samples - 1)
            self.components_ = top_eigenvectors
        else:
            # linear PCA
            cov_matrix = np.dot(X_centered.T, X_centered) / (X_centered.shape[0] - 1)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            idx = eigenvalues.argsort()[::-1]
            top_eigenvalues = eigenvalues[idx[:self.n_components]]
            top_eigenvectors = eigenvectors[:, idx[:self.n_components]]
            self.components_ = top_eigenvectors

    def transform(self, X:np.ndarray):
        # X: [n_samples, n_features]
        X_centered = self.X_centered
        if self.kernel_f is not None:
            # kernel PCA transform
            transformed = np.dot(self.kernel_f(X_centered, X_centered), self.components_)
        else:
            # linear PCA transform
            transformed = np.dot(X_centered, self.components_.T)
        return transformed

class KMeans:
    def __init__(self, n_clusters:int=3, max_iter:int=10) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centers = None
        self.labels = None

    # Randomly initialize the centers
    def initialize_centers(self, points):
        # points: (n_samples, n_dims,)
        n, d = points.shape

        self.k = self.n_clusters
        self.centers = np.zeros((self.k, d))
        for k in range(self.k):
            # use more random points to initialize centers, make kmeans more stable
            random_index = np.random.choice(n, size=10, replace=False)
            self.centers[k] = points[random_index].mean(axis=0)

        return self.centers

    # Assign each point to the closest center
    def assign_points(self, points):
        # points: (n_samples, n_dims,)
        # return labels: (n_samples, )
        # n_samples, _ = points.shape
        # self.labels = np.zeros(n_samples)
        # TODO: Compute the distance between each point and each center
        # and Assign each point to the closest center
        self.labels = np.argmin(np.linalg.norm(points[:, np.newaxis] -
                                               self.centers,
                                               axis=-1)**2,
                                axis=1)
        return self.labels

    # Update the centers based on the new assignment of points
    def update_centers(self, points):
        # points: (n_samples, n_dims,)
        # TODO: Update the centers based on the new assignment of points
        unique_labels, label_counts = np.unique(self.labels, return_counts=True)
        new_centers = np.zeros_like(self.centers)
        for i, label in enumerate(unique_labels):
            if label_counts[i] > 0:  # Avoid division by zero
                new_centers[label] = np.mean(points[self.labels == label], axis=0)
        self.centers = new_centers

    # k-means clustering
    def fit(self, points):
        # points: (n_samples, n_dims,)
        # TODO: Implement k-means clustering
        self.initialize_centers(points)
        for _ in trange(self.max_iter, desc="Clustering"):
            old_centers = np.copy(self.centers)
            self.assign_points(points)
            self.update_centers(points)
            # Check for convergence (no significant change in centers)
            if np.allclose(old_centers, self.centers):
                break
        # pass

    # Predict the closest cluster each sample in X belongs to
    def predict(self, points):
        # points: (n_samples, n_dims,)
        # return labels: (n_samples, )
        return self.assign_points(points)

def load_data():
    words = [
        'computer', 'laptop', 'minicomputers', 'PC', 'software', 'Macbook',
        'king', 'queen', 'monarch','prince', 'ruler','princes', 'kingdom', 'royal',
        'man', 'woman', 'boy', 'teenager', 'girl', 'robber','guy','person','gentleman',
        'banana', 'pineapple','mango','papaya','coconut','potato','melon',
        'shanghai','HongKong','chinese','Xiamen','beijing','Guilin',
        'disease', 'infection', 'cancer', 'illness',
        'twitter', 'facebook', 'chat', 'hashtag', 'link', 'internet',
    ]
    w2v = KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary = True)
    vectors = []
    for w in words:
        vectors.append(w2v[w].reshape(1, 300))
    vectors = np.concatenate(vectors, axis=0)
    return words, vectors

@logger.catch
def main():
    logger.info("PCA + KMeans")
    logger.info("Loading data...")
    words, data = load_data()
    logger.info("Fitting PCA...")
    pca = PCA(n_components=2)
    pca.fit(data)
    logger.info("Transforming data...")
    data_pca = pca.transform(data)

    logger.info("Fitting KMeans...")
    kmeans = KMeans(n_clusters=7)
    kmeans.fit(data_pca)
    logger.info("Predicting clusters...")
    clusters = kmeans.predict(data_pca)

    # plot the data

    logger.info("Plotting...")
    plt.figure()
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=clusters)
    for i in range(len(words)):
        plt.annotate(words[i], data_pca[i, :])
    plt.title("PB21051056")
    plt.savefig("PCA_KMeans.png")
    logger.info("Done!")

if __name__=='__main__':
    main()