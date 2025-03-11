# dire.py

"""
Provides the main class for dimension reduction
"""

#
# Imports
#

from jax import jit, lax, vmap, random, device_put
import functools
import jax.numpy as jnp
import numpy as np
import faiss
from scipy.sparse.csgraph import laplacian
from scipy.optimize import curve_fit
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import PCA, KernelPCA
from tqdm import tqdm
import gc
import pandas as pd
import plotly.express as px
import sys
import os
from loguru import logger


#
# Main class for DImension REduction
#
class DiRe:
    """
    Dimension Reduction (DiRe) is a class designed to reduce the dimensionality of high-dimensional
    data using various embedding techniques and optimization algorithms. It supports embedding
    initialization methods such as random and spectral embeddings and utilizes k-nearest neighbors
    for manifold learning.

    Parameters
    ----------
    dimension: (int) Embedding dimension, default 2.
    n_neighbors: (int) Number of nearest neighbors to consider for each point, default 16.
    init_embedding_type: (str)
        Method to initialize the embedding; choices are:
         - 'random' for random projection based on the Johnson-Lindenstrauss lemma;
         - 'spectral' for spectral embedding (with sim_kernel as similarity kernel);
         - 'pca' for PCA embedding (classical, no kernel).
         By default, 'random'.
    sim_kernel: (callable)
        A similarity kernel function that transforms a distance metric to a similarity score.
        The function should have the form `lambda distance: float -> similarity: float`; default `None`.
    max_iter_layout: (int)
        Maximum number of iterations to run the layout optimization, default 128.
    min_dist: (float)
        Minimum distance scale for distribution kernel, default 0.01.
    spread: (float)
        Spread of the distribution kernel, default 1.0.
    cutoff: (float)
        Cutoff for clipping forces during layout optimization, default 42.0.
    n_sample_dirs: (int)
        Number of directions to sample in random sampling, default 8.
    sample_size: (int)
        Number of samples per direction in random sampling, default 16.
    neg_ratio: (int)
        Ratio of negative to positive samples in random sampling, default 8.
    my_logger: (logger.Logger or `None`)
        Custom logger for logging events; if None, a default logger is created, default `None`.
    verbose: (bool)
        Flag to enable verbose output, default `True`.

    Attributes
    ----------
    dimension: int
        Target dimensionality of the output space.
    n_neighbors: int
        Number of neighbors to consider in the k-nearest neighbors graph.
    init_embedding_type: str
        Chosen method for initial embedding.
    sim_kernel: callable
        Similarity kernel function to be used if 'init_embedding_type' is 'spectral', by default `None`.
    pca_kernel: callable
        Kernel function to be used if 'init_embedding_type' is 'pca', by default `None`.
    max_iter_layout: int
        Maximum iterations for optimizing the layout.
    min_dist: float
        Minimum distance for repulsion used in the distribution kernel.
    spread: float
        Spread between the data points used in the distribution kernel.
    cutoff: float
        Maximum cutoff for forces during optimization.
    n_sample_dirs: int
        Number of random directions sampled.
    sample_size: int
        Number of samples per random direction.
    neg_ratio: int
        Ratio of negative to positive samples in the sampling process.
    logger: logger.Logger or `None`
        Logger used for logging informational and warning messages.

    Methods
    -------
    fit_transform(data)
        A convenience method that fits the model and then transforms the data.
        The separate `fit` and `transform` methods can only be used one after
        another because dimension reduction is applied to the dataset as a whole.
    visualize(labels=None, point_size=2)
        Visualizes the transformed data, optionally using labels to color the points.
    """

    def __init__(self,
                 dimension=2,
                 n_neighbors=16,
                 init_embedding_type='random',
                 sim_kernel=None,
                 pca_kernel=None,
                 max_iter_layout=128,
                 min_dist=1e-2,
                 spread=1.0,
                 cutoff=42.0,
                 n_sample_dirs=8,
                 sample_size=16,
                 neg_ratio=8,
                 my_logger=None,
                 verbose=True):
        """
        Class constructor
        """

        #
        self.dimension = dimension
        """ Embedding dimension """
        self.n_neighbors = n_neighbors
        """ Number of neighbors for kNN computations"""
        self.init_embedding_type = init_embedding_type
        """ Type of the initial embedding (PCA, random, spectral) """
        self.sim_kernel = sim_kernel
        """ Similarity kernel """
        self.pca_kernel = pca_kernel
        """ PCA kernel """
        self.max_iter_layout = max_iter_layout
        """ Max iterations for the force layout """
        self.min_dist = min_dist
        """ Min distance between points in layout """
        self.spread = spread
        """ Layout spread """
        self.cutoff = cutoff
        """ Cutoff for layout displacement """
        self.n_sample_dirs = n_sample_dirs
        """ Number of sampling directions for layout"""
        self.sample_size = sample_size
        """ Sample size for attraction """
        self.neg_ratio = neg_ratio
        """ Ratio for repulsion sample size """
        self.init_embedding = None
        """ Initial embedding """
        self.layout = None
        """ Layout output """
        self.a = None
        """ Probability kernel parameter """
        self.b = None
        """ Probability kernel parameter """
        self.data = None
        """ Higher-dimensional data """
        self.n_samples = None
        """ Number of data points """
        self.data_dim = None
        """ Dimension of data """
        self.distances = None
        """ Distances in the kNN graph """
        self.indices = None
        """ Neighbor indices in the kNN graph """
        self.nearest_neighbor_distances = None
        """ Neighbor indices in the kNN graph, excluding the point itself """
        self.row_idx = None
        """ Row indices for nearest neighbors """
        self.col_idx = None
        """ Column indices for nearest neighbors """
        self.adjacency = None
        """ kNN adjacency matrix """
        #
        if my_logger is None:
            logger.remove()
            sink = sys.stdout if verbose else open(os.devnull, 'w')
            logger.add(sink, level="INFO")
            self.logger = logger
            """ System logger """
        else:
            self.logger = my_logger

    #
    # Fitting the distribution kernel with given min_dist and spread
    #

    def find_ab_params(self, min_dist=0.01, spread=1.0):
        """
        Rational function approximation to the probabilistic t-kernel
        """

        #
        self.logger.info('find_ab_params ...')

        def curve(x, a, b):
            return 1.0 / (1.0 + a * x ** (2 * b))
        #
        xv = np.linspace(0, spread * 3, 300)
        yv = np.zeros(xv.shape)
        yv[xv < min_dist] = 1.0
        yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
        params, _ = curve_fit(curve, xv, yv)
        #
        self.logger.info(f'a = {params[0]}, b = {params[1]}')
        self.logger.info('find_ab_params done ...')
        #
        return params[0], params[1]

    #
    # Fitting on data
    #

    def fit(self, data):
        """
        Fit the model to data: create the kNN graph and fit the probability kernel to force layout parameters.

        Parameters
        ----------
        data: (numpy.ndarray)
            High-dimensional data to fit the model. Shape (n_samples, n_features).

        Returns
        -------
        self: The DiRe instance fitted to data.
        """

        #
        self.logger.info('fit ...')
        #
        self.data = data

        self.n_samples = self.data.shape[0]
        self.data_dim = self.data.shape[1]

        self.make_knn_adjacency()

        self.a, self.b = self.find_ab_params(self.min_dist, self.spread)
        #
        self.logger.info('fit done ...')
        #
        return self

    #
    # Transforming data
    #

    def transform(self):
        """
        Transform the fitted data into a lower-dimensional layout.

        Returns
        -------
        numpy.ndarray:
            The lower-dimensional data embedding. Shape (n_samples, dimension).
        """

        #
        self.logger.info('transform ...')
        #
        if self.init_embedding_type == 'random':
            self.do_random_embedding()
        elif self.init_embedding_type == 'spectral':
            self.do_spectral_embedding()
        elif self.init_embedding_type == 'pca':
            self.do_pca_embedding()
        else:
            self.logger.warning(f'transform ERROR: embedding method "{self.init_embedding_type}" not available')
            return None
        #
        self.do_layout()
        #
        self.logger.info('transform done ...')
        #
        return self.layout

    #
    # Fit and transform combined
    #

    def fit_transform(self, data):
        """
        Fit the model to data and transform it into the low-dimensional layout.

        Parameters
        ----------
        data: (numpy.ndarray)
            High-dimensional data to fit and transform. Shape (n_samples, n_features)

        Returns
        -------
        numpy.ndarray:
            The lower-dimensional data embedding. Shape (n_samples, dimension)
        """

        #
        self.data = data
        self.fit(self.data)
        #
        return self.transform()

    #
    # Computing the kNN adjacency matrix (sparse)
    #

    def make_knn_adjacency(self):
        """
        Internal routine building the adjacency matrix for the kNN graph
        """
        #
        self.logger.info('make_knn_adjacency ...')
        #
        self.data = np.ascontiguousarray(self.data.astype(np.float32))
        n_neighbors = self.n_neighbors + 1  # Including the point itself
        data_dim = self.data.shape[1]

        try:
            res = faiss.StandardGpuResources()
            index = faiss.GpuIndexFlatL2(res, data_dim)
            self.logger.info('Using GPU for kNN search')
        except AttributeError as e:
            # No GPU available
            self.logger.warning('GPU resources unavailable, using CPU. ERROR: %s', str(e))
            index = faiss.IndexFlatL2(data_dim)
        except Exception as e:
            # Handle other potential exceptions
            self.logger.warning('An error occurred, trying CPU fallback. ERROR: %s', str(e))
            index = faiss.IndexFlatL2(data_dim)

        index.add(self.data)

        self.distances, self.indices = index.search(self.data, n_neighbors)

        self.nearest_neighbor_distances = self.distances[:, 1]  # Excluding the point itself
        self.row_idx = np.repeat(np.arange(self.n_samples), n_neighbors)
        self.col_idx = self.indices.ravel()

        data_values = self.distances.ravel()
        adj_mat = csr_matrix((data_values, (self.row_idx, self.col_idx)), shape=(self.n_samples, self.n_samples))
        self.adjacency = adj_mat

        del index
        gc.collect()
        #
        self.logger.info('make_knn_adjacency done ...')

    #
    # PCA embedding for the dataset (classical, no kernel)
    #

    def do_pca_embedding(self):
        """
        Internal routine for initial PCA embedding
        """
        #
        self.logger.info('do_pca_embedding ...')
        #
        if self.pca_kernel is not None:
            self.logger.info('kernelized embedding ...')
            pca = KernelPCA(n_components=self.dimension, kernel=self.pca_kernel)
            self.init_embedding = pca.fit_transform(self.data)
        else:
            self.logger.info('standard embedding ...')
            pca = PCA(n_components=self.dimension)
            self.init_embedding = pca.fit_transform(self.data)
        #
        self.logger.info('do_pca_embedding done ...')

    #
    # Spectral embedding for the kNN graph (weighted) with
    # a similarity kernel to transform distances (default: no kernel)
    #

    def do_spectral_embedding(self):
        """
        Internal routine for initial spectral embedding
        """
        #
        self.logger.info('do_spectral_embedding ...')
        #
        if self.sim_kernel is not None:
            data_values = self.sim_kernel(self.adjacency.data)
            adj_mat = csr_matrix((data_values, (self.row_idx, self.col_idx)), shape=(self.n_samples, self.n_samples))
        else:
            adj_mat = self.adjacency
        #
        lap = laplacian(adj_mat+adj_mat.T, normed=True)
        k = self.dimension+1
        _, eigenvectors = eigsh(lap, k, which='SM')
        self.init_embedding = eigenvectors[:, 1:k]
        #
        self.logger.info('do_spectral_embedding done ...')

    #
    # Random projection embedding
    #

    def do_random_embedding(self):
        """
        Internal routine for initial random projections embedding
        """
        #
        self.logger.info('do_random_embedding ...')
        #
        key = random.PRNGKey(13)
        rand_basis = random.normal(key, (self.dimension, self.data_dim))

        data_matrx = device_put(self.data)
        rand_basis = device_put(rand_basis)

        self.init_embedding = data_matrx @ rand_basis.T
        #
        self.logger.info('do_random_embedding done ...')

    #
    # Random sampling of the closest n_samples by using random
    # projections in n_dirs directions. Negative samples are
    # added at the ratio of neg_ratio negative samples for each
    # positive sample taken.
    #

    def do_rand_sampling(self, key, arr, n_samples, n_dirs, neg_ratio):
        """
        Random sampling of the closest n_samples by using random
        projections in n_dirs directions. Negative samples are
        added at the ratio of neg_ratio negative samples for each
        positive sample taken.
        """
        #
        self.logger.info('do_rand_sampling ...')
        #
        sampled_indices_list = []
        arr_len = len(arr)

        for vec in rand_directions(key, self.dimension, n_dirs):
            arr_proj = vec @ arr.T
            indices_sort = jnp.argsort(arr_proj)
            indices = vmap_get_slice(indices_sort, n_samples, jnp.arange(arr_len))
            indices = indices[indices_sort]
            sampled_indices_list += [indices]

        n_neg_samples = int(neg_ratio * n_samples)
        neg_indices = random.randint(key, (arr_len, n_neg_samples), 0, arr_len)
        sampled_indices_list += [neg_indices]

        sampled_indices = jnp.concatenate(sampled_indices_list, axis=-1)
        #
        self.logger.info('do_rand_sampling done ...')
        #
        return sampled_indices

    #
    # Create layout (auxiliary functions defined at the bottom)
    #

    def do_layout(self):
        """
        Internal routine for creating the lower-dimensional layout
        """
        #
        self.logger.info('do_layout ...')
        #
        cutoff = jnp.array([self.cutoff])
        num_iterations = self.max_iter_layout
        batch_size = self.sample_size

        if batch_size == 'auto':
            batch_size = int(self.n_neighbors * np.log(self.n_samples))
            batch_size = max(min(512, batch_size), 32)

        n_dirs = self.n_sample_dirs
        neg_ratio = self.neg_ratio

        init_pos_jax = device_put(self.init_embedding)
        init_pos_jax -= init_pos_jax.mean(axis=0)
        init_pos_jax /= init_pos_jax.std(axis=0)

        indices_jax = device_put(self.indices)

        key = random.PRNGKey(42)

        for iter_id in tqdm(range(num_iterations)):
            logger.debug(f'Iteration {iter_id + 1}')
            indices_emb_jax = self.do_rand_sampling(key,
                                                    init_pos_jax,
                                                    batch_size,
                                                    n_dirs,
                                                    neg_ratio)
            indices_emb_jax = device_put(indices_emb_jax)

            # Calculate positions and distances for attraction forces
            v_pos = init_pos_jax[:, None, :]
            u_pos = init_pos_jax[indices_jax]
            position_diff = u_pos - v_pos
            distance_geom = jnp.linalg.norm(position_diff, axis=2, keepdims=True)
            mask = (distance_geom > 0)
            direction = jnp.where(mask, position_diff / distance_geom, 0.0)

            # Attraction forces
            grad_coeff_att_vals = jnp.where(mask,
                                            1.0 * vmap_coeff_att(distance_geom, self.a, self.b)
                                            - 1.0 * vmap_coeff_rep(distance_geom, self.a, self.b),
                                            0.0
                                            )
            # Here repulsion between genuine kNN neighbours is subtracted (so that it won't exist if added
            # later on based on lower-dimensional layout neighbours
            attraction_force = jnp.sum(grad_coeff_att_vals * direction, axis=1)

            # Calculate positions and distances for repulsion forces
            u_pos = init_pos_jax[indices_emb_jax]
            position_diff = u_pos - v_pos  # Reusing v_pos
            distance_geom = jnp.linalg.norm(position_diff, axis=2, keepdims=True)
            mask = (distance_geom > 0)
            direction = jnp.where(mask, position_diff / distance_geom, 0.0)

            # Repulsion forces
            grad_coeff_rep_vals = jnp.where(mask,
                                            vmap_coeff_rep(distance_geom, self.a, self.b),
                                            0.0
                                            )
            repulsion_force = jnp.sum(grad_coeff_rep_vals * direction, axis=1)

            # Combining forces
            alpha = 1.0 - iter_id / num_iterations
            net_force = alpha * (attraction_force + repulsion_force)
            net_force = jnp.clip(net_force, -cutoff, cutoff)
            init_pos_jax += net_force

        init_pos_jax -= init_pos_jax.mean(axis=0)
        init_pos_jax /= init_pos_jax.std(axis=0)
        self.layout = np.asarray(init_pos_jax)
        #
        self.logger.info('do_layout done ...')
        #

    #
    # Visualize the layout
    #

    def visualize(self, labels=None, point_size=2):
        """
        Generate a visualization of the data in the transformed space using a scatter plot.
        Supports both 2D and 3D visualizations depending on the specified dimension.

        Parameters
        ----------
        labels: (numpy.ndarray or `None`)
            Labels for each data point to color the points in the visualization, default `None`.
        point_size: (int)
            Size of points in the scatter plot, default 2.

        Returns
        -------
        plotly.graph_objs._figure.Figure or `None`:
            A plotly figure object if the visualization is successful; None if no layout is available.
        """

        #
        if self.layout is None:
            self.logger.warning('visualize ERROR: no layout available')
            return None

        if self.dimension == 2:
            self.logger.info('visualize: 2D ...')
            datadf = pd.DataFrame(self.layout, columns=['x', 'y'])
            if labels is not None:
                datadf['label'] = labels
                fig = px.scatter(datadf, x='x', y='y', color='label')
            else:
                fig = px.scatter(datadf, x='x', y='y')
            fig.update_traces(marker=dict(size=point_size))
            return fig
        #
        if self.dimension == 3:
            self.logger.info('visualize: 3D ...')
            datadf = pd.DataFrame(self.layout, columns=['x', 'y', 'z'])
            if labels is not None:
                datadf['label'] = labels
                fig = px.scatter_3d(datadf, x='x', y='y', z='z', color='label')
            else:
                fig = px.scatter_3d(datadf, x='x', y='y', z='z')
            fig.update_traces(marker=dict(size=point_size))
            return fig
        #
        self.logger.warning('visualize ERROR: dimension > 3')
        return None
##


#
# Auxiliary functions
#


#
# Probability kernel
#
@functools.partial(jit, static_argnums=(1, 2))
def distribution_kernel(x, a, b):
    """
    Probability kernel
    """
    return 1.0 / (1.0 + a * x ** (2 * b))


#
# Attraction force
#
@functools.partial(jit, static_argnums=(1, 2))
def grad_coeff_att(x, a, b):
    """
    Attraction force
    """
    y = distribution_kernel(1.0 / x, a, b)
    return 1.0 * y


#
# Repulsion force
#
@functools.partial(jit, static_argnums=(1, 2))
def grad_coeff_rep(x, a, b):
    """
    Repulsion force
    """
    y = distribution_kernel(x, a, b)
    return -1.0 * y


#
# Sampling random directions
#
@functools.partial(jit, static_argnums=(1, 2))
def rand_directions(key, dim=2, num=100):
    """
    Sampling random directions
    """
    points = random.normal(key, (num, dim))
    norms = jnp.sqrt(jnp.sum(points * points, axis=-1))
    return points / norms[:, None]


#
# Dynamic slice around i of (almost) k elements
#
@functools.partial(jit, static_argnums=(1,))
def get_slice(arr, k, i):
    """
    Dynamic slice around i of (almost) k elements
    """
    return lax.dynamic_slice(arr, (i - k // 2,), (k,))


#
# Vectorized versions of the above functions
#

""" Vectorized grad_coeff_att """
vmap_coeff_att = vmap(grad_coeff_att, in_axes=(0, None, None))

""" Vectorized grad_coeff_rep """
vmap_coeff_rep = vmap(grad_coeff_rep, in_axes=(0, None, None))

""" Vectorized get_slice """
vmap_get_slice = vmap(get_slice, in_axes=(None, None, 0))