# dire.py

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
from tqdm import tqdm
import gc
import pandas as pd
import plotly.express as px
import sys, os
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
    dimension : int, optional
        The target dimensionality of the output space, by default 2.
    n_neighbors : int, optional
        The number of nearest neighbors to consider for each point, by default 16.
    init_embedding_type : str, optional
        Method to initialize the embedding; choices are 'random' or 'spectral', by default 'random'.
    max_iter_layout : int, optional
        Maximum number of iterations to run the layout optimization, by default 128.
    min_dist : float, optional
        Minimum distance scale for distribution kernel, by default 0.01.
    spread : float, optional
        Spread of the distribution kernel, by default 1.0.
    cutoff : float, optional
        Cutoff for clipping forces during layout optimization, by default 42.0.
    n_sample_dirs : int, optional
        Number of directions to sample in random sampling, by default 8.
    sample_size : int, optional
        Number of samples per direction in random sampling, by default 16.
    neg_ratio : int, optional
        Ratio of negative to positive samples in random sampling, by default 8.
    my_logger : logger.Logger or None, optional
        Custom logger for logging events; if None, a default logger is created, by default None.
    verbose : bool, optional
        Flag to enable verbose output, by default True.

    Attributes
    ----------
    dimension : int
        Target dimensionality of the output space.
    n_neighbors : int
        Number of neighbors to consider in the k-nearest neighbors graph.
    init_embedding_type : str
        Chosen method for initial embedding.
    max_iter_layout : int
        Maximum iterations for optimizing the layout.
    min_dist : float
        Minimum distance used in the distribution kernel.
    spread : float
        Spread used in the distribution kernel.
    cutoff : float
        Maximum cutoff for forces during optimization.
    n_sample_dirs : int
        Number of random directions sampled.
    sample_size : int
        Number of samples per random direction.
    neg_ratio : int
        Ratio of negative to positive samples in the sampling process.
    logger : logger.Logger
        Logger used for logging informational and warning messages.

    Methods
    -------
    fit(data)
        Fits the model to the data using k-nearest neighbors and initializes embedding.
    transform()
        Transforms the data into a lower-dimensional space based on the fitted model.
    fit_transform(data)
        A convenience method that fits the model and then transforms the data.
    visualize(labels=None, point_size=1)
        Visualizes the transformed data, optionally using labels to color the points.
    """

    def __init__(self,
                 dimension=2,
                 n_neighbors=16,
                 init_embedding_type='random',
                 max_iter_layout=128,
                 min_dist=1e-2,
                 spread=1.0,
                 cutoff=42.0,
                 n_sample_dirs=8,
                 sample_size=16,
                 neg_ratio=8,
                 my_logger=None,
                 verbose=True):
        #
        self.dimension = dimension
        self.n_neighbors = n_neighbors
        self.init_embedding_type = init_embedding_type
        self.max_iter_layout = max_iter_layout
        self.min_dist = min_dist
        self.spread = spread
        self.cutoff = cutoff
        self.n_sample_dirs = n_sample_dirs
        self.sample_size = sample_size
        self.neg_ratio = neg_ratio
        #
        self.init_embedding = None
        self.layout = None
        self.a = None
        self.b = None
        #
        if my_logger is None:
            logger.remove()
            sink = sys.stdout if verbose else open(os.devnull, 'w')
            logger.add(sink, level="INFO")
            self.logger = logger
        else:
            self.logger = my_logger

    #
    # Fitting the distribution kernel with given min_dist and spread
    #

    def find_ab_params(self, min_dist=0.01, spread=1.0):
        #
        self.logger.info('find_ab_params ...')
        #
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
        Fit the model to the data using k-nearest neighbors to create an adjacency matrix and
        initialize the embedding based on the specified method.

        Parameters
        ----------
        data : numpy.ndarray
            High-dimensional data to fit the model. Shape (n_samples, n_features)

        Returns
        -------
        self : DiRe
            The DiRe instance fitted to the data.
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
        Transform the fitted data into a lower-dimensional space using the learned parameters and embeddings.

        Returns
        -------
        layout : numpy.ndarray
            The transformed data in the lower-dimensional space. Shape (n_samples, dimension)
        """

        #
        self.logger.info('transform ...')
        #
        if self.init_embedding_type == 'random':
            self.do_random_embedding()
        elif self.init_embedding_type == 'spectral':
            self.do_spectral_embedding()
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
        Fit the model to the data and transform it using the embedding and layout optimization.

        Parameters
        ----------
        data : numpy.ndarray
            High-dimensional data to fit and transform. Shape (n_samples, n_features)

        Returns
        -------
        layout : numpy.ndarray
            The transformed data in the lower-dimensional space. Shape (n_samples, dimension)
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
        except Exception as e:
            self.logger.warning('GPU resources not available, using CPU. ERROR: %s', str(e))
            index = faiss.IndexFlatL2(data_dim)

        index.add(self.data)

        distances, indices = index.search(self.data, n_neighbors)

        self.distances = distances
        self.indices = indices

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
    # Spectral embedding for the kNN graph (unweighted)
    #

    def do_spectral_embedding(self):
        #
        self.logger.info('do_spectral_embedding ...')
        #
        lap = laplacian(self.adjacency + self.adjacency.T, normed=True)
        k = self.dimension + 1
        _, eigenvectors = eigsh(lap, k, which='SM')
        self.init_embedding = eigenvectors[:, 1:k]
        #
        self.logger.info('do_spectral_embedding done ...')

    #
    # Random projection embedding
    #

    def do_random_embedding(self):
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
        #
        self.logger.info('do_rand_sampling ...')
        #
        sampled_indices_list = []
        N = len(arr)

        for dir in rand_direction(key, self.dimension, n_dirs):
            arr_proj = dir @ arr.T
            indices_sort = jnp.argsort(arr_proj)
            indices = vmap_get_slice(indices_sort, n_samples, jnp.arange(N))
            indices = indices[indices_sort]
            sampled_indices_list += [indices]

        n_neg_samples = int(neg_ratio * n_samples)
        neg_indices = random.randint(key, (N, n_neg_samples), 0, N)
        sampled_indices_list += [neg_indices]

        sampled_indices = jnp.concatenate(sampled_indices_list, axis=-1)
        #
        self.logger.info('do_rand_sampling done ...')
        #
        return sampled_indices

    #
    # Create layout
    #

    def do_layout(self):
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

    def visualize(self, labels=None, point_size=1):
        """
        Generate a visualization of the data in the transformed space using a scatter plot.
        Supports both 2D and 3D visualizations depending on the specified dimension.

        Parameters
        ----------
        labels : numpy.ndarray or None, optional
            Labels for each data point to color the points in the visualization, by default None.
        point_size : int, optional
            Size of points in the scatter plot, by default 1.

        Returns
        -------
        fig : plotly.graph_objs._figure.Figure or None
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
    return 1.0 / (1.0 + a * x ** (2 * b))

#
# Attraction force
#
@functools.partial(jit, static_argnums=(1, 2))
def grad_coeff_att(x, a, b):
    y = distribution_kernel(1.0 / x, a, b)
    return 1.0 * y

#
# Repulsion force
#
@functools.partial(jit, static_argnums=(1, 2))
def grad_coeff_rep(x, a, b):
    y = distribution_kernel(x, a, b)
    return -1.0 * y

#
# Sampling random directions
#
@functools.partial(jit, static_argnums=(1, 2))
def rand_direction(key, dim=2, num=100):
    points = random.normal(key, (num, dim))
    norms = jnp.sqrt(jnp.sum(points * points, axis=-1))
    return points / norms[:, None]

#
# Dynamic slice around i of (almost) k elements
#
@functools.partial(jit, static_argnums=(1,))
def get_slice(arr, k, i):
    return lax.dynamic_slice(arr, (i - k // 2,), (k,))

#
# Vectorized versions of the above functions
#
vmap_coeff_att = vmap(grad_coeff_att, in_axes=(0, None, None))
vmap_coeff_rep = vmap(grad_coeff_rep, in_axes=(0, None, None))
vmap_get_slice = vmap(get_slice, in_axes=(None, None, 0))
