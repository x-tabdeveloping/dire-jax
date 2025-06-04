# dire.py

"""
Provides the main class for dimensionality reduction.

The DiRe (Dimensionality Reduction) class implements a modern approach to
dimensionality reduction, leveraging JAX for efficient computation. It uses 
force-directed layout techniques combined with k-nearest neighbor graph 
construction to generate meaningful low-dimensional embeddings of 
high-dimensional data.
"""

#
# Imports
#

import sys
import os
import gc
import functools

# JAX-related imports
import jax
from jax import jit, lax, vmap, random, device_put, device_get
import jax.numpy as jnp

# Scientific and numerical libraries
import numpy as np
from scipy.sparse.csgraph import laplacian
from scipy.optimize import curve_fit
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import PCA, KernelPCA

# Data structures and visualization
import pandas as pd
import plotly.express as px

# Utilities
from tqdm import tqdm
from loguru import logger

# Nearest neighbor search
from .hpindex import HPIndex

#
# Double precision support
#
jax.config.update("jax_enable_x64", True)

#
# Main class for Dimensionality Reduction
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
    sample_size: int or 'auto'
        Number of samples per random direction, unless chosen automatically with 'auto'.
    neg_ratio: int
        Ratio of negative to positive samples in the sampling process.
    logger: logger.Logger or `None`
        Logger used for logging informational and warning messages.
    verbose: bool
        Logger output flag (True = output logger messages, False = flush to null)
    memm: dictionary or `None`
        Memory manager: a dictionary with the batch / memory tile size for different
        hardware architectures. Accepts 'tpu', 'gpu' and 'other' as keys. Values must
        be positive integers.
    mpa: bool
        Mixed Precision Arithmetic flag (True = use MPA, False = always use float64)

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
                 verbose=True,
                 memm=None,
                 mpa=True):
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
        self._init_embedding = None
        """ Initial embedding """
        self._layout = None
        """ Layout output """
        self._a = None
        """ Probability kernel parameter """
        self._b = None
        """ Probability kernel parameter """
        self._data = None
        """ Higher-dimensional data """
        self._n_samples = None
        """ Number of data points """
        self._data_dim = None
        """ Dimension of data """
        self._distances_np = None
        self._distances_jax = None
        """ Distances in the kNN graph """
        self._indices_np = None
        self._indices_jax = None
        """ Neighbor indices in the kNN graph """
        self._nearest_neighbor_distances = None
        """ Neighbor indices in the kNN graph, excluding the point itself """
        self._row_idx = None
        """ Row indices for nearest neighbors """
        self._col_idx = None
        """ Column indices for nearest neighbors """
        self._adjacency = None
        """ kNN adjacency matrix """
        #
        if my_logger is None:
            logger.remove()
            sink = sys.stdout if verbose else open(os.devnull, 'w', encoding='utf-8')
            logger.add(sink, level="INFO")
            self.logger = logger
            """ System logger """
        else:
            self.logger = my_logger
        # Memory manager to be adjusted for each particular type of hardware
        # Below are some minimalist settings that may give less than satisfactory performance
        self.memm = {'gpu': 16384, 'tpu': 8192, 'other': 8192} if memm is None else memm
        # Using Mixed Precision Arithmetic flag (True = MPA, False = no MPA)
        self.mpa = mpa

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
        self._data = data

        self._n_samples = self._data.shape[0]
        self._data_dim = self._data.shape[1]

        self.logger.info(f'Dimension {self._data_dim}, number of samples {self._n_samples}')

        self.make_knn_adjacency()

        self._a, self._b = self.find_ab_params(self.min_dist, self.spread)
        #
        self.logger.info('fit done ...')
        #
        return self

    #
    # Transform fitted data into lower-dimensional representation
    #

    def transform(self):
        """
        Transform the fitted data into a lower-dimensional layout.
        
        This method applies the selected embedding initialization technique
        to the data that has already been fitted (creating the kNN graph), 
        and then optimizes the layout using force-directed placement.
        
        The transformation process involves:

        1. Creating an initial embedding using the specified method
           (random projection, PCA, or spectral embedding)
        2. Optimizing the layout with attractive and repulsive forces
        
        Returns
        -------
        numpy.ndarray
            The lower-dimensional data embedding with shape (n_samples, dimension).
            Points are arranged to preserve the local structure of the original data.
            
        Raises
        ------
        ValueError
            If an unsupported embedding initialization method is specified.
        """
        self.logger.info('transform ...')

        # Create initial embedding based on specified initialization method
        if self.init_embedding_type == 'random':
            self.do_random_embedding()
        elif self.init_embedding_type == 'spectral':
            self.do_spectral_embedding()
        elif self.init_embedding_type == 'pca':
            self.do_pca_embedding()
        else:
            error_msg = f'Unsupported embedding method: "{self.init_embedding_type}"'
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Optimize the layout using force-directed placement
        self.do_layout()

        self.logger.info('transform done ...')

        return self._layout

    def fit_transform(self, data):
        """
        Fit the model to data and transform it into a low-dimensional layout.
        
        This is a convenience method that combines the fitting and transformation
        steps. It first builds the kNN graph and then creates the optimized
        layout in a single operation.
        
        Parameters
        ----------
        data : numpy.ndarray
            High-dimensional data to fit and transform.
            Shape (n_samples, n_features)
            
        Returns
        -------
        numpy.ndarray
            The lower-dimensional embedding of the data.
            Shape (n_samples, dimension)
            
        Notes
        -----
        This method is more memory-efficient than calling fit() and transform()
        separately, as it avoids storing intermediate results.
        """
        # Store the data and perform fitting (build kNN graph)
        self._data = data
        self.fit(self._data)

        # Transform the data (create initial embedding and optimize layout)
        return self.transform()

    #
    # Computing the kNN adjacency matrix (sparse)
    #

    def make_knn_adjacency(self, batch_size=None):
        """
        Internal routine building the adjacency matrix for the kNN graph.

        This method computes the k-nearest neighbors for each point in the dataset
        and constructs a sparse adjacency matrix representing the kNN graph.
        It attempts to use GPU acceleration if available, with a fallback to CPU.

        For large datasets, it uses batching to limit memory usage.

        Parameters
        ----------
        batch_size : int or None, optional
            Number of samples to process at once. If None, a suitable value
            will be automatically determined based on dataset size.

        The method sets the following instance attributes:
        - distances: Distances to the k nearest neighbors (including self)
        - indices: Indices of the k nearest neighbors (including self)
        - nearest_neighbor_distances: Distances to the nearest neighbors (excluding self)
        - row_idx, col_idx: Indices for constructing the sparse adjacency matrix
        - adjacency: Sparse adjacency matrix of the kNN graph
        """
        self.logger.info('make_knn_adjacency ...')

        # Ensure data is in the right format for HPIndex
        self._data = np.ascontiguousarray(self._data.astype(np.float64))
        n_neighbors = self.n_neighbors + 1  # Including the point itself

        # Determine appropriate batch size for memory efficiency
        if batch_size is None:
            # Process in chunks to reduce peak memory usage
            if jax.devices()[0].platform == 'tpu':
                batch_size = min(self.memm['tpu'], self._n_samples)
            elif jax.devices()[0].platform == 'gpu':
                batch_size = min(self.memm['gpu'], self._n_samples)
            else:
                batch_size = min(self.memm['other'], self._n_samples)

        self.logger.info(f'Using batch size: {batch_size}')
        self.logger.debug(f"[KNN] Using precision: {'float32' if self.mpa else 'float64'}")

        if self.mpa:
            self._indices_jax, self._distances_jax = HPIndex.knn_tiled(
                self._data, self._data, n_neighbors, batch_size, batch_size, dtype=jnp.float32)
        else:
            self._indices_jax, self._distances_jax = HPIndex.knn_tiled(
                self._data, self._data, n_neighbors, batch_size, batch_size, dtype=jnp.float64)

        # Wait until ready
        self._indices_jax.block_until_ready()
        self._distances_jax.block_until_ready()

        # Store results in numpy
        self._indices_np = device_get(self._indices_jax).astype(np.int64)
        self._distances_np = device_get(self._distances_jax).astype(np.float64)

        # Extract nearest neighbor distances (excluding self)
        self._nearest_neighbor_distances = self._distances_np[:, 1:]

        # Create indices for sparse matrix construction
        self._row_idx = np.repeat(np.arange(self._n_samples), n_neighbors)
        self._col_idx = self._indices_np.ravel()

        # Create sparse adjacency matrix (memory efficient)
        data_values = self._distances_np.ravel()
        self._adjacency = csr_matrix(
            (data_values, (self._row_idx, self._col_idx)),
            shape=(self._n_samples, self._n_samples)
        )

        # Clean up resources
        gc.collect()

        self.logger.info('make_knn_adjacency done ...')

    #
    # Initialize embedding using different techniques
    #

    def do_pca_embedding(self):
        """
        Initialize embedding using Principal Component Analysis (PCA).
        
        This method creates an initial embedding of the data using PCA, which finds
        a linear projection of the high-dimensional data into a lower-dimensional space
        that maximizes the variance. If a kernel is specified, Kernel PCA is used instead,
        which can capture nonlinear relationships.
        
        Sets the init_embedding attribute with the PCA projection of the data.
        """
        self.logger.info('do_pca_embedding ...')

        if self.pca_kernel is not None:
            # Use Kernel PCA for nonlinear dimensionality reduction
            self.logger.info('Using kernelized PCA embedding...')
            pca = KernelPCA(
                n_components=self.dimension,
                kernel=self.pca_kernel
            )
            self._init_embedding = pca.fit_transform(self._data)
        else:
            # Use standard PCA for linear dimensionality reduction
            self.logger.info('Using standard PCA embedding...')
            pca = PCA(n_components=self.dimension)
            self._init_embedding = pca.fit_transform(self._data)

        self.logger.info('do_pca_embedding done ...')

    def do_spectral_embedding(self):
        """
        Initialize embedding using Spectral Embedding.
        
        This method creates an initial embedding of the data using spectral embedding,
        which is based on the eigenvectors of the graph Laplacian. It relies on the
        kNN graph structure to find a lower-dimensional representation that preserves
        local relationships.
        
        If a similarity kernel is specified, it is applied to transform the distances
        in the adjacency matrix before computing the Laplacian.
        
        Sets the init_embedding attribute with the spectral embedding of the data.
        """
        self.logger.info('do_spectral_embedding ...')

        # Apply similarity kernel if provided
        if self.sim_kernel is not None:
            self.logger.info('Applying similarity kernel to adjacency matrix...')
            # Transform distances using the similarity kernel
            data_values = self.sim_kernel(self._adjacency.data)
            # Create a new adjacency matrix with transformed values
            adj_mat = csr_matrix(
                (data_values, (self._row_idx, self._col_idx)),
                shape=(self._n_samples, self._n_samples)
            )
        else:
            adj_mat = self._adjacency

        # Make the adjacency matrix symmetric by adding it to its transpose
        symmetric_adj = adj_mat + adj_mat.T

        # Compute the normalized Laplacian
        lap = laplacian(symmetric_adj, normed=True)

        # Find the k smallest eigenvectors (k = dimension + 1)
        k = self.dimension + 1
        _, eigenvectors = eigsh(lap, k, which='SM')

        # Skip the first eigenvector (corresponds to constant function)
        self._init_embedding = eigenvectors[:, 1:k]

        self.logger.info('do_spectral_embedding done ...')

    def do_random_embedding(self):
        """
        Initialize embedding using Random Projection.
        
        This method creates an initial embedding of the data using random projection,
        which is a simple and computationally efficient technique for dimensionality
        reduction. It projects the data onto a randomly generated basis, providing
        a good starting point for further optimization.
        
        Random projection is supported by the Johnson-Lindenstrauss lemma, which
        guarantees that the distances between points are approximately preserved
        under certain conditions.
        
        Sets the init_embedding attribute with the random projection of the data.
        """
        self.logger.info('do_random_embedding ...')

        # Create a random projection matrix
        key = random.PRNGKey(13)  # Fixed seed for reproducibility
        rand_basis = random.normal(key, (self.dimension, self._data_dim))

        # Move data and projection matrix to device memory
        data_matrix = device_put(self._data)
        rand_basis = device_put(rand_basis)

        # Project data onto random basis
        self._init_embedding = data_matrix @ rand_basis.T

        self.logger.info('do_random_embedding done ...')

    #
    # Efficient sampling for force-directed layout
    #

    def do_rand_sampling(self, key, arr, n_samples, n_dirs, neg_ratio):
        """
        Sample points for force calculation using random projections.
        
        This method implements an efficient sampling strategy to identify points
        for applying attractive and repulsive forces during layout optimization.
        It uses random projections to quickly identify nearby points in different
        directions, and also adds random negative samples for repulsion.
        
        Parameters
        ----------
        key : jax.random.PRNGKey
            Random number generator key
        arr : jax.numpy.ndarray
            Array of current point positions
        n_samples : int
            Number of samples to take in each direction
        n_dirs : int
            Number of random directions to sample
        neg_ratio : int
            Ratio of negative samples to positive samples
            
        Returns
        -------
        jax.numpy.ndarray
            Array of sampled indices for force calculations
            
        Notes
        -----
        The sampling strategy works as follows:
        1. Generate n_dirs random unit vectors
        2. Project the points onto each vector
        3. For each point, take the n_samples closest points in each direction
        4. Add random negative samples for repulsion
        5. Combine all sampled indices
        
        This approach is more efficient than a full nearest neighbor search
        while still capturing the important local relationships.
        """
        self.logger.info('do_rand_sampling ...')

        sampled_indices_list = []
        arr_len = len(arr)

        # Get random unit vectors for projections
        key, subkey = random.split(key)
        direction_vectors = rand_directions(subkey, self.dimension, n_dirs)

        # For each direction, sample points based on projections
        for vec in direction_vectors:
            # Project points onto the direction vector
            arr_proj = vec @ arr.T

            # Sort indices by projection values
            indices_sort = jnp.argsort(arr_proj)

            # For each point, take n_samples points around it in sorted order
            vmap_get_slice = vmap(get_slice, in_axes=(None, None, 0))
            indices = vmap_get_slice(indices_sort, n_samples, jnp.arange(arr_len))

            # Reorder indices back to original ordering
            indices = indices[indices_sort]

            # Add to list of sampled indices
            sampled_indices_list.append(indices)

        # Generate random negative samples for repulsion
        n_neg_samples = int(neg_ratio * n_samples)
        key, subkey = random.split(key)
        neg_indices = random.randint(subkey, (arr_len, n_neg_samples), 0, arr_len)
        sampled_indices_list.append(neg_indices)

        # Combine all sampled indices
        sampled_indices = jnp.concatenate(sampled_indices_list, axis=-1)

        self.logger.info('do_rand_sampling done ...')

        return sampled_indices

    #
    # Create layout using force-directed optimization
    #

    def do_layout(self, large_dataset_mode=None, force_cpu=False):
        """
        Optimize the layout using force-directed placement with JAX kernels.

        This method takes the initial embedding and iteratively refines it using
        attractive and repulsive forces to create a meaningful low-dimensional
        representation of the high-dimensional data. The algorithm applies:

        1. Attraction forces between points that are neighbors in the high-dimensional space
        2. Repulsion forces between randomly sampled points in the low-dimensional space
        3. Gradual cooling (decreasing force impact) as iterations progress

        The final layout is normalized to have zero mean and unit standard deviation.

        Parameters
        ----------
        large_dataset_mode : bool or None, optional
            If True, use memory-efficient techniques for large datasets.
            If None, automatically determine based on dataset size.
        force_cpu : bool, optional
            If True, force computations on CPU instead of GPU, which can
            be helpful for large datasets that exceed GPU memory.
        """
        self.logger.info('do_layout ...')

        # Setup parameters
        cutoff = jnp.array([self.cutoff])
        num_iterations = self.max_iter_layout

        # Handle automatic batch size calculation if needed
        sample_size = self.sample_size
        if sample_size == 'auto':
            # Scale batch size based on dataset size and neighborhood size
            sample_size = int(self.n_neighbors * np.log(self._n_samples))
            # Ensure batch size is reasonable (not too small or large)
            sample_size = max(min(512, sample_size), 32)

        # Determine if we should use memory-efficient mode for large datasets
        if large_dataset_mode is None:
            large_dataset_mode = ((self._n_samples > 65536) or
                                  (jax.devices()[0].platform == 'tpu'))

        # Other parameters
        n_dirs = self.n_sample_dirs
        neg_ratio = self.neg_ratio

        # Debug initial embedding precision
        self.logger.debug(f"[LAYOUT] Initial embedding precision: {self._init_embedding.dtype}")

        # we shall use force_cpu only as a flag passed to the routine
        # force_cpu = force_cpu or large_dataset_mode and (jax.devices()[0].platform == 'tpu')

        # Initialize and normalize positions
        if force_cpu:
            self.logger.info("Forcing computations on CPU")
            cpu_device = jax.devices('cpu')[0]
            init_pos_jax = device_put(self._init_embedding, device=cpu_device)
            neighbor_indices_jax = device_put(self._indices_np, device=cpu_device)
        else:
            init_pos_jax = device_put(self._init_embedding)
            neighbor_indices_jax = device_put(self._indices_jax)

        init_pos_jax -= init_pos_jax.mean(axis=0)  # Center positions
        init_pos_jax /= init_pos_jax.std(axis=0)  # Normalize variance

        # Set random seed for reproducibility
        key = random.PRNGKey(42)

        # Optimization loop
        for iter_id in tqdm(range(num_iterations)):
            logger.debug(f'Iteration {iter_id + 1}')

            # Sample random points for repulsion
            sample_indices_jax = self.do_rand_sampling(
                key,
                init_pos_jax,
                sample_size,
                n_dirs,
                neg_ratio
            )

            if force_cpu:
                cpu_device = jax.devices('cpu')[0]
                sample_indices_jax = device_put(sample_indices_jax, device=cpu_device)
            else:
                sample_indices_jax = device_put(sample_indices_jax)

            # Split computation for memory efficiency if needed
            if large_dataset_mode:
                # Process in chunks to reduce peak memory usage
                if jax.devices()[0].platform == 'tpu':
                    chunk_size = min(self.memm['tpu'], self._n_samples)
                elif jax.devices()[0].platform == 'gpu':
                    chunk_size = min(self.memm['gpu'], self._n_samples)
                else:
                    chunk_size = min(self.memm['other'], self._n_samples)
                    # this is actually inefficient, but let's postpone

                all_forces = []

                self.logger.info(f"Using memory tiling with tile size: {chunk_size}")

                for chunk_start in range(0, self._n_samples, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, self._n_samples)
                    chunk_indices = jnp.arange(chunk_start, chunk_end)

                    # Process this chunk using our kernelized function
                    chunk_force = self._compute_forces(
                        init_pos_jax,
                        chunk_indices,
                        neighbor_indices_jax[chunk_indices],
                        sample_indices_jax[chunk_indices],
                        alpha=1.0 - iter_id / num_iterations
                    )

                    all_forces.append(chunk_force)

                    # Explicitly clean up to reduce memory pressure
                    gc.collect()

                # Combine results from all chunks
                net_force = jnp.concatenate(all_forces, axis=0)

            else:
                # Process all points at once for smaller datasets
                net_force = self._compute_forces(
                    init_pos_jax,
                    jnp.arange(self._n_samples),
                    neighbor_indices_jax,
                    sample_indices_jax,
                    alpha=1.0 - iter_id / num_iterations
                )

            # Clip forces to prevent extreme movements
            net_force = jnp.clip(net_force, -cutoff, cutoff)

            # Update positions
            init_pos_jax += net_force

            # Ensure we're not accumulating unnecessary computation graphs in JAX
            init_pos_jax.block_until_ready()

            # Clean up resources
            gc.collect()

        # Normalize final layout
        init_pos_jax -= init_pos_jax.mean(axis=0)
        init_pos_jax /= init_pos_jax.std(axis=0)

        # Store final layout
        self._layout = np.asarray(init_pos_jax)

        # Clear any cached values to free memory
        gc.collect()

        self.logger.info('do_layout done ...')

    # Modified _compute_forces method to use the kernel
    def _compute_forces(self, positions, chunk_indices, neighbor_indices, sample_indices, alpha=1.0):
        """
        Compute attractive and repulsive forces for points using JAX kernels.

        This method uses JAX-optimized kernels to efficiently compute forces
        between points during layout optimization.

        Parameters
        ----------
        positions : jax.numpy.ndarray
            Current point positions
        chunk_indices : jax.numpy.ndarray
            Current batch indices
        neighbor_indices : jax.numpy.ndarray
            Indices of neighbors for attractive forces
        sample_indices : jax.numpy.ndarray
            Indices of points for repulsive forces
        alpha : float
            Cooling factor that scales force magnitude

        Returns
        -------
        jax.numpy.ndarray
            Net force vectors for each point
        """

        if self.mpa:
            positions = positions.astype(jnp.float32)
        else:
            positions = positions.astype(jnp.float64)

        self.logger.debug(f"[FORCE] Computing forces on device: {positions.device}")
        self.logger.debug(f"[FORCE] Using precision: {positions.dtype}")

        # Call the JAX-optimized kernel
        return compute_forces_kernel(
            positions,
            chunk_indices,
            neighbor_indices,
            sample_indices,
            alpha,
            self._a,
            self._b,
        )

    #
    # Visualize the layout
    #

    def visualize(self, labels=None, point_size=2, title=None, colormap=None, width=800, height=600, opacity=0.7):
        """
        Generate an interactive visualization of the data in the transformed space.
        
        This method creates a scatter plot visualization of the embedded data, supporting
        both 2D and 3D visualizations depending on the specified dimension. Points can be
        colored by provided labels for clearer visualization of clusters or categories.
        
        Parameters
        ----------
        labels : numpy.ndarray or None, optional
            Labels for each data point to color the points in the visualization.
            If None, all points will have the same color. Default is None.
        point_size : int or float, optional
            Size of points in the scatter plot. Default is 2.
        title : str or None, optional
            Title for the visualization. If None, a default title will be used. Default is None.
        colormap : str or None, optional
            Name of the colormap to use for labels (e.g., 'viridis', 'plasma'). 
            If None, the default Plotly colormap will be used. Default is None.
        width : int, optional
            Width of the figure in pixels. Default is 800.
        height : int, optional
            Height of the figure in pixels. Default is 600.
        opacity : float, optional
            Opacity of the points (0.0 to 1.0). Default is 0.7.
            
        Returns
        -------
        plotly.graph_objs._figure.Figure or None
            A Plotly figure object if the visualization is successful; 
            None if no layout is available or dimension > 3.

        Notes
        -----
        For 3D visualizations, you can rotate, zoom, and pan the plot interactively.
        For both 2D and 3D, hover over points to see their coordinates and labels.
        """
        # Check if layout is available
        if self._layout is None:
            self.logger.warning('visualize ERROR: no layout available')
            return None

        # Set default title if not provided
        if title is None:
            title = f"{self.init_embedding_type.capitalize()} Initialized {self.dimension}D Embedding"

        # Common visualization parameters
        vis_params = {
            'color': 'label' if labels is not None else None,
            'color_continuous_scale': colormap,
            'opacity': opacity,
            'title': title,
            'hover_data': ['label'] if labels is not None else None,
        }

        # Create 2D visualization
        if self.dimension == 2:
            self.logger.info('visualize: 2D ...')

            # Create dataframe for plotting
            datadf = pd.DataFrame(self._layout, columns=['x', 'y'])

            # Add labels if provided
            if labels is not None:
                datadf['label'] = labels

            # Create scatter plot
            fig = px.scatter(
                datadf,
                x='x',
                y='y',
                **vis_params
            )

            # Update layout
            fig.update_layout(
                width=width,
                height=height,
                xaxis_title='x',
                yaxis_title='y',
            )

        # Create 3D visualization
        elif self.dimension == 3:
            self.logger.info('visualize: 3D ...')

            # Create dataframe for plotting
            datadf = pd.DataFrame(self._layout, columns=['x', 'y', 'z'])

            # Add labels if provided
            if labels is not None:
                datadf['label'] = labels

            # Create 3D scatter plot
            fig = px.scatter_3d(
                datadf,
                x='x',
                y='y',
                z='z',
                **vis_params
            )

            # Update layout
            fig.update_layout(
                width=width,
                height=height,
                scene={
                    "xaxis_title": 'x',
                    "yaxis_title": 'y',
                    "zaxis_title": 'z',
                }
            )

        # Return None for higher dimensions
        else:
            self.logger.warning('visualize ERROR: dimension > 3')
            return None

        # Update marker properties
        fig.update_traces(marker={"size": point_size})

        return fig


#
# Kernel for force-directed layout
#

@functools.partial(jit, static_argnums=(5, 6))
def compute_forces_kernel(positions, chunk_indices, neighbor_indices, sample_indices, alpha, a, b):
    """
    JAX-optimized kernel for computing attractive and repulsive forces.

    Parameters
    ----------
    positions : jax.numpy.ndarray
        Current point positions
    chunk_indices : jax.numpy.ndarray
        Current batch indices
    neighbor_indices : jax.numpy.ndarray
        Indices of neighbors for attractive forces
    sample_indices : jax.numpy.ndarray
        Indices of points for repulsive forces
    alpha : float
        Cooling factor that scales force magnitude
    a : float
        Attraction parameter
    b : float
        Repulsion parameter

    Returns
    -------
    jax.numpy.ndarray
        Net force vectors for each point
    """

    # ===== Attraction Forces =====
    def compute_attraction(chunk_idx, neighbors_idx):
        # Get positions of current point and its neighbors
        point_pos = positions[chunk_idx]
        neighbor_pos = positions[neighbors_idx]

        # Compute position differences and distances
        diff = neighbor_pos - point_pos
        dist = jnp.linalg.norm(diff, axis=1, keepdims=True)

        # Avoid division by zero
        mask = dist > 0
        direction = jnp.where(mask, diff / dist, 0.0)

        # Compute attraction-repulsion coefficients
        grad_coeff = jnp.where(
            mask,
            1.0 * jax_coeff_att(dist, a, b) + 1.0 * jax_coeff_rep(dist, a, b),
            0.0
        )

        # Sum forces from all neighbors
        return jnp.sum(grad_coeff * direction, axis=0)

    # ===== Repulsion Forces =====
    def compute_repulsion(chunk_idx, sample_idx):
        # Get positions of current point and sampled points
        point_pos = positions[chunk_idx]
        sample_pos = positions[sample_idx]

        # Compute position differences and distances
        diff = sample_pos - point_pos
        dist = jnp.linalg.norm(diff, axis=1, keepdims=True)

        # Avoid division by zero
        mask = dist > 0
        direction = jnp.where(mask, diff / dist, 0.0)

        # Compute repulsion coefficients
        grad_coeff = jnp.where(mask, jax_coeff_rep(dist, a, b), 0.0)

        # Sum forces from all sampled points
        return jnp.sum(grad_coeff * direction, axis=0)

    # Vectorize force computation across all points
    attraction_forces = vmap(compute_attraction)(chunk_indices, neighbor_indices)
    repulsion_forces = vmap(compute_repulsion)(chunk_indices, sample_indices)

    # Combine forces with cooling factor
    return alpha * (attraction_forces + repulsion_forces)


#
# Auxiliary functions for force-directed layout
#

@jax.jit
def distribution_kernel(dist, a, b):
    """
    Probability kernel that maps distances to similarity scores.

    This is a rational function approximation of a t-distribution.

    Parameters
    ----------
    dist : jax.numpy.ndarray
        Distance values
    a : float
        Scale parameter that controls the steepness of the distribution
    b : float
        Shape parameter that controls the tail behavior

    Returns
    -------
    float or jax.numpy.ndarray
        Similarity score(s) between 0 and 1
    """
    return 1.0 / (1.0 + a * dist ** (2 * b))


# Helper functions for force calculations
@jax.jit
def jax_coeff_att(dist, a, b):
    """JAX-optimized attraction coefficient function."""
    return 1.0 * distribution_kernel(1/dist, a, b)


@jax.jit
def jax_coeff_rep(dist, a, b):
    """JAX-optimized repulsion coefficient function."""
    return -1.0 * distribution_kernel(dist, a, b)


@functools.partial(jit, static_argnums=(1, 2))
def rand_directions(key, dim=2, num=100):
    """
    Sample unit vectors in random directions.
    
    Parameters
    ----------
    key : jax.random.PRNGKey
        Random number generator key
    dim : int
        Dimensionality of the vectors
    num : int
        Number of random directions to sample
        
    Returns
    -------
    jax.numpy.ndarray
        Array of shape (num, dim) containing unit vectors
    """
    points = random.normal(key, (num, dim))
    norms = jnp.sqrt(jnp.sum(points * points, axis=-1))
    return points / norms[:, None]


@functools.partial(jit, static_argnums=(1,))
def get_slice(arr, k, i):
    """
    Extract a slice of size k centered around index i.
    
    Parameters
    ----------
    arr : jax.numpy.ndarray
        Input array
    k : int
        Size of the slice
    i : int
        Center index position
        
    Returns
    -------
    jax.numpy.ndarray
        Slice of the input array
    """
    return lax.dynamic_slice(arr, (i - k // 2,), (k,))
