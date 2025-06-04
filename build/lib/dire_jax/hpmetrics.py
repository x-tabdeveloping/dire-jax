# hpmetrics.py

"""
Auxiliary functions for high-performance benchmarking metrics
"""

#
# Imports
#

import gc
import numpy as np
import jax.numpy as jnp
import jax
from jax import jit, lax, random, device_get
import ot
from ripser import ripser
from fastdtw import fastdtw
from twed import twed
from persim import wasserstein, bottleneck
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from .hpindex import HPIndex

#
# Auxiliary functions
#


@jit
def welford_update(carry, new_value, finite_threshold=1e12):
    """
    Update running mean and variance using Welford's algorithm,
    ignoring values beyond the given finite_threshold.

    Parameters
    ----------
    carry : tuple(int, float, float)
        (count, mean, M2) — intermediate stats.
    new_value : float
        Incoming value to incorporate.
    finite_threshold : float
        Max magnitude allowed for inclusion.

    Returns
    -------
    (carry, None): Updated carry and dummy output for lax.scan.
    """

    (count, mean, M2) = carry
    is_finite = jnp.isfinite(new_value) & (jnp.abs(new_value) < finite_threshold)
    count = count + is_finite  # Only increment count if new_value is not too large
    delta = new_value - mean
    mean += delta * is_finite / count
    delta2 = new_value - mean
    M2 += delta * delta2 * is_finite
    return (count, mean, M2), None


@jit
def welford_finalize(agg):
    """
    Finalize the computation of mean and variance from the aggregate statistics.

    Parameters
    ----------
    agg: (tuple) A tuple containing the aggregated statistics:
                 - count: (int) The total count of valid (non-NaN) entries.
                 - mean: (float) The computed mean of the dataset.
                 - M2: (float) The computed sum of squares of differences from the mean.

    Returns
    -------
    tuple: A tuple containing the final mean and standard deviation of the dataset.
    """

    count, mean, M2 = agg
    variance = jnp.where(count > 1, M2 / (count - 1), 0.0)
    return mean, jnp.sqrt(variance)


@jit
def welford(data):
    """
    Compute the mean and standard deviation of a dataset using Welford's algorithm.

    Parameters
    ----------
    data: (jax.numpy.ndarray) An array of data points, potentially containing NaNs which are ignored.

    Returns
    -------
    tuple: A tuple containing the mean and standard deviation of the valid entries in the dataset.
    """

    init_agg = (0, 0.0, 0.0)  # initial aggregate: count, mean, M2
    agg, _ = lax.scan(welford_update, init_agg, data)
    return welford_finalize(agg)


#
# Local metrics (based on the kNN graph) 
#


#
# Make the kNN graph of given data with k=n_neighbors
#
def make_knn_graph(data, n_neighbors, batch_size=None):
    """
    Compute the distances to nearest neighbors and their indices in the kNN graph of data.

    Parameters
    ----------
    data : numpy.ndarray
        High-dimensional data points.
    n_neighbors : int
        Number of nearest neighbors to find for each point.
    batch_size : int or None, optional
        Number of samples to process at once. If None, a suitable value
        will be automatically determined based on dataset size.

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        Tuple containing:
        - distances: Array of shape (n_samples, n_neighbors+1) with distances to nearest neighbors
        - indices: Array of shape (n_samples, n_neighbors+1) with indices of nearest neighbors
        
        The first column contains each point's self-reference (distance 0.0 and own index).
        The remaining columns contain the n_neighbors nearest neighbors in ascending order of distance.

    """
    # Get data size
    n_samples = data.shape[0]

    # Determine appropriate batch size for memory efficiency
    if batch_size is None:
        # Process in chunks to reduce peak memory usage
        if jax.devices()[0].platform == 'tpu':
            batch_size = min(8192, n_samples)
        elif jax.devices()[0].platform == 'gpu':
            batch_size = min(16384, n_samples)
        else:
            batch_size = min(8192, n_samples)

    # Convert data to the required format for kNN search
    data_np = np.ascontiguousarray(data.astype(np.float32))

    jax_indices, jax_distances = HPIndex.knn_tiled(
        data_np, data_np, n_neighbors+1, batch_size, batch_size)

    jax_indices = jax_indices.block_until_ready()
    jax_distances = jax_distances.block_until_ready()

    indices = device_get(jax_indices)
    distances = device_get(jax_distances)

    # Clean up resources
    del jax_indices, jax_distances
    gc.collect()

    return distances, indices


#
# Embedding stress
#
def compute_stress(data, layout, n_neighbors, eps=1e-6):
    """
    Compute the stress of an embedding based on the distances in the original high-dimensional
    space and the embedded space, using a ratio of distances.

    Parameters
    ----------
    data: (numpy.ndarray) High-dimensional data points.
    layout: (numpy.ndarray) Embedded data points.
    n_neighbors: (int) Number of nearest neighbors to consider for each point.
    eps: (float) Parameter to prevent zero division if mean distortion is near zero, default 1e-6.

    Returns
    -------
    float: The normalized stress value indicating the quality of the embedding.
    """

    # Computing kNN distances and indices for higher-dimensional data
    distances, indices = make_knn_graph(data, n_neighbors)

    # HPIndex returns L2 distances squared (sic!)
    # Higher-dimensional distances
    distances = jnp.sqrt(distances)

    # Lower-dimensional distances
    distances_emb = jnp.linalg.norm(layout[:, None] - layout[indices], axis=-1)

    # Removing zero distance to self
    distances = distances[:, 1:]
    distances_emb = distances_emb[:, 1:]

    # Computing normalized (= scaling adjusted) stress
    ratios = jnp.absolute(distances / distances_emb - 1.0)
    stress_mean, stress_std = welford(ratios.ravel())

    # Avoiding division by 0 if stress is small
    stress_normalized = 0.0 if stress_mean < eps else stress_std.item() / stress_mean.item()

    return stress_normalized


#
# Neighborhood preservation score
#
def compute_neighbor_score(data, layout, n_neighbors):
    """
    Computes the neighborhood preservation score between high-dimensional data and its corresponding low-dimensional
    layout.
    
    The function evaluates how well the neighborhood relationships are preserved when data is projected from
    a high-dimensional space to a lower-dimensional space using the K-nearest neighbors approach. This involves
    comparing the nearest neighbors in the original space with those in the reduced space.

    Parameters
    ----------
    data: (numpy.ndarray) A NumPy array of shape (n_samples, data_dim) containing
                        the original high-dimensional data.
    layout: (numpy.ndarray) A NumPy array of shape (n_samples, embed_dim) containing
                        the lower-dimensional embedding of the data.
    n_neighbors: (int) The number of nearest neighbors to consider for each data point.

    Returns
    -------
    list: A list containing two floats:
          - neighbor_mean: (float) The mean of the neighborhood preservation scores.
          - neighbor_std: (float) The standard deviation of the neighborhood preservation scores.
    """

    # Computing kNN indices for higher-dimensional data
    _, indices_data = make_knn_graph(data, n_neighbors)
    # Removing self from the set of indices
    indices_data = indices_data[:, 1:]

    # Computing kNN indices for the layout
    _, indices_embed = make_knn_graph(layout, n_neighbors)
    # Removing self from the set of indices
    indices_embed = indices_embed[:, 1:]

    # Sorting indices for efficient search
    indices_data = np.sort(indices_data, axis=1)
    indices_embed = np.sort(indices_embed, axis=1)

    # Compute preservation scores for each point neighborhood
    preservation_scores = np.mean(indices_data == indices_embed, axis=1)

    # Mean and std over all points
    neighbor_mean, neighbor_std = welford(jnp.asarray(preservation_scores.ravel()))

    return [neighbor_mean.item(), neighbor_std.item()]


#
# Computing local metrics based on the kNN graph:
#
# 1. Embedding stress (scaling adjusted);
# 2. Neighborhood preservation score (mean, std).
#
def compute_local_metrics(data, layout, n_neighbors, memory_efficient=None):
    """
    Compute local metrics of the (data, layout) pair.

    Parameters
    ----------
    data: (numpy.ndarray) High-dimensional data points.
    layout: (numpy.ndarray) Low-dimensional data points corresponding to the high-dimensional data.
    n_neighbors: (int) Number of closest neighbors for the kNN graph.
    memory_efficient: (bool or None) If True, use memory-efficient algorithms for large datasets.
                     If None, automatically determine based on dataset size.

    Returns
    -------
    dict: A dictionary containing computed scores of each type (stress, neighborhood preservation).
    """
    # Determine if we should use memory-efficient mode for large datasets
    if memory_efficient is None:
        memory_efficient = data.shape[0] > 32768

    # For very large datasets, subsample before computing metrics
    if memory_efficient and data.shape[0] > 131072:
        # Use a reasonable sample size that maintains statistical validity
        sample_size = 32768
        indices = np.random.choice(data.shape[0], sample_size, replace=False)
        data_sample = data[indices]
        layout_sample = layout[indices]

        metrics = {'stress': compute_stress(data_sample, layout_sample, n_neighbors),
                   'neighbor': compute_neighbor_score(data_sample, layout_sample, n_neighbors),
                   'note': f"Metrics computed on {sample_size} randomly sampled points due to large dataset size"}

        # Add note about subsampling

    else:
        metrics = {
            'stress': compute_stress(data, layout, n_neighbors),
            'neighbor': compute_neighbor_score(data, layout, n_neighbors)
        }

    return metrics


#
# Global metrics based on persistence (co)homology
#


#
# Auxiliary functions
#


#
# Bernoulli trial subsampling
#
def threshold_subsample(*arrays, threshold, rng_key):
    """
    Subsample multiple arrays based on a specified threshold.
    The function generates random numbers and selects the samples where the random number is less than the threshold.

    Parameters
    ----------
    *arrays: (tuple of numpy.ndarray)
        The input data arrays to be subsampled. Each array should have the same number of samples (rows).
    threshold: (float)
        Probability threshold for subsampling; only samples with generated random numbers below this value are kept.
    rng_key: Random key or random generator used for generating random numbers, ensuring reproducibility.

    Returns
    -------
    tuple: A tuple containing the subsampled arrays in the same order as the input arrays.
    """

    # Check that all arrays have the same number of samples
    n_samples = arrays[0].shape[0]
    for array in arrays:
        assert array.shape[0] == n_samples, "All input arrays must have the same number of rows."

    random_numbers = random.uniform(rng_key, shape=(n_samples,))
    selected_indices = random_numbers < threshold

    return tuple(array[selected_indices] for array in arrays)


#
# Producing diagrams in dimensions up to dim (inclusive) for data and layout
#
def diagrams(data, layout, max_dim, subsample_threshold, rng_key):
    """
    Generate persistence diagrams for high-dimensional and low-dimensional data up to a specified dimension,
    after subsampling both datasets based on a threshold. The subsampling is performed to reduce the dataset size
    and potentially highlight more relevant features when computing topological summaries.

    Parameters
    ----------
    data: (numpy.ndarray) High-dimensional data points.
    layout: (numpy.ndarray) Low-dimensional data points corresponding to the high-dimensional data.
    max_dim: (int) Maximum dimension of homology groups to compute.
    subsample_threshold: (float) Threshold used for subsampling the data points.
    rng_key: Random key used for generating random numbers for subsampling, ensuring reproducibility.

    Returns
    -------
    dict: A dictionary containing two keys, 'data' and 'layout', each associated with arrays of persistence diagrams
          for the respective high-dimensional and low-dimensional datasets.
    """

    data_hd, data_ld = threshold_subsample(data, layout,
                                           threshold=subsample_threshold,
                                           rng_key=rng_key)

    diags_hd = ripser(data_hd, maxdim=max_dim)['dgms']
    diags_ld = ripser(data_ld, maxdim=max_dim)['dgms']

    return {'data': diags_hd, 'layout': diags_ld}


#
# Betti curve of a diagram (in a single given dimension)
#
def betti_curve(diagram, n_steps=100):
    """
    Compute the Betti curve from a persistence diagram, which is a function of the number of features
    that persist at different filtration levels. This curve provides a summary of topological features
    across scales.

    Parameters
    ----------
    diagram: (list of tuples) A persistence diagram represented as a list of tuples (birth, death) indicating
                              the range over which each topological feature persists.
    n_steps: (int, optional) The number of steps or points in the filtration range at which to evaluate the Betti number.

    Returns
    -------
    tuple: A tuple of two numpy arrays:
        - The first array represents the evenly spaced filtration values.
        - The second array represents the Betti numbers at each filtration value.
    """

    if len(diagram) == 0:
        return np.linspace(0, n_steps, n_steps), np.zeros(n_steps)
    max_dist = np.max([x[1] for x in diagram if x[1] != np.inf])
    axis_x = np.linspace(0, max_dist, n_steps)
    axis_y = np.zeros(n_steps)
    for i, x in enumerate(axis_x):
        for b, d in diagram:
            if b < x < d:
                axis_y[i] += 1
    return axis_x, axis_y


#
# Metrics (DTW, TWED, Wasserstein, etc) on Betti curves and persistence diagrams
#

#
# Compute normalized Dynamic Time Warp (DTW) distance (using Euclidean metric)
#
def compute_dtw(axis_x_hd, axis_y_hd, axis_x_ld, axis_y_ld, norm_factor=1.0):
    """
    Compute normalized Dynamic Time Warp (DTW) distance (using Euclidean metric)
    between two Betti curves represented as time series with time dimension x and values y.

    Parameters
    ----------
    axis_x_hd: (numpy.ndarray) Time axis of the high-dimensional Betti curve.
    axis_y_hd: (numpy.ndarray) Values of the high-dimensional Betti curve.
    axis_x_ld: (numpy.ndarray) Time axis of the low-dimensional Betti curve.
    axis_y_ld: (numpy.ndarray) Values of the low-dimensional Betti curve.
    norm_factor: (float) Normalization factor, default 1.0.

    Returns
    -------
    float: Normalized DTW distance between two Betti curves.
    """

    seq0 = np.array(list(zip(axis_x_hd, axis_y_hd)))
    seq1 = np.array(list(zip(axis_x_ld, axis_y_ld)))
    dist_dtw, _ = fastdtw(seq0, seq1, dist=2)  # dist_dtw recorded, path unused
    dist_dtw *= norm_factor

    return dist_dtw


#
# Compute normalized Time Warp Edit Distance (TWED) using Euclidean metric
#
def compute_twed(axis_x_hd, axis_y_hd, axis_x_ld, axis_y_ld, norm_factor=1.0):
    """
    Compute normalized Time Warp Edit Distance (TWED) distance using Euclidean metric
    between two Betti curves represented as time series with time dimension x and values y.

    Parameters
    ----------
    axis_x_hd: (numpy.ndarray) Time axis of the high-dimensional Betti curve.
    axis_y_hd: (numpy.ndarray) Values of the high-dimensional Betti curve.
    axis_x_ld: (numpy.ndarray) Time axis of the low-dimensional Betti curve.
    axis_y_ld: (numpy.ndarray) Values of the low-dimensional Betti curve.
    norm_factor: (float) Normalization factor, default 1.0.

    Returns
    -------
    float: Normalized TWED distance between two Betti curves.
    """

    dist_twed = twed(axis_y_hd.reshape(-1, 1),
                     axis_y_ld.reshape(-1, 1),
                     axis_x_hd,
                     axis_x_ld,
                     p=2,
                     )
    dist_twed *= norm_factor

    return dist_twed


#
# Compute normalized Earth Mover Distance (EMD) distance (using Euclidean metric)
#
def compute_emd(axis_x_hd, axis_y_hd, axis_x_ld, axis_y_ld, adjust_mass=False, norm_factor=1.0):
    """
    Compute normalized Earth Mover Distance (EMD) distance (using Euclidean metric)
    between two Betti curves represented as time series with time dimension x and values y.

    Parameters
    ----------
    axis_x_hd: (numpy.ndarray) Time axis of the high-dimensional Betti curve.
    axis_y_hd: (numpy.ndarray) Values of the high-dimensional Betti curve.
    axis_x_ld: (numpy.ndarray) Time axis of the low-dimensional Betti curve.
    axis_y_ld: (numpy.ndarray) Values of the low-dimensional Betti curve.
    adjust_mass: (bool) Use to adjust mass (by default, EMD is computed for unit mass curves);
                default `False`.
    norm_factor: (float) Normalization factor, default 1.0.

    Returns
    -------
    float: Normalized EMD distance between two Betti curves.
    """

    sum_hd = np.sum(axis_y_hd)
    sum_ld = np.sum(axis_y_ld)
    axis_y_hd_ = axis_y_hd / sum_hd
    axis_y_ld_ = axis_y_ld / sum_ld
    dist_emd = ot.emd2_1d(axis_x_hd, axis_x_ld, axis_y_hd_, axis_y_ld_, metric='euclidean')

    if adjust_mass:
        dist_emd *= np.max([sum_hd / sum_ld, sum_ld / sum_hd])
    dist_emd *= norm_factor

    return dist_emd


#
# Compute normalized Wasserstein distance
#
def compute_wasserstein(diag_hd, diag_ld, norm_factor=1.0):
    """
    Compute normalized Wasserstein distance between two persistence diagrams
    (usually one of high-dimensional data and one of low-dimensional data).

    Parameters
    ----------
    diag_hd: (list of tuples) Persistence diagram for the high-dimensional data.
    diag_ld: (list of tuples) Persistence diagram for the low-dimensional data.
    norm_factor: (float) Normalization factor, default 1.0.

    Returns
    -------
    float: Normalized Wasserstein distance between persistence diagrams.
    """

    dist_wass = wasserstein(diag_hd, diag_ld)
    dist_wass *= norm_factor

    return dist_wass


#
# Compute normalized bottleneck distance
#
def compute_bottleneck(diag_hd, diag_ld, norm_factor=1.0):
    """
    Compute normalized bottleneck distance between two persistence diagrams
    (usually one of high-dimensional data and one of low-dimensional data).

    Parameters
    ----------
    diag_hd: (list of tuples) Persistence diagram for the high-dimensional data.
    diag_ld: (list of tuples) Persistence diagram for the low-dimensional data.
    norm_factor: (float) Normalization factor, default 1.0.

    Returns
    -------
    float: Normalized bottleneck distance between persistence diagrams.
    """

    dist_bott = bottleneck(diag_hd, diag_ld)
    dist_bott *= norm_factor

    return dist_bott


#
# Computing global metrics based on persistence homology:
#
# 1. DTW, TWED, EMD for Betti curves;
# 2. Wasserstein, bottleneck for diagrams.
#
# together with diagrams and Betti curves, if necessary.
#
def compute_global_metrics(data, layout, dimension, subsample_threshold, rng_key, n_steps=100, metrics_only=True):
    """
    Compute and compare persistence metrics between high-dimensional and low-dimensional data representations.
    The function calculates the Dynamic Time Warp (DTW), Time Warp Edit Distance (TWED), and Earth Mover Distance
    based on Betti curves derived from persistence diagrams. The function also calculate the Wasserstein distance
    and the bottleneck distance based on persistence diagrams. This evaluation helps quantify the topological
    fidelity of dimensionality reduction.

    Parameters
    ----------
    data: (numpy.ndarray) High-dimensional data points.
    layout: (numpy.ndarray) Low-dimensional data points corresponding to the high-dimensional data.
    dimension: (int) The maximum dimension for which to compute persistence diagrams.
    subsample_threshold: (float) Threshold used for subsampling the data.
    rng_key: Random key used for generating random numbers for subsampling, ensuring reproducibility.
    n_steps: (int, optional) The number of steps or points in the filtration range for computing Betti curves.
    metrics_only: (bool) If True, return metrics only; otherwise diagrams and Betti curves are also returned;
                default `True`.

    Returns
    -------
    If metrics_only is True:
        dict(dict): A dictionary containing one item 'metrics' that is a dictionary of lists of computed distances for
        each of the metrics (DTW, TWED, EMD, Wasserstein, and bottleneck). Each list is populated according to the
        dimensions in which the distances were computed.
    If metrics_only is False:
        dict(dict, dict, dict): A dictionary containing three items:
        - 'metrics': A dictionary of metrics, as described above;
        - 'diags': A dictionary of diagrams for the initial data and for the layout;
        - 'bettis': A dictionary of Betti curves for the initial data and for the layout.
        Each dictionary is a dictionary of lists. Each list is populated according to the dimensions in which
        the distances, diagrams, or curves were computed.
    """
    metrics = {'dtw': [],
               'twed': [],
               'emd': [],
               'wass': [],
               'bott': []}

    betti_curves = {'data': [],
                    'layout': []}

    data_hd, data_ld = threshold_subsample(data, layout,
                                           threshold=subsample_threshold,
                                           rng_key=rng_key)
    n_points = data_hd.shape[0]
    assert n_points == data_ld.shape[0]

    diags = diagrams(data_hd, data_ld, dimension, subsample_threshold, rng_key)

    for diag_hd, diag_ld in zip(diags['data'], diags['layout']):
        axis_x_hd, axis_y_hd = betti_curve(diag_hd, n_steps=n_steps)
        axis_x_ld, axis_y_ld = betti_curve(diag_ld, n_steps=n_steps)

        betti_curves['data'].append((axis_x_hd, axis_y_hd))
        betti_curves['layout'].append((axis_x_ld, axis_y_ld))

        # Computing DTW distance (normalized)
        dist_dtw = compute_dtw(axis_x_hd, axis_y_hd, axis_x_ld, axis_y_ld, norm_factor=1.0/n_points)

        # Computing TWED distance (normalized)
        dist_twed = compute_twed(axis_x_hd, axis_y_hd, axis_x_ld, axis_y_ld, norm_factor=1.0/n_points)

        # Computing EMD distance (normalized)
        dist_emd = compute_emd(axis_x_hd, axis_y_hd, axis_x_ld, axis_y_ld, adjust_mass=True, norm_factor=1.0/n_points)

        # Computing Wasserstein distance (normalized)
        dist_wass = compute_wasserstein(diag_hd, diag_ld, norm_factor=1.0/n_points)

        # Computing bottleneck distance (without normalization)
        dist_bott = compute_bottleneck(diag_hd, diag_ld, norm_factor=1.0)

        # Adding metrics to dictionary 
        metrics['dtw'].append(dist_dtw)
        metrics['twed'].append(dist_twed)
        metrics['emd'].append(dist_emd)
        metrics['wass'].append(dist_wass)
        metrics['bott'].append(dist_bott)

    if metrics_only:
        return {'metrics': metrics}
    return {'metrics': metrics, 'diags': diags, 'bettis': betti_curves}


#
# Metrics for quality and context
#


#
# Compute linear SVM accuracy of given labelled data X with labels y
#
def compute_svm_accuracy(X, y, test_size=0.3, reg_param=1.0, max_iter=100, random_state=42):
    """
    Compute linear SVM classifier accuracy for given labelled data X with labels y.

    Parameters
    ----------
    X: (numpy.ndarray) Data.
    y: (numpy.ndarray) Data labels.
    test_size: (float) Test size (between 0.0 and 1.0) for the train / test split, default 0.3.
    reg_param: (float) Regularization parameter for SVM, default 1.0.
    max_iter: (int) Maximal number of iterations for SVM training, default 100.
    random_state: (int) Random state for reproducibility, default 42.

    Returns
    -------
    float: Accuracy of the linear SVM model on the test set.
    """

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize and train the Linear SVM model using liblinear
    model = LinearSVC(C=reg_param, max_iter=max_iter)  
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


#
# Compute SVM score (context preservation measure) by comparing
# linear SVM classifier accuracies on the high-dimensional data
# and on the low-dimensional embedding
#
def compute_svm_score(data, layout, labels, subsample_threshold, rng_key, **kwargs):
    """
    Compute SVM score (context preservation measure) by comparing linear SVM classifier accuracies
    on the high-dimensional data and on the low-dimensional embedding.

    Parameters
    ----------
    data: (numpy.ndarray)  High-dimensional data.
    layout: (numpy.ndarray) Low-dimensional embedding.
    labels: (numpy.ndarray) Data labels.
    subsample_threshold: (float) Threshold used for subsampling the data.
    rng_key: Random key used for generating random numbers for subsampling, ensuring reproducibility.
    kwargs: Other keyword arguments used by the various scores above.

    Returns
    -------
    float: SVM context preservation score.
    """

    X_hd, X_ld, y = threshold_subsample(data, layout, labels,
                                        threshold=subsample_threshold,
                                        rng_key=rng_key)

    svm_test_size = kwargs.pop('test_size', 0.3)
    svm_reg_param = kwargs.pop('reg_param', 1.0)
    svm_max_iter = kwargs.pop('max_iter', 100)
    svm_random_state = kwargs.pop('random_state', 42)

    svm_acc_hd = compute_svm_accuracy(X_hd, y,
                                      test_size=svm_test_size,
                                      reg_param=svm_reg_param,
                                      max_iter=svm_max_iter,
                                      random_state=svm_random_state)

    svm_acc_ld = compute_svm_accuracy(X_ld, y,
                                      test_size=svm_test_size,
                                      reg_param=svm_reg_param,
                                      max_iter=svm_max_iter,
                                      random_state=svm_random_state)

    svm_score = np.min([svm_acc_hd/svm_acc_ld, svm_acc_ld/svm_acc_hd])
    svm_score = np.log(svm_score)

    out = np.array([svm_acc_hd, svm_acc_ld, svm_score]).astype(np.float32)

    return out


#
# Compute kNN classifier accuracy
#
def compute_knn_accuracy(X, y, n_neighbors=16, test_size=0.3, random_state=42):
    """
    Compute kNN classifier accuracy for given labelled data X with labels y.

    Parameters
    ----------
    X: (numpy.ndarray) Data.
    y: (numpy.ndarray) Data labels.
    test_size: (float) Test size (between 0.0 and 1.0) for the train / test split, default 0.3.
    n_neighbors: (int) Number of neighbors for kNN classification, default 16.
    random_state: (int) Random state for reproducibility, default 42.

    Returns
    -------
    accuracy: (float) Accuracy of the KNN model on the test set.
    """

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Standardize the data (KNN can be sensitive to the scale of the data)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize and train the KNN model
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy


#
# Compute kNN context preservation score
#
def compute_knn_score(data, layout, labels, n_neighbors=16, **kwargs):
    """
    Compute kNN score (context preservation measure) by comparing kNN classifier accuracies
    on the high-dimensional data and on the low-dimensional embedding.

    Parameters
    ----------
    data: (numpy.ndarray)  High-dimensional data.
    layout: (numpy.ndarray) Low-dimensional embedding.
    labels: (numpy.ndarray) Data labels.
    n_neighbors: (int) Number of nearest neighbors for kNN classifier, default 16.
    kwargs: Other keyword arguments used by the various scores above.

    Returns
    -------
    float: kNN context preservation score.
    """

    test_size = kwargs.pop('test_size', 0.3)
    random_state = kwargs.pop('random_state', 42)

    knn_acc_hd = compute_knn_accuracy(data, labels,
                                      n_neighbors=n_neighbors,
                                      test_size=test_size,
                                      random_state=random_state)
    knn_acc_ld = compute_knn_accuracy(layout, labels,
                                      n_neighbors=n_neighbors,
                                      test_size=test_size,
                                      random_state=random_state)

    knn_score = np.log(knn_acc_ld/knn_acc_hd)

    out = np.array([knn_acc_hd, knn_acc_ld, knn_score]).astype(np.float32)

    return out


#
# Compute quality measures for dimensionality reduction
#
def compute_quality_measures(data, layout, n_neighbors=None):
    """
    Compute quality measures for assessing the quality of dimensionality reduction.
    
    This function calculates various metrics that evaluate how well the low-dimensional
    representation preserves important properties of the high-dimensional data.
    
    Parameters
    ----------
    data : numpy.ndarray
        High-dimensional data points.
    layout : numpy.ndarray
        Low-dimensional embedding of the data.
        
    Returns
    -------
    dict
        Dictionary of quality measures including:
        - trustworthiness: Measures if points that are close in the embedding are also close in original space
        - continuity: Measures if points that are close in original space are also close in the embedding
        - shepard_correlation: Correlation between pairwise distances in original and embedded spaces
    """
    # Calculate pairwise distances in original space
    n_samples = min(16384, data.shape[0])  # Limit computation for very large datasets

    # this has to be removed as sampling is already defined ...
    if data.shape[0] > n_samples:
        # Random sampling for large datasets
        indices = np.random.choice(data.shape[0], n_samples, replace=False)
        data_subset = data[indices]
        layout_subset = layout[indices]
    else:
        data_subset = data
        layout_subset = layout
    
    # Use contiguous arrays for efficient distance computation
    data_np = np.ascontiguousarray(data_subset.astype(np.float32))
    layout_np = np.ascontiguousarray(layout_subset.astype(np.float32))

    # Compute all pairwise distances (excluding self-distances)
    if n_neighbors is None:  # Number of neighbors to consider
        n_neighbors = min(13, int(np.log(n_samples)))
    
    # High-dimensional distances and indices
    hd_indices, hd_distances = make_knn_graph(data_np, n_neighbors)
    hd_indices = hd_indices[:, 1:]  # Skip the first column (self)
    hd_distances = hd_distances[:, 1:]  # Skip the first column (self)
    hd_distances = np.sqrt(hd_distances)

    # Low-dimensional distances
    ld_indices, ld_distances = make_knn_graph(layout_np, n_neighbors)
    ld_indices = ld_indices[:, 1:]  # Skip the first column (self)
    ld_distances = ld_distances[:, 1:]  # Skip the first column (self)
    ld_distances = np.sqrt(ld_distances)
    
    # Calculate trustworthiness (are neighbors in embedding also neighbors in original space?)
    def calculate_trustworthiness():
        # Vectorized implementation for better performance
        trust_sum = 0
        
        for i in range(n_samples):
            # Get neighbors in the embedding
            ld_neighbor_indices = ld_indices[i]
            
            # Get neighbors in the original space
            hd_neighbor_indices = hd_indices[i]
            
            # Find points that are neighbors in the embedding but not in the original space
            ld_neighbors = set(ld_neighbor_indices)
            hd_neighbors = set(hd_neighbor_indices)
            violators = list(ld_neighbors - hd_neighbors)
            
            if violators:
                # Get the distances to violators in the original space
                for j in violators:
                    # Calculate rank based on distance
                    orig_dists = hd_distances[i]
                    dist_to_j = orig_dists[j]
                    # Count how many points are closer than j to i
                    orig_rank = np.sum(orig_dists < dist_to_j)
                    
                    # Penalty based on how far j is in original space
                    trust_sum += (orig_rank - n_neighbors)
        
        # Normalize the trustworthiness score
        norm = 2.0 / (n_samples * n_neighbors * (2 * n_samples - 3 * n_neighbors - 1))
        trust_score = 1 - norm * trust_sum
        
        return trust_score
    
    # Calculate continuity (are neighbors in original space also neighbors in embedding?)
    def calculate_continuity():
        # Vectorized implementation for better performance
        cont_sum = 0
        
        for i in range(n_samples):
            # Get neighbors in the original space
            hd_neighbor_indices = hd_indices[i]
            
            # Get neighbors in the embedding
            ld_neighbor_indices = ld_indices[i]
            
            # Find points that are neighbors in the original space but not in the embedding
            hd_neighbors = set(hd_neighbor_indices)
            ld_neighbors = set(ld_neighbor_indices)
            violators = list(hd_neighbors - ld_neighbors)
            
            if violators:
                # Get the distances to violators in the embedding
                for j in violators:
                    # Calculate rank based on distance
                    embed_dists = ld_distances[i]
                    dist_to_j = embed_dists[j]
                    # Count how many points are closer than j to i
                    embed_rank = np.sum(embed_dists < dist_to_j)
                    
                    # Penalty based on how far j is in the embedding
                    cont_sum += (embed_rank - n_neighbors)
        
        # Normalize the continuity score
        norm = 2.0 / (n_samples * n_neighbors * (2 * n_samples - 3 * n_neighbors - 1))
        cont_score = 1 - norm * cont_sum
        
        return cont_score
    
    # Calculate Shepard diagram correlation
    def calculate_shepard_correlation():
        # Sample pairs for correlation
        n_pairs = min(131072, n_samples * (n_samples - 1) // 2)
        
        # Generate random pairs
        i_indices = np.random.randint(0, n_samples, n_pairs)
        j_indices = np.random.randint(0, n_samples, n_pairs)
        
        # Ensure i != j
        mask = i_indices != j_indices
        i_indices = i_indices[mask]
        j_indices = j_indices[mask]
        
        # Calculate distances
        hd_dists = np.linalg.norm(data_subset[i_indices] - data_subset[j_indices], axis=1)
        ld_dists = np.linalg.norm(layout_subset[i_indices] - layout_subset[j_indices], axis=1)
        
        # Calculate correlation
        correlation = np.corrcoef(hd_dists, ld_dists)[0, 1]
        
        return correlation
    
    # Compute and return metrics
    trustworthiness = calculate_trustworthiness()
    continuity = calculate_continuity()
    shepard_correlation = calculate_shepard_correlation()
    
    return {
        'trustworthiness': trustworthiness,
        'continuity': continuity,
        'shepard_correlation': shepard_correlation
    }


#
# Compute context measures (context preservation)
#
def compute_context_measures(data, layout, labels, subsample_threshold, n_neighbors, rng_key, **kwargs):
    """
    Compute measures of how well the embedding preserves the context of the data.
    
    Parameters
    ----------
    data : numpy.ndarray
        High-dimensional data points.
    layout : numpy.ndarray
        Low-dimensional embedding of the data.
    labels : numpy.ndarray
        Data labels needed for context preservation analysis.
    subsample_threshold : float
        Threshold used for subsampling the data.
    n_neighbors : int
        Number of neighbors for the kNN graph.
    rng_key : jax.random.PRNGKey
        Random key for reproducible subsampling.
    **kwargs
        Additional keyword arguments for the scoring functions.

    Returns
    -------
    dict
        Dictionary of context preservation measures, including
        SVM and kNN classification performance comparisons.
    """
    measures = {
        'svm': compute_svm_score(data, layout, labels, subsample_threshold, rng_key, **kwargs),
        'knn': compute_knn_score(data, layout, labels, n_neighbors, **kwargs)
    }

    return measures
