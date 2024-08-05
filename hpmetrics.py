# hpmetrics.py

#
# Imports
#
import numpy as np
import jax.numpy as jnp
from jax import jit, lax, random
import faiss
import gc
import ot
from ripser import ripser
from fastdtw import fastdtw
#from pytwed import twed
from persim import wasserstein, bottleneck
import plotly.express as px
import pandas as pd
from utils import make_knn_adjacency, timing
from scipy.spatial import procrustes
from scipy.stats import spearmanr

#import numpy as np
from scipy.spatial import procrustes
from scipy.stats import spearmanr


# Function to compute Procrustes distance
def procrustes_distance(X, Y):
    ddim = X.shape[1] - Y.shape[1]
    if ddim > 0: 
        Z= np.pad(Y, ((0, 0), (0, ddim)), mode='constant', constant_values = 0)
        _, _, disparity = procrustes(X, Z)
    elif ddim < 0:
        W= np.pad(X, ((0, 0), (0, -ddim)), mode='constant', constant_values =0)
        _, _, disparity = procrustes(W, Y)
    else:
        _, _, disparity = procrustes(X, Y)
    return disparity

# Function to compute Spearman's rank correlation
def spearman_rank_corr(X, Y):
    distances_X = np.linalg.norm(X[:, np.newaxis] - X, axis=2)
    distances_Y = np.linalg.norm(Y[:, np.newaxis] - Y, axis=2)
    rank_corr, _ = spearmanr(distances_X.flatten(), distances_Y.flatten())
    return rank_corr

# Example function for repeated random sampling
def compute_quality_measures(X, Y, num_samples=None, sample_size=None, key=random.PRNGKey(0)):
    if num_samples is None:
        num_samples = int(np.log(X.shape[0]))
    if sample_size is None:
        sample_size = int(np.sqrt(X.shape[0]))
    procrustes_distances = []
    spearman_correlations = []

    keys = random.split(key, num_samples)
    for k in keys:
        sample_indices = random.choice(k, X.shape[0], (sample_size,), replace=False)
        #sample_indices = random.sample(range(X.shape[0]), sample_size)
        X_sample = X[sample_indices]
        Y_sample = Y[sample_indices]

        procrustes_distances.append(procrustes_distance(X_sample, Y_sample))
        spearman_correlations.append(spearman_rank_corr(X_sample, Y_sample))
    
    procrustes_mean = np.mean(procrustes_distances)
    procrustes_var = np.var(procrustes_distances)
    spearman_mean = np.mean(spearman_correlations)
    spearman_var = np.var(spearman_correlations)

    return {
        'procrustes_mean': procrustes_mean,
        'procrustes_variance': procrustes_var,
        'spearman_mean': spearman_mean,
        'spearman_variance': spearman_var
    }

# Example usage
X = np.random.rand(1000, 50)  # Original high-dimensional data
Y = np.random.rand(1000, 2)   # Reduced low-dimensional data
results = compute_quality_measures(X, Y, num_samples=100, sample_size=200)
print(results)


#
# Auxiliary functions
#

@jit
def welford_update(carry, new_value):
    """
    Update the running mean and variance calculations with a new value using Welford's algorithm.

    Parameters:
    carry (tuple): A tuple containing three elements:
                   - count (int): The count of valid (non-NaN) entries processed so far.
                   - mean (float): The running mean of the dataset.
                   - M2 (float): The running sum of squares of differences from the current mean.
    new_value (float): The new value to include in the statistics.

    Returns:
    tuple: Updated carry tuple (count, mean, M2) and None as a placeholder for loop compatibility.
    """
    (count, mean, M2) = carry
    is_finite = jnp.isfinite(new_value)
    count = count + is_finite  # Only increment count if new_value is finite
    delta = new_value - mean
    mean += delta * is_finite / count
    delta2 = new_value - mean
    M2 += delta * delta2 * is_finite
    return (count, mean, M2), None

@jit
def welford_finalize(agg):
    """
    Finalize the computation of mean and variance from the aggregate statistics.

    Parameters:
    agg (tuple): A tuple containing the aggregated statistics:
                 - count (int): The total count of valid (non-NaN) entries.
                 - mean (float): The computed mean of the dataset.
                 - M2 (float): The computed sum of squares of differences from the mean.

    Returns:
    tuple: A tuple containing the final mean and standard deviation of the dataset.
    """
    count, mean, M2 = agg
    variance = jnp.where(count > 1, M2 / (count - 1), 0.0)
    return mean, jnp.sqrt(variance)

@jit
def welford(data):
    """
    Compute the mean and standard deviation of a dataset using Welford's algorithm.

    Parameters:
    data (jax.numpy.ndarray): An array of data points, potentially containing NaNs which are ignored.

    Returns:
    tuple: A tuple containing the mean and standard deviation of the valid entries in the dataset.
    """
    init_agg = (0, 0.0, 0.0)  # initial aggregate: count, mean, M2
    agg, _ = lax.scan(welford_update, init_agg, data)
    return welford_finalize(agg)

#
# Local metrics (based on the kNN graph) 
#

#
# Embedding stress
#
def compute_stress(data, layout, n_neighbors, eps=1e-6):
    """
    Computes the stress of an embedding based on the distances in the original high-dimensional
    space and the embedded space, using a ratio of distances.

    Parameters
    ----------
    data : numpy.ndarray
        High-dimensional data points.
    layout : numpy.ndarray
        Embedded data points.
    n_neighbors : int
        Number of nearest neighbors to consider for each point.

    Returns
    -------
    float
        The normalized stress value indicating the quality of the embedding.
    """
    data_dim = data.shape[1]
    try:
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, data_dim)
    except Exception as e:
        index = faiss.IndexFlatL2(data_dim)

    data_np = np.ascontiguousarray(data.astype(np.float32))
    index.add(data_np)
    distances, indices = index.search(data_np, n_neighbors + 1)

    # FAISS uses L2 distances squared (sic!)
    # higher-dimensional distances
    distances = jnp.sqrt(distances)

    del index
    gc.collect()

    # lower-dimensional distances
    distances_emb = jnp.linalg.norm(layout[:, None] - layout[indices], axis=-1)

    distances = distances[:, 1:]
    distances_emb = distances_emb[:, 1:]

    ratios = jnp.absolute(distances / distances_emb - 1.0)
    stress_mean, stress_std = welford(ratios.ravel())

    stress_normalized = 0.0 if stress_mean < eps else stress_std.item() / stress_mean.item()

    return stress_normalized

#
# Neighborhood preservation score
#
def compute_neighbor_score(data, layout, n_neighbors):
    """
    Computes the neighborhood preservation score between high-dimensional data and its corresponding low-dimensional layout.
    
    The function evaluates how well the neighborhood relationships are preserved when data is projected from a high-dimensional space to a lower-dimensional space using the K-nearest neighbors approach. This involves comparing the nearest neighbors in the original space with those in the reduced space.

    Parameters:
    data (np.ndarray): A NumPy array of shape (n_samples, data_dim) containing the original high-dimensional data.
    layout (np.ndarray): A NumPy array of shape (n_samples, embed_dim) containing the lower-dimensional embedding of the data.
    n_neighbors (int): The number of nearest neighbors to consider for each data point.

    Returns:
    tuple: A tuple containing two floats:
        - neighbor_mean (float): The mean of the neighborhood preservation scores.
        - neighbor_std (float): The standard deviation of the neighborhood preservation scores.
    """
    data_dim = data.shape[1]
    embed_dim = layout.shape[1]

    # Create FAISS index for high-dimensional data
    try:
        res_data = faiss.StandardGpuResources()
        index_data = faiss.GpuIndexFlatL2(res_data, data_dim)
    except Exception as e:
        index_data = faiss.IndexFlatL2(data_dim)

    data_np = np.ascontiguousarray(data.astype(np.float32))
    index_data.add(data_np)
    _, indices_data = index_data.search(data_np, n_neighbors + 1)
    indices_data = indices_data[:, 1:]

    # Clear resources
    del index_data
    gc.collect()

    # Create FAISS index for layout
    try:
        res_embed = faiss.StandardGpuResources()
        index_embed = faiss.GpuIndexFlatL2(res_embed, embed_dim)
    except Exception as e:
        index_embed = faiss.IndexFlatL2(embed_dim)
        
    layout_np = np.ascontiguousarray(layout.astype(np.float32))
    index_embed.add(layout_np)
    _, indices_embed = index_embed.search(layout_np, n_neighbors + 1)
    indices_embed = indices_embed[:, 1:]

    # Clear resources
    del index_embed
    gc.collect()

    indices_data = jnp.sort(indices_data, axis=1)
    indices_embed = jnp.sort(indices_embed, axis=1)

    # Compute preservation scores
    preservation_scores = jnp.mean(indices_data == indices_embed, axis=1)
    neighbor_mean, neighbor_std = welford(preservation_scores.ravel())

    return neighbor_mean.item(), neighbor_std.item()

def compute_local_metrics(data, layout, n_neighbors):
    stress_normalized = compute_stress(data, layout, n_neighbors)
    neighbor_score, _ = compute_neighbor_score(data, layout, n_neighbors)
    ldict = {"stress": stress_normalized, 'neighbor' : neighbor_score}
    return ldict


#
# Global metrics based on persistence (co)homology
#

#
# Bernoulli trial subsampling
#
def threshold_subsample(data_hd, data_ld, threshold=1.0, rng_key=42):
    """
    Subsample high-dimensional and corresponding low-dimensional data based on a specified threshold.
    The function generates random numbers and selects the samples where the random number is less than the threshold.
    
    Parameters:
    data_hd (numpy.ndarray): High-dimensional data points.
    data_ld (numpy.ndarray): Low-dimensional data points corresponding to high-dimensional data.
    threshold (float): Probability threshold for subsampling; only samples with generated random numbers below this value are kept.
    rng_key: Random key used for generating random numbers, ensuring reproducibility.

    Returns:
    tuple: A tuple containing two numpy arrays; the subsampled high-dimensional data and the subsampled low-dimensional data.
    """
    n_samples = data_hd.shape[0]
    assert n_samples == data_ld.shape[0]
    random_numbers = random.uniform(rng_key, shape=(n_samples,))
    selected_indices = random_numbers < threshold
    return data_hd[selected_indices], data_ld[selected_indices]

#
# Producing diagrams in dimensions up to dim (inclusive) for data and layout
#
def diagrams(data, layout, max_dim, subsample_threshold, do_knn = 100, rng_key=42):
    """
    Generates persistence diagrams for high-dimensional and low-dimensional data up to a specified dimension,
    after subsampling both datasets based on a threshold. The subsampling is performed to reduce the dataset size
    and potentially highlight more relevant features when computing topological summaries.

    Parameters:
    data (numpy.ndarray): High-dimensional data points.
    layout (numpy.ndarray): Low-dimensional data points corresponding to the high-dimensional data.
    max_dim (int): Maximum dimension of homology groups to compute.
    subsample_threshold (float): Threshold used for subsampling the data points.
    rng_key: Random key used for generating random numbers for subsampling, ensuring reproducibility.

    Returns:
    dict: A dictionary containing two keys, 'data' and 'layout', each associated with arrays of persistence diagrams
          for the respective high-dimensional and low-dimensional datasets.
    """
    data_hd, data_ld = threshold_subsample(data, layout, subsample_threshold, rng_key)
    if do_knn is not None:
        adj_hd = make_knn_adjacency(data_hd, n_neighbors=do_knn)
        adj_ld = make_knn_adjacency(data_ld, n_neighbors=do_knn)
        diags_hd = ripser(adj_hd, distance_matrix = True)['dgms']
        diags_ld = ripser(adj_ld, distance_matrix = True)['dgms']
    else:
        diags_hd = ripser(data_hd, maxdim=max_dim)['dgms']
        diags_ld = ripser(data_ld, maxdim=max_dim,)['dgms']
    return {'data': diags_hd, 'layout': diags_ld}

#
# Betti curve of a diagram (in one given dimension)
#
def betti_curve(diagram, n_steps=100):
    """
    Computes the Betti curve from a persistence diagram, which is a function of the number of features
    that persist at different filtration values. This curve provides a summary of topological features
    across scales.

    Parameters:
    diagram (list of tuples): A persistence diagram represented as a list of tuples (birth, death) indicating
                              the range over which each topological feature persists.
    n_steps (int, optional): The number of steps or points in the filtration range at which to evaluate the Betti number.

    Returns:
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
# Do persistence analysis: compute the Betti curves and find the following distances between them:
# 1. Dynamic Time Warp (DTW);
# 2. Time Warp Edit Distance (TWED);
# 3. Earth Mover Distance (EMD).
# Each distance is normalised by the number of points (wrt subsampling) and the Earth Mover 
# Distance is computed by normalising by the number of points, and then multiplying by the mass 
# ratio (EMD requires same mass of both distributions) to account for the difference   
#
def do_persistence_analysis(data, layout, dimension, subsample_threshold, rng_key=random.PRNGKey(42), n_steps=100):
    """
    Perform a comprehensive persistence analysis by subsampling data, computing persistence diagrams, and
    calculating distances between Betti curves of high-dimensional and low-dimensional data. This analysis
    includes computing distances such as Dynamic Time Warp (DTW), Time Warp Edit Distance (TWED), and
    Earth Mover Distance.

    Parameters:
    data (numpy.ndarray): High-dimensional data points.
    layout (numpy.ndarray): Low-dimensional data points corresponding to the high-dimensional data.
    dimension (int): The dimension up to which persistence diagrams are computed.
    subsample_threshold (float): The threshold used for subsampling the data points.
    rng_key: Random key used for generating random numbers for subsampling, ensuring reproducibility.
    n_steps (int, optional): The number of steps or points in the filtration range for computing Betti curves.

    Returns:
    None: This function primarily prints results and may plot curves, but does not return data directly.
    """
    # Subsample with a threshold
    print("Subsampling data ...")
    data_hd, data_ld = threshold_subsample(data, layout, subsample_threshold, rng_key)
    n_points = data_hd.shape[0]
    print("done")

    # Creating persistence diagrams up to dimension
    print(f"Computing persistence up to dimension {dimension}...")
    diags = diagrams(data_hd, data_ld, dimension, subsample_threshold, rng_key)
    print("done")

    dim = 0
    for diag_hd, diag_ld in zip(diags['data'], diags['layout']):
        # Computing Betti curves
        print(f"Computing Betti curve for dimension {dim} ...")
        axis_x_hd, axis_y_hd = betti_curve(diag_hd, n_steps=n_steps)
        axis_x_ld, axis_y_ld = betti_curve(diag_ld, n_steps=n_steps)
        # Normalising Betti curves for plotting
        axis_x_hd_, axis_y_hd_ = axis_x_hd / np.max(axis_x_hd), axis_y_hd
        axis_x_ld_, axis_y_ld_ = axis_x_ld / np.max(axis_x_ld), axis_y_ld
        print("done")
        # Plotting *normalised* Betti curves
        df_hd = pd.DataFrame({'x': axis_x_hd_,
                              'y': axis_y_hd_,
                              'Type': f'Betti {dim} High-dimensional'})
        df_ld = pd.DataFrame({'x': axis_x_ld_,
                              'y': axis_y_ld_,
                              'Type': f'Betti {dim} Low-dimensional'})
        df = pd.concat([df_hd, df_ld])
        fig = px.line(df, x='x', y='y', color='Type', labels={'x': 'Filtration Level', 'y': 'Rank'}, title=f'Betti {dim} High-dimensional vs Low-dimensional')
        fig.show()
        # Computing DTW distance
        seq0 = np.array(list(zip(axis_x_hd, axis_y_hd)))
        seq1 = np.array(list(zip(axis_x_ld, axis_y_ld)))
        dist_dtw, path = fastdtw(seq0, seq1, dist=2)
        dist_dtw /= n_points
        print(f"Distance (DTW) for dimension {dim}: {dist_dtw}")
        # Computing TWED distance
        dist_twed = twed(axis_y_hd, axis_y_ld, axis_x_hd, axis_x_ld, p=2, fast=True)
        dist_twed /= n_points
        print(f"Distance (TWED) for dimension {dim}: {dist_twed}")
        # Computing EMD distance
        sum_hd = np.sum(axis_y_hd)
        sum_ld = np.sum(axis_y_ld)
        axis_y_hd_ = axis_y_hd / sum_hd
        axis_y_ld_ = axis_y_ld / sum_ld
        dist_emd = ot.emd2_1d(axis_x_hd, axis_x_ld, axis_y_hd_, axis_y_ld_, metric='sqeuclidean')
        dist_emd = dist_emd / n_points * np.max([sum_hd / sum_ld, sum_ld / sum_hd])
        print(f"Distance (EMD) for dimension {dim}: {dist_emd}")
        dim += 1
    return None

#
# Computing global metrics:
# DTW, TWED, EMD for Betti curves;
# 
#
def compute_global_metrics(data, layout, dimension, subsample_threshold=1.0, rng_key=random.PRNGKey(42), n_steps=100, do_knn = 100):
    """
    Computes and compares persistence metrics between high-dimensional and low-dimensional data representations.
    The function calculates the Dynamic Time Warp (DTW), Time Warp Edit Distance (TWED), and Earth Mover Distance
    based on Betti curves derived from persistence diagrams. This evaluation helps quantify the topological
    fidelity of dimensionality reduction.

    Parameters:
    data (numpy.ndarray): High-dimensional data points.
    layout (numpy.ndarray): Low-dimensional data points corresponding to the high-dimensional data.
    dimension (int): The maximum dimension for which to compute persistence diagrams.
    subsample_threshold (float): Threshold used for subsampling the data.
    rng_key: Random key used for generating random numbers for subsampling, ensuring reproducibility.
    n_steps (int, optional): The number of steps or points in the filtration range for computing Betti curves.

    Returns:
    dict: A dictionary containing lists of computed distances for each of the three metrics (DTW, TWED, and EMD).
          Each list corresponds to a dimension in which the distances were computed.
    """
    metrics = {'dtw': [], 'emd': [], 'wass': [], 'bot': []}
    data_hd, data_ld = threshold_subsample(data, layout, subsample_threshold, rng_key)
    n_points = data_hd.shape[0]
    assert n_points == data_ld.shape[0]

    diags = diagrams(data_hd, data_ld, dimension, subsample_threshold, do_knn, rng_key)

    for diag_hd, diag_ld in zip(diags['data'], diags['layout']):
        axis_x_hd, axis_y_hd = betti_curve(diag_hd, n_steps=n_steps)
        axis_x_ld, axis_y_ld = betti_curve(diag_ld, n_steps=n_steps)
        # Computing DTW distance (normalized)
        seq0 = np.array(list(zip(axis_x_hd, axis_y_hd)))
        seq1 = np.array(list(zip(axis_x_ld, axis_y_ld)))
        dist_dtw, path = fastdtw(seq0, seq1, dist=2)
        dist_dtw /= n_points
        # Computing TWED distance (normalized)
        #dist_twed = twed(axis_y_hd, axis_y_ld, axis_x_hd, axis_x_ld, p=2, fast=True)
        #dist_twed /= n_points
        # Computing EMD distance (normalized)
        sum_hd = np.sum(axis_y_hd)
        sum_ld = np.sum(axis_y_ld)
        axis_y_hd_ = axis_y_hd / sum_hd
        axis_y_ld_ = axis_y_ld / sum_ld
        dist_emd = ot.emd2_1d(axis_x_hd, axis_x_ld, axis_y_hd_, axis_y_ld_, metric='sqeuclidean')
        dist_emd = dist_emd / n_points * np.max([sum_hd / sum_ld, sum_ld / sum_hd])
        # Computing Wasserstein distance (normalized)
        dist_wass = wasserstein(diag_hd, diag_ld)
        dist_wass /= n_points
        # Computing bottleneck distance (without normalization)
        dist_bot = bottleneck(diag_hd, diag_ld)
        # Adding metrics to dictionary 
        metrics['dtw'].append(dist_dtw)
        #metrics['twed'].append(dist_twed)
        metrics['emd'].append(dist_emd)
        metrics['wass'].append(dist_wass)
        metrics['bot'].append(dist_bot)

    return metrics

def do_metrics(reducer, data, n_neighbors = 100, **kwargs):
    mdict = {}
    with timing('timing', mdict):
        layout = reducer.fit_transform(data)
    ldict = compute_local_metrics(data, layout, n_neighbors)
    mdict.update(ldict)
    gdict = compute_global_metrics(data, layout, 1.0)
    mdict.update(gdict)
    ndict = compute_quality_measures(data, layout)
    mdict.update(ndict)
    return mdict
    
