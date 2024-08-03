# dire_utils.py

#
# Imports
#
import time
import pandas as pd
import plotly.express as px
from hpmetrics import *
from jax import random

#
# Auxiliary functions for plotting etc that are not included in other packages and classes
#

#
# Display a given layout with labels and with given point size
#
def display_layout(layout, labels=None, point_size=1):
  #
  dimension = layout.shape[1]
  #
  if dimension==2:

    datadf = pd.DataFrame(layout, columns=['x','y'])

    if labels is not None:
      datadf['label'] = labels
      fig = px.scatter(datadf, x='x', y='y', color='label', title=f'Layout in dimension {dimension}')
    else:
      fig = px.scatter(datadf, x='x', y='y', title=f'Layout in dimension {dimension}')

    fig.update_traces(marker=dict(size=point_size))
    return fig
  #
  if dimension==3:

    datadf = pd.DataFrame(layout, columns=['x','y','z'])

    if labels is not None:
      datadf['label'] = labels
      fig = px.scatter_3d(datadf, x='x', y='y', z='z', color='label', title=f'Layout in dimension {dimension}')
    else:
      fig = px.scatter_3d(datadf, x='x', y='y', z='z', title=f'Layout in dimension {dimension}')

    fig.update_traces(marker=dict(size=point_size))
    return fig
  #
  return None

#
# Running a benchmark on given reducer with (features, labels)
#
def run_benchmark(reducer, features, labels, subsample_threshold=0.1, rng_key=random.PRNGKey(42), point_size=4):
    """
    Runs a benchmarking process for dimensionality reduction using the provided reducer.

    Parameters:
    reducer (object): The dimensionality reduction model with a `fit_transform` method.
                      It should also have an `n_neighbors` attribute for computing neighborhood scores.
    features (ndarray): The high-dimensional feature data to be reduced.
    labels (ndarray): The labels corresponding to the feature data.
    subsample_threshold (float, optional): The threshold for subsampling during persistence analysis. Defaults to 0.1.
    rng_key (object, optional): The random number generator key. Defaults to random.PRNGKey(42).
    point_size (int, optional): The size of the points in the display layout. Defaults to 4.

    Returns:
    None: This function does not return anything. It performs the embedding, displays the layout, 
          conducts persistence analysis, prints the embedding stress and neighborhood preservation score,
          and times the embedding process.
    
    """
    start_time = time.time()
    embedding = reducer.fit_transform(features)
    end_time = time.time()
    
    embedding_time = end_time - start_time
    print(f'Embedding time: {embedding_time:.4f} seconds')

    display_layout(embedding, 
                   labels=labels.astype('str') if labels else None, 
                   point_size=point_size).show()
    
    do_persistence_analysis(features,
                            embedding,
                            dimension=1,
                            subsample_threshold=subsample_threshold,
                            rng_key=rng_key)
    
    embedding_stress = compute_stress(features, embedding, reducer.n_neighbors)
    
    embedding_neighbor_score = compute_neighbor_score(features, embedding, reducer.n_neighbors)
    
    print(f'Embedding stress (scaling adjusted): {embedding_stress}')
    print(f'Neighborhood preservation (mean, std): {embedding_neighbor_score}')
    
    return None
