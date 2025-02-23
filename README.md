# DiRe - JAX

### A new DImensionality REduction package written in JAX 

We offer a new dimension reduction tool called DiRe - JAX that is benchmarked against the existing approaches: UMAP (original and Rapids.AI versions), and tSNE (Rapids.AI version)

### Quick start

First, do the usual

    pip install dire_jax

Then, another usual

    import dire_jax

and afterwards, for example, this 

```python
from sklearn.datasets import make_blobs

n_samples  = 100_000
n_features = 1_000
n_centers  = 12
features_blobs, labels_blobs = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_centers, random_state=42)

reducer_blobs = DiRe(dimension=2,
                     n_neighbors=16,
                     init_embedding_type='pca',
                     max_iter_layout=32,
                     min_dist=1e-4,
                     spread=1.0,
                     cutoff=4.0,
                     n_sample_dirs=8,
                     sample_size=16,
                     neg_ratio=32,
                     verbose=False,)

viz_benchmark(reducer_blobs,
              features_blobs,
              labels=labels_blobs,
              dimension=1,
              subsample_threshold=0.1,
              rng_key=random.PRNGKey(42),
              point_size=4)
```

### Working paper

Our working paper is available in the repository. Also, check out the Jupyter notebook with benchmarking results. 

### Contributing

Create a fork, contribute, and make a merge request. Thanks! 