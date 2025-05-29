# DiRe-JAX Documentation

This directory contains the Sphinx documentation for the DiRe-JAX package.

## Important Information about JAX GPU/TPU Support

For GPU or TPU acceleration, JAX needs to be specifically installed with hardware support. The default JAX installation through pip doesn't include GPU/TPU support.

To enable GPU/TPU acceleration:

* For **GPU** support, follow the [JAX GPU installation instructions](https://github.com/google/jax#installation)
* For **TPU** support, follow the [JAX TPU installation instructions](https://github.com/google/jax#tpu-tpu-vm)

Installing JAX with hardware acceleration can significantly improve the performance of DiRe-JAX, especially for larger datasets.

## Building the Documentation

### Prerequisites

To build the documentation, you need to have Sphinx and the required extensions installed:

```bash
pip install sphinx sphinx_rtd_theme
```

### Building

From this directory, run:

```bash
# On Unix/Linux/macOS
make html

# On Windows
make.bat html
```

The built documentation will be available in the `build/html` directory.

## Documentation Structure

- `source/` - Source files for the documentation
  - `api/` - API reference documentation
  - `conf.py` - Sphinx configuration
  - `*.rst` - Documentation pages

## Updating the Documentation

When updating the package, please also update the documentation to reflect any changes.

### Note on JAX Hardware Acceleration

The documentation includes important information about JAX GPU/TPU support. Please ensure this information is preserved when rebuilding the documentation.

In `installation.rst`, there is a section about JAX GPU/TPU support that explains how to install JAX with hardware acceleration. This information is crucial for users who want to leverage GPU or TPU resources for better performance.