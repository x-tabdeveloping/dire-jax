#!/bin/bash

# Script to build the documentation for DiRe-JAX

# Ensure pip packages are installed
echo "Installing required packages..."
pip install sphinx sphinx_rtd_theme

# Build the documentation
echo "Building documentation..."
cd docs
make clean
make html

echo "Documentation built successfully. Open docs/build/html/index.html to view."