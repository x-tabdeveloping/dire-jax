from pathlib import Path
from setuptools import setup, find_packages

# Read the long description from README.md
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Comment on GPU extra vs CPU extra
print("For CPU, use [cpu] extra, and faiss-cpu will be installed automatically via pip.")
print("For GPU, use [gpu] extra, and manually install faiss-gpu via conda.")

# Comment on utils extra
print("For benchmarking, metrics and utilities, use [utils] extra.")

# Core dependencies (dire.py)
core_deps = [
    "jax",
    "numpy",
    "scipy",
    "tqdm",
    "pandas",
    "plotly",
    "loguru",
    "scikit-learn"
]

# Dependencies for utils and metrics (dire_utils.py and hpmetrics.py)
utils_deps = [
    "ripser",
    "persim",
    "fastdtw",
    "pytwed",
    "pot"
]

setup(
    name="dire-jax",
    version="0.0.1",
    author="Alexander Kolpakov, Igor Rivin",
    author_email="akolpakov@uaustin.org, rivin@temple.edu",
    description="A JAX-based Dimension Reducer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sashakolpakov/dire-jax",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=core_deps,
    extras_require={
        "cpu": ["faiss-cpu"],
        "gpu": [],  # faiss-gpu installed manually via conda
        "utils": utils_deps,
    },
)
