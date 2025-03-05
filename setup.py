from setuptools import setup, find_packages

# read the contents of the README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='dire-jax',
    version='0.0.0',
    author='Alexander Kolpakov, Igor Rivin',
    author_email='akolpakov@uaustin.org, rivin@temple.edu',
    description='A JAX-based Dimension Reducer',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        ],
    python_requires='>=3.7',
    install_requires=['jax',
                      'numpy',
                      'scipy',
                      'tqdm',
                      'pandas',
                      'plotly',
                      'kaleido',
                      'loguru',
                      'ripser',
                      'fastdtw',
                      'pytwed',
                      'pot',
                      'scikit-learn'],
    extras_require={
        'cpu': ['faiss-cpu'],
        'gpu': []  # GPU users must install faiss-gpu via conda
    },
)
