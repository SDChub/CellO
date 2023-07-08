import os
import sys
from setuptools import setup, find_packages

install_requires = [
    "Cython>=0.29.17",
    "quadprog>=0.1.6",
    "numpy>=1.17.1",
    "scikit-learn>=0.22.2.post1",
    "scipy>=1.3.1",
    "pandas>=0.23.4",
    "dill>=0.3.1.1",
    "h5py>=2.10.0",
    "anndata>=0.7.1",
    "matplotlib",
    "pygraphviz"
]

if sys.version_info[:2] < (3, 6):
    raise RuntimeError("Python version >=3.6 required.")

with open("README.rst", "r", encoding="utf-8") as fh:
    readme = fh.read()

setup(
    name="cello_classify",
    version="2.1.1",
    description="CellO",
    author="Matthew N. Bernstein",
    author_email="mbernstein@morgridge.org",
    packages=[
        "cello",
        "cello.onto_lib_py3",
        "cello.models",
        "cello.graph_lib"
    ],
    license="MIT License",
    install_requires=install_requires,
    long_description=readme,
    include_package_data=True,
    zip_safe=True,
    url="https://github.com/deweylab/CellO",
    entry_points={
        'console_scripts': [
            'cello_predict = cello.cello_predict:main',
            'cello_train_model = cello.cello_train_model:main',
            'cello_quantify_sample = cello.cello_quantify_sample:main'
        ]
    },
    keywords=[
        "scRNA-seq",
        "cell-type",
        "cell-ontology",
        "gene-expression",
        "computational-biology",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ]
)


