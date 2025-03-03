import os
from setuptools import setup, find_packages

# 读取requirements.txt文件
with open('requirements.txt') as f:
    required = f.read().splitlines()

# 读取README.md文件
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="new_cello",
    version="0.1.0",
    description="Enhanced CellO: Cell Ontology-based classification with deep learning support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="CellO Team",
    author_email="cello@example.com",
    url="https://github.com/example/new_cello",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=required,
    entry_points={
        'console_scripts': [
            'new_cello_predict=new_cello.cli.predict:main',
            'new_cello_train=new_cello.cli.train:main',
            'new_cello_visualize=new_cello.cli.visualize:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)  
