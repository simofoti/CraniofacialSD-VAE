#!/bin/bash

virtualenv -p python3 ./id-generator-env
source ./id-generator-env/bin/activate

export PYTHON_V=38
export CUDA=cu113
export TORCH=1.12.1
export TORCHVISION=0.13.1
export TORCH_GEOM_VERSION=1.12.0

pip install cmake
pip install trimesh pyrender tqdm matplotlib rtree openmesh tb-nightly av seaborn xlrd openpyxl

pip3 install torch==${TORCH} torchvision==${TORCHVISION} --default-timeout=1000 --extra-index-url https://download.pytorch.org/whl/${CUDA}

pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-${TORCH_GEOM_VERSION}+${CUDA}.html

pip install pytorch3d==0.6.0  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py${PYTHON_V}_${CUDA}_pyt1120/download.html

pip install jupyterlab ipywidgets