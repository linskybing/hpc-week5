#! /bin/bash

module purge
module load miniconda3/conda24.5.0_py3.9 nvhpc-24.11_hpcx-2.20_cuda-12.6

conda create -n mnist_cuda python=3.9 -y
conda activate mnist_cuda

pip install --upgrade pip
pip3 install torch torchvision nvtx