#! /bin/bash
#SBATCH -A ACD110018
#SBATCH -p gtest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -J allreduce


module purge
module load miniconda3/conda24.5.0_py3.9 nvhpc-24.11_hpcx-2.20_cuda-12.6
export WORKDIR="/work/u8644434/hpc-week5/profile" # your dir
cd $WORKDIR

conda activate mnist_cuda

nsys profile -t cuda,osrt,nvtx -o baseline -w true python3 main.py

make clean
make
ncu --clock-control none --set=full -o report transpose