#!/bin/bash
#SBATCH --job-name=mein-gpu-job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1          
#SBATCH --time=01:00:00            
#SBATCH --partition=earth-4        
#SBATCH --constraint=rhel8
#SBATCH --gres=gpu:l40s:1          

module load USS/2022
module load gcc/9.4.0-pe5.34
module load cuda/11.6.2
module load lsfm-init-miniconda/1.0.0   

cd /cfs/earth/scratch/nogernic/PA2/src/U-net
conda activate unet_gpu        
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" #without this it does not work, does not find libcudart.so so also no GPU

dos2unix train.py

python train.py
