#!/bin/bash
#SBATCH --job-name=pnet_aerial_v3
#SBATCH --output=Slurm-%j/slurm-%j.out
#SBATCH --error=Slurm-%j/slurm-%j.err
#SBATCH --partition=earth-5
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00

# Create output directory for SLURM logs
mkdir -p Slurm-${SLURM_JOB_ID}

# Load modules
module purge
module load USS/2022
module load gcc/9.4.0-pe5.34
module load cuda/11.6.2
module load python/3.9.12-pe5.34


module list

# Activate virtual environment
VENV_PATH="/cfs/earth/scratch/nogernic/PA2/venv_pointnet"
source ${VENV_PATH}/bin/activate

# GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""


cd /cfs/earth/scratch/nogernic/PA2/src/PointnetPP

python pointnet_train_aerial_optimized.py

