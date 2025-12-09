#!/bin/bash
#SBATCH --job-name=pointnet_advanced
#SBATCH --time=08:00:00
#SBATCH --partition=earth-5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=Slurm-%j/slurm-%j.out
#SBATCH --error=Slurm-%j/slurm-%j.err

# Create output directory
mkdir -p Slurm-${SLURM_JOB_ID}

# Load modules
module purge
module load USS/2022
module load gcc/9.4.0-pe5.34
module load cuda/11.6.2
module load python/3.9.12-pe5.34

# Activate virtual environment
source /cfs/earth/scratch/nogernic/PA2/venv_pointnet/bin/activate

# Print environment info
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Hostname: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=========================================="

# Run advanced training with Focal Loss + SPG Loss
python pointnet_train_advanced.py \
    --data_dir /cfs/earth/scratch/nogernic/Pointnet2_PyTorch/data_full \
    --epochs 50 \
    --batch_size 24 \
    --learning_rate 0.001 \
    --num_points 16384

echo "=========================================="
echo "Advanced training completed!"
echo "=========================================="
