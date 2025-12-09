#!/bin/bash
#SBATCH --job-name=pnet_finetune
#SBATCH --output=Slurm-%j/slurm-%j.out
#SBATCH --error=Slurm-%j/slurm-%j.err
#SBATCH --partition=earth-5
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00

# Create output directory for SLURM logs
mkdir -p Slurm-${SLURM_JOB_ID}

echo "========================================"
echo "INTELLIGENT FINE-TUNING - ADVANCED MODEL"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""

# Load modules
module purge
module load stack/2024-06 gcc/12.2.0 cuda/12.1.1

echo "Loaded modules:"
module list
echo ""

# Activate virtual environment
VENV_PATH="/cfs/earth/scratch/nogernic/PA2/venv_pointnet"
source ${VENV_PATH}/bin/activate

echo "Virtual environment: ${VENV_PATH}"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# Run fine-tuning
echo "========================================"
echo "Starting intelligent fine-tuning..."
echo "========================================"
echo ""

cd /cfs/earth/scratch/nogernic/PA2/src/PointnetPP

python pointnet_finetune_advanced.py

echo ""
echo "========================================"
echo "Intelligent fine-tuning completed!"
echo "Date: $(date)"
echo "========================================"
