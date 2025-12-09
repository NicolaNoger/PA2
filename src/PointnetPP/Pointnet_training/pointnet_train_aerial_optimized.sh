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

echo "========================================"
echo "POINTNET++ AERIAL OPTIMIZED (V3)"
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo ""
echo "🎯 Key Improvements:"
echo "  ✅ Radii: 1.0m-10.0m (vs 0.1-0.8m indoor)"
echo "  ✅ Sampling: 2048-32 points (vs 1024-16)"
echo "  ✅ Dice Loss + Focal Loss combination"
echo "  ✅ Dropout: 0.6 (vs 0.5)"
echo "  ✅ Weight Decay: 5e-5 (vs 1e-4)"
echo ""
echo "📈 Expected: +2-3% accuracy improvement"
echo "========================================"
echo ""

# Load modules
module purge
module load USS/2022
module load gcc/9.4.0-pe5.34
module load cuda/11.6.2
module load python/3.9.12-pe5.34


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

# Run training
echo "========================================"
echo "Starting aerial-optimized training..."
echo "========================================"
echo ""

cd /cfs/earth/scratch/nogernic/PA2/src/PointnetPP

python pointnet_train_aerial_optimized.py

echo ""
echo "========================================"
echo "Training completed!"
echo "Date: $(date)"
echo "========================================"
