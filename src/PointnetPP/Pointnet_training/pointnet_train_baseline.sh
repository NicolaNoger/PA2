#!/bin/bash
#SBATCH --job-name=pointnet_training
#SBATCH --partition=earth-5
#SBATCH --constraint=rhel8
#SBATCH --time=02-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1

                            
echo "#SBATCH --gres=gpu:l40s:1"

# Print job info
echo "=========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE MB"
echo "=========================================="

module load USS/2022
module load gcc/9.4.0-pe5.34
module load cuda/11.6.2
module load python/3.9.12-pe5.34

cd /cfs/earth/scratch/nogernic/PA2/src/PointnetPP

# Activate virtual environment
source /cfs/earth/scratch/nogernic/PA2/venv_pointnet/bin/activate

# Verify Python and CUDA
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

dos2unix pointnet_train_baseline.py

python pointnet_train_baseline.py
