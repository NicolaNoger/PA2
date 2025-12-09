#!/bin/bash
#SBATCH --job-name=finetune_pointnet
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=earth-5
#SBATCH --gres=gpu:1


# Print job info
echo "=========================================="
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE MB"
echo "=========================================="

# Load modules
module load USS/2022
module load gcc/9.4.0-pe5.34
module load cuda/11.6.2
module load python/3.9.12-pe5.34

# Activate virtual environment
source /cfs/earth/scratch/nogernic/PA2/venv_pointnet/bin/activate

# Verify environment
python --version
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Convert line endings (just in case)
dos2unix pointnet_finetune.py 2>/dev/null || true

# Run fine-tuning
python pointnet_finetune.py

echo ""
echo "Fine-tuning job completed at: $(date)"
