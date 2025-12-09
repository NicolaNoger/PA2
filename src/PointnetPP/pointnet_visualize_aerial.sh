#!/bin/bash
#SBATCH --job-name=visualize_aerial_v3
#SBATCH --time=02:00:00
#SBATCH --partition=earth-5
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1


# Load modules
module load USS/2022
module load gcc/9.4.0-pe5.34
module load cuda/11.6.2
module load python/3.9.12-pe5.34

# Activate virtual environment
source /cfs/earth/scratch/nogernic/PA2/venv_pointnet/bin/activate

# Change to script directory
cd /cfs/earth/scratch/nogernic/PA2/src/PointnetPP

# Run visualization
echo "Starting PointNet++ aerial V3 visualization..."
echo "========================================"
# Ensure the PointNet2_PyTorch package and the compiled ops lib are on PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/PointNet2_PyTorch:$(pwd)/PointNet2_PyTorch/pointnet2_ops_lib

echo "PYTHONPATH=$PYTHONPATH"

python pointnet_visualize_aerial.py
echo "========================================"
echo "Visualization complete!"
