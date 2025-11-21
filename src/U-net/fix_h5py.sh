#!/bin/bash
# Fix h5py compatibility issue for loading Keras weights

echo "Fixing h5py compatibility for TensorFlow 2.15.0..."

# Load modules
module load lsfm-init-miniconda/1.0.0

# Activate environment
eval "$(conda shell.bash hook)"
conda activate unet_gpu

echo "Current h5py version:"
python -c "import h5py; print(h5py.__version__)"

echo ""
echo "Upgrading h5py to compatible version..."
pip install --upgrade h5py

echo ""
echo "New h5py version:"
python -c "import h5py; print(h5py.__version__)"

echo ""
echo "Testing weight loading..."
python -c "
import tensorflow as tf
try:
    model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)
    print('✓ Weight loading works!')
except Exception as e:
    print(f'✗ Still failing: {e}')
"

echo ""
echo "Done!"
