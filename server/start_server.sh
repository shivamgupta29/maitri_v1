#!/bin/bash

# Set environment variables to prevent TensorFlow mutex issues on macOS
export OMP_NUM_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
export TF_CPP_MIN_LOG_LEVEL=2

# Additional macOS specific settings
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

echo "Starting model server with optimized threading settings..."
python3 model_server.py
