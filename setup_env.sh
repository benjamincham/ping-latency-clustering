#!/bin/bash
# This script sets up a conda environment for the project with all required dependencies.
# Usage: source setup_env.sh

# Name of the environment
env_name="ping-latency-env"

# Create conda environment with Python 3.8 (if it doesn't already exist)
if ! conda info --envs | grep -q "$env_name"; then
    echo "Creating conda environment '$env_name' with Python 3.8..."
    conda create -y -n "$env_name" python=3.8
else
    echo "Conda environment '$env_name' already exists."
fi

echo "Activating environment..."
conda activate "$env_name"

# Install dependencies from conda-forge first
conda install -y -c conda-forge numpy=1.24.3 pandas matplotlib seaborn scikit-learn hdbscan

# Install any remaining Python dependencies
pip install -r requirements.txt

echo "\nSetup complete! To activate the environment in the future, run:"
echo "  conda activate $env_name"
