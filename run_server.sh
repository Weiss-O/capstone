#!/bin/bash
CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate sam2
export PYTHONPATH=/home/owen_weiss/sam2

python python/SAM.py
