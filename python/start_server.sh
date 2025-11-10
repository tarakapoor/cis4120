#!/bin/bash
# Start the FastAPI server with the conda environment

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate ddpm_bc_env

# Start the server
cd "$(dirname "$0")"
uvicorn server:app --reload --port 8000
