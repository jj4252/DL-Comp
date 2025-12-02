#!/bin/bash
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00-12:00:00
#SBATCH --mem=256GB
#SBATCH --job-name=download_birdsnap
#SBATCH --output=logs/%x/%j.out
#SBATCH --error=logs/%x/%j.err


# simply download Birdsnap

PROJECT_ROOT=/gpfs/data/fieremanslab/dayne/projects/DL-Final-Competition

source /gpfs/home/dl5635/.bashrc
conda activate ssl-vision
cd ${PROJECT_ROOT}
mkdir -p logs/${SLURM_JOB_NAME}

python scripts/download_birdsnap.py
