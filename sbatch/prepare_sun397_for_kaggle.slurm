#!/bin/bash
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00-12:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=prepare_sun397_for_kaggle
#SBATCH --output=logs/%x/%j.out
#SBATCH --error=logs/%x/%j.err

PROJECT_ROOT=/gpfs/data/fieremanslab/dayne/projects/DL-Final-Competition
TESTSET_3_DIR=/gpfs/data/fieremanslab/dayne/projects/fall2025_finalproject/testset_3

source /gpfs/home/dl5635/.bashrc
conda activate ssl-vision
cd ${PROJECT_ROOT}
mkdir -p logs/${SLURM_JOB_NAME}

cd ${TESTSET_3_DIR}
python prepare_sun397_for_kaggle.py --download_dir ./raw_data --output_dir ./data
