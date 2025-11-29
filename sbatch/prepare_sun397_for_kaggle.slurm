#!/bin/bash
#SBATCH --partition=cpu_short
#SBATCH --account=ds_ga_3001_003-2025fa
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00-12:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=prepare_sun397_for_kaggle
#SBATCH --output=logs/%x/%j.out
#SBATCH --error=logs/%x/%j.err


VENV_DIR=/scratch/jj4252/venvs/distill_1114
cd /scratch/jj4252/Nov_14_distill/ssl_project/DL-Comp
source "$VENV_DIR/bin/activate"
mkdir -p logs/${SLURM_JOB_NAME}

cd /scratch/jj4252/Nov_14_distill/fall2025_finalproject/testset_3


python prepare_sun397_for_kaggle.py --download_dir ./raw_data --output_dir ./data
