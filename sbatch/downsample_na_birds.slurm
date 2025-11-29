#!/bin/bash
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00-12:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=downsample_na_birds
#SBATCH --output=logs/%x/%j.out
#SBATCH --error=logs/%x/%j.err

source /gpfs/home/dl5635/.bashrc
conda activate ssl-vision
cd /gpfs/data/fieremanslab/dayne/projects/DL-Final-Competition
mkdir -p logs/${SLURM_JOB_NAME}


python scripts/downsample_na_birds.py \
    --data_dir data/NABirds/images \
    --output_dir data/NABirds/images_downsampled \
    --num_workers ${SLURM_CPUS_PER_TASK}
