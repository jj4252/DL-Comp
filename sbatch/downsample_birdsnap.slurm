#!/bin/bash
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00-12:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=downsample_birdsnap
#SBATCH --output=logs/%x/%j.out
#SBATCH --error=logs/%x/%j.err


source /gpfs/home/dl5635/.bashrc
conda activate ssl-vision
cd /gpfs/data/fieremanslab/dayne/projects/DL-Final-Competition
mkdir -p logs/${SLURM_JOB_NAME}


python scripts/downsample_images.py \
    --data_dir data/Birdsnap/train/raw \
    --output_dir data/Birdsnap/train/downsampled \
    --num_workers ${SLURM_CPUS_PER_TASK}
