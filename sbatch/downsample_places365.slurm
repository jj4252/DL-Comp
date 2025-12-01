#!/bin/bash
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00-12:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=downsample_places365
#SBATCH --output=logs/%x/%j.out
#SBATCH --error=logs/%x/%j.err


source /gpfs/home/dl5635/.bashrc
conda activate ssl-vision
cd /gpfs/data/fieremanslab/dayne/projects/DL-Final-Competition
mkdir -p logs/${SLURM_JOB_NAME}


RAW_DIR=data/Places365/data_256_standard

IMG_START_IDX=1
IMG_END_IDX=300
SUBSET_DIR=data/Places365/subset_${IMG_START_IDX}-${IMG_END_IDX}

DOWNSAMPLED_DIR=data/Places365/subset_${IMG_START_IDX}-${IMG_END_IDX}_downsampled


python scripts/symlink_places365.py \
    --input_dir ${RAW_DIR} \
    --output_dir ${SUBSET_DIR} \
    --img_start_idx ${IMG_START_IDX} \
    --img_end_idx ${IMG_END_IDX}


python scripts/downsample_images.py \
    --data_dir ${SUBSET_DIR} \
    --output_dir ${DOWNSAMPLED_DIR} \
    --num_workers ${SLURM_CPUS_PER_TASK}
