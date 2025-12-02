#!/bin/bash
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00-12:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=downsample_open_images
#SBATCH --output=logs/%x/%j.out
#SBATCH --error=logs/%x/%j.err

PROJECT_ROOT=/gpfs/data/fieremanslab/dayne/projects/DL-Final-Competition

IMAGE_LIST_FILE=data/Open_Images/train_sampled_200k.txt
DATA_DIR=data/Open_Images/train/raw_200k
OUTPUT_DIR=data/Open_Images/train/downsampled_200k

source /gpfs/home/dl5635/.bashrc
conda activate ssl-vision
cd ${PROJECT_ROOT}
mkdir -p logs/${SLURM_JOB_NAME}

python scripts/downsample_open_images.py \
    --image_list_file=${IMAGE_LIST_FILE} \
    --data_dir=${DATA_DIR} \
    --output_dir=${OUTPUT_DIR} \
    --num_workers=${SLURM_CPUS_PER_TASK}
