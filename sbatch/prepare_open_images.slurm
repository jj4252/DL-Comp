#!/bin/bash
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00-12:00:00
#SBATCH --mem=256GB
#SBATCH --job-name=prepare_open_images
#SBATCH --output=logs/%x/%j.out
#SBATCH --error=logs/%x/%j.err


# simply download Open Images and downsample them (one sbatch pipeline)

PROJECT_ROOT=/gpfs/data/fieremanslab/dayne/projects/DL-Final-Competition

# IMAGE_LIST_FILE is created by scripts/sample_open_images.py
IMAGE_LIST_FILE=data/Open_Images/train_sampled_2200k.txt
DATA_DIR=data/Open_Images/train/raw_2200k
DOWNSAMPLE_DIR=data/Open_Images/train/downsampled_2200k

source /gpfs/home/dl5635/.bashrc
conda activate ssl-vision
cd ${PROJECT_ROOT}
mkdir -p logs/${SLURM_JOB_NAME}

python scripts/download_open_images.py \
    ${IMAGE_LIST_FILE} \
    --download_folder=${DATA_DIR} \
    --num_processes=${SLURM_CPUS_PER_TASK}

python scripts/downsample_open_images.py \
    --image_list_file=${IMAGE_LIST_FILE} \
    --data_dir=${DATA_DIR} \
    --output_dir=${DOWNSAMPLE_DIR} \
    --num_workers=${SLURM_CPUS_PER_TASK}
