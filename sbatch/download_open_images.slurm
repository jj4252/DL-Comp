#!/bin/bash
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00-12:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=download_open_images
#SBATCH --output=logs/%x/%j.out
#SBATCH --error=logs/%x/%j.err


source /gpfs/home/dl5635/.bashrc
conda activate ssl-vision
cd /gpfs/data/fieremanslab/dayne/projects/DL-Final-Competition
mkdir -p logs/${SLURM_JOB_NAME}


IMAGE_LIST_FILE=data/Open_Images/open_images_image_list.txt
DOWNLOAD_FOLDER=data/Open_Images

python scripts/download_open_images.py $IMAGE_LIST_FILE  --download_folder=$DOWNLOAD_FOLDER --num_processes=$SLURM_CPUS_PER_TASK
