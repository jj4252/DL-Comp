#!/bin/bash
#SBATCH --partition=cpu_short
#SBATCH --account=ds_ga_3001_003-2025fa
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00-12:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=downsample_open_images
#SBATCH --output=/scratch/jj4252/Nov_14_distill/ssl_project/DL-Comp/logs/%x/%j.out
#SBATCH --error=/scratch/jj4252/Nov_14_distill/ssl_project/DL-Comp/logs/%x/%j.err

VENV_DIR=/scratch/jj4252/venvs/distill_1114
cd /scratch/jj4252/Nov_14_distill/ssl_project/DL-Comp
source "$VENV_DIR/bin/activate"
mkdir -p logs/${SLURM_JOB_NAME}


IMAGE_LIST_FILE=data/Open_Images/open_images_image_list.txt
DATA_DIR=data/Open_Images
OUTPUT_DIR=data/Open_Images_downsampled

python scripts/downsample.py \
    --image_list_file=${IMAGE_LIST_FILE} \
    --data_dir=${DATA_DIR} \
    --output_dir=${OUTPUT_DIR} \
    --num_workers=${SLURM_CPUS_PER_TASK}
