#!/bin/bash
#SBATCH --partition=n2c48m24
#SBATCH --account=ds_ga_3001_003-2025fa
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00-12:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=download_open_images
#SBATCH --output=logs/%x/%j.out
#SBATCH --error=logs/%x/%j.err


VENV_DIR=/scratch/jj4252/venvs/distill_1114
cd /scratch/jj4252/Nov_14_distill/ssl_project/DL-Comp
source "$VENV_DIR/bin/activate"
mkdir -p logs/${SLURM_JOB_NAME}

# Download CSV files if they don't exist
if [ ! -f "validation-images-with-rotation.csv" ]; then
    echo "Downloading validation-images-with-rotation.csv..."
    curl -L -O https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv
fi

if [ ! -f "test-images-with-rotation.csv" ]; then
    echo "Downloading test-images-with-rotation.csv..."
    curl -L -O https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv
fi

# Generate image list file if it doesn't exist
IMAGE_LIST_FILE=data/Open_Images/open_images_image_list.txt
if [ ! -f "$IMAGE_LIST_FILE" ]; then
    echo "Generating image list file..."
    python scripts/generate_open_images_file.py
fi

# Download images
DOWNLOAD_FOLDER=data/Open_Images
echo "Starting image download..."
python scripts/download_open_images.py $IMAGE_LIST_FILE  --download_folder=$DOWNLOAD_FOLDER --num_processes=$SLURM_CPUS_PER_TASK
