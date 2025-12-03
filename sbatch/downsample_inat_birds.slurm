#!/bin/bash
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00-12:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=downsample_inat_birds
#SBATCH --output=logs/%x/%j.out
#SBATCH --error=logs/%x/%j.err

PROJECT_ROOT=/gpfs/data/fieremanslab/dayne/projects/DL-Final-Competition

source /gpfs/home/dl5635/.bashrc
conda activate ssl-vision
cd ${PROJECT_ROOT}
mkdir -p logs/${SLURM_JOB_NAME}


python scripts/downsample_images.py \
    --data_dir data/iNat_birds/NA_1 \
    --output_dir data/iNat_birds/NA_1_downsampled \
    --num_workers ${SLURM_CPUS_PER_TASK}

python scripts/downsample_images.py \
    --data_dir data/iNat_birds/non_NA_1 \
    --output_dir data/iNat_birds/non_NA_1_downsampled \
    --num_workers ${SLURM_CPUS_PER_TASK}

python scripts/downsample_images.py \
    --data_dir data/iNat_birds/NA_2 \
    --output_dir data/iNat_birds/NA_2_downsampled \
    --num_workers ${SLURM_CPUS_PER_TASK}

python scripts/downsample_images.py \
    --data_dir data/iNat_birds/non_NA_2 \
    --output_dir data/iNat_birds/non_NA_2_downsampled \
    --num_workers ${SLURM_CPUS_PER_TASK}
