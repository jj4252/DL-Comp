#!/bin/bash
#SBATCH --partition=a100_short
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_single_gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dl5635@nyu.edu
#SBATCH --output="logs/%x/%j.out"
#SBATCH --error=logs/%x/%j.err

TRAIN_EXP_ID=${SLURM_JOB_ID}
PROJECT_ROOT=/gpfs/data/fieremanslab/dayne/projects/DL-Final-Competition

export HYDRA_FULL_ERROR=1
export PYTHONPATH=${PROJECT_ROOT}:$PYTHONPATH
export TIMM_FUSED_ATTN=1  # Enable Flash Attention in timm

module load cuda/12.6
source /gpfs/home/dl5635/.bashrc
conda activate ssl-vision
cd ${PROJECT_ROOT}
mkdir -p logs/${SLURM_JOB_NAME}

echo "=================================================="
echo "Training Experiment ID: ${TRAIN_EXP_ID}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "SLURM_NTASKS=$SLURM_NTASKS"
echo "Start Time: $(date)"
echo "=================================================="

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# ViT-Small
python scripts/train.py \
    training.num_workers=16 \
    experiment_name=${TRAIN_EXP_ID}
