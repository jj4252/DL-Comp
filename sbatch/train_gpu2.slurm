#!/bin/bash
#SBATCH --partition=a100_short
#SBATCH --nodes=1
#SBATCH --tasks-per-node=2 # 1 task (process) per GPU
#SBATCH --cpus-per-task=16   # 32 logical CPUs (2 * 16)
#SBATCH --time=2-00:00:00
#SBATCH --mem=256GB
#SBATCH --gres=gpu:2
#SBATCH --job-name=train_gpu2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dl5635@nyu.edu
#SBATCH --output="logs/%x/%j.out"
#SBATCH --error=logs/%x/%j.err

TRAIN_EXP_ID=${SLURM_JOB_ID}


echo "=================================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Training Experiment ID: ${TRAIN_EXP_ID}"
echo "Start Time: $(date)"
echo "=================================================="


module load cuda/12.6
source /gpfs/home/dl5635/.bashrc
conda activate ssl-vision
cd /gpfs/data/fieremanslab/dayne/projects/DL-Final-Competition
mkdir -p logs/${SLURM_JOB_NAME}

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
export HYDRA_FULL_ERROR=1
export PYTHONPATH=/gpfs/data/fieremanslab/dayne/projects/DL-Final-Competition:$PYTHONPATH

START_TIME=$(date +%s)

# ViT-Small (default config)
torchrun --standalone --nproc_per_node=$SLURM_NTASKS scripts/train.py \
  distributed.enabled=true \
  experiment_name=${TRAIN_EXP_ID}

# Print training duration
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
DAYS=$((ELAPSED/86400))
HOURS=$(((ELAPSED%86400)/3600))
MINUTES=$(((ELAPSED%3600)/60))
SECONDS=$((ELAPSED%60))

echo "=================================================="
echo "Training Duration: ${DAYS}d ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "End Time: $(date)"
echo "=================================================="
