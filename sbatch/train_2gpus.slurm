#!/bin/bash
#SBATCH --partition=a100_short
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1    # no need to set to 2 since using torchrun
#SBATCH --cpus-per-task=32    # 2 GPUs settings: cpus-per-task == 2 * num_workers
#SBATCH --time=2-00:00:00
#SBATCH --mem=256GB
#SBATCH --gres=gpu:2
#SBATCH --job-name=train_2gpus
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dl5635@nyu.edu
#SBATCH --output="logs/%x/%j.out"
#SBATCH --error=logs/%x/%j.err

TRAIN_EXP_ID=${SLURM_JOB_ID}
PROJECT_ROOT=/gpfs/data/fieremanslab/dayne/projects/DL-Final-Competition

export HYDRA_FULL_ERROR=1
export PYTHONPATH=${PROJECT_ROOT}:$PYTHONPATH
export TIMM_FUSED_ATTN=1  # Enable Flash Attention in timm
export TORCHRUN_PROC_NUM=2 # number of GPUs to use

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

START_TIME=$(date +%s)

# ViT-Small (default config)
torchrun --standalone --nproc_per_node=${TORCHRUN_PROC_NUM} scripts/train.py \
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
