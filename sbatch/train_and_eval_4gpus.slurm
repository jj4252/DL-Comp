#!/bin/bash
#SBATCH --partition=a100_short
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --time=2-10:00:00
#SBATCH --mem=256GB
#SBATCH --gres=gpu:4
#SBATCH --job-name=train_and_eval_4gpus
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dl5635@nyu.edu
#SBATCH --output="logs/%x/%j.out"
#SBATCH --error=logs/%x/%j.err


TRAIN_EVAL_EXP_ID=${SLURM_JOB_ID}
PROJECT_ROOT=/gpfs/data/fieremanslab/dayne/projects/DL-Final-Competition

EVAL_CKPT_DIR=checkpoints/${TRAIN_EVAL_EXP_ID}

export HYDRA_FULL_ERROR=1
export PYTHONPATH=${PROJECT_ROOT}:$PYTHONPATH
export TIMM_FUSED_ATTN=1  # Enable Flash Attention in timm
export TORCHRUN_PROC_NUM=4 # number of GPUs to use

module load cuda/12.6
source /gpfs/home/dl5635/.bashrc
conda activate ssl-vision
cd ${PROJECT_ROOT}
mkdir -p logs/${SLURM_JOB_NAME}

echo "=================================================="
echo "Training and Evaluation Experiment ID: ${TRAIN_EVAL_EXP_ID}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "SLURM_NTASKS=$SLURM_NTASKS"
echo "Start Time: $(date)"
echo "=================================================="

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

START_TIME=$(date +%s)

torchrun --standalone --nproc_per_node=${TORCHRUN_PROC_NUM} scripts/train.py \
  distributed.enabled=true \
  experiment_name=${TRAIN_EVAL_EXP_ID} \
  training.num_epochs=250 \
  training.batch_size=320 \
  training.num_workers=10 \
  model.vit.patch_size=6 \
  model.dino.multi_crop_augmentation.local_crops_number=2

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

python scripts/eval_submission.py --config-name=submission \
    experiment_name=${TRAIN_EVAL_EXP_ID} \
    evaluation.checkpoint_dir=${CUB200_BEST_CKPT_DIR} \
    "evaluation.checkpoints=[checkpoint_epoch_250.pth, checkpoint_epoch_240.pth, checkpoint_epoch_230.pth, checkpoint_epoch_220.pth, checkpoint_epoch_210.pth, checkpoint_epoch_200.pth, checkpoint_epoch_190.pth, checkpoint_epoch_180.pth, checkpoint_epoch_170.pth, checkpoint_epoch_160.pth]"
