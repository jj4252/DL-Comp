#!/bin/bash
#SBATCH --partition=a100_short
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1    # no need to set to 2 since using torchrun
#SBATCH --cpus-per-task=32    # 2 GPUs settings: cpus-per-task == 2 * num_workers
#SBATCH --time=2-10:00:00
#SBATCH --mem=256GB
#SBATCH --gres=gpu:2
#SBATCH --job-name=train_and_eval
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dl5635@nyu.edu
#SBATCH --output="logs/%x/%j.out"
#SBATCH --error=logs/%x/%j.err


TRAIN_EVAL_EXP_ID=${SLURM_JOB_ID}


echo "=================================================="
echo "Training and Evaluation Experiment ID: ${TRAIN_EVAL_EXP_ID}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "SLURM_NTASKS=$SLURM_NTASKS"
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
export TIMM_FUSED_ATTN=1  # Enable Flash Attention in timm
export TORCHRUN_PROC_NUM=2

START_TIME=$(date +%s)

# ViT-Small (default config)
torchrun --standalone --nproc_per_node=${TORCHRUN_PROC_NUM} scripts/train.py \
  distributed.enabled=true \
  experiment_name=${TRAIN_EVAL_EXP_ID} \
  optimizer.lr=0.001 \
  model.dino.multi_crop_augmentation.local_crops_number=6 \
  checkpoint.resume_from=/gpfs/data/fieremanslab/dayne/projects/DL-Final-Competition/checkpoints/14309937/checkpoint_epoch_140.pth


# ViT-Base (global batch size = 384, 768 dim)
# reduced learing rate following a linear scaling rule
# scheduler.warmup_epochs = 30 (good for larger model?)
# torchrun --standalone --nproc_per_node=${TORCHRUN_PROC_NUM} scripts/train.py \
#   distributed.enabled=true \
#   experiment_name=${TRAIN_EVAL_EXP_ID} \
#   model.vit.embed_dim=768 \
#   model.vit.depth=12 \
#   model.vit.num_heads=12 \
#   training.batch_size=192 \
#   checkpoint.resume_from=/gpfs/data/fieremanslab/dayne/projects/DL-Final-Competition/checkpoints/14033350/checkpoint_epoch_150.pth \
#   training.num_epochs=210 \
#   optimizer.lr=0.000375 \
#   scheduler.warmup_epochs=30


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

CKPT_DIR=/gpfs/data/fieremanslab/dayne/projects/DL-Final-Competition/checkpoints/${TRAIN_EVAL_EXP_ID}

CUB200_BEST_CKPT_DIR=${CKPT_DIR}/best_cub200
MINI_IMAGE_NET_BEST_CKPT_DIR=${CKPT_DIR}/best_mini_imagenet
SUN397_BEST_CKPT_DIR=${CKPT_DIR}/best_sun397

# eval on CUB-200 with linear probing
python scripts/eval_submission.py --config-name=submission \
    experiment_name=cub200_${TRAIN_EVAL_EXP_ID} \
    'evaluation/data@evaluation.data.0=cub200' \
    evaluation.checkpoint_dir=${CUB200_BEST_CKPT_DIR} \

# eval on mini-ImageNet with linear probing
python scripts/eval_submission.py --config-name=submission \
    experiment_name=mini_imagenet_${TRAIN_EVAL_EXP_ID} \
    'evaluation/data@evaluation.data.0=mini_imagenet' \
    evaluation.checkpoint_dir=${MINI_IMAGE_NET_BEST_CKPT_DIR} \


# eval on SUN397 with linear probing
python scripts/eval_submission.py --config-name=submission \
    experiment_name=sun397_${TRAIN_EVAL_EXP_ID} \
    'evaluation/data@evaluation.data.0=sun397' \
    evaluation.checkpoint_dir=${SUN397_BEST_CKPT_DIR}
