#!/bin/bash
#SBATCH --partition=a100_short
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --mem=256GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_dino_v2
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


# ViT-Tiny
# python scripts/train.py \
#     experiment_name=${TRAIN_EXP_ID} \
#     model.vit.embed_dim=192 \
#     model.vit.depth=12 \
#     model.vit.num_heads=3

# ViT-Small
python scripts/train.py \
    experiment_name=${TRAIN_EXP_ID} \
    # checkpoint.resume_from=/gpfs/data/fieremanslab/dayne/projects/DL-Final-Competition/checkpoints/train_dino_v2_13205191/checkpoint_epoch_60.pth

# ViT-Base
# python scripts/train.py \
#     experiment_name=${TRAIN_EXP_ID} \
#     model.vit.embed_dim=768 \
#     model.vit.depth=12 \
#     model.vit.num_heads=12
