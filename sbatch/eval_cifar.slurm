#!/bin/bash
#SBATCH --partition=a100_short
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=eval_cifar
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dl5635@nyu.edu
#SBATCH --output="logs/%x/%j.out"
#SBATCH --error=logs/%x/%j.err

EVAL_EXP_ID=${SLURM_JOB_ID}

echo "=================================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Evaluation Experiment ID: ${EVAL_EXP_ID}"
echo "Start Time: $(date)"
echo "=================================================="


module load cuda/12.6
source /gpfs/home/dl5635/.bashrc
conda activate ssl-vision
cd /gpfs/data/fieremanslab/dayne/projects/DL-Final-Competition
mkdir -p logs/${SLURM_JOB_NAME}


echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
export HYDRA_FULL_ERROR=1
export PYTHONPATH=/gpfs/data/fieremanslab/dayne/projects/DL-Final-Competition:$PYTHONPATH


# ViT-Tiny
# python scripts/eval_cifar.py \
#     experiment_name=${EVAL_EXP_ID} \
#     data=cifar10 \
#     model.vit.image_size=32 \
#     model.vit.patch_size=4 \
#     model.vit.embed_dim=192 \
#     model.vit.depth=12 \
#     model.vit.num_heads=3 \
#     evaluation.checkpoint_path=/gpfs/data/fieremanslab/dayne/projects/DL-Final-Competition/checkpoints/train_dino_v2_13201936/checkpoint_final.pth

# ViT-Small
python scripts/eval_cifar.py \
    experiment_name=${EVAL_EXP_ID} \
    data=cifar100 \
    evaluation.checkpoint_path=/gpfs/data/fieremanslab/dayne/projects/DL-Final-Competition/checkpoints/train_dino_v2_gpu2_13223123/checkpoint_epoch_118.pth \
    # evaluation.linear_probe_epochs=100

# ViT-Base
# python scripts/eval_cifar.py \
#     experiment_name=${EVAL_EXP_ID} \
#     data=cifar10 \
#     model.vit.embed_dim=768 \
#     model.vit.depth=12 \
#     model.vit.num_heads=12 \
#     evaluation.checkpoint_path=/gpfs/data/fieremanslab/dayne/projects/DL-Final-Competition/checkpoints/train_dino_v2_13201936/checkpoint_final.pth
