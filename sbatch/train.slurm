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
#SBATCH --mail-user=jj4252@nyu.edu
#SBATCH --output="logs/%x/%j.out"
#SBATCH --error=logs/%x/%j.err

TRAIN_EXP_ID=${SLURM_JOB_ID}


echo "=================================================="
echo "Training Experiment ID: ${TRAIN_EXP_ID}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "SLURM_NTASKS=$SLURM_NTASKS"
echo "Start Time: $(date)"
echo "=================================================="


module load cuda/12.6
source /gpfs/home/jj4252/.bashrc
conda activate ssl-vision
cd /scratch/jj4252/Nov_14_distill/ssl_project/DL-Comp
mkdir -p logs/${SLURM_JOB_NAME}

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
export HYDRA_FULL_ERROR=1
export PYTHONPATH=/scratch/jj4252/Nov_14_distill/ssl_project/DL-Comp:$PYTHONPATH
export TIMM_FUSED_ATTN=1  # Enable Flash Attention in timm


# ViT-Small
python scripts/train.py \
    # checkpoint.resume_from=/scratch/jj4252/Nov_14_distill/ssl_project/DL-Comp/checkpoints/train_dino_v2_13205191/checkpoint_epoch_60.pth \
    training.num_workers=16 \
    experiment_name=${TRAIN_EXP_ID}

# ViT-Base
# python scripts/train.py \
#     experiment_name=${TRAIN_EXP_ID} \
#     training.num_workers=16 \
#     model.vit.embed_dim=768 \
#     model.vit.depth=12 \
#     model.vit.num_heads=12
