#!/bin/bash
#SBATCH --partition=c12m85-a100-1
#SBATCH --account=ds_ga_3001_003-2025fa
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=2-00:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_moco
#SBATCH --mail-user=jj4252@nyu.edu
#SBATCH --output=/scratch/jj4252/Nov_14_distill/ssl_project/DL-Comp/logs/%x/%j.out
#SBATCH --error=/scratch/jj4252/Nov_14_distill/ssl_project/DL-Comp/logs/%x/%j.err

TRAIN_EXP_ID=${SLURM_JOB_ID}


echo "=================================================="
echo "MoCo V3 Training Experiment ID: ${TRAIN_EXP_ID}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "SLURM_NTASKS=$SLURM_NTASKS"
echo "Start Time: $(date)"
echo "=================================================="


module load cuda/12.6 2>/dev/null || echo "Note: module command not available, assuming CUDA is available"
VENV_DIR=/scratch/jj4252/venvs/distill_1114
cd /scratch/jj4252/Nov_14_distill/ssl_project/DL-Comp
source "$VENV_DIR/bin/activate"
mkdir -p logs/${SLURM_JOB_NAME}

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
export HYDRA_FULL_ERROR=1
export PYTHONPATH=/scratch/jj4252/Nov_14_distill/ssl_project/DL-Comp:$PYTHONPATH
export TIMM_FUSED_ATTN=1  # Enable Flash Attention in timm


# MoCo V3 Training with ViT-Small
python scripts/train.py \
    --config-name train_moco \
    training.num_workers=12 \
    experiment_name=${TRAIN_EXP_ID}

# MoCo V3 Training with ViT-Base (uncomment to use)
# python scripts/train.py \
#     --config-name train_moco \
#     experiment_name=${TRAIN_EXP_ID} \
#     training.num_workers=12 \
#     model.vit.embed_dim=768 \
#     model.vit.depth=12 \
#     model.vit.num_heads=12


