#!/bin/bash
#SBATCH --partition=a100_short
#SBATCH --account=ds_ga_3001_003-2025fa
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1    # no need to set to 2 since using torchrun
#SBATCH --cpus-per-task=16  # DataLoader in each process gets 8 num_workers (curernt setting) -> 8 * 2 (worker processes) + 2 (main processes) = 18
#SBATCH --time=2-10:00:00
#SBATCH --mem=256GB
#SBATCH --gres=gpu:2
#SBATCH --job-name=train_gpu2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jj4252@nyu.edu
#SBATCH --output=/scratch/jj4252/Nov_14_distill/ssl_project/DL-Comp/logs/%x/%j.out
#SBATCH --error=/scratch/jj4252/Nov_14_distill/ssl_project/DL-Comp/logs/%x/%j.err

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
VENV_DIR=/scratch/jj4252/venvs/distill_1114
cd /scratch/jj4252/Nov_14_distill/ssl_project/DL-Comp
source "$VENV_DIR/bin/activate"
mkdir -p logs/${SLURM_JOB_NAME}

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
export HYDRA_FULL_ERROR=1
export PYTHONPATH=/scratch/jj4252/Nov_14_distill/ssl_project/DL-Comp:$PYTHONPATH
export TIMM_FUSED_ATTN=1  # Enable Flash Attention in timm
export TORCHRUN_PROC_NUM=2

START_TIME=$(date +%s)

# ViT-Small (default config)
# torchrun --standalone --nproc_per_node=${TORCHRUN_PROC_NUM} scripts/train.py \
#   distributed.enabled=true \
#   experiment_name=${TRAIN_EXP_ID}

# ViT-Base (global batch size = 256, 768 dim)
torchrun --standalone --nproc_per_node=${TORCHRUN_PROC_NUM} scripts/train.py \
  distributed.enabled=true \
  experiment_name=${TRAIN_EXP_ID} \
  model.vit.embed_dim=768 \
  model.vit.depth=12 \
  model.vit.num_heads=12 \
  training.batch_size=192 \
  training.num_epochs=200


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

