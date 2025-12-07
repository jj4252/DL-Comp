#!/bin/bash
#SBATCH --partition=g2-standard-12
#SBATCH --account=ds_ga_3001_003-2025fa
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=eval_linear_probe
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jj4252@nyu.edu
#SBATCH --output=/scratch/jj4252/Nov_14_distill/ssl_project/DL-Comp/logs/%x/%j.out
#SBATCH --error=/scratch/jj4252/Nov_14_distill/ssl_project/DL-Comp/logs/%x/%j.err

EVAL_EXP_ID=${SLURM_JOB_ID}

echo "=================================================="
echo "Linear Probing Evaluation"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Evaluation Experiment ID: ${EVAL_EXP_ID}"
echo "Start Time: $(date)"
echo "=================================================="

module load cuda/12.6
VENV_DIR=/scratch/jj4252/venvs/distill_1114
cd /scratch/jj4252/Nov_14_distill/ssl_project/DL-Comp
source "$VENV_DIR/bin/activate"
mkdir -p logs/${SLURM_JOB_NAME}

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
export HYDRA_FULL_ERROR=1
export PYTHONPATH=/scratch/jj4252/Nov_14_distill/ssl_project/DL-Comp:$PYTHONPATH

# Checkpoint directory - update this path to your checkpoint directory
CHECKPOINT_DIR=./checkpoints

# Checkpoint file
CHECKPOINT_FILE=checkpoint_epoch_160.pth

# Evaluate on CUB-200 with linear probing
echo ""
echo "=================================================="
echo "Evaluating on CUB-200"
echo "=================================================="
python scripts/eval_linear_probe.py \
    --config-name eval_linear_probe \
    experiment_name=cub200_${EVAL_EXP_ID} \
    'evaluation/data@evaluation.data.0=cub200' \
    evaluation.checkpoint_dir=${CHECKPOINT_DIR} \
    evaluation.checkpoints=[${CHECKPOINT_FILE}] \
    evaluation.linear_probe_epochs=100 \
    evaluation.linear_probe_lr=0.001 \
    evaluation.lr_schedule=cosine

# Evaluate on mini-ImageNet with linear probing
echo ""
echo "=================================================="
echo "Evaluating on mini-ImageNet"
echo "=================================================="
python scripts/eval_linear_probe.py \
    --config-name eval_linear_probe \
    experiment_name=mini_imagenet_${EVAL_EXP_ID} \
    'evaluation/data@evaluation.data.0=mini_imagenet' \
    evaluation.checkpoint_dir=${CHECKPOINT_DIR} \
    evaluation.checkpoints=[${CHECKPOINT_FILE}] \
    evaluation.linear_probe_epochs=100 \
    evaluation.linear_probe_lr=0.001 \
    evaluation.lr_schedule=cosine

# Evaluate on SUN397 with linear probing
echo ""
echo "=================================================="
echo "Evaluating on SUN397"
echo "=================================================="
python scripts/eval_linear_probe.py \
    --config-name eval_linear_probe \
    experiment_name=sun397_${EVAL_EXP_ID} \
    'evaluation/data@evaluation.data.0=sun397' \
    evaluation.checkpoint_dir=${CHECKPOINT_DIR} \
    evaluation.checkpoints=[${CHECKPOINT_FILE}] \
    evaluation.linear_probe_epochs=100 \
    evaluation.linear_probe_lr=0.001 \
    evaluation.lr_schedule=cosine

echo ""
echo "=================================================="
echo "All evaluations completed!"
echo "=================================================="
echo "End time: $(date)"
echo "✓ Job completed!"

