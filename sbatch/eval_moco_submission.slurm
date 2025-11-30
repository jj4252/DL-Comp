#!/bin/bash
#SBATCH --partition=c12m85-a100-1
#SBATCH --account=ds_ga_3001_003-2025fa
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=eval_moco
#SBATCH --mail-user=jj4252@nyu.edu
#SBATCH --output=/scratch/jj4252/Nov_14_distill/ssl_project/DL-Comp/logs/%x/%j.out
#SBATCH --error=/scratch/jj4252/Nov_14_distill/ssl_project/DL-Comp/logs/%x/%j.err

EVAL_EXP_ID=${SLURM_JOB_ID}

echo "=================================================="
echo "MoCo V3 Evaluation"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Evaluation Experiment ID: ${EVAL_EXP_ID}"
echo "Start Time: $(date)"
echo "=================================================="

module load cuda/12.6 2>/dev/null || echo "Note: module command not available, assuming CUDA is available"
VENV_DIR=/scratch/jj4252/venvs/distill_1114
cd /scratch/jj4252/Nov_14_distill/ssl_project/DL-Comp
source "$VENV_DIR/bin/activate"
mkdir -p logs/${SLURM_JOB_NAME}

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
export HYDRA_FULL_ERROR=1
export PYTHONPATH=/scratch/jj4252/Nov_14_distill/ssl_project/DL-Comp:$PYTHONPATH

# Checkpoint directory
MOCO_CHECKPOINT_DIR=/scratch/jj4252/Nov_14_distill/ssl_project/DL-Comp/checkpoints/122222

# Evaluate on testset_1 (CUB-200) with linear probing
echo ""
echo "=================================================="
echo "Evaluating on testset_1 (CUB-200)"
echo "=================================================="
python scripts/eval_moco_submission.py \
    --config-name eval_moco_submission \
    experiment_name=testset1_${EVAL_EXP_ID} \
    'evaluation/data@evaluation.data.0=testset1' \
    evaluation.checkpoint_dir=${MOCO_CHECKPOINT_DIR} \
    evaluation.checkpoint_file=checkpoint_epoch_250.pth

# Evaluate on testset_2 (MiniImageNet) with linear probing
echo ""
echo "=================================================="
echo "Evaluating on testset_2 (MiniImageNet)"
echo "=================================================="
python scripts/eval_moco_submission.py \
    --config-name eval_moco_submission \
    experiment_name=testset2_${EVAL_EXP_ID} \
    'evaluation/data@evaluation.data.0=testset2' \
    evaluation.checkpoint_dir=${MOCO_CHECKPOINT_DIR} \
    evaluation.checkpoint_file=checkpoint_epoch_250.pth

# Evaluate on testset_3 (SUN397) with linear probing
echo ""
echo "=================================================="
echo "Evaluating on testset_3 (SUN397)"
echo "=================================================="
python scripts/eval_moco_submission.py \
    --config-name eval_moco_submission \
    experiment_name=testset3_${EVAL_EXP_ID} \
    'evaluation/data@evaluation.data.0=testset3' \
    evaluation.checkpoint_dir=${MOCO_CHECKPOINT_DIR} \
    evaluation.checkpoint_file=checkpoint_epoch_250.pth

echo ""
echo "=================================================="
echo "All evaluations completed!"
echo "=================================================="
echo "End time: $(date)"
echo "✓ Job completed!"

