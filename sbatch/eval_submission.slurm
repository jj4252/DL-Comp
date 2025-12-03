#!/bin/bash
#SBATCH --partition=a100_dev
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=eval_submission
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dl5635@nyu.edu
#SBATCH --output="logs/%x/%j.out"
#SBATCH --error=logs/%x/%j.err

EVAL_EXP_ID=${SLURM_JOB_ID}
PROJECT_ROOT=/gpfs/data/fieremanslab/dayne/projects/DL-Final-Competition

CKPT_DIR=checkpoints/14758573

export HYDRA_FULL_ERROR=1
export PYTHONPATH=${PROJECT_ROOT}:$PYTHONPATH

module load cuda/12.6
source /gpfs/home/dl5635/.bashrc
conda activate ssl-vision
cd ${PROJECT_ROOT}
mkdir -p logs/${SLURM_JOB_NAME}

echo "=================================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Evaluation Experiment ID: ${EVAL_EXP_ID}"
echo "Start Time: $(date)"
echo "=================================================="

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

python scripts/eval_submission.py --config-name=submission \
    experiment_name=${EVAL_EXP_ID} \
    evaluation.checkpoint_dir=${CKPT_DIR} \
    "evaluation.checkpoints=[checkpoint_epoch_200.pth, checkpoint_epoch_190.pth, checkpoint_epoch_180.pth, checkpoint_epoch_170.pth, checkpoint_epoch_160.pth]"
