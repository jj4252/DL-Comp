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
PROJECT_ROOT=/gpfs/data/fieremanslab/dayne/projects/DL-Final-Competition

# change this to the checkpoint path you want to evaluate
CKPT_PATH=checkpoints/train_dino_v2_gpu2_13223123/checkpoint_epoch_118.pth

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

python scripts/eval_cifar.py \
    experiment_name=${EVAL_EXP_ID} \
    data=cifar100 \
    evaluation.checkpoint_path=${CKPT_PATH}
