#!/bin/bash
#SBATCH --partition=a100_short
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=eval_submission
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dl5635@nyu.edu
#SBATCH --output="logs/%x/%j.out"
#SBATCH --error=logs/%x/%j.err

EXPERIMENT_ID=${SLURM_JOB_NAME}_${SLURM_JOB_ID}


echo "=================================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Experiment: ${EXPERIMENT_ID}"
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

# ViT-Small
python scripts/eval_submission.py --config-name=submission \
    experiment_name=${EXPERIMENT_ID} \
    data=submission_cub200 \
    evaluation.checkpoint_path=/gpfs/data/fieremanslab/dayne/projects/DL-Final-Competition/checkpoints/train_dino_v2_gpu2_13223123/checkpoint_epoch_102.pth
