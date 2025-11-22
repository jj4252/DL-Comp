#!/bin/bash
#SBATCH --partition=a100_short
#SBATCH --nodes=1
#SBATCH --tasks-per-node=2 # 1 task (process) per GPU
#SBATCH --cpus-per-task=16   # 32 logical CPUs (2 * 16)
#SBATCH --time=2-00:00:00
#SBATCH --mem=256GB
#SBATCH --gres=gpu:2
#SBATCH --job-name=train_dino_v2_gpu2
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

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
export HYDRA_FULL_ERROR=1

# ViT-Small (default config)
torchrun --standalone --nproc_per_node=$SLURM_NTASKS ssl_vision/train.py \
  distributed.enabled=true \
  experiment_name=${EXPERIMENT_ID}
