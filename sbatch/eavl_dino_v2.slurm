#!/bin/bash
#SBATCH --partition=a100_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=eval_dino_v2
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

HYDRA_FULL_ERROR=1 python ssl_vision/eval.py \
    experiment_name=${EXPERIMENT_ID} \
    data=cifar100 \
    model.vit.img_size=32 \
    model.vit.patch_size=4 \
    evaluation.checkpoint_path=/gpfs/data/fieremanslab/dayne/projects/DL-Final-Competition/checkpoints/train_dino_v2_13200170/checkpoint_epoch_180.pth
