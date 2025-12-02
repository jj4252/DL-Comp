#!/bin/bash
#SBATCH --partition=a100_dev
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=4:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=eval_knn
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dl5635@nyu.edu
#SBATCH --output="logs/%x/%j.out"
#SBATCH --error=logs/%x/%j.err

EVAL_EXP_ID=${SLURM_JOB_ID}
PROJECT_ROOT=/gpfs/data/fieremanslab/dayne/projects/DL-Final-Competition

# change this
CKPT_DIR=checkpoints/train_dino_v2_gpu2_13397059

CUB200_BEST_CKPT_DIR=${CKPT_DIR}/best_cub200
MINI_IMAGE_NET_BEST_CKPT_DIR=${CKPT_DIR}/best_mini_imagenet
SUN397_BEST_CKPT_DIR=${CKPT_DIR}/best_sun397

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

# eval on CUB-200 with k-NN
python scripts/eval_submission.py --config-name=submission \
    experiment_name=cub200_${EVAL_EXP_ID} \
    evaluation.method=knn \
    'evaluation/data@evaluation.data.0=cub200' \
    evaluation.checkpoint_dir=${CUB200_BEST_CKPT_DIR} \

# eval on mini-ImageNet with k-NN
python scripts/eval_submission.py --config-name=submission \
    experiment_name=mini_imagenet_${EVAL_EXP_ID} \
    evaluation.method=knn \
    'evaluation/data@evaluation.data.0=mini_imagenet' \
    evaluation.checkpoint_dir=${MINI_IMAGE_NET_BEST_CKPT_DIR} \

# eval on SUN397 with k-NN
python scripts/eval_submission.py --config-name=submission \
    experiment_name=sun397_${EVAL_EXP_ID} \
    evaluation.method=knn \
    'evaluation/data@evaluation.data.0=sun397' \
    evaluation.checkpoint_dir=${SUN397_BEST_CKPT_DIR}
