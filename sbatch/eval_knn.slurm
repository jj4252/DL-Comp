#!/bin/bash
#SBATCH --partition=a100_short
#SBATCH --account=ds_ga_3001_003-2025fa
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=eval_knn
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jj4252@nyu.edu
#SBATCH --output=/scratch/jj4252/Nov_14_distill/ssl_project/DL-Comp/logs/%x/%j.out
#SBATCH --error=/scratch/jj4252/Nov_14_distill/ssl_project/DL-Comp/logs/%x/%j.err

EVAL_EXP_ID=${SLURM_JOB_ID}

echo "=================================================="
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

CUB200_BEST_CKPT_DIR=/scratch/jj4252/Nov_14_distill/ssl_project/DL-Comp/checkpoints/train_dino_v2_gpu2_13397059/a_best
MINI_IMAGE_NET_BEST_CKPT_DIR=/scratch/jj4252/Nov_14_distill/ssl_project/DL-Comp/checkpoints/train_dino_v2_gpu2_13397059/a_best
SUN397_BEST_CKPT_DIR=/scratch/jj4252/Nov_14_distill/ssl_project/DL-Comp/checkpoints/train_dino_v2_gpu2_13397059/a_best

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

