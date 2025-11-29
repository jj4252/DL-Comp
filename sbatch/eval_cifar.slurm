#!/bin/bash
#SBATCH --partition=a100_short
#SBATCH --account=ds_ga_3001_003-2025fa
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=eval_cifar
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jj4252@nyu.edu
#SBATCH --output="logs/%x/%j.out"
#SBATCH --error=logs/%x/%j.err

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


# ViT-Small
python scripts/eval_cifar.py \
    experiment_name=${EVAL_EXP_ID} \
    data=cifar100 \
    evaluation.checkpoint_path=/scratch/jj4252/Nov_14_distill/ssl_project/DL-Comp/checkpoints/train_dino_v2_gpu2_13223123/checkpoint_epoch_118.pth \
    # evaluation.linear_probe_epochs=100

# ViT-Base
# python scripts/eval_cifar.py \
#     experiment_name=${EVAL_EXP_ID} \
#     data=cifar10 \
#     model.vit.embed_dim=768 \
#     model.vit.depth=12 \
#     model.vit.num_heads=12 \
#     evaluation.checkpoint_path=/scratch/jj4252/Nov_14_distill/ssl_project/DL-Comp/checkpoints/train_dino_v2_13201936/checkpoint_final.pth
