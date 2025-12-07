#!/bin/bash
#SBATCH --partition=n2c48m24
#SBATCH --account=ds_ga_3001_003-2025fa
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=download_fall2025_dl
#SBATCH --output=/scratch/jj4252/Nov_14_distill/ssl_project/DL-Comp/logs/%x/%j.out
#SBATCH --error=/scratch/jj4252/Nov_14_distill/ssl_project/DL-Comp/logs/%x/%j.err

VENV_DIR=/scratch/jj4252/venvs/distill_1114
cd /scratch/jj4252/Nov_14_distill/ssl_project/DL-Comp
source "$VENV_DIR/bin/activate"
mkdir -p logs/${SLURM_JOB_NAME}

echo "=========================================="
echo "Downloading fall2025_deeplearning dataset"
echo "=========================================="
echo "Start time: $(date)"
echo ""

# Make script executable and run it
chmod +x scripts/download_fall2025_deeplearning.sh
bash scripts/download_fall2025_deeplearning.sh

echo ""
echo "End time: $(date)"
echo "✓ Download job completed!"


