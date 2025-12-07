#!/bin/bash
#SBATCH --partition=n2c48m24
#SBATCH --account=ds_ga_3001_003-2025fa
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=prepare_miniimagenet
#SBATCH --output=/scratch/jj4252/Nov_14_distill/ssl_project/DL-Comp/logs/%x/%j.out
#SBATCH --error=/scratch/jj4252/Nov_14_distill/ssl_project/DL-Comp/logs/%x/%j.err

VENV_DIR=/scratch/jj4252/venvs/distill_1114
cd /scratch/jj4252/Nov_14_distill/ssl_project/DL-Comp
source "$VENV_DIR/bin/activate"
mkdir -p logs/${SLURM_JOB_NAME}

echo "=========================================="
echo "Preparing MiniImageNet for Kaggle"
echo "=========================================="
echo "Start time: $(date)"
echo ""

# Change to testset_2 directory
cd /scratch/jj4252/Nov_14_distill/ssl_project/DL-Comp/fall2025_finalproject/testset_2

# Install datasets package if needed
echo "Installing datasets package..."
pip install datasets

# Run the preparation script
echo "Running prepare_miniimagenet_for_kaggle.py..."
python prepare_miniimagenet_for_kaggle.py --download_dir ./raw_data --output_dir ./data

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "MiniImageNet preparation completed successfully!"
    echo "=========================================="
    echo "End time: $(date)"
    echo "✓ Job completed!"
else
    echo ""
    echo "=========================================="
    echo "MiniImageNet preparation failed"
    echo "=========================================="
    exit 1
fi


