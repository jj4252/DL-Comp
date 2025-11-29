#!/bin/bash
module load cuda/12.6
source ~/.bashrc
conda activate ssl-vision

export HYDRA_FULL_ERROR=1
export PYTHONPATH=/gpfs/data/fieremanslab/dayne/projects/DL-Final-Competition:$PYTHONPATH
export TIMM_FUSED_ATTN=1  # Enable Flash Attention in timm
export TORCHRUN_PROC_NUM=2


torchrun --standalone --nproc_per_node=${TORCHRUN_PROC_NUM} scripts/train.py --config-name=debug \
  model.vit.embed_dim=384 \
  model.vit.depth=12 \
  model.vit.num_heads=6 \
  training.batch_size=256
