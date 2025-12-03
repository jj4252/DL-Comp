# DL-Final-Competition


Downloading subset of Open Images train Dataset

```bash
cd data/Open_Images
curl -L -O https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv
curl -L -O https://storage.googleapis.com/openimages/v7/oidv7-train-annotations-human-imagelabels.csv
curl -L -O https://storage.googleapis.com/openimages/v6/oidv6-train-images-with-labels-with-rotation.csv

# From the PROJECT ROOT:
# Run all cells in the notebook/mini_imagenet_class.ipynb
# Run all cells in the notebook/open_images_analysis2.ipynb


python scripts/sample_open_images.py --start 1 --end 150000

python scripts/download_open_images.py data/Open_Images/mini_only_1-150000.txt  --download_folder=data/Open_Images/train/raw/mini_only_1-150000 --num_processes=5

python scripts/download_open_images.py data/Open_Images/non_only_1-150000.txt  --download_folder=data/Open_Images/train/raw/non_only_1-150000 --num_processes=5

python scripts/download_open_images.py data/Open_Images/mixed_1-150000.txt  --download_folder=data/Open_Images/train/raw/mixed_1-150000 --num_processes=5

# or on HPC Cluster
sbatch sbatch/download_open_images.slurm
```
