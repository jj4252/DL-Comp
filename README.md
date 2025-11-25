# DL-Final-Competition


Downloading subset of Open Images Dataset (validation and test are use for SSL training)

```bash
# From the project root directory
curl -L -O https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv
curl -L -O https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv

python scripts/generate_open_images_file.py

python scripts/download_open_images.py data/open_images_image_list.txt  --download_folder=data/Open_Images --num_processes=5

# or on HPC Cluster
sbatch sbatch/download_open_images.slurm
```
