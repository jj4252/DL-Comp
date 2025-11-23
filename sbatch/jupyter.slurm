#!/bin/bash
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00-12:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=jupyter
#SBATCH --output=logs/%x/%j.log


# get tunneling info
XDG_RUNTIME_DIR=""
port=8890
node=$(hostname -s)
user=$(whoami)

case "$SLURM_SUBMIT_HOST" in
bigpurple-ln1)
 login_node=bigpurple1.nyumc.org
 ;;
bigpurple-ln2)
 login_node=bigpurple2.nyumc.org
 ;;
bigpurple-ln3)
 login_node=bigpurple3.nyumc.org
 ;;
bigpurple-ln4)
 login_node=bigpurple4.nyumc.org
 ;;
esac

# print tunneling instructions jupyter-log
echo -e "
MacOS or linux terminal command to create your ssh tunnel:
ssh -N -L ${port}:${node}:${port} ${user}@${login_node}

For more info and how to connect from windows,
   see research.computing.yale.edu/jupyter-nb
Here is the MobaXterm info:

Forwarded port:same as remote port
Remote server: ${node}
Remote port: ${port}
SSH server: ${login_node}
SSH login: $user
SSH port: 22
Use a Browser on your local machine to go to:
http://localhost:${port}  (prefix w/ https:// if using password)
Use the token string from the URL printed below and add it to the URL above
"

# load modules or conda environments here

module load cuda/12.6
source /gpfs/home/dl5635/.bashrc
conda activate ssl-vision
cd /gpfs/data/fieremanslab/dayne/projects/DL-Final-Competition
mkdir -p logs/${SLURM_JOB_NAME}
jupyter-notebook --no-browser --port=${port} --ip=${node}
