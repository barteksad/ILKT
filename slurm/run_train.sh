#!/usr/bin/env bash
#SBATCH --partition common
#SBATCH --qos=1gpu4h
#SBATCH --gres=gpu:1
#SBATCH --time 4:00:00
#SBATCH --job-name=train
#SBATCH --output=slurm_logs/train-%A-%a.log

# echo file content to logs
script_path=$(readlink -f "$0")
cat $script_path

cd /home/barteksad/ILKT
source env/bin/activate

wandb online
export HYDRA_FULL_ERROR=1

srun python src/train.py