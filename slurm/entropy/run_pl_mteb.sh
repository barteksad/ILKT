#!/usr/bin/env bash
#SBATCH --partition common
#SBATCH --qos=1gpu4h
#SBATCH --gres=gpu:1
#SBATCH --time 4:00:00
#SBATCH --job-name=train
#SBATCH --output=slurm_logs/pl-mteb-%A.log

# echo file content to logs
script_path=$(readlink -f "$0")
cat $script_path

MODEL_NAME="ILKT/2024-06-03_20-17-15"
MODEL_NAME_NO_ORG="2024-06-03_20-17-15"

wandb online

source env/bin/activate
cd ~/ILKT/benchmarks

srun python python run_mteb_polish.py $MODEL_NAME
mteb create_meta --results_folder results/pl/$MODEL_NAME_NO_ORG/$MODEL_NAME_NO_ORG --output_path model_card.md
srun python create_hf_model_card.py $MODEL_NAME