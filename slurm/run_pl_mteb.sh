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
MODEL_TYPE="T"

wandb online

cd ~/ILKT/benchmarks/src-pl-mteb
source env/bin/activate

echo '[{"model_name": "'"$MODEL_NAME"'","model_abbr": "","model_type": "'"$MODEL_TYPE"'"}]' > configs/transformer_embeddings.json

srun python run_evaluation.py --models_config configs/transformer_embeddings.json
srun python create_hf_model_card.py $MODEL_NAME