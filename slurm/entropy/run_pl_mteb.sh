#!/usr/bin/env bash
#SBATCH --partition common
#SBATCH --qos=1gpu1d
#SBATCH --gres=gpu:1
#SBATCH --time 16:00:00
#SBATCH --job-name=train
#SBATCH --array=1-1
#SBATCH --output=slurm_logs/pl-mteb-%A-%a.log

# echo file content to logs
script_path=$(readlink -f "$0")
cat $script_path

MODEL_NAMES=(
"2024-06-24_22-31-18" 
"2024-06-24_22-31-23" 
"2024-06-24_22-31-28" 
"2024-06-24_22-31-33" 
"2024-06-24_22-31-39" 
"2024-06-24_22-31-34" 
) 
EPOCHS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35)


wandb online

source env/bin/activate
cd ~/ILKT/benchmarks
export TOKENIZERS_PARALLELISM=false

# for loop over epochs
MODEL_NAME_NO_ORG_NO_EPOCH=${MODEL_NAMES[$SLURM_ARRAY_TASK_ID]}
for EPOCH in ${EPOCHS[@]}; do
    set +e
    MODEL_NAME_NO_ORG="${MODEL_NAME_NO_ORG_NO_EPOCH}_epoch_${EPOCH}"
    MODEL_NAME="ILKT/${MODEL_NAME_NO_ORG}"
    srun python run_mteb_polish.py $MODEL_NAME
    srun python parse_results.py create_meta --results_folder results/pl/$MODEL_NAME_NO_ORG/$MODEL_NAME_NO_ORG/no_revision_available --output_path model_card_$MODEL_NAME_NO_ORG.md --overwrite
    srun python create_hf_model_card.py $MODEL_NAME
done