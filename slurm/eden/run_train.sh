#!/usr/bin/env bash
#SBATCH --partition short
#SBATCH --account=mi2lab-normal
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time 16:00:00
#SBATCH --job-name=inp_exp
#SBATCH --output=slurm_logs/inp_exp-%A.log

script_path=$(readlink -f "$0")
cat $script_path

WANDB_MODE=online
HYDRA_FULL_ERROR=1

source /etc/profile.d/slurm.sh
source /mnt/evafs/groups/mi2lab/bsobieski/scripts/conda/import.sh
cd ~/ILKT
cload ~/my_conda.tar
conda activate ilkt
pip install pyarrow==16.1.0
source .env
wandb login $WANDB_API_KEY
git config --global credential.helper store
huggingface-cli login --token $HF_TOKEN
wandb online
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

srun python src/train.py --config-name train_config