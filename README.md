to run bash session on entropy cluster:
```
srun --partition=common --qos=1gpu4h --time=1:00:00 --gres=gpu:1 --pty /bin/bash
```

instalation on entropy cluster (on GPU node, not connect node):
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

benchmark installation:
```
cd benchmarks/pl-mteb
python -m venv env
source env/bin/activate
pip install -r requirements.txt
pip install scipy==1.10.1
pip install mteb==1.12.5
```

to configure wandb install it with pip and run
```
wandb login $WANDB_API_KEY
```

to submit a training job:
```
sbatch slurm/run_train.sh
```

to submit a benchmark job:
First specify the model path in benchmarks/pl-mteb/configs/transformer_embeddings.json
```
sbatch slurm/run_pl_mteb.sh
```

Datasets types follows the one availabe [here](https://huggingface.co/datasets/sentence-transformers/embedding-training-data)
