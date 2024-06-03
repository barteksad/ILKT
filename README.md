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
pip install pydantic==2.7.2
```

to configure wandb install it with pip and run
```
wandb login $WANDB_API_KEY
```

to configure huggingface read/write install huggingface_cli with pip and run
```
git config --global credential.helper store
huggingface-cli login
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


AWS setup through [sky-pilot](https://skypilot.readthedocs.io/en/latest/getting-started/installation.html)
```
pip install "skypilot-nightly[aws]" boto3
```
And you need to create Access Key and have quota for spot instances with GPU!

to run pl-mteb with sky-pilot:
```
sky spot launch \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --env HF_TOKEN=$HF_TOKEN \
    --env MODEL_NAME=$MODEL_NAME \
    sky/sky_run_pl-mteb.yml
```

by default sky-pilot uses quite powerful instance as a controll node which in cases when you want to run one spot machine with GPU results in paying more for control node.
It can be overriden:
```
~/.sky/config.yaml

jobs:
  controller:
    resources:
      cloud: aws
      region: us-east-1
      cpus: 2
```
