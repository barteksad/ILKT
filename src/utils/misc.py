import wandb
from pathlib import Path
import omegaconf
from omegaconf import DictConfig


def extract_output_dir(config: DictConfig) -> Path:
    """
    Extracts path to output directory created by Hydra as pathlib.Path instance
    """
    date = "/".join(list(config._metadata.resolver_cache["now"].values()))
    output_dir = Path.cwd() / "outputs" / date
    return output_dir


def preprocess_config(config):
    config.exp.log_dir = extract_output_dir(config)


def setup_wandb(config):
    group, name = str(config.exp.log_dir).split("/")[-2:]
    wandb_config = omegaconf.OmegaConf.to_container(
        config, resolve=True, throw_on_missing=True
    )
    wandb.init(
        project=config.wandb.project,
        dir=config.exp.log_dir,
        group=group,
        name=name,
        entity=config.wandb.entity,
        config=wandb_config,
        sync_tensorboard=True,
    )
