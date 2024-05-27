from typing import Any, Dict

from transformers import PretrainedConfig


class ILKTConfig(PretrainedConfig):

    model_type = "ILKT"

    def __init__(
        self,
        backbone_config: Dict[str, Any],
        embedding_head_config: Dict[str, Any],
        mlm_head_config: Dict[str, Any],
        **kwargs
    ):

        self.backbone_config = backbone_config
        self.embedding_head_config = embedding_head_config
        self.mlm_head_config = mlm_head_config

        super().__init__(**kwargs)
