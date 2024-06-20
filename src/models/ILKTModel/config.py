from typing import Any, Dict, List, Tuple

from transformers import PretrainedConfig


class ILKTConfig(PretrainedConfig):

    model_type = "ILKT"

    def __init__(
        self,
        backbone_config: Dict[str, Any] = {},
        embedding_head_config: Dict[str, Any] = {},
        mlm_head_config: Dict[str, Any] = {},
        cls_head_config: Dict[str, Any] = {},
        cls_heads: List[Tuple[int, str]] = [],
        max_length: int = 512,
        **kwargs
    ):
        self.backbone_config = backbone_config
        self.embedding_head_config = embedding_head_config
        self.mlm_head_config = mlm_head_config
        self.cls_head_config = cls_head_config
        self.cls_heads = cls_heads
        self.max_length = False
        self.output_hidden_states = False

        # TODO:
        # make config a proper HF config, save max length ets, don't know how it works exactly in hf ecosystem

        super().__init__(**kwargs)
