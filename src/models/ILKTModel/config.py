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
        **kwargs
    ):

        self.backbone_config = dict(**backbone_config)
        self.embedding_head_config = dict(**embedding_head_config)
        self.mlm_head_config = dict(**mlm_head_config)
        self.cls_head_config = dict(**cls_head_config)
        self.cls_heads = cls_heads

        super().__init__(**kwargs)
