from typing import Any

from omegaconf import DictConfig
from transformers import PretrainedConfig


class Config(PretrainedConfig):
    model_type = "nuclei_graph_transformer"

    def __init__(
        self,
        ffn: DictConfig,
        self_attn: DictConfig,
        node_features: int,
        dim: int,
        hidden_dim: int,
        pos_dim: int,
        num_heads: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
        layer_scale_init: float,
        **kwargs: Any,
    ) -> None:
        self.ffn = ffn
        self.self_attn = self_attn
        self.node_features = node_features
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.pos_dim = pos_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.layer_scale_init = layer_scale_init
        super().__init__(**kwargs)
