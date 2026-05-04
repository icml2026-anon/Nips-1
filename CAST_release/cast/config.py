import os
import torch
from dataclasses import dataclass, field
from typing import Optional
import math
import yaml


@dataclass
class CASTConfig:
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    seq_len: int = 13
    input_channels: int = 1
    raw_channels: int = 1
    num_classes: int = 2

    sem_hidden_dim: int = 64
    sparsity_lambda: float = 1.0
    dag_threshold: float = 0.1
    pretrain_epochs: int = 100
    dag_rho_init: float = 1.0
    dag_alpha_init: float = 0.0
    dag_outer_iter: int = 10
    dag_inner_iter: int = 300
    sigma_z: float = 1.0

    embedding_dim: int = 32
    khop: int = 2
    lambda_g: float = 0.1
    pagerank_damping: float = 0.15

    gate_hidden_dim: int = 64
    phase_scale: float = math.pi / 4.0
    sparsity_mlp_hidden: int = 64

    kan_type: str = "taylor"
    kan_order: int = 8
    kan_hidden_dim: int = 64
    kan_num_layers: int = 3
    kan_output_dim: int = 64
    bspline_order: int = 3
    fourier_omega: float = 1.0
    jacobi_a: float = 1.0
    jacobi_b: float = 1.0

    learning_rate: float = 1e-3
    rsgd_lr: float = 1e-3
    batch_size: int = 128
    finetune_epochs: int = 100
    lambda_dag: float = 0.1
    lambda_hyp: float = 0.01
    lambda_sp: float = 0.01
    weight_decay: float = 1e-4

    focal_alpha: float = 0.7
    focal_gamma: float = 1.0
    dropout: float = 0.1

    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    ablate_no_causal: bool = False
    ablate_no_hyperbolic: bool = False
    ablate_no_sparsity_gate: bool = False
    ablate_no_causal_attn: bool = False
    ablate_no_focal: bool = False

    @staticmethod
    def resolve_yaml_path(dataset: str, task: Optional[str] = None,
                          config_dir: str = "configs") -> str:
        if task:
            name = f"{dataset}_{task}"
        else:
            name = dataset
        path = os.path.join(config_dir, f"{name}.yaml")
        return path

    @classmethod
    def from_yaml(cls, yaml_path: str, **kwargs):
        overrides = {}
        if os.path.exists(yaml_path):
            with open(yaml_path, "r") as f:
                overrides = yaml.safe_load(f) or {}
        overrides.update({k: v for k, v in kwargs.items() if v is not None})
        return cls(**overrides)

    def apply_yaml(self, yaml_path: str):
        if not os.path.exists(yaml_path):
            return
        with open(yaml_path, "r") as f:
            overrides = yaml.safe_load(f) or {}
        for k, v in overrides.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self.__post_init__()

    def __post_init__(self):
        if self.embedding_dim % 2 != 0:
            self.embedding_dim = self.embedding_dim + 1
        self.kan_input_dim = self.embedding_dim * 2
