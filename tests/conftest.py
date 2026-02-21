"""
Fixtures compartilhadas para os testes unitários do projeto ExperimentoBERT-PLI.
"""
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Modelo mínimo para ser usado nos testes de otimizador, checkpoint, etc.
# ---------------------------------------------------------------------------

class TinyModel(nn.Module):
    """Modelo PyTorch mínimo — apenas um Linear — para uso nos testes."""

    def __init__(self, in_features: int = 8, out_features: int = 2):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def tiny_model() -> TinyModel:
    return TinyModel()


# ---------------------------------------------------------------------------
# Fábrica de configs mock
# ---------------------------------------------------------------------------

def make_mock_config(
    optimizer: str = "adam",
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    warmup_ratio: float = 0.1,
    step_size: int = 1,
    lr_multiplier: float = 0.9,
    epoch: int = 10,
    output_time: int = 1,
    test_time: int = 1,
    model_path: str = "/tmp/models",
    model_name: str = "test_model",
    tensorboard_path: str = "/tmp/tb",
    delimiter: str = " ",
) -> MagicMock:
    """Retorna um mock de ConfigParser com os valores especificados."""
    _str = {
        ("train", "optimizer"): optimizer,
        ("output", "model_path"): model_path,
        ("output", "model_name"): model_name,
        ("output", "tensorboard_path"): tensorboard_path,
        ("output", "delimiter"): delimiter,
    }
    _float = {
        ("train", "learning_rate"): learning_rate,
        ("train", "weight_decay"): weight_decay,
        ("train", "warmup_ratio"): warmup_ratio,
        ("train", "lr_multiplier"): lr_multiplier,
    }
    _int = {
        ("train", "step_size"): step_size,
        ("train", "epoch"): epoch,
        ("output", "output_time"): output_time,
        ("output", "test_time"): test_time,
    }

    config = MagicMock()
    # Retorna valor mapeado ou string sentinela para chaves não configuradas,
    # evitando KeyError quando init_all acessa seções como ("model", "model_name").
    config.get.side_effect = lambda section, key: _str.get((section, key), f"mock_{section}_{key}")
    config.getfloat.side_effect = lambda section, key: _float[(section, key)]
    config.getint.side_effect = lambda section, key: _int[(section, key)]
    return config


@pytest.fixture
def mock_config():
    """Config padrão com optimizer=adam."""
    return make_mock_config()


@pytest.fixture
def bert_adam_config():
    """Config com optimizer=bert_adam e warmup_ratio=0.1."""
    return make_mock_config(optimizer="bert_adam", learning_rate=2e-5, weight_decay=0.01)
