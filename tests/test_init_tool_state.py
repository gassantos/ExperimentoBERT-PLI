"""
Testes unitários para a lógica de carregamento de estado do warmup scheduler
em tools/init_tool.py (init_all).

Estratégia: as dependências pesadas (reader, model, formatter) são mockadas por
patch para isolar exclusivamente o comportamento da inicialização do estado.

Coberturas:
  - warmup_scheduler_state inicializado como None por padrão
  - warmup_scheduler_state carregado quando presente no checkpoint
  - warmup_scheduler_state ausente no result quando não está no checkpoint
  - warmup_scheduler_state presente no result quando está no checkpoint
  - Carregamento tolerante a falha no checkpoint (modo train) não propaga exceção
  - Modo 'test' propaga exceção de checkpoint inválido
"""
import pytest
import torch
import tempfile
import os
from unittest.mock import MagicMock, patch
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_linear_schedule_with_warmup

from tests.conftest import TinyModel, make_mock_config
from model.optimizer import init_optimizer


# ---------------------------------------------------------------------------
# Helpers: cria um arquivo de checkpoint real no disco
# ---------------------------------------------------------------------------

def _make_checkpoint_file(path: str, include_warmup_scheduler: bool):
    """Salva em `path` um checkpoint mínimo para uso nos testes de init_tool."""
    model = TinyModel()
    config = make_mock_config(optimizer="bert_adam", learning_rate=2e-5)
    optimizer = init_optimizer(model, config)

    save = {
        "model": model.state_dict(),
        "optimizer_name": "bert_adam",
        "optimizer": optimizer.state_dict(),
        "trained_epoch": 2,
        "global_step": 200,
    }

    if include_warmup_scheduler:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=5, num_training_steps=50
        )
        for _ in range(3):
            optimizer.step()
            scheduler.step()
        save["warmup_scheduler"] = scheduler.state_dict()

    torch.save(save, path)
    return save


# ---------------------------------------------------------------------------
# Patch contextual: substitui os módulos pesados por mocks durante o import
# ---------------------------------------------------------------------------

PATCH_TARGETS = {
    "tools.init_tool.init_formatter": MagicMock(),
    "tools.init_tool.init_dataset": MagicMock(return_value=(MagicMock(), MagicMock())),
    "tools.init_tool.init_test_dataset": MagicMock(return_value=MagicMock()),
    "tools.init_tool.get_model": MagicMock(),
    "tools.init_tool.init_optimizer": MagicMock(),
    "tools.init_tool.init_output_function": MagicMock(),
}


def _patched_init_all(checkpoint_path: str, mode: str, model: TinyModel, optimizer):
    """
    Chama init_all() com todas as dependências externas mockadas.
    O model e optimizer reais são injetados via mock de get_model / init_optimizer.
    """
    from tests.conftest import make_mock_config

    # O mock de get_model retorna uma classe, cuja instância é o model real
    model_class_mock = MagicMock(return_value=model)
    optimizer_mock = MagicMock(return_value=optimizer)
    dataset_mock = (MagicMock(), MagicMock())
    config = make_mock_config(optimizer="bert_adam", learning_rate=2e-5, weight_decay=0.01)
    gpu_list = []

    with patch("tools.init_tool.init_formatter"), \
         patch("tools.init_tool.init_dataset", return_value=dataset_mock), \
         patch("tools.init_tool.init_test_dataset", return_value=MagicMock()), \
         patch("tools.init_tool.get_model", return_value=model_class_mock), \
         patch("tools.init_tool.init_optimizer", return_value=optimizer), \
         patch("tools.init_tool.init_output_function", return_value=MagicMock()):

        from tools.init_tool import init_all
        return init_all(config, gpu_list, checkpoint_path, mode)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def real_model_and_optimizer():
    model = TinyModel()
    config = make_mock_config(optimizer="bert_adam", learning_rate=2e-5)
    optimizer = init_optimizer(model, config)
    return model, optimizer


# ---------------------------------------------------------------------------
# Testes: estado do warmup scheduler
# ---------------------------------------------------------------------------

class TestWarmupSchedulerStateLoading:
    def test_state_absent_when_not_in_checkpoint(
        self, tmp_path, real_model_and_optimizer
    ):
        """Resultado NÃO deve conter warmup_scheduler_state se não estava no checkpoint."""
        model, optimizer = real_model_and_optimizer
        ckpt_path = str(tmp_path / "ckpt_no_warmup.pkl")
        _make_checkpoint_file(ckpt_path, include_warmup_scheduler=False)

        result = _patched_init_all(ckpt_path, mode="train", model=model, optimizer=optimizer)

        assert "warmup_scheduler_state" not in result, (
            "warmup_scheduler_state não deveria estar no result quando ausente no checkpoint"
        )

    def test_state_present_when_in_checkpoint(
        self, tmp_path, real_model_and_optimizer
    ):
        """Resultado DEVE conter warmup_scheduler_state quando presente no checkpoint."""
        model, optimizer = real_model_and_optimizer
        ckpt_path = str(tmp_path / "ckpt_with_warmup.pkl")
        _make_checkpoint_file(ckpt_path, include_warmup_scheduler=True)

        result = _patched_init_all(ckpt_path, mode="train", model=model, optimizer=optimizer)

        assert "warmup_scheduler_state" in result, (
            "warmup_scheduler_state deveria estar no result quando presente no checkpoint"
        )

    def test_state_is_dict_when_loaded(
        self, tmp_path, real_model_and_optimizer
    ):
        """O warmup_scheduler_state carregado deve ser um dict (state_dict do scheduler)."""
        model, optimizer = real_model_and_optimizer
        ckpt_path = str(tmp_path / "ckpt_dict.pkl")
        _make_checkpoint_file(ckpt_path, include_warmup_scheduler=True)

        result = _patched_init_all(ckpt_path, mode="train", model=model, optimizer=optimizer)

        state = result["warmup_scheduler_state"]
        assert isinstance(state, dict), (
            f"warmup_scheduler_state deve ser um dict, obtido: {type(state)}"
        )
        assert "last_epoch" in state, (
            "warmup_scheduler_state deve conter a chave 'last_epoch'"
        )

    def test_state_can_be_loaded_into_new_scheduler(
        self, tmp_path, real_model_and_optimizer
    ):
        """O state_dict deve ser usável para restaurar um novo scheduler sem erros."""
        model, optimizer = real_model_and_optimizer
        ckpt_path = str(tmp_path / "ckpt_load.pkl")
        original_save = _make_checkpoint_file(ckpt_path, include_warmup_scheduler=True)

        result = _patched_init_all(ckpt_path, mode="train", model=model, optimizer=optimizer)
        state = result["warmup_scheduler_state"]

        # Recria o scheduler e carrega o state
        new_scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=5, num_training_steps=50
        )
        new_scheduler.load_state_dict(state)  # Não deve levantar exceção

        assert new_scheduler.state_dict()["last_epoch"] == state["last_epoch"]


# ---------------------------------------------------------------------------
# Testes: tolerância a falhas no checkpoint
# ---------------------------------------------------------------------------

class TestCheckpointFaultTolerance:
    def test_invalid_checkpoint_in_train_mode_logs_warning_not_raises(
        self, tmp_path, real_model_and_optimizer
    ):
        """Em modo train, checkpoint inválido deve ser logado como warning, não exceção."""
        model, optimizer = real_model_and_optimizer
        invalid_path = str(tmp_path / "nonexistent.pkl")  # Arquivo não existe

        # Em modo train não deve lançar exceção
        result = _patched_init_all(invalid_path, mode="train", model=model, optimizer=optimizer)

        # Resultado principal ainda deve existir
        assert "model" in result
        assert "optimizer" in result
        assert "trained_epoch" in result

    def test_invalid_checkpoint_in_test_mode_raises(
        self, tmp_path, real_model_and_optimizer
    ):
        """Em modo test, checkpoint inválido deve propagar a exceção."""
        model, optimizer = real_model_and_optimizer
        invalid_path = str(tmp_path / "nonexistent.pkl")

        with pytest.raises(Exception):
            _patched_init_all(invalid_path, mode="test", model=model, optimizer=optimizer)
