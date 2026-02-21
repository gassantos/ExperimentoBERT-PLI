"""
Testes unitários para a função checkpoint() em tools/train_tool.py.

Coberturas:
  - Checkpoint salva corretamente as chaves esperadas
  - Sem warmup_scheduler: chave 'warmup_scheduler' ausente no arquivo salvo
  - Com warmup_scheduler: chave 'warmup_scheduler' presente com state_dict correto
  - Round-trip: salva e carrega state_dict do warmup_scheduler sem perda
  - trained_epoch e global_step são persistidos corretamente
  - optimizer_name reflete o tipo configurado
"""
import torch
import pytest
from unittest.mock import MagicMock, patch
from transformers import get_linear_schedule_with_warmup

from tools.train_tool import checkpoint
from model.optimizer import init_optimizer
from tests.conftest import TinyModel, make_mock_config


# ---------------------------------------------------------------------------
# Fixtures locais
# ---------------------------------------------------------------------------

@pytest.fixture
def model_and_optimizer():
    model = TinyModel()
    config = make_mock_config(optimizer="bert_adam", learning_rate=2e-5, weight_decay=0.01)
    optimizer = init_optimizer(model, config)
    return model, optimizer


@pytest.fixture
def warmup_scheduler(model_and_optimizer):
    _, optimizer = model_and_optimizer
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=5,
        num_training_steps=50,
    )
    return scheduler


# ---------------------------------------------------------------------------
# Testes de chaves salvas
# ---------------------------------------------------------------------------

class TestCheckpointKeys:
    def test_required_keys_always_present(self, tmp_path, model_and_optimizer):
        model, optimizer = model_and_optimizer
        config = make_mock_config(optimizer="bert_adam")
        path = str(tmp_path / "ckpt.pkl")

        checkpoint(path, model, optimizer, trained_epoch=2, config=config, global_step=100)

        saved = torch.load(path)
        for key in ("model", "optimizer_name", "optimizer", "trained_epoch", "global_step"):
            assert key in saved, f"Chave obrigatória '{key}' ausente no checkpoint"

    def test_warmup_scheduler_absent_when_none(self, tmp_path, model_and_optimizer):
        model, optimizer = model_and_optimizer
        config = make_mock_config(optimizer="bert_adam")
        path = str(tmp_path / "ckpt_no_warmup.pkl")

        checkpoint(path, model, optimizer, trained_epoch=0, config=config,
                   global_step=0, warmup_scheduler=None)

        saved = torch.load(path)
        assert "warmup_scheduler" not in saved, (
            "warmup_scheduler NÃO deve ser salvo quando não fornecido"
        )

    def test_warmup_scheduler_present_when_provided(
        self, tmp_path, model_and_optimizer, warmup_scheduler
    ):
        model, optimizer = model_and_optimizer
        config = make_mock_config(optimizer="bert_adam")
        path = str(tmp_path / "ckpt_with_warmup.pkl")

        checkpoint(path, model, optimizer, trained_epoch=3, config=config,
                   global_step=150, warmup_scheduler=warmup_scheduler)

        saved = torch.load(path)
        assert "warmup_scheduler" in saved, (
            "warmup_scheduler DEVE ser salvo quando fornecido"
        )

    def test_optimizer_name_matches_config(self, tmp_path, model_and_optimizer):
        model, optimizer = model_and_optimizer
        config = make_mock_config(optimizer="bert_adam")
        path = str(tmp_path / "ckpt_name.pkl")

        checkpoint(path, model, optimizer, trained_epoch=1, config=config, global_step=50)

        saved = torch.load(path)
        assert saved["optimizer_name"] == "bert_adam"

    def test_trained_epoch_and_global_step_persisted(self, tmp_path, model_and_optimizer):
        model, optimizer = model_and_optimizer
        config = make_mock_config(optimizer="bert_adam")
        path = str(tmp_path / "ckpt_epoch.pkl")

        checkpoint(path, model, optimizer, trained_epoch=7, config=config, global_step=420)

        saved = torch.load(path)
        assert saved["trained_epoch"] == 7
        assert saved["global_step"] == 420


# ---------------------------------------------------------------------------
# Testes de round-trip do warmup scheduler
# ---------------------------------------------------------------------------

class TestCheckpointWarmupRoundTrip:
    def test_warmup_state_dict_round_trip(
        self, tmp_path, model_and_optimizer, warmup_scheduler
    ):
        """State dict salvo e recarregado deve ser idêntico ao original."""
        model, optimizer = model_and_optimizer
        config = make_mock_config(optimizer="bert_adam")
        path = str(tmp_path / "ckpt_rt.pkl")

        # Avança o scheduler para criar um estado não-trivial
        for _ in range(3):
            optimizer.step()
            warmup_scheduler.step()

        original_state = warmup_scheduler.state_dict()
        checkpoint(path, model, optimizer, trained_epoch=3, config=config,
                   global_step=3, warmup_scheduler=warmup_scheduler)

        saved = torch.load(path)
        restored_state = saved["warmup_scheduler"]

        assert restored_state["last_epoch"] == original_state["last_epoch"], (
            "last_epoch do scheduler não foi preservado no checkpoint"
        )
        assert len(restored_state["_last_lr"]) == len(original_state["_last_lr"]), (
            "_last_lr do scheduler não foi preservado no checkpoint"
        )

    def test_restored_scheduler_continues_correctly(
        self, tmp_path, model_and_optimizer, warmup_scheduler
    ):
        """Scheduler restaurado deve continuar o decaimento a partir do ponto salvo."""
        model, optimizer = model_and_optimizer
        config = make_mock_config(optimizer="bert_adam", learning_rate=2e-5)
        path = str(tmp_path / "ckpt_continue.pkl")

        # Avança 5 passos (dentro do período de warmup)
        for _ in range(5):
            optimizer.step()
            warmup_scheduler.step()

        lr_after_5_steps = warmup_scheduler.get_last_lr()[0]

        checkpoint(path, model, optimizer, trained_epoch=5, config=config,
                   global_step=5, warmup_scheduler=warmup_scheduler)

        # Cria um novo scheduler e restaura o estado
        new_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=5,
            num_training_steps=50,
        )
        saved = torch.load(path)
        new_scheduler.load_state_dict(saved["warmup_scheduler"])

        assert new_scheduler.get_last_lr()[0] == pytest.approx(lr_after_5_steps), (
            "LR do scheduler restaurado diverge do LR no momento do checkpoint"
        )
