"""
Testes unitários para a lógica do warmup scheduler introduzida em tools/train_tool.py
durante a migração D1 (BertAdam → AdamW).

Coberturas:
  - Cálculo correto de total_steps e num_warmup_steps
  - Tipo do scheduler retornado é compatível com PyTorch LRScheduler
  - LR durante warmup aumenta monotonicamente (fase de aquecimento)
  - LR após warmup decai monotonicamente (fase de decaimento)
  - warmup_ratio=0 resulta em zero passos de warmup
  - Restauração do state_dict do scheduler não altera o LR corrente
"""
import pytest
import torch
from transformers import get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR

from tests.conftest import TinyModel, make_mock_config
from model.optimizer import init_optimizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scheduler(num_warmup_steps: int, num_training_steps: int, lr: float = 2e-5):
    model = TinyModel()
    config = make_mock_config(optimizer="bert_adam", learning_rate=lr)
    optimizer = init_optimizer(model, config)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return scheduler, optimizer


# ---------------------------------------------------------------------------
# Cálculo de steps (lógica espelhada de train_tool.py)
# ---------------------------------------------------------------------------

class TestStepsCalculation:
    @pytest.mark.parametrize("dataset_len, remaining_epochs, warmup_ratio, expected_warmup", [
        (100, 5,  0.1,   50),   # 100 * 5 * 0.1 = 50
        (200, 3,  0.15,  90),   # 200 * 3 * 0.15 = 90
        (50,  10, 0.0,    0),   # warmup_ratio = 0 → 0 warmup steps
        (80,  4,  0.2,   64),   # 80 * 4 * 0.2 = 64
    ])
    def test_warmup_steps_formula(self, dataset_len, remaining_epochs, warmup_ratio, expected_warmup):
        """
        A fórmula em train_tool.py é:
            total_steps      = len(dataset) * (epoch - (trained_epoch + 1))
            num_warmup_steps = int(warmup_ratio * total_steps)
        """
        total_steps = dataset_len * remaining_epochs
        num_warmup_steps = int(warmup_ratio * total_steps)
        assert num_warmup_steps == expected_warmup, (
            f"warmup_steps={num_warmup_steps}, esperado={expected_warmup}"
        )

    def test_zero_warmup_ratio_yields_zero_warmup_steps(self):
        total_steps = 500
        warmup_ratio = 0.0
        assert int(warmup_ratio * total_steps) == 0

    def test_full_warmup_ratio_yields_total_steps(self):
        total_steps = 100
        warmup_ratio = 1.0
        assert int(warmup_ratio * total_steps) == total_steps


# ---------------------------------------------------------------------------
# Tipo do scheduler
# ---------------------------------------------------------------------------

class TestSchedulerType:
    def test_scheduler_is_pytorch_lr_scheduler(self):
        scheduler, _ = _make_scheduler(num_warmup_steps=10, num_training_steps=100)
        assert isinstance(scheduler, LambdaLR), (
            "get_linear_schedule_with_warmup deve retornar uma LambdaLR"
        )

    def test_scheduler_has_state_dict(self):
        scheduler, _ = _make_scheduler(num_warmup_steps=5, num_training_steps=50)
        state = scheduler.state_dict()
        assert isinstance(state, dict)
        assert "last_epoch" in state


# ---------------------------------------------------------------------------
# Comportamento de LR durante warmup e decaimento
# ---------------------------------------------------------------------------

class TestSchedulerLRBehavior:
    def test_lr_increases_during_warmup(self):
        """Durante o warmup, cada passo deve aumentar (ou manter) o LR."""
        num_warmup = 10
        num_training = 100
        scheduler, optimizer = _make_scheduler(num_warmup, num_training)

        lrs = []
        for _ in range(num_warmup):
            lrs.append(scheduler.get_last_lr()[0])
            optimizer.step()
            scheduler.step()

        for i in range(1, len(lrs)):
            assert lrs[i] >= lrs[i - 1], (
                f"LR caiu durante warmup: passo {i-1}={lrs[i-1]:.2e} → passo {i}={lrs[i]:.2e}"
            )

    def test_lr_decays_after_warmup(self):
        """Após o warmup, o LR deve decair a cada passo."""
        num_warmup = 5
        num_training = 20
        scheduler, optimizer = _make_scheduler(num_warmup, num_training)

        # Avança pelo período de warmup
        for _ in range(num_warmup):
            optimizer.step()
            scheduler.step()

        # Coleta LRs no período de decaimento
        lrs = []
        for _ in range(num_warmup, num_training):
            lrs.append(scheduler.get_last_lr()[0])
            optimizer.step()
            scheduler.step()

        for i in range(1, len(lrs)):
            assert lrs[i] <= lrs[i - 1], (
                f"LR aumentou após warmup: passo {i-1}={lrs[i-1]:.2e} → passo {i}={lrs[i]:.2e}"
            )

    def test_lr_reaches_zero_at_end(self):
        """O LR deve ser aproximadamente 0 ao final do treinamento."""
        num_warmup = 2
        num_training = 10
        scheduler, optimizer = _make_scheduler(num_warmup, num_training)

        for _ in range(num_training):
            optimizer.step()
            scheduler.step()

        final_lr = scheduler.get_last_lr()[0]
        assert final_lr == pytest.approx(0.0, abs=1e-10), (
            f"LR ao final deveria ser ~0, obtido={final_lr:.2e}"
        )


# ---------------------------------------------------------------------------
# Restauração de estado
# ---------------------------------------------------------------------------

class TestSchedulerStateRestoration:
    def test_state_dict_restore_preserves_lr(self):
        """Após restaurar state_dict, o LR corrente deve ser igual ao original."""
        num_warmup = 5
        num_training = 50
        scheduler, optimizer = _make_scheduler(num_warmup, num_training)

        # Avança 8 passos
        for _ in range(8):
            optimizer.step()
            scheduler.step()

        original_lr = scheduler.get_last_lr()[0]
        saved_state = scheduler.state_dict()

        # Cria scheduler "zerado" e restaura
        scheduler2, _ = _make_scheduler(num_warmup, num_training)
        scheduler2.load_state_dict(saved_state)

        assert scheduler2.get_last_lr()[0] == pytest.approx(original_lr), (
            "LR após restaurar state_dict diverge do LR original"
        )

    def test_state_dict_restore_preserves_last_epoch(self):
        num_warmup = 3
        num_training = 30
        scheduler, optimizer = _make_scheduler(num_warmup, num_training)

        for _ in range(6):
            optimizer.step()
            scheduler.step()

        saved_state = scheduler.state_dict()

        scheduler2, _ = _make_scheduler(num_warmup, num_training)
        scheduler2.load_state_dict(saved_state)

        assert scheduler2.state_dict()["last_epoch"] == saved_state["last_epoch"]
