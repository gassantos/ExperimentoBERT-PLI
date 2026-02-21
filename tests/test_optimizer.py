"""
Testes unitários para model/optimizer.py (init_optimizer).

Coberturas:
  - Tipo correto do otimizador para cada variante suportada
  - bert_adam usa AdamW do pacote `transformers` (não pytorch_pretrained_bert)
  - Learning rate e weight_decay são transmitidos corretamente
  - Otimizador desconhecido levanta NotImplementedError
"""
import pytest
import torch.optim as optim

from model.optimizer import init_optimizer
from tests.conftest import TinyModel, make_mock_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make(optimizer_type: str, lr: float = 1e-3, wd: float = 0.01):
    model = TinyModel()
    config = make_mock_config(optimizer=optimizer_type, learning_rate=lr, weight_decay=wd)
    return init_optimizer(model, config), model, config


# ---------------------------------------------------------------------------
# Testes de tipo
# ---------------------------------------------------------------------------

class TestOptimizerType:
    def test_adam_returns_torch_adam(self):
        opt, *_ = _make("adam")
        assert isinstance(opt, optim.Adam), (
            "optimizer='adam' deve retornar torch.optim.Adam"
        )

    def test_adamw_returns_torch_adamw(self):
        opt, *_ = _make("adamw")
        assert isinstance(opt, optim.AdamW), (
            "optimizer='adamw' deve retornar torch.optim.AdamW"
        )

    def test_sgd_returns_torch_sgd(self):
        opt, *_ = _make("sgd")
        assert isinstance(opt, optim.SGD), (
            "optimizer='sgd' deve retornar torch.optim.SGD"
        )

    def test_bert_adam_returns_torch_adamw(self):
        """bert_adam deve retornar torch.optim.AdamW (transformers.AdamW removido em >=4.46)."""
        opt, *_ = _make("bert_adam", lr=2e-5, wd=0.01)
        assert isinstance(opt, optim.AdamW), (
            "optimizer='bert_adam' deve retornar torch.optim.AdamW"
        )

    def test_bert_adam_not_pytorch_pretrained_bert(self):
        """Garantia de que pytorch_pretrained_bert.BertAdam NÃO é mais usado."""
        opt, *_ = _make("bert_adam")
        # Se a importação antiga ainda fosse usada, a classe seria registrada
        # num módulo diferente.
        assert "pytorch_pretrained_bert" not in type(opt).__module__, (
            "bert_adam não deve ser originário de pytorch_pretrained_bert"
        )

    def test_unknown_optimizer_raises_not_implemented(self):
        model = TinyModel()
        config = make_mock_config(optimizer="nonexistent_opt")
        with pytest.raises(NotImplementedError):
            init_optimizer(model, config)


# ---------------------------------------------------------------------------
# Testes de hiperparâmetros
# ---------------------------------------------------------------------------

class TestOptimizerHyperparams:
    @pytest.mark.parametrize("opt_type", ["adam", "adamw", "sgd", "bert_adam"])
    def test_learning_rate_is_applied(self, opt_type):
        lr = 3e-4
        opt, *_ = _make(opt_type, lr=lr)
        actual_lr = opt.param_groups[0]["lr"]
        assert actual_lr == pytest.approx(lr), (
            f"LR esperado={lr}, obtido={actual_lr} para optimizer='{opt_type}'"
        )

    @pytest.mark.parametrize("opt_type", ["adam", "adamw", "sgd", "bert_adam"])
    def test_weight_decay_is_applied(self, opt_type):
        wd = 0.05
        opt, *_ = _make(opt_type, wd=wd)
        actual_wd = opt.param_groups[0]["weight_decay"]
        assert actual_wd == pytest.approx(wd), (
            f"weight_decay esperado={wd}, obtido={actual_wd} para optimizer='{opt_type}'"
        )

    def test_bert_adam_lr_matches_config(self, bert_adam_config):
        model = TinyModel()
        opt = init_optimizer(model, bert_adam_config)
        assert opt.param_groups[0]["lr"] == pytest.approx(2e-5)

    def test_bert_adam_weight_decay_matches_config(self, bert_adam_config):
        model = TinyModel()
        opt = init_optimizer(model, bert_adam_config)
        assert opt.param_groups[0]["weight_decay"] == pytest.approx(0.01)
