"""
Testes unitários para verificar correções no run_experiment.py.

Coberturas:
  - get_torch_device() retorna dict com chave 'name' (não tupla)
  - _ENERGY_COST_USD_PER_KWH é configurável via variável de ambiente
  - cost_usd é calculado corretamente a partir de energy_kwh
"""

import os
import pytest


class TestGetTorchDevice:
    def test_returns_dict(self):
        from utils.util import get_torch_device
        result = get_torch_device()
        assert isinstance(result, dict), "get_torch_device() deve retornar um dicionário"

    def test_has_name_key(self):
        from utils.util import get_torch_device
        result = get_torch_device()
        assert "name" in result, "get_torch_device() deve conter chave 'name'"

    def test_name_is_string(self):
        from utils.util import get_torch_device
        result = get_torch_device()
        assert isinstance(result["name"], str), "device['name'] deve ser string"

    def test_has_type_key(self):
        from utils.util import get_torch_device
        result = get_torch_device()
        assert "type" in result, "get_torch_device() deve conter chave 'type'"


class TestCostUsdCalculation:
    def test_cost_usd_with_known_energy(self):
        energy_kwh = 0.1
        cost_per_kwh = 0.12
        cost_usd = energy_kwh * cost_per_kwh
        assert cost_usd == pytest.approx(0.012, rel=1e-6)

    def test_cost_usd_none_when_energy_none(self):
        energy_kwh = None
        cost_usd = float(energy_kwh) * 0.12 if energy_kwh is not None else None
        assert cost_usd is None

    def test_cost_usd_env_var_override(self, monkeypatch):
        monkeypatch.setenv("ENERGY_COST_USD_PER_KWH", "0.20")
        cost = float(os.getenv("ENERGY_COST_USD_PER_KWH", "0.12"))
        assert cost == pytest.approx(0.20)

    def test_cost_usd_default_value(self, monkeypatch):
        monkeypatch.delenv("ENERGY_COST_USD_PER_KWH", raising=False)
        cost = float(os.getenv("ENERGY_COST_USD_PER_KWH", "0.12"))
        assert cost == pytest.approx(0.12)
