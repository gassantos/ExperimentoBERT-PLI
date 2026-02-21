"""
Testes para a configuração do grid search (gridsearch/config/grid_search.json).

Coberturas:
  - Arquivo JSON existe e é válido
  - Chave 'hyperparameters' presente
  - Chave 'optimizer' presente nos hiperparâmetros
  - 'bert_adam' (com underscore) está na lista de otimizadores — fix D1
  - 'bertadam' (sem underscore, nome antigo com bug) NÃO está na lista
  - Todos os valores de optimizer são strings reconhecidas por init_optimizer
"""
import json
import pytest
from pathlib import Path


GRID_SEARCH_CONFIG = Path(__file__).parents[1] / "gridsearch" / "config" / "grid_search.json"

KNOWN_OPTIMIZER_TYPES = {"adam", "adamw", "sgd", "bert_adam"}


@pytest.fixture(scope="module")
def grid_config() -> dict:
    assert GRID_SEARCH_CONFIG.exists(), (
        f"Arquivo de configuração do grid search não encontrado: {GRID_SEARCH_CONFIG}"
    )
    with open(GRID_SEARCH_CONFIG, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def optimizer_values(grid_config) -> list:
    assert "hyperparameters" in grid_config, (
        "Chave 'hyperparameters' não encontrada no grid_search.json"
    )
    assert "optimizer" in grid_config["hyperparameters"], (
        "Chave 'optimizer' não encontrada em hyperparameters"
    )
    return grid_config["hyperparameters"]["optimizer"]


# ---------------------------------------------------------------------------
# Estrutura do arquivo
# ---------------------------------------------------------------------------

class TestGridSearchConfigStructure:
    def test_file_is_valid_json(self, grid_config):
        assert isinstance(grid_config, dict), "grid_search.json deve ser um objeto JSON"

    def test_hyperparameters_key_present(self, grid_config):
        assert "hyperparameters" in grid_config

    def test_optimizer_key_present_in_hyperparameters(self, grid_config):
        assert "optimizer" in grid_config["hyperparameters"]

    def test_optimizer_is_a_list(self, optimizer_values):
        assert isinstance(optimizer_values, list), (
            "O campo 'optimizer' no grid deve ser uma lista de strings"
        )

    def test_optimizer_list_is_non_empty(self, optimizer_values):
        assert len(optimizer_values) > 0


# ---------------------------------------------------------------------------
# Correção do bug de naming (bert_adam vs bertadam)
# ---------------------------------------------------------------------------

class TestOptimizerNamingFix:
    def test_bert_adam_with_underscore_present(self, optimizer_values):
        """Após o fix D1, 'bert_adam' (com underscore) deve estar na lista."""
        assert "bert_adam" in optimizer_values, (
            f"'bert_adam' não encontrado em {optimizer_values}. "
            "Verificar fix aplicado em gridsearch/config/grid_search.json."
        )

    def test_bertadam_without_underscore_absent(self, optimizer_values):
        """O nome com bug ('bertadam', sem underscore) NÃO deve estar na lista."""
        assert "bertadam" not in optimizer_values, (
            f"'bertadam' (nome com bug) ainda presente em {optimizer_values}."
        )

    def test_no_duplicate_bert_adam(self, optimizer_values):
        bert_adam_occurrences = optimizer_values.count("bert_adam")
        assert bert_adam_occurrences == 1, (
            f"'bert_adam' aparece {bert_adam_occurrences} vez(es), esperado 1"
        )


# ---------------------------------------------------------------------------
# Todos os valores de optimizer são reconhecidos
# ---------------------------------------------------------------------------

class TestOptimizerValuesRecognized:
    def test_all_optimizer_values_are_known(self, optimizer_values):
        unknown = [v for v in optimizer_values if v not in KNOWN_OPTIMIZER_TYPES]
        assert not unknown, (
            f"Optimizer(s) desconhecido(s) no grid config: {unknown}. "
            f"Valores conhecidos: {KNOWN_OPTIMIZER_TYPES}"
        )

    @pytest.mark.parametrize("expected_opt", ["adam", "adamw", "sgd", "bert_adam"])
    def test_standard_optimizers_all_present(self, optimizer_values, expected_opt):
        assert expected_opt in optimizer_values, (
            f"Optimizer '{expected_opt}' ausente no grid config"
        )
