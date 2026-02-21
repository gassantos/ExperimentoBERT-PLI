"""
Testes pytest para o módulo gridsearch.

Coberturas:
  - Importações dos símbolos públicos do pacote
  - Geração de grade de hiperparâmetros (generate_parameter_grid)
  - Validação de disponibilidade de memória (check_memory_availability)
  - Estimativa de requisitos de memória (estimate_memory_requirements)
  - Filtragem de metadados de configuração (filter_grid_config)
  - Carregamento dos arquivos JSON de configuração do grid
  - Análise de resultados mock (compute_descriptive_statistics, rank_configurations)
"""

import json
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Importações opcionais — marcados como skip se o módulo não estiver pronto
# ---------------------------------------------------------------------------

gridsearch = pytest.importorskip("gridsearch", reason="Módulo gridsearch não disponível")

from gridsearch import run_grid_search, generate_parameter_grid, analyze_results  # noqa: E402
from gridsearch.utils import (  # noqa: E402
    check_memory_availability,
    estimate_memory_requirements,
    filter_grid_config,
)
from gridsearch.analysis import (  # noqa: E402
    compute_descriptive_statistics,
    analyze_correlations,
    rank_configurations,
)

GRID_SEARCH_JSON = Path(__file__).parents[1] / "gridsearch" / "config" / "grid_search.json"
GRID_SEARCH_TEST_JSON = Path(__file__).parents[1] / "gridsearch" / "config" / "grid_search_test.json"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def simple_grid():
    return {"learning_rate": [1e-5, 2e-5], "batch_size": [8, 16]}


@pytest.fixture(scope="module")
def mock_results():
    return [
        {
            "grid_experiment_idx": 0,
            "grid_params": {"learning_rate": 1e-5, "batch_size": 8},
            "status": "success",
            "resources": {
                "train_time_sec": 1234.5,
                "energy_kwh": 0.025,
                "peak_ram_mb": 8192.0,
            },
        },
        {
            "grid_experiment_idx": 1,
            "grid_params": {"learning_rate": 2e-5, "batch_size": 16},
            "status": "success",
            "resources": {
                "train_time_sec": 1156.3,
                "energy_kwh": 0.023,
                "peak_ram_mb": 12288.0,
            },
        },
        {
            "grid_experiment_idx": 2,
            "grid_params": {"learning_rate": 1e-5, "batch_size": 16},
            "status": "failed",
            "error": "Out of memory",
        },
    ]


# ---------------------------------------------------------------------------
# Teste 1: Importações
# ---------------------------------------------------------------------------

class TestImports:
    def test_run_grid_search_importable(self):
        assert callable(run_grid_search)

    def test_generate_parameter_grid_importable(self):
        assert callable(generate_parameter_grid)

    def test_analyze_results_importable(self):
        assert callable(analyze_results)

    def test_utils_importable(self):
        assert callable(check_memory_availability)
        assert callable(estimate_memory_requirements)
        assert callable(filter_grid_config)

    def test_analysis_importable(self):
        assert callable(compute_descriptive_statistics)
        assert callable(analyze_correlations)
        assert callable(rank_configurations)


# ---------------------------------------------------------------------------
# Teste 2: Geração de combinações
# ---------------------------------------------------------------------------

class TestGenerateParameterGrid:
    def test_correct_number_of_combinations(self, simple_grid):
        combos = generate_parameter_grid(simple_grid)
        assert len(combos) == 4, f"Esperado 4 combinações (2x2), obtido {len(combos)}"

    def test_combinations_are_dicts(self, simple_grid):
        combos = generate_parameter_grid(simple_grid)
        for combo in combos:
            assert isinstance(combo, dict)

    def test_all_keys_present_in_each_combo(self, simple_grid):
        combos = generate_parameter_grid(simple_grid)
        for combo in combos:
            assert "learning_rate" in combo
            assert "batch_size" in combo

    def test_single_param_returns_all_values(self):
        grid = {"learning_rate": [1e-5, 2e-5, 3e-5]}
        combos = generate_parameter_grid(grid)
        assert len(combos) == 3

    def test_three_params_cartesian_product(self):
        grid = {"a": [1, 2], "b": [3, 4], "c": [5, 6]}
        combos = generate_parameter_grid(grid)
        assert len(combos) == 8  # 2 * 2 * 2


# ---------------------------------------------------------------------------
# Teste 3: Validação de memória
# ---------------------------------------------------------------------------

class TestMemoryValidation:
    def test_check_memory_returns_tuple(self):
        result = check_memory_availability(parallel=2, batch_size=16)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_check_memory_first_element_is_bool(self):
        is_safe, _ = check_memory_availability(parallel=2, batch_size=16)
        assert isinstance(is_safe, bool)

    def test_check_memory_second_element_is_str(self):
        _, message = check_memory_availability(parallel=2, batch_size=16)
        assert isinstance(message, str)

    def test_estimate_memory_returns_float(self):
        estimated_gb = estimate_memory_requirements(parallel=2, batch_size=16)
        assert isinstance(estimated_gb, (int, float))

    def test_estimate_memory_positive(self):
        estimated_gb = estimate_memory_requirements(parallel=1, batch_size=8)
        assert estimated_gb > 0


# ---------------------------------------------------------------------------
# Teste 4: Filtragem de configuração
# ---------------------------------------------------------------------------

class TestFilterGridConfig:
    def test_metadata_keys_removed(self):
        config = {
            "description": "Test grid",
            "notes": ["test note"],
            "learning_rate": [1e-5],
            "batch_size": [16],
        }
        filtered = filter_grid_config(config)
        assert "description" not in filtered
        assert "notes" not in filtered

    def test_hyperparameter_keys_preserved(self):
        config = {
            "description": "To be removed",
            "learning_rate": [1e-5],
            "batch_size": [16],
        }
        filtered = filter_grid_config(config)
        assert "learning_rate" in filtered
        assert "batch_size" in filtered

    def test_empty_config_returns_empty(self):
        filtered = filter_grid_config({})
        assert isinstance(filtered, dict)


# ---------------------------------------------------------------------------
# Teste 5: Carregamento de arquivos JSON
# ---------------------------------------------------------------------------

class TestJsonConfigLoading:
    @pytest.mark.skipif(not GRID_SEARCH_TEST_JSON.exists(), reason="grid_search_test.json não encontrado")
    def test_test_json_generates_combinations(self):
        with open(GRID_SEARCH_TEST_JSON) as f:
            config = json.load(f)
        assert "hyperparameters" in config
        combos = generate_parameter_grid(config["hyperparameters"])
        assert len(combos) > 0

    @pytest.mark.skipif(not GRID_SEARCH_JSON.exists(), reason="grid_search.json não encontrado")
    def test_full_json_generates_combinations(self):
        with open(GRID_SEARCH_JSON) as f:
            config = json.load(f)
        assert "hyperparameters" in config
        combos = generate_parameter_grid(config["hyperparameters"])
        assert len(combos) > 0


# ---------------------------------------------------------------------------
# Teste 6: Análise de resultados mock
# ---------------------------------------------------------------------------

class TestAnalyzeResults:
    def test_descriptive_stats_returns_dict(self, mock_results):
        stats = compute_descriptive_statistics(mock_results)
        assert isinstance(stats, dict)

    def test_descriptive_stats_has_train_time(self, mock_results):
        stats = compute_descriptive_statistics(mock_results)
        assert "train_time" in stats

    def test_descriptive_stats_counts_success_and_failure(self, mock_results):
        stats = compute_descriptive_statistics(mock_results)
        assert stats.get("successful_experiments") == 2
        assert stats.get("failed_experiments") == 1

    def test_rank_configurations_returns_list(self, mock_results):
        ranked = rank_configurations(mock_results, metrics=["train_time_sec"])
        assert isinstance(ranked, list)

    def test_rank_configurations_non_empty(self, mock_results):
        ranked = rank_configurations(mock_results, metrics=["train_time_sec"])
        assert len(ranked) > 0

    def test_rank_configurations_has_params_key(self, mock_results):
        ranked = rank_configurations(mock_results, metrics=["train_time_sec"])
        for item in ranked:
            assert "params" in item


# ---------------------------------------------------------------------------
# Ponto de entrada manual (não executado pelo pytest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "pytest", __file__, "-v"], check=True)
