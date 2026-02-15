"""
Grid Search Module - BERT-PLI
==============================

Módulo completo para busca em grade de hiperparâmetros com recursos avançados:
- Execução paralela configurável
- Salvamento incremental de estado
- Análise estatística de resultados
- Recuperação automática de falhas
- Métricas de recursos computacionais

Componentes principais:
- core: Execução do grid search
- analysis: Análise e visualização de resultados
- config: Configurações de espaço de busca
- scripts: Scripts auxiliares de execução

Autor: Gustavo Alexandre
Data: 2026-02-15
"""

__version__ = "1.0.0"
__author__ = "BERT-PLI Team"

from .core import (
    run_grid_search,
    generate_parameter_grid,
    create_config_for_combination,
    run_single_experiment,
    analyze_results,
    generate_summary_report
)

__all__ = [
    'run_grid_search',
    'generate_parameter_grid',
    'create_config_for_combination',
    'run_single_experiment',
    'analyze_results',
    'generate_summary_report'
]
