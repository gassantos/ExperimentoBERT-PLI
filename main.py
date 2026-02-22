"""
Main Entry Point - BERT-PLI Experiment Runner
==============================================

Script principal para orquestrar a execução de experimentos.
Centraliza a execução de experimentos únicos ou grid search.

Uso simples (com defaults):
    python -m main

Uso avançado:
    python -m main --mode grid --config config/experiments/BertPLI.config
    python -m main --mode single --config config/experiments/BertPLI2.config
    python -m main --mode grid --parallel 2

Autor: Gustavo Alexandre
Data: 2026-02-17
"""

import argparse
import json
import logging
import sys
import multiprocessing
from pathlib import Path
from typing import Optional

# Deve-se usar 'spawn' para compatibilidade com CUDA
#  - https://pytorch.org/docs/stable/multiprocessing.html#best-practices
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

from gridsearch.core import run_grid_search, _LOGFILE
from utils.paths import PathManager
from utils.log_setup import setup_main_logging

# Configura logging multiprocessing-safe antes de qualquer log.
# QueueListener é iniciado aqui e parado no finally do main().
_log_listener = setup_main_logging(_LOGFILE)
logger = logging.getLogger(__name__)

# =========================
# CONFIGURAÇÕES PADRÃO
# =========================
DEFAULT_CONFIG = "config/experiments/BertPLI.config"
DEFAULT_GRID_CONFIG = "gridsearch/config/grid_search_test.json"
DEFAULT_MODE = "grid"
DEFAULT_PARALLEL = 2


def validate_paths(config_path: str, grid_config_path: Optional[str] = None) -> bool:
    """
    Valida se os caminhos de configuração existem.
    
    Args:
        config_path: Caminho do arquivo de configuração base
        grid_config_path: Caminho do arquivo de grid config (opcional)
        
    Returns:
        True se todos os arquivos existem, False caso contrário
    """
    config_file = Path(config_path)
    if not config_file.exists():
        logger.error(f"Arquivo de configuração não encontrado: {config_path}")
        return False
    
    if grid_config_path:
        grid_file = Path(grid_config_path)
        if not grid_file.exists():
            logger.error(f"Arquivo de grid config não encontrado: {grid_config_path}")
            return False
    
    return True


def run_single_experiment(config_path: str):
    """
    Executa um único experimento.
    
    Args:
        config_path: Caminho do arquivo de configuração
    """
    # Import lazy para evitar inicialização de CUDA no processo principal
    from run_experiment import execute_experiment
    
    logger.info("=" * 70)
    logger.info("MODO: Experimento Único")
    logger.info(f"Configuração: {config_path}")
    logger.info("=" * 70)
    
    if not validate_paths(config_path):
        sys.exit(1)
    
    execute_experiment(config_path)
    logger.info("Experimento concluído com sucesso!")


def run_grid_search_experiments(
    base_config_path: str,
    grid_config_path: str,
    parallel: int = 1,
    resume: bool = False
):
    """
    Executa grid search de hiperparâmetros.
    
    Args:
        base_config_path: Caminho do arquivo de configuração base
        grid_config_path: Caminho do arquivo JSON com grade de hiperparâmetros
        parallel: Número de processos paralelos
        resume: Se True, retoma execução anterior
    """
    logger.info("=" * 70)
    logger.info("MODO: Grid Search")
    logger.info(f"Configuração base: {base_config_path}")
    logger.info(f"Grid config: {grid_config_path}")
    logger.info(f"Execução: {'Paralela (' + str(parallel) + ' workers)' if parallel > 1 else 'Sequencial'}")
    logger.info("=" * 70)
    
    if not validate_paths(base_config_path, grid_config_path):
        sys.exit(1)
    
    # Carrega configuração da grade
    with open(grid_config_path, 'r', encoding='utf-8') as f:
        grid_config = json.load(f)
    
    # Executa grid search
    results = run_grid_search(
        base_config_path=base_config_path,
        grid_config=grid_config,
        resume=resume,
        parallel=parallel
    )
    
    logger.info("Grid search concluído com sucesso!")
    logger.info(f"Total de experimentos executados: {len(results)}")


def main():
    """Função principal que orquestra a execução."""
    parser = argparse.ArgumentParser(
        description="BERT-PLI Experiment Runner - Execução centralizada de experimentos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Exemplos de uso:

  # Execução padrão (grid search minimal)
  python -m main

  # Experimento único
  python -m main --mode single

  # Grid search com configuração específica
  python -m main --mode grid --grid-config gridsearch/config/grid_search_test.json

  # Grid search paralelo com 4 workers
  python -m main --mode grid --parallel 4

  # Retomar grid search interrompido
  python -m main --mode grid --resume

Configurações padrão:
  - Modo: {DEFAULT_MODE}
  - Config: {DEFAULT_CONFIG}
  - Grid config: {DEFAULT_GRID_CONFIG}
  - Parallel: {DEFAULT_PARALLEL}
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "grid"],
        default=DEFAULT_MODE,
        help=f"Modo de execução: 'single' para um único experimento, 'grid' para grid search (padrão: {DEFAULT_MODE})"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help=f"Caminho do arquivo de configuração base (padrão: {DEFAULT_CONFIG})"
    )
    
    parser.add_argument(
        "--grid-config",
        type=str,
        default=DEFAULT_GRID_CONFIG,
        help=f"Caminho do arquivo JSON com grade de hiperparâmetros (padrão: {DEFAULT_GRID_CONFIG})"
    )
    
    parser.add_argument(
        "--parallel",
        type=int,
        default=DEFAULT_PARALLEL,
        help=f"Número de processos paralelos para grid search (padrão: {DEFAULT_PARALLEL})"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Retoma execução anterior de grid search usando estado salvo"
    )
    
    args = parser.parse_args()
    
    # Print informações iniciais
    logger.info("=" * 70)
    logger.info("BERT-PLI Experiment Runner")
    logger.info("=" * 70)
    logger.info(f"Diretório base: {PathManager.BASE_DIR}")
    logger.info("")
    
    try:
        if args.mode == "single":
            run_single_experiment(args.config)
        elif args.mode == "grid":
            run_grid_search_experiments(
                base_config_path=args.config,
                grid_config_path=args.grid_config,
                parallel=args.parallel,
                resume=args.resume
            )
        else:
            parser.error(f"Modo inválido: {args.mode}")
    
    except KeyboardInterrupt:
        logger.warning("\nExecução interrompida pelo usuário")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Erro durante execução: {e}", exc_info=True)
        sys.exit(1)
    finally:
        _log_listener.stop()


if __name__ == "__main__":
    main()
