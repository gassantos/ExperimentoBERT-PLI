"""
Utilitários para Grid Search - BERT-PLI
========================================

Funções auxiliares para validação de recursos, estimativas
e operações comuns do grid search.

Autor: Gustavo Alexandre
Data: 2026-02-15
"""

import psutil
import logging
from typing import Tuple

from utils.paths import PathManager

logger = logging.getLogger(__name__)


def get_system_memory_info() -> Tuple[float, float, float]:
    """
    Obtém informações de memória do sistema.
    
    Returns:
        Tupla com (total_gb, available_gb, percent_used)
    """
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024 ** 3)
    available_gb = mem.available / (1024 ** 3)
    percent_used = mem.percent
    
    return total_gb, available_gb, percent_used


def estimate_memory_requirements(parallel: int, batch_size: int = 16) -> float:
    """
    Estima requisitos de memória RAM para grid search.
    
    Args:
        parallel: Número de workers paralelos
        batch_size: Tamanho do batch (afeta uso de memória)
        
    Returns:
        Memória estimada em GB
    """
    # Estimativas baseadas em BERT-base com mixed precision
    BASE_MEMORY_PER_PROCESS = {
        8: 2.0,   # batch_size 8
        16: 2.5,  # batch_size 16
        32: 3.5,  # batch_size 32
    }
    
    memory_per_worker = BASE_MEMORY_PER_PROCESS.get(batch_size, 2.5)
    system_overhead = 2.0  # SO + outros processos
    grid_search_overhead = 0.5  # Processo principal
    
    total_estimated = (parallel * memory_per_worker) + system_overhead + grid_search_overhead
    
    return total_estimated


def check_memory_availability(parallel: int, batch_size: int = 16) -> Tuple[bool, str]:
    """
    Verifica se há memória suficiente para execução.
    
    Args:
        parallel: Número de workers paralelos
        batch_size: Tamanho do batch
        
    Returns:
        Tupla (is_safe, message)
    """
    total_gb, available_gb, percent_used = get_system_memory_info()
    required_gb = estimate_memory_requirements(parallel, batch_size)
    
    # Margem de segurança de 20%
    safe_threshold = required_gb * 1.2
    
    if available_gb < safe_threshold:
        message = (
            f"AVISO DE MEMÓRIA:\n"
            f"  RAM Total: {total_gb:.1f} GB\n"
            f"  RAM Disponível: {available_gb:.1f} GB\n"
            f"  RAM Necessária (estimada): {required_gb:.1f} GB\n"
            f"  Margem de segurança: {safe_threshold:.1f} GB\n"
            f"  Uso atual: {percent_used:.1f}%\n\n"
            f"  RECOMENDAÇÃO: Reduzir --parallel para {max(1, parallel // 2)} ou executar modo sequencial"
        )
        return False, message
    else:
        message = (
            f"Memória disponível suficiente:\n"
            f"  RAM Disponível: {available_gb:.1f} GB\n"
            f"  RAM Necessária: {required_gb:.1f} GB\n"
            f"  Margem: {available_gb - required_gb:.1f} GB"
        )
        return True, message


def estimate_execution_time(
    num_experiments: int,
    parallel: int,
    minutes_per_experiment: int = 10
) -> Tuple[int, int]:
    """
    Estima tempo total de execução.
    
    Args:
        num_experiments: Número total de experimentos
        parallel: Número de workers paralelos
        minutes_per_experiment: Tempo médio por experimento
        
    Returns:
        Tupla (hours, minutes)
    """
    total_minutes = (num_experiments * minutes_per_experiment) / parallel
    hours = int(total_minutes // 60)
    minutes = int(total_minutes % 60)
    
    return hours, minutes


def filter_grid_config(config: dict) -> dict:
    """
    Filtra configuração de grid search removendo campos de metadados.
    
    Suporta dois formatos:
    1. Hiperparâmetros em campo "hyperparameters": {"hyperparameters": {"lr": [...]}}
    2. Hiperparâmetros no nível raiz: {"lr": [...], "batch": [...]}
    
    Args:
        config: Dicionário de configuração completo
        
    Returns:
        Dicionário apenas com parâmetros de busca
    """
    # Se existe campo "hyperparameters", usa ele diretamente
    if "hyperparameters" in config:
        return config["hyperparameters"]
    
    # Caso contrário, filtra metadados do nível raiz
    metadata_fields = {
        "description",
        "experiment_base",
        "output_dir",
        "parallel_workers",
        "notes",
        "recommendations"
    }
    
    # Remove metadados e mantém apenas listas (hiperparâmetros válidos)
    filtered = {
        k: v for k, v in config.items()
        if k not in metadata_fields and isinstance(v, list)
    }
    
    return filtered


def validate_grid_config(config: dict) -> Tuple[bool, str]:
    """
    Valida configuração de grid search.
    
    Args:
        config: Dicionário de configuração
        
    Returns:
        Tupla (is_valid, message)
    """
    filtered = filter_grid_config(config)
    
    if not filtered:
        return False, "Nenhum parâmetro de busca encontrado na configuração"
    
    # Verifica se todos os valores são listas
    for key, value in filtered.items():
        if not isinstance(value, list):
            return False, f"Parâmetro '{key}' deve ser uma lista de valores"
        
        if len(value) == 0:
            return False, f"Parâmetro '{key}' não pode ter lista vazia"
    
    # Calcula total de experimentos
    total = 1
    for value in filtered.values():
        total *= len(value)
    
    if total > 1000:
        message = (
            f"⚠️  AVISO: Grid search gerará {total} experimentos.\n"
            f"  Isso pode levar muito tempo. Considere reduzir o espaço de busca."
        )
        logger.warning(message)
    
    return True, f"Configuração válida: {total} experimentos serão gerados"


def create_experiment_name(params: dict, idx: int, base_name: str = "grid") -> str:
    """
    Cria nome descritivo para experimento.
    
    Args:
        params: Dicionário de parâmetros
        idx: Índice do experimento
        base_name: Nome base
        
    Returns:
        Nome formatado do experimento
    """
    param_suffix = "_".join([
        f"{k}{v}".replace(".", "").replace("-", "").replace("e", "")
        for k, v in params.items()
    ])
    
    return f"{base_name}{idx:03d}_{param_suffix}"


def ensure_output_directories():
    """Cria diretórios de saída necessários."""
    dirs = [
        PathManager.EXPERIMENTS_DIR / "grid_search",
        PathManager.EXPERIMENTS_DIR / "grid_search" / "configs",
        PathManager.EXPERIMENTS_DIR / "grid_search" / "analysis",
        PathManager.EXPERIMENTS_DIR / "metrics"
    ]
    
    for dir_path in dirs:
        PathManager.ensure_dir(dir_path)
        logger.debug(f"Diretório criado/verificado: {dir_path}")
