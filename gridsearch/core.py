"""
Grid Search Core - BERT-PLI
============================

Motor de execução para busca em grade de hiperparâmetros.
Versão modularizada com validações de memória integradas.

Uso como módulo:
    from gridsearch import run_grid_search
    
Uso CLI:
    python -m gridsearch.core --config config/experiments/BertPLI.config \
                              --search-config gridsearch/config/grid_search.json \
                              --parallel 2

Autor: Gustavo Alexandre
Data: 2026-02-15
"""

import argparse
import json
import logging
import os
import sys
# import multiprocessing
import configparser
import itertools
from datetime import datetime
from typing import Dict, List, Any
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils.device import get_torch_device
from utils.paths import PathManager
from utils.log_setup import setup_worker_logging, _LOG_QUEUE
from .utils import (
    check_memory_availability,
    filter_grid_config,
    ensure_output_directories
)

_TDATE = datetime.now().strftime("%Y-%m-%d")
_LOGFILE = PathManager.LOGS_DIR / f"grid_search_{_TDATE}.log"
device_type = get_torch_device()['type']

# Logging configurado via setup_main_logging() em run_grid_search().
# Não chamamos basicConfig aqui para evitar dupla inicialização nos workers.
logger = logging.getLogger(__name__)


# Diretórios
GRID_OUTPUT_DIR = PathManager.EXPERIMENTS_DIR / "grid_search"
GRID_CONFIGS_DIR = GRID_OUTPUT_DIR / "configs"
GRID_STATE_FILE = GRID_OUTPUT_DIR / f"grid_search_state_{device_type}_{_TDATE}.json"
GRID_RESULTS_FILE = GRID_OUTPUT_DIR / f"grid_search_results_{device_type}_{_TDATE}.json"
GRID_SUMMARY_FILE = GRID_OUTPUT_DIR / f"grid_search_summary_{device_type}_{_TDATE}.txt"

# Configurações de custo
# Tarifa média de energia em USD/kWh (pode ser configurada via variável de ambiente)
ENERGY_COST_USD_PER_KWH = float(os.getenv("ENERGY_COST_USD_PER_KWH", "0.12"))

# Criar diretórios
ensure_output_directories()


# ============================================================================
# GERAÇÃO DE COMBINAÇÕES
# ============================================================================

def generate_parameter_grid(grid_config: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """
    Gera todas as combinações possíveis de hiperparâmetros.
    
    Args:
        grid_config: Dicionário com listas de valores para cada hiperparâmetro
        
    Returns:
        Lista de dicionários, cada um representando uma combinação única
        
    Exemplo:
        >>> grid = {
        ...     "learning_rate": [1e-5, 2e-5],
        ...     "batch_size": [8, 16]
        ... }
        >>> generate_parameter_grid(grid)
        [
            {"learning_rate": 1e-5, "batch_size": 8},
            {"learning_rate": 1e-5, "batch_size": 16},
            {"learning_rate": 2e-5, "batch_size": 8},
            {"learning_rate": 2e-5, "batch_size": 16}
        ]
    """
    # Filtra metadados da configuração
    grid_config = filter_grid_config(grid_config)
    
    keys = list(grid_config.keys())
    values = list(grid_config.values())
    
    # Gera produto cartesiano
    combinations = list(itertools.product(*values))
    
    # Converte para lista de dicionários
    param_grid = []
    for combo in combinations:
        param_dict = dict(zip(keys, combo))
        param_grid.append(param_dict)
    
    logger.info(f"Geradas {len(param_grid)} combinações de hiperparâmetros")
    return param_grid


def create_config_for_combination(
    base_config_path: str,
    params: Dict[str, Any],
    experiment_idx: int
) -> str:
    """
    Cria um arquivo de configuração específico para uma combinação de parâmetros.
    
    Args:
        base_config_path: Caminho do arquivo de configuração base
        params: Dicionário com os parâmetros a serem modificados
        experiment_idx: Índice do experimento na grade
        
    Returns:
        Caminho do novo arquivo de configuração criado
    """
    config = configparser.ConfigParser()
    config.read(base_config_path)
    
    # Atualiza seção [train] com hiperparâmetros
    if "learning_rate" in params:
        config.set("train", "learning_rate", str(params["learning_rate"]))
    
    if "batch_size" in params:
        config.set("train", "batch_size", str(params["batch_size"]))
    
    if "optimizer" in params:
        config.set("train", "optimizer", params["optimizer"])
    
    if "dropout" in params:
        config.set("model", "dropout", str(params["dropout"]))
    
    if "seed" in params:
        config.set("experiment", "seed", str(params["seed"]))
    
    # Atualiza nome do experimento
    base_name = config.get("experiment", "name")
    
    # Gera nome descritivo
    param_suffix = "_".join([
        f"{k}{v}".replace(".", "").replace("-", "")
        for k, v in params.items()
    ])
    
    new_name = f"{base_name}_grid{experiment_idx:03d}_{param_suffix}"
    config.set("experiment", "name", new_name)
    
    # Atualiza descrição
    description = f"Grid Search Experiment {experiment_idx}\n"
    description += "Hyperparameters:\n"
    for k, v in params.items():
        description += f"  - {k}: {v}\n"
    config.set("experiment", "description", description)
    
    # Salva nova configuração
    new_config_path = GRID_CONFIGS_DIR / f"grid_exp_{experiment_idx:03d}.config"
    with open(new_config_path, 'w') as f:
        config.write(f)
    
    logger.debug(f"Config criada: {new_config_path}")
    return str(new_config_path)


# ============================================================================
# EXECUÇÃO DE EXPERIMENTOS
# ============================================================================

def run_single_experiment(
    experiment_idx: int,
    config_path: str,
    params: Dict[str, Any],
    gpu_list: List[int] | None = None,
) -> Dict[str, Any]:
    """
    Executa um único experimento e retorna os resultados.

    Args:
        experiment_idx: Índice do experimento
        config_path: Caminho do arquivo de configuração
        params: Parâmetros do experimento
        gpu_list: GPUs a utilizar (ex: [0] ou [1]). None = detecta automaticamente.

    Returns:
        Dicionário com resultados do experimento
    """
    # Import lazy para evitar inicialização de CUDA no processo principal
    from run_experiment import execute_experiment
    
    logger.info(f"[{experiment_idx}] Iniciando experimento com parâmetros: {params}")
    
    try:
        # Executa experimento nas GPUs designadas
        execute_experiment(config_path, gpu_list=gpu_list)
        
        # Coleta resultados do arquivo JSON mais recente gerado
        metrics_dir = PathManager.BASE_DIR / "output" / "experiments" / "metrics"
        json_files = sorted(metrics_dir.glob("*.json"), key=os.path.getmtime)
        
        if not json_files:
            raise FileNotFoundError("Nenhum arquivo de resultados encontrado")
        
        latest_result = json_files[-1]
        with open(latest_result, 'r') as f:
            result_data = json.load(f)
        
        # Adiciona parâmetros ao resultado
        result_data["grid_params"] = params
        result_data["grid_experiment_idx"] = experiment_idx
        result_data["status"] = "success"
        
        logger.info(f"[{experiment_idx}] Experimento concluído com sucesso")
        return result_data
        
    except Exception as e:
        logger.error(f"[{experiment_idx}] Erro no experimento: {str(e)}")
        logger.debug(traceback.format_exc())
        
        return {
            "grid_experiment_idx": experiment_idx,
            "grid_params": params,
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def run_grid_search(
    base_config_path: str,
    grid_config: Dict[str, List[Any]],
    resume: bool = False,
    parallel: int = 1,
    gpu_ids: List[int] | None = None,
) -> List[Dict[str, Any]]:
    """
    Executa busca em grade completa.

    Args:
        base_config_path: Caminho da configuração base
        grid_config: Configuração da grade de hiperparâmetros
        resume: Se True, retoma execução anterior
        parallel: Número de processos paralelos (1 = sequencial)
        gpu_ids: Lista explícita de GPUs disponíveis para distribuição
                 round-robin entre workers (ex: [0, 1, 2, 3]).
                 None = detecta automaticamente via torch.cuda.

    Returns:
        Lista com resultados de todos os experimentos
    """
    # Carrega estado anterior se existir
    completed_experiments = set()
    all_results = []
    
    if resume and GRID_STATE_FILE.exists():
        logger.info("Retomando execução anterior...")
        with open(GRID_STATE_FILE, 'r', encoding='utf-8') as f:
            state = json.load(f)
            completed_experiments = set(state.get("completed_experiments", []))
            all_results = state.get("results", [])
        logger.info(f"Encontrados {len(completed_experiments)} experimentos já concluídos")
    
    # Gera grade de parâmetros
    param_grid = generate_parameter_grid(grid_config)
    total_experiments = len(param_grid)
    
    logger.info(f"Total de experimentos: {total_experiments}")
    logger.info(f"Já concluídos: {len(completed_experiments)}")
    logger.info(f"Restantes: {total_experiments - len(completed_experiments)}")
    
    # Prepara experimentos pendentes
    pending_experiments = []
    for idx, params in enumerate(param_grid):
        if idx in completed_experiments:
            continue
        
        config_path = create_config_for_combination(base_config_path, params, idx)
        pending_experiments.append((idx, config_path, params))
    
    if not pending_experiments:
        logger.info("Todos os experimentos já foram concluídos!")
        return all_results
    
    # Validação de memória antes de executar
    if parallel > 1:
        max_batch_size = max([p.get('batch_size', 16) for _, _, p in pending_experiments])
        is_safe, mem_message = check_memory_availability(parallel, max_batch_size)
        logger.info(f"\n{mem_message}\n")
        
        if not is_safe:
            response = input("Deseja continuar mesmo assim? (s/N): ")
            if response.lower() != 's':
                logger.info("Execução cancelada pelo usuário")
                sys.exit(0)
    
    # Distribui GPUs entre workers em round-robin (um worker → uma GPU)
    import torch as _torch
    _available_gpus: List[int] = (
        gpu_ids
        if gpu_ids is not None
        else list(range(_torch.cuda.device_count()))
    )
    def _gpu_for(idx: int) -> List[int] | None:
        """Retorna [gpu_id] para o worker `idx`, ou None quando não há GPUs."""
        if not _available_gpus:
            return None
        return [_available_gpus[idx % len(_available_gpus)]]

    # Executa experimentos
    if parallel > 1:
        logger.info(
            "Executando em modo paralelo com %d workers | GPUs disponíveis: %s",
            parallel, _available_gpus or "CPU"
        )
        with ProcessPoolExecutor(
            max_workers=parallel,
            initializer=setup_worker_logging,
            initargs=(_LOG_QUEUE,),
        ) as executor:
            futures = {
                executor.submit(run_single_experiment, idx, cfg, params, _gpu_for(idx)): idx
                for idx, cfg, params in pending_experiments
            }
            
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    all_results.append(result)
                    completed_experiments.add(idx)
                    
                    # Salva estado incremental
                    save_state(completed_experiments, all_results)
                    
                    logger.info(f"Progresso: {len(completed_experiments)}/{total_experiments}")
                    
                except Exception as e:
                    logger.error(f"Erro ao executar experimento {idx}: {e}")
    else:
        logger.info("Executando em modo sequencial | GPUs disponíveis: %s", _available_gpus or "CPU")
        for idx, config_path, params in pending_experiments:
            result = run_single_experiment(idx, config_path, params, _gpu_for(idx))
            all_results.append(result)
            completed_experiments.add(idx)
            
            # Salva estado incremental
            save_state(completed_experiments, all_results)
            
            logger.info(f"Progresso: {len(completed_experiments)}/{total_experiments}")
    
    return all_results


def save_state(completed_experiments: set, results: List[Dict[str, Any]]):
    """Salva estado da execução para permitir retomada."""
    state = {
        "timestamp": datetime.now().isoformat(),
        "completed_experiments": list(completed_experiments),
        "results": results
    }
    
    with open(GRID_STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)


# ============================================================================
# ANÁLISE DE RESULTADOS
# ============================================================================

def analyze_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analisa resultados e identifica as melhores configurações por múltiplos critérios.
    
    Critérios de análise:
        - Tempo de treinamento (train_time_sec)
        - Eficiência energética (energy_kwh)
        - Uso de memória RAM (peak_ram_mb)
        - Emissão de carbono (emissions_kg_co2)
        - Custo financeiro (cost_usd, calculado a partir de energy_kwh)
    
    Args:
        results: Lista com resultados de todos os experimentos
        
    Returns:
        Dicionário com análise dos resultados incluindo:
        - best_by_time: Melhor configuração por tempo
        - best_by_energy: Melhor configuração por energia
        - best_by_ram: Melhor configuração por memória
        - best_by_carbon: Melhor configuração por emissão de CO2
        - best_by_cost: Melhor configuração por custo financeiro
    """
    logger.info("Analisando resultados...")
    
    # Filtra experimentos bem-sucedidos
    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") == "failed"]
    
    logger.info(f"Experimentos bem-sucedidos: {len(successful)}")
    logger.info(f"Experimentos falhos: {len(failed)}")
    
    if not successful:
        logger.warning("Nenhum experimento foi concluído com sucesso!")
        return {
            "total_experiments": len(results),
            "successful": 0,
            "failed": len(failed),
            "best_config": None
        }
    
    # Ordena por tempo de treinamento (menor é melhor)
    sorted_by_time = sorted(
        successful,
        key=lambda x: float(x.get("resources", {}).get("train_time_sec", float('inf')))
    )
    
    # Ordena por eficiência energética (menor é melhor)
    sorted_by_energy = sorted(
        successful,
        key=lambda x: float(x.get("resources", {}).get("energy_kwh", float('inf')))
        if x.get("resources", {}).get("energy_kwh") is not None else float('inf')
    )
    
    # Ordena por uso de RAM (menor é melhor)
    sorted_by_ram = sorted(
        successful,
        key=lambda x: float(x.get("resources", {}).get("peak_ram_mb", float('inf')))
        if x.get("resources", {}).get("peak_ram_mb") is not None else float('inf')
    )
    
    # Ordena por emissão de CO2 (menor é melhor)
    sorted_by_carbon = sorted(
        successful,
        key=lambda x: float(x.get("resources", {}).get("emissions_kg_co2", float('inf')))
        if x.get("resources", {}).get("emissions_kg_co2") is not None else float('inf')
    )
    
    # Calcula custo financeiro e ordena (menor é melhor)
    for result in successful:
        energy_kwh = result.get("resources", {}).get("energy_kwh")
        if energy_kwh is not None:
            cost_usd = float(energy_kwh) * ENERGY_COST_USD_PER_KWH
            result["resources"]["cost_usd"] = cost_usd
        else:
            result["resources"]["cost_usd"] = None
    
    sorted_by_cost = sorted(
        successful,
        key=lambda x: float(x.get("resources", {}).get("cost_usd", float('inf')))
        if x.get("resources", {}).get("cost_usd") is not None else float('inf')
    )
    
    analysis = {
        "timestamp": datetime.now().isoformat(),
        "total_experiments": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "energy_cost_usd_per_kwh": ENERGY_COST_USD_PER_KWH,
        
        "best_by_time": {
            "experiment_idx": sorted_by_time[0]["grid_experiment_idx"],
            "params": sorted_by_time[0]["grid_params"],
            "train_time_sec": sorted_by_time[0]["resources"]["train_time_sec"]
        } if sorted_by_time else None,
        
        "best_by_energy": {
            "experiment_idx": sorted_by_energy[0]["grid_experiment_idx"],
            "params": sorted_by_energy[0]["grid_params"],
            "energy_kwh": sorted_by_energy[0]["resources"].get("energy_kwh")
        } if sorted_by_energy and sorted_by_energy[0]["resources"].get("energy_kwh") else None,
        
        "best_by_ram": {
            "experiment_idx": sorted_by_ram[0]["grid_experiment_idx"],
            "params": sorted_by_ram[0]["grid_params"],
            "peak_ram_mb": sorted_by_ram[0]["resources"].get("peak_ram_mb")
        } if sorted_by_ram and sorted_by_ram[0]["resources"].get("peak_ram_mb") else None,
        
        "best_by_carbon": {
            "experiment_idx": sorted_by_carbon[0]["grid_experiment_idx"],
            "params": sorted_by_carbon[0]["grid_params"],
            "emissions_kg_co2": sorted_by_carbon[0]["resources"].get("emissions_kg_co2")
        } if sorted_by_carbon and sorted_by_carbon[0]["resources"].get("emissions_kg_co2") else None,
        
        "best_by_cost": {
            "experiment_idx": sorted_by_cost[0]["grid_experiment_idx"],
            "params": sorted_by_cost[0]["grid_params"],
            "cost_usd": sorted_by_cost[0]["resources"].get("cost_usd")
        } if sorted_by_cost and sorted_by_cost[0]["resources"].get("cost_usd") else None,
        
        "all_results": results
    }
    
    return analysis


def generate_summary_report(analysis: Dict[str, Any]) -> str:
    """
    Gera relatório textual resumido dos resultados.
    
    Args:
        analysis: Dicionário com análise dos resultados
        
    Returns:
        String formatada com o relatório
    """
    report = []
    report.append("=" * 80)
    report.append("GRID SEARCH - RELATÓRIO DE RESULTADOS")
    report.append("=" * 80)
    report.append(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    report.append("")
    
    report.append("RESUMO GERAL:")
    report.append(f"  Total de experimentos: {analysis['total_experiments']}")
    report.append(f"  Bem-sucedidos: {analysis['successful']}")
    report.append(f"  Falhos: {analysis['failed']}")
    report.append("")
    
    if analysis.get("best_by_time"):
        report.append("MELHOR CONFIGURAÇÃO (Tempo de Treinamento):")
        best = analysis["best_by_time"]
        report.append(f"  Experimento: {best['experiment_idx']}")
        report.append(f"  Tempo: {best['train_time_sec']} segundos")
        report.append("  Parâmetros:")
        for k, v in best["params"].items():
            report.append(f"    - {k}: {v}")
        report.append("")
    
    if analysis.get("best_by_energy"):
        report.append("MELHOR CONFIGURAÇÃO (Eficiência Energética):")
        best = analysis["best_by_energy"]
        report.append(f"  Experimento: {best['experiment_idx']}")
        report.append(f"  Energia: {best['energy_kwh']} kWh")
        report.append("  Parâmetros:")
        for k, v in best["params"].items():
            report.append(f"    - {k}: {v}")
        report.append("")
    
    if analysis.get("best_by_ram"):
        report.append("MELHOR CONFIGURAÇÃO (Uso de Memória RAM):")
        best = analysis["best_by_ram"]
        report.append(f"  Experimento: {best['experiment_idx']}")
        report.append(f"  RAM Pico: {best['peak_ram_mb']} MB")
        report.append("  Parâmetros:")
        for k, v in best["params"].items():
            report.append(f"    - {k}: {v}")
        report.append("")
    
    if analysis.get("best_by_carbon"):
        report.append("MELHOR CONFIGURAÇÃO (Menor Emissão de Carbono):")
        best = analysis["best_by_carbon"]
        report.append(f"  Experimento: {best['experiment_idx']}")
        report.append(f"  Emissão CO2: {best['emissions_kg_co2']:.6f} kg")
        report.append("  Parâmetros:")
        for k, v in best["params"].items():
            report.append(f"    - {k}: {v}")
        report.append("")
    
    if analysis.get("best_by_cost"):
        report.append("MELHOR CONFIGURAÇÃO (Menor Custo Financeiro):")
        best = analysis["best_by_cost"]
        report.append(f"  Experimento: {best['experiment_idx']}")
        report.append(f"  Custo: ${best['cost_usd']:.4f} USD")
        report.append(f"  (Tarifa: ${analysis['energy_cost_usd_per_kwh']:.4f}/kWh)")
        report.append("  Parâmetros:")
        for k, v in best["params"].items():
            report.append(f"    - {k}: {v}")
        report.append("")
    
    report.append("=" * 80)
    
    # Adiciona estatísticas gerais
    if analysis['successful'] > 0:
        report.append("")
        report.append("ESTATÍSTICAS GERAIS DOS EXPERIMENTOS BEM-SUCEDIDOS:")
        report.append("")
        
        # Calcula estatísticas agregadas
        all_successful = [r for r in analysis['all_results'] if r.get('status') == 'success']
        
        # Tempo total
        total_time = sum(
            float(r.get('resources', {}).get('train_time_sec', 0))
            for r in all_successful
        )
        report.append(f"  Tempo total de treinamento: {total_time:.2f} segundos ({total_time/3600:.2f} horas)")
        
        # Energia total
        total_energy = sum(
            float(r.get('resources', {}).get('energy_kwh', 0) or 0)
            for r in all_successful
        )
        if total_energy > 0:
            report.append(f"  Energia total consumida: {total_energy:.4f} kWh")
        
        # CO2 total
        total_co2 = sum(
            float(r.get('resources', {}).get('emissions_kg_co2', 0) or 0)
            for r in all_successful
        )
        if total_co2 > 0:
            report.append(f"  Emissão total de CO2: {total_co2:.6f} kg ({total_co2*1000:.2f} g)")
        
        # Custo total
        total_cost = sum(
            float(r.get('resources', {}).get('cost_usd', 0) or 0)
            for r in all_successful
        )
        if total_cost > 0:
            report.append(f"  Custo financeiro total: ${total_cost:.4f} USD")
        
        report.append("")
        report.append("=" * 80)
    
    return "\n".join(report)


# ============================================================================
# INTERFACE CLI
# ============================================================================

def main():
    """Ponto de entrada CLI para execução de grid search via ``python -m gridsearch.core``.

    Analisa argumentos de linha de comando e despacha para uma das operações:

    - Busca em grade completa (``--config`` + ``--search-config``)
    - Retomada de execução interrompida (``--resume``)
    - Análise de resultados existentes (``--analyze-only``)
    """
    parser = argparse.ArgumentParser(
        description="Grid Search para hiperparâmetros do BERT-PLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:

  # Busca completa com configuração JSON
  python -m gridsearch.core --config config/experiments/BertPLI.config \\
                            --search-config gridsearch/config/grid_search.json

  # Execução paralela com 4 processos
  python -m gridsearch.core --config config/experiments/BertPLI.config \\
                            --search-config gridsearch/config/grid_search.json \\
                            --parallel 4

  # Retomar execução interrompida
  python -m gridsearch.core --resume

  # Analisar resultados existentes
  python -m gridsearch.core --analyze-only
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Caminho do arquivo de configuração base"
    )
    
    parser.add_argument(
        "--search-config",
        type=str,
        help="Caminho do arquivo JSON com grade de hiperparâmetros"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Retoma execução anterior usando estado salvo"
    )
    
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Número de processos paralelos (padrão: 1 = sequencial)"
    )
    
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Apenas analisa resultados existentes sem executar novos experimentos"
    )
    
    args = parser.parse_args()
    
    # Modo: apenas análise
    if args.analyze_only:
        # Tenta encontrar o arquivo de resultados: com data de hoje, sem data, ou o mais recente
        results_file = None
        if GRID_RESULTS_FILE.exists():
            results_file = GRID_RESULTS_FILE
        else:
            # Fallback 1: arquivo sem data
            fallback_no_date = GRID_OUTPUT_DIR / "grid_search_results.json"
            if fallback_no_date.exists():
                results_file = fallback_no_date
                logger.warning(f"Arquivo do dia não encontrado. Usando: {results_file}")
            else:
                # Fallback 2: arquivo com data mais recente disponível
                candidates = sorted(GRID_OUTPUT_DIR.glob("grid_search_results_*.json"), reverse=True)
                if candidates:
                    results_file = candidates[0]
                    logger.warning(f"Arquivo do dia não encontrado. Usando o mais recente: {results_file}")
        
        if results_file is None:
            logger.error(f"Nenhum arquivo de resultados encontrado em: {GRID_OUTPUT_DIR}")
            sys.exit(1)
        
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        analysis = analyze_results(results)
        report = generate_summary_report(analysis)
        
        print("\n" + report)
        
        # Salva com data e também como arquivo canônico sem data
        with open(GRID_SUMMARY_FILE, 'w', encoding='utf-8') as f:
            f.write(report)
        
        canonical_summary = GRID_OUTPUT_DIR / "grid_search_summary.txt"
        with open(canonical_summary, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Relatório salvo em: {GRID_SUMMARY_FILE}")
        logger.info(f"Relatório canônico salvo em: {canonical_summary}")
        return
    
    # Modo: retomar execução
    if args.resume:
        if not GRID_STATE_FILE.exists():
            logger.error(f"Arquivo de estado não encontrado: {GRID_STATE_FILE}")
            sys.exit(1)
        
        logger.info("Retomando execução...")
        # Continua com os mesmos parâmetros
        
    else:
        # Modo: nova execução
        if not args.config or not args.search_config:
            parser.error("--config e --search-config são obrigatórios para nova execução")
        
        if not os.path.exists(args.config):
            logger.error(f"Arquivo de configuração não encontrado: {args.config}")
            sys.exit(1)
        
        if not os.path.exists(args.search_config):
            logger.error(f"Arquivo de busca não encontrado: {args.search_config}")
            sys.exit(1)
        
        # Carrega configuração da grade
        with open(args.search_config, 'r', encoding='utf-8') as f:
            grid_config = json.load(f)
        
        logger.info(f"Configuração base: {args.config}")
        logger.info(f"Grade de hiperparâmetros: {args.search_config}")
        logger.info(f"Modo de execução: {'Paralelo (' + str(args.parallel) + ' workers)' if args.parallel > 1 else 'Sequencial'}")
        
        # Executa grid search
        results = run_grid_search(
            base_config_path=args.config,
            grid_config=grid_config,
            resume=args.resume,
            parallel=args.parallel
        )
        
        # Salva resultados completos
        with open(GRID_RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Resultados completos salvos em: {GRID_RESULTS_FILE}")
        
        # Analisa e gera relatório
        analysis = analyze_results(results)
        report = generate_summary_report(analysis)
        
        print("\n" + report)
        
        # Salva com data e também como arquivo canônico sem data
        with open(GRID_SUMMARY_FILE, 'w', encoding='utf-8') as f:
            f.write(report)
        
        canonical_summary = GRID_OUTPUT_DIR / "grid_search_summary.txt"
        with open(canonical_summary, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Relatório salvo em: {GRID_SUMMARY_FILE}")
        logger.info(f"Relatório canônico salvo em: {canonical_summary}")


if __name__ == "__main__":
    main()
