"""
Grid Search Analysis - BERT-PLI
================================

Ferramentas de análise e visualização de resultados do grid search.

Autor: Gustavo Alexandre
Data: 2026-02-15
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import statistics

from utils.paths import PathManager

logger = logging.getLogger(__name__)

# Diretórios
GRID_OUTPUT_DIR = PathManager.EXPERIMENTS_DIR / "grid_search"
GRID_RESULTS_FILE = GRID_OUTPUT_DIR / "grid_search_results.json"
ANALYSIS_DIR = GRID_OUTPUT_DIR / "analysis"


# ============================================================================
# ESTATÍSTICAS DESCRITIVAS
# ============================================================================

def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """
    Calcula estatísticas descritivas de uma lista de valores.
    
    Args:
        values: Lista de valores numéricos
        
    Returns:
        Dicionário com estatísticas (média, mediana, desvio padrão, etc.)
    """
    if not values:
        return {
            "mean": None,
            "median": None,
            "stdev": None,
            "min": None,
            "max": None,
            "count": 0
        }
    
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "count": len(values)
    }


def compute_descriptive_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computa estatísticas descritivas para todas as métricas.
    
    Métricas analisadas:
        - train_time_sec: Tempo de treinamento em segundos
        - energy_kwh: Consumo de energia em kWh
        - peak_ram_mb: Uso de memória RAM em MB
        - emissions_kg_co2: Emissões de CO2 em kg
        - cost_usd: Custo financeiro em USD
    
    Args:
        results: Lista de resultados dos experimentos
        
    Returns:
        Dicionário com estatísticas para cada métrica
    """
    logger.info("Computando estatísticas descritivas...")
    
    successful = [r for r in results if r.get("status") == "success"]
    
    if not successful:
        logger.warning("Nenhum resultado válido para análise")
        return {}
    
    # Coleta valores de cada métrica
    train_times = []
    energy_values = []
    ram_values = []
    carbon_values = []
    cost_values = []
    
    for result in successful:
        resources = result.get("resources", {})
        
        if "train_time_sec" in resources:
            train_times.append(float(resources["train_time_sec"]))
        
        if "energy_kwh" in resources and resources["energy_kwh"] is not None:
            energy_values.append(float(resources["energy_kwh"]))
        
        if "peak_ram_mb" in resources and resources["peak_ram_mb"] is not None:
            ram_values.append(float(resources["peak_ram_mb"]))
        
        if "emissions_kg_co2" in resources and resources["emissions_kg_co2"] is not None:
            carbon_values.append(float(resources["emissions_kg_co2"]))
        
        if "cost_usd" in resources and resources["cost_usd"] is not None:
            cost_values.append(float(resources["cost_usd"]))
    
    return {
        "train_time": calculate_statistics(train_times),
        "energy_kwh": calculate_statistics(energy_values),
        "peak_ram_mb": calculate_statistics(ram_values),
        "emissions_kg_co2": calculate_statistics(carbon_values),
        "cost_usd": calculate_statistics(cost_values),
        "total_experiments": len(results),
        "successful_experiments": len(successful),
        "failed_experiments": len(results) - len(successful)
    }


# ============================================================================
# ANÁLISE POR HIPERPARÂMETRO
# ============================================================================

def analyze_by_hyperparameter(
    results: List[Dict[str, Any]],
    param_name: str,
    metric_name: str = "train_time_sec"
) -> Dict[Any, Dict[str, float]]:
    """
    Analisa o impacto de um hiperparâmetro específico em uma métrica.
    
    Args:
        results: Lista de resultados
        param_name: Nome do hiperparâmetro a analisar
        metric_name: Nome da métrica a medir (padrão: train_time_sec)
                    Opções: train_time_sec, energy_kwh, peak_ram_mb, 
                            emissions_kg_co2, cost_usd
        
    Returns:
        Dicionário mapeando valores do hiperparâmetro para estatísticas
    """
    logger.info(f"Analisando impacto de '{param_name}' em '{metric_name}'...")
    
    successful = [r for r in results if r.get("status") == "success"]
    
    # Agrupa por valor do hiperparâmetro
    grouped = {}
    
    for result in successful:
        params = result.get("grid_params", {})
        
        if param_name not in params:
            continue
        
        param_value = params[param_name]
        
        # Extrai valor da métrica
        if metric_name in ["train_time_sec", "energy_kwh", "peak_ram_mb", 
                          "emissions_kg_co2", "cost_usd"]:
            resources = result.get("resources", {})
            metric_value = resources.get(metric_name)
        else:
            metric_value = result.get(metric_name)
        
        if metric_value is None:
            continue
        
        if param_value not in grouped:
            grouped[param_value] = []
        
        grouped[param_value].append(float(metric_value))
    
    # Calcula estatísticas para cada valor
    analysis = {}
    for param_value, metric_values in grouped.items():
        analysis[param_value] = calculate_statistics(metric_values)
    
    return analysis


def find_best_value_per_hyperparameter(
    results: List[Dict[str, Any]],
    metric_name: str = "train_time_sec",
    minimize: bool = True
) -> Dict[str, Any]:
    """
    Identifica o melhor valor para cada hiperparâmetro individualmente.
    
    Args:
        results: Lista de resultados
        metric_name: Métrica a otimizar
        minimize: Se True, procura menor valor; se False, maior
        
    Returns:
        Dicionário com melhor valor de cada hiperparâmetro
    """
    logger.info("Identificando melhores valores para cada hiperparâmetro...")
    
    successful = [r for r in results if r.get("status") == "success"]
    
    if not successful:
        return {}
    
    # Identifica todos os hiperparâmetros presentes
    all_params = set()
    for result in successful:
        params = result.get("grid_params", {})
        all_params.update(params.keys())
    
    best_values = {}
    
    for param_name in all_params:
        analysis = analyze_by_hyperparameter(results, param_name, metric_name)
        
        if not analysis:
            continue
        
        # Encontra o valor que minimiza/maximiza a média
        best_value = None
        best_mean = None
        
        for param_value, stats in analysis.items():
            mean = stats.get("mean")
            
            if mean is None:
                continue
            
            if best_mean is None:
                best_value = param_value
                best_mean = mean
            elif minimize and mean < best_mean:
                best_value = param_value
                best_mean = mean
            elif not minimize and mean > best_mean:
                best_value = param_value
                best_mean = mean
        
        best_values[param_name] = {
            "best_value": best_value,
            "mean_metric": best_mean,
            "all_values": analysis
        }
    
    return best_values


# ============================================================================
# ANÁLISE DE CORRELAÇÕES
# ============================================================================

def compute_correlation(x: List[float], y: List[float]) -> Optional[float]:
    """
    Calcula correlação de Pearson entre duas listas de valores.
    
    Args:
        x: Lista de valores da variável X
        y: Lista de valores da variável Y
        
    Returns:
        Coeficiente de correlação de Pearson ou None se não for possível calcular
    """
    if len(x) != len(y) or len(x) < 2:
        return None
    
    try:
        n = len(x)
        
        mean_x = statistics.mean(x)
        mean_y = statistics.mean(y)
        
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        
        denominator_x = sum((x[i] - mean_x) ** 2 for i in range(n))
        denominator_y = sum((y[i] - mean_y) ** 2 for i in range(n))
        
        denominator = (denominator_x * denominator_y) ** 0.5
        
        if denominator == 0:
            return None
        
        return numerator / denominator
        
    except Exception as e:
        logger.warning(f"Erro ao calcular correlação: {e}")
        return None


def analyze_correlations(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Analisa correlações entre hiperparâmetros numéricos e métricas.
    
    Args:
        results: Lista de resultados
        
    Returns:
        Dicionário com coeficientes de correlação
    """
    logger.info("Analisando correlações...")
    
    successful = [r for r in results if r.get("status") == "success"]
    
    if not successful:
        return {}
    
    # Coleta dados numéricos
    learning_rates = []
    batch_sizes = []
    dropouts = []
    train_times = []
    energies = []
    rams = []
    
    for result in successful:
        params = result.get("grid_params", {})
        resources = result.get("resources", {})
        
        # Hiperparâmetros
        if "learning_rate" in params:
            learning_rates.append(float(params["learning_rate"]))
        
        if "batch_size" in params:
            batch_sizes.append(float(params["batch_size"]))
        
        if "dropout" in params:
            dropouts.append(float(params["dropout"]))
        
        # Métricas
        if "train_time_sec" in resources:
            train_times.append(float(resources["train_time_sec"]))
        
        if "energy_kwh" in resources and resources["energy_kwh"] is not None:
            energies.append(float(resources["energy_kwh"]))
        
        if "peak_ram_mb" in resources and resources["peak_ram_mb"] is not None:
            rams.append(float(resources["peak_ram_mb"]))
    
    correlations = {}
    
    # Correlações com tempo de treinamento
    if learning_rates and train_times and len(learning_rates) == len(train_times):
        correlations["learning_rate_vs_train_time"] = compute_correlation(learning_rates, train_times)
    
    if batch_sizes and train_times and len(batch_sizes) == len(train_times):
        correlations["batch_size_vs_train_time"] = compute_correlation(batch_sizes, train_times)
    
    if dropouts and train_times and len(dropouts) == len(train_times):
        correlations["dropout_vs_train_time"] = compute_correlation(dropouts, train_times)
    
    # Correlações com energia
    if learning_rates and energies and len(learning_rates) == len(energies):
        correlations["learning_rate_vs_energy"] = compute_correlation(learning_rates, energies)
    
    if batch_sizes and energies and len(batch_sizes) == len(energies):
        correlations["batch_size_vs_energy"] = compute_correlation(batch_sizes, energies)
    
    # Correlações com RAM
    if batch_sizes and rams and len(batch_sizes) == len(rams):
        correlations["batch_size_vs_ram"] = compute_correlation(batch_sizes, rams)
    
    return correlations


# ============================================================================
# RANKING DE CONFIGURAÇÕES
# ============================================================================

def rank_configurations(
    results: List[Dict[str, Any]],
    metrics: List[str] = None,
    weights: List[float] = None
) -> List[Dict[str, Any]]:
    """
    Cria ranking de configurações baseado em múltiplas métricas.
    
    Args:
        results: Lista de resultados
        metrics: Lista de nomes das métricas (padrão: ["train_time_sec", "energy_kwh"])
                Opções: train_time_sec, energy_kwh, peak_ram_mb, 
                       emissions_kg_co2, cost_usd
        weights: Pesos para cada métrica (padrão: igual para todas)
        
    Returns:
        Lista ordenada de configurações com scores
    """
    if metrics is None:
        metrics = ["train_time_sec", "energy_kwh"]
    
    if weights is None:
        weights = [1.0] * len(metrics)
    
    if len(metrics) != len(weights):
        raise ValueError("Número de métricas e pesos deve ser igual")
    
    logger.info(f"Criando ranking com métricas: {metrics}")
    
    successful = [r for r in results if r.get("status") == "success"]
    
    if not successful:
        return []
    
    # Normaliza valores de cada métrica
    normalized_data = []
    
    for metric in metrics:
        values = []
        
        for result in successful:
            if metric in ["train_time_sec", "energy_kwh", "peak_ram_mb", 
                         "emissions_kg_co2", "cost_usd"]:
                resources = result.get("resources", {})
                value = resources.get(metric)
            else:
                value = result.get(metric)
            
            if value is not None:
                values.append(float(value))
            else:
                values.append(None)
        
        # Normalização min-max (0 = melhor, 1 = pior)
        valid_values = [v for v in values if v is not None]
        
        if valid_values:
            min_val = min(valid_values)
            max_val = max(valid_values)
            range_val = max_val - min_val if max_val != min_val else 1.0
            
            normalized = []
            for v in values:
                if v is None:
                    normalized.append(1.0)  # Penalidade máxima
                else:
                    normalized.append((v - min_val) / range_val)
        else:
            normalized = [0.0] * len(values)
        
        normalized_data.append(normalized)
    
    # Calcula score ponderado
    ranked = []
    
    for i, result in enumerate(successful):
        score = 0.0
        
        for j, metric in enumerate(metrics):
            score += weights[j] * normalized_data[j][i]
        
        ranked.append({
            "experiment_idx": result.get("grid_experiment_idx"),
            "params": result.get("grid_params"),
            "score": score,
            "metrics": {
                metric: normalized_data[j][i]
                for j, metric in enumerate(metrics)
            }
        })
    
    # Ordena por score (menor é melhor)
    ranked.sort(key=lambda x: x["score"])
    
    return ranked


# ============================================================================
# EXPORTAÇÃO DE ANÁLISES
# ============================================================================

def export_analysis_to_json(analysis_data: Dict[str, Any], output_file: Path):
    """
    Exporta análise completa para JSON.
    
    Args:
        analysis_data: Dados da análise
        output_file: Arquivo de saída
    """
    logger.info(f"Exportando análise para {output_file}")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, indent=2, default=str)
    
    logger.info("Análise exportada com sucesso")


def generate_analysis_report(results: List[Dict[str, Any]]) -> str:
    """
    Gera relatório detalhado de análise.
    
    Args:
        results: Lista de resultados
        
    Returns:
        String formatada com relatório completo
    """
    report = []
    report.append("=" * 80)
    report.append("ANÁLISE DETALHADA - GRID SEARCH")
    report.append("=" * 80)
    report.append("")
    
    # Estatísticas descritivas
    stats = compute_descriptive_statistics(results)
    
    report.append("ESTATÍSTICAS DESCRITIVAS:")
    report.append("")
    
    if "train_time" in stats:
        report.append("  Tempo de Treinamento (segundos):")
        t = stats["train_time"]
        if t["count"] > 0:
            report.append(f"    Média: {t['mean']:.2f}")
            report.append(f"    Mediana: {t['median']:.2f}")
            report.append(f"    Desvio Padrão: {t['stdev']:.2f}")
            report.append(f"    Mínimo: {t['min']:.2f}")
            report.append(f"    Máximo: {t['max']:.2f}")
        report.append("")
    
    if "energy_kwh" in stats:
        report.append("  Energia Consumida (kWh):")
        e = stats["energy_kwh"]
        if e["count"] > 0:
            report.append(f"    Média: {e['mean']:.4f}")
            report.append(f"    Mediana: {e['median']:.4f}")
            report.append(f"    Desvio Padrão: {e['stdev']:.4f}")
            report.append(f"    Mínimo: {e['min']:.4f}")
            report.append(f"    Máximo: {e['max']:.4f}")
        report.append("")
    
    if "peak_ram_mb" in stats:
        report.append("  Uso de RAM Pico (MB):")
        r = stats["peak_ram_mb"]
        if r["count"] > 0:
            report.append(f"    Média: {r['mean']:.2f}")
            report.append(f"    Mediana: {r['median']:.2f}")
            report.append(f"    Desvio Padrão: {r['stdev']:.2f}")
            report.append(f"    Mínimo: {r['min']:.2f}")
            report.append(f"    Máximo: {r['max']:.2f}")
        report.append("")
    
    # Correlações
    correlations = analyze_correlations(results)
    
    if correlations:
        report.append("CORRELAÇÕES:")
        report.append("")
        for corr_name, corr_value in correlations.items():
            if corr_value is not None:
                report.append(f"  {corr_name}: {corr_value:.3f}")
        report.append("")
    
    # Ranking top 10
    ranked = rank_configurations(results)
    
    if ranked:
        report.append("TOP 10 CONFIGURAÇÕES:")
        report.append("")
        
        for i, config in enumerate(ranked[:10], 1):
            report.append(f"  {i}. Experimento {config['experiment_idx']} (Score: {config['score']:.3f})")
            report.append(f"     Parâmetros: {config['params']}")
            report.append("")
    
    report.append("=" * 80)
    
    return "\n".join(report)


# ============================================================================
# INTERFACE CLI
# ============================================================================

def main():
    """Executa análise standalone dos resultados."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Análise de resultados do Grid Search")
    
    parser.add_argument(
        "--results-file",
        type=str,
        default=str(GRID_RESULTS_FILE),
        help="Caminho do arquivo de resultados JSON"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ANALYSIS_DIR),
        help="Diretório de saída para análises"
    )
    
    args = parser.parse_args()
    
    # Carrega resultados
    results_file = Path(args.results_file)
    
    if not results_file.exists():
        logger.error(f"Arquivo de resultados não encontrado: {results_file}")
        sys.exit(1)
    
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    logger.info(f"Carregados {len(results)} resultados")
    
    # Executa análises
    stats = compute_descriptive_statistics(results)
    correlations = analyze_correlations(results)
    ranked = rank_configurations(results)
    best_per_param = find_best_value_per_hyperparameter(results)
    
    # Compila análise completa
    full_analysis = {
        "descriptive_statistics": stats,
        "correlations": correlations,
        "ranking": ranked,
        "best_values_per_hyperparameter": best_per_param
    }
    
    # Exporta análise
    output_dir = Path(args.output_dir)
    export_analysis_to_json(full_analysis, output_dir / "full_analysis.json")
    
    # Gera e salva relatório
    report = generate_analysis_report(results)
    
    print("\n" + report)
    
    report_file = output_dir / "analysis_report.txt"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_file, 'w') as f:
        f.write(report)
    
    logger.info(f"Relatório salvo em: {report_file}")


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    main()
