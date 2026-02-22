"""run_experiment.py — Motor de execução de experimentos BERT-PLI
==============================================================
Executa um experimento rastreável de ponta a ponta:

- Carrega e aplica a configuração em cascata (default → experimento)
- Mede tempo de execução, uso de RAM e emissões de CO₂ (via codecarbon)
- Captura stdout do loop de treino sem bloquear a saída em tempo real
- Persiste artefatos em JSON (por execução) e CSV (histórico acumulado)

Uso direto::

    uv run python run_experiment.py config/experiments/BertPLI.config [gpu_id ...]

Uso programático::

    from run_experiment import execute_experiment
    execute_experiment("config/experiments/BertPLI.config", gpu_list=[0, 1])
"""
from pathlib import Path
import os
import re
import sys
import threading
import subprocess
import configparser
import time
import uuid
import json
import csv
import logging
from datetime import datetime, timezone
from utils.device import get_torch_device
from utils.paths import PathManager
import torch
import psutil
from utils.util import print_system_info
from tools.eval_tool import (
    convert_test_results_to_task1, 
    compute_metrics
)

try:
    from codecarbon import EmissionsTracker
except ImportError:
    EmissionsTracker = None

from warnings import filterwarnings
filterwarnings("ignore", category=UserWarning)

# Logging configurado pelo processo que invoca execute_experiment().
# Aqui apenas obtemos o logger do módulo — sem basicConfig.
logger = logging.getLogger(__name__)

# Constantes de diretórios e arquivos
_METRICS_DIR = PathManager.EXPERIMENTS_DIR / "metrics"
PathManager.ensure_dir(_METRICS_DIR)

# =========================
# UTILS
# =========================
def now_iso() -> str:
    """Retorna o instante atual em formato ISO 8601 com timezone UTC."""
    return datetime.now(timezone.utc).isoformat()


def load_config(path: str) -> configparser.ConfigParser:
    """Lê e retorna um ConfigParser padrão da stdlib a partir do arquivo.

    Args:
        path: Caminho para o arquivo ``.config``.

    Returns:
        ``configparser.ConfigParser`` com as seções e chaves carregadas.
    """
    cfg = configparser.ConfigParser()
    cfg.read(path)
    return cfg


def estimate_bert_flops(
    seq_len: int,
    hidden_size: int = 768,
    num_layers: int = 12,
    num_heads: int = 12,
) -> float:
    """Estima os GFLOPs de uma passagem forward pelo BERT.

    A estimativa cobre atenção multi-head (QKV + produto attention)
    e os dois blocos FFN por camada.

    Args:
        seq_len: Comprimento da sequência de tokens.
        hidden_size: Dimensão oculta do modelo. Padrão: 768 (BERT-base).
        num_layers: Número de camadas Transformer. Padrão: 12.
        num_heads: Número de cabeças de atenção. Padrão: 12.

    Returns:
        GFLOPs estimados por forward pass.
    """
    attention = (
        4 * seq_len * hidden_size * hidden_size +
        2 * num_heads * seq_len * seq_len * (hidden_size // num_heads)
    )
    ffn = 8 * seq_len * hidden_size * hidden_size
    return num_layers * (attention + ffn) / 1e9  # GFLOPs

# Exibe informações do sistema para o experimento
print_system_info()

# info de processamento (CPU/GPU)
_torch_device_info = get_torch_device()
device_name = _torch_device_info['name']

# Tarifa de energia (USD/kWh) — configurável via variável de ambiente
_ENERGY_COST_USD_PER_KWH = float(os.getenv("ENERGY_COST_USD_PER_KWH", "0.12"))

# =========================
# MAIN WRAPPER
# =========================
def execute_experiment(config_path: str, gpu_list: list[int] | None = None) -> None:
    """Executa um experimento completo de forma rastreável.

    Realiza treino in-process com captura de stdout, amostragem contínua
    de RAM, rastreamento de emissões de CO₂ (opcional) e persistência
    de artefatos em JSON e CSV.

    Args:
        config_path: Caminho para o arquivo ``.config`` do experimento.
        gpu_list: Lista de IDs de GPU a utilizar. ``None`` seleciona GPU 0
            se disponível, ou CPU caso contrário.

    Side effects:
        - Escreve ``output/experiments/metrics/<nome>_<data>.json``
        - Acrescenta linha em ``output/experiments/metrics/experiment_summary_<data>.csv``
        - Acrescenta linha em ``output/experiments/metrics/EmissionsCO2_<device>_<data>.csv``
          (quando ``enable_monitoring = true``)
    """
    cfg = load_config(config_path)

    exp =   cfg["experiment"]
    train = cfg["train"]
    env =   cfg["environment"]
    mon =   cfg["monitoring"]

    experiment_id = str(uuid.uuid4())
    device_type = _torch_device_info['type']

    # Sincroniza o CUDA para garantir que as medições de tempo sejam precisas
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.perf_counter()
    start_iso = now_iso()
    DATE_EXEC = datetime.now().strftime("%Y%m%d_%H%M%S")

    # -------- ENERGY TRACKER --------
    tracker = None
    if EmissionsTracker and mon.getboolean("enable_monitoring"):
        tracker = EmissionsTracker(
            project_name=exp["name"],
            output_dir=_METRICS_DIR.as_posix(),
            log_level="error",
            output_file=f"EmissionsCO2_{device_type}_{datetime.now().strftime('%Y%m%d')}.csv"
        )
        tracker.start()

    # -------- EXEC TRAIN (IN-PROCESS) --------
    # Importa de forma lazy para evitar inicialização de CUDA no processo
    # principal antes da configuração de gpu_list.
    from utils.config import create_config
    from tools.init_tool import init_all
    from tools.train_tool import train as run_train

    # Usa as GPUs fornecidas pelo chamador; se não informado, seleciona GPU 0
    # (quando disponível) ou executa em CPU — mantém compatibilidade retroativa.
    if gpu_list is None:
        gpu_list = [0] if torch.cuda.is_available() else []

    # _TeeStream: escreve simultaneamente no terminal e em um buffer em memória.
    # Permite capturar o stdout do loop de treino sem perder a saída em tempo real.
    class _TeeStream:
        """Stream que escreve simultaneamente no terminal e em um buffer em memória.

        Permite capturar o stdout do loop de treino sem perder a saída em
        tempo real. Implementa a interface mínima exigida pelo Python e pelo
        Transformers ≥5.x (``write``, ``flush``, ``isatty``, ``fileno``).
        """

        def __init__(self, original) -> None:
            """Args:
                original: Stream original (normalmente ``sys.stdout``) a ser
                    mantido em paralelo com a captura em memória.
            """
            self.original = original
            self.lines: list = []
        def write(self, text: str) -> None:
            """Escreve ``text`` no stream original e acrescenta à lista ``lines``."""
            self.original.write(text)
            self.lines.append(text)
        def flush(self) -> None:
            """Propaga flush para o stream original."""
            self.original.flush()
        def isatty(self) -> bool:         # exigido pelo transformers >=5.x (loading_report)
            """Delega ``isatty()`` ao stream original; retorna ``False`` se não implementado."""
            return getattr(self.original, "isatty", lambda: False)()
        def fileno(self) -> int:          # necessário para logging handlers
            """Retorna o descritor de arquivo do stream original."""
            return self.original.fileno()

    tee = _TeeStream(sys.stdout)
    sys.stdout = tee

    # Amostragem de RAM em thread daemon (1 amostra/segundo).
    # Usa o PID do processo atual — mais preciso do que medir um subprocesso
    # filho, e permite incluir VRAM via torch.cuda.memory_allocated().
    ram_samples: list = []
    _stop_ram = threading.Event()

    def _sample_ram() -> None:
        """Amostra o uso de RAM (RSS) do processo atual a cada segundo.

        Executa em loop até que ``_stop_ram`` seja acionado. Projetada para
        ser executada em thread daemon — termina silenciosamente se o
        processo pai encerrar.
        """
        proc = psutil.Process(os.getpid())
        while not _stop_ram.is_set():
            try:
                ram_samples.append(proc.memory_info().rss / (1024 ** 2))
            except psutil.NoSuchProcess:
                break
            _stop_ram.wait(timeout=1.0)

    ram_thread = threading.Thread(target=_sample_ram, daemon=True)
    ram_thread.start()

    status = "failed"
    stdout = ""
    stderr = ""
    output_lines: list = []

    try:
        config = create_config(config_path)
        parameters = init_all(config, gpu_list, None, "train")
        run_train(parameters, config, gpu_list)
        status = "success"
    except Exception as exc:
        logger.error("Treinamento falhou: %s", exc, exc_info=True)
        stderr = str(exc)
    finally:
        _stop_ram.set()
        ram_thread.join(timeout=5)
        sys.stdout = tee.original
        output_lines = tee.lines
        stdout = "".join(output_lines)

    # -------- STOP ENERGY TRACKER --------
    emissions_kg = None
    energy_kwh = None
    
    if tracker:
        emissions_kg = tracker.stop()  # Retorna kg de CO2
        # Acessa os dados finais para obter energia em kWh
        try:
            if hasattr(tracker, 'final_emissions_data') and tracker.final_emissions_data is not None:
                energy_kwh = tracker.final_emissions_data.energy_consumed  # kWh
            elif hasattr(tracker, '_total_energy') and tracker._total_energy is not None:
                energy_kwh = tracker._total_energy.kWh
        except Exception as e:
            logger.warning("Não foi possível obter energy_kwh do tracker: %s", e)

    # Sincronização CUDA para garantir que todas as operações sejam 
    # concluídas antes de medir o tempo final
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    exec_time = time.perf_counter() - start_time
    end_iso = now_iso()

    # -------- METRICS (proxy / external) --------
    avg_ram = sum(ram_samples) / len(ram_samples) if ram_samples else None
    peak_ram = max(ram_samples) if ram_samples else None

    # Load profiling metrics if available
    profiling_path = Path(cfg.get("output", "model_path")) / cfg.get("output", "model_name") / "profiling_metrics.json"
    total_gflops = 0
    avg_gflops_per_batch = 0
    
    if profiling_path.exists():
        try:
            with open(profiling_path, "r") as f:
                profiling_data = json.load(f)
                total_gflops = profiling_data.get("total_gflops", 0)
                avg_gflops_per_batch = profiling_data.get("avg_gflops_per_batch", 0)
                logger.info(f"Loaded profiling metrics: {avg_gflops_per_batch:.2f} GFLOPs/batch")
        except Exception as e:
            logger.warning(f"Could not load profiling metrics: {e}")
            # Fallback to estimation
            total_gflops = estimate_bert_flops(seq_len=256)
    else:
        # Fallback to estimation if profiling not available
        logger.warning("Profiling metrics not found, using estimation")
        total_gflops = estimate_bert_flops(seq_len=256)

    # -------- EVAL METRICS --------
    # Padrão de regex para extrair métricas do log de validação do train_tool
    _VALID_RE = re.compile(
        r"valid set: micro_prec_query=([\d.]+),\s*micro_recall_query=([\d.]+),\s*micro_f1_query=([\d.]+),\s*accuracy=([\d.]+)"
    )

    eval_metrics = {}
    if status == "success":
        try:
            run_test_at_end = cfg.getboolean("eval", "run_test_at_end", fallback=False)
            pool_out_mode = cfg.getboolean("output", "pool_out", fallback=False)
            if pool_out_mode:
                # Pipeline de dois estágios: BERT como extrator de embeddings.
                # As métricas de validação já foram computadas durante o treino e
                # estão impressas no stdout capturado — extraímos a última ocorrência.
                last_match = None
                for line in output_lines:
                    m = _VALID_RE.search(line)
                    if m:
                        last_match = m
                if last_match:
                    eval_metrics = {
                        "precision": float(last_match.group(1)),
                        "recall":    float(last_match.group(2)),
                        "f1_score":  float(last_match.group(3)),
                        "accuracy":  float(last_match.group(4)),
                        "source":    "validation_log",
                    }
                    logger.info(
                        "Métricas (validação final, pool_out): P=%.4f  R=%.4f  F1=%.4f  Acc=%.4f",
                        eval_metrics["precision"],
                        eval_metrics["recall"],
                        eval_metrics["f1_score"],
                        eval_metrics["accuracy"],
                    )
                else:
                    logger.warning(
                        "pool_out=True mas nenhuma linha 'valid set:' encontrada no stdout. "
                        "Métricas de avaliação indisponíveis."
                    )
            elif run_test_at_end:
                model_out_path = Path(cfg.get("output", "model_path")) / cfg.get("output", "model_name")
                labels_path = cfg.get("data", "test_labels_file", fallback="data/task1_test_labels_2024.json")
                test_result_path = model_out_path / "test_results.json"

                # Localiza o último checkpoint salvo (por número de época)
                last_epoch = cfg.getint("train", "epoch") - 1
                checkpoint_path = model_out_path / f"{last_epoch}.pkl"
                if not checkpoint_path.exists():
                    pkl_files = sorted(
                        model_out_path.glob("*.pkl"),
                        key=lambda p: int(p.stem) if p.stem.isdigit() else -1,
                    )
                    checkpoint_path = pkl_files[-1] if pkl_files else None

                if checkpoint_path and checkpoint_path.exists() and Path(labels_path).exists():
                    logger.info("Executando avaliação com checkpoint: %s", checkpoint_path)
                    test_proc = subprocess.run(
                        [
                            "uv", "run", "python", "scripts/test.py",
                            "-c", config_path,
                            "-g", "0",
                            "--checkpoint", str(checkpoint_path),
                            "--result", str(test_result_path),
                        ],
                        capture_output=True,
                        text=True,
                    )
                    if test_proc.returncode == 0 and test_result_path.exists():
                        task1_predicted = convert_test_results_to_task1(str(test_result_path))
                        task1_path = model_out_path / "test_results_task1.json"
                        with open(task1_path, "w") as f:
                            json.dump(task1_predicted, f)
                        eval_metrics = compute_metrics(labels_path, str(task1_path))
                        logger.info(
                            "Métricas de avaliação: P=%.4f  R=%.4f  F1=%.4f  Acc=%.4f",
                            eval_metrics["precision"],
                            eval_metrics["recall"],
                            eval_metrics["f1_score"],
                            eval_metrics.get("accuracy", 0.0),
                        )
                    else:
                        logger.warning(
                            "Subprocess de teste falhou (código %d). Métricas de avaliação indisponíveis.\n%s",
                            test_proc.returncode,
                            test_proc.stderr[-500:],
                        )
                else:
                    logger.warning(
                        "Checkpoint (%s) ou labels (%s) não encontrado. Métricas de avaliação indisponíveis.",
                        checkpoint_path,
                        labels_path,
                    )
        except Exception as exc:
            logger.warning("Erro ao calcular métricas de avaliação: %s", exc)

    # =========================
    # JSON OUTPUT
    # =========================
    
    # Padronização do log filename
    id = exp["name"]
    optmzr = train["optimizer"]
    lr = f"lr{train['learning_rate']}".replace('-', '')
    bs = f"bs{train['batch_size']}"
    ep = f"ep{train['epoch']}"

    json_filename = f"{id}_{optmzr}_{lr}_{bs}_{ep}_{DATE_EXEC}.json"

    cost_usd = float(energy_kwh) * _ENERGY_COST_USD_PER_KWH if energy_kwh is not None else None

    result = {
        "experiment": {
            "id": experiment_id,
            "config_name": json_filename,
            "seed": int(exp["seed"]),
            "status": status,
            "date": DATE_EXEC,
            "timestamp_start": start_iso,
            "timestamp_end": end_iso
        },
        "environment": {
            "device_type": device_type,
            "device_name": device_name,
            "precision": env["precision"]
        },
        "hyperparameters": {
            "optimizer": train["optimizer"],
            "learning_rate": float(train["learning_rate"]),
            "avg_gflops_per_batch": avg_gflops_per_batch,
            "batch_size": int(train["batch_size"]),
            "epoch": int(train["epoch"])
        },
        "resources": {
            "train_time_sec": f"{exec_time:.2f}",
            "energy_kwh": energy_kwh,
            "emissions_kg_co2": emissions_kg,
            "cost_usd": cost_usd,
            "avg_ram_mb": avg_ram,
            "peak_ram_mb": peak_ram,
            "total_gflops": total_gflops
        },
        "evaluation": eval_metrics if eval_metrics else None,
        "logs": {
            "stdout_tail": stdout[-1000:],
            "stderr_tail": stderr[-1000:]
        }
    }

    json_path = _METRICS_DIR / json_filename
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    # =========================
    # CSV AGGREGATION
    # =========================
    csv_filename = f"experiment_summary_{device_type}{datetime.now().strftime('%Y%m%d')}.csv"
    CSV_PATH = _METRICS_DIR / csv_filename
    write_header = not CSV_PATH.exists()

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow([
                "experiment_id",
                "config_name",
                "seed",
                "device_type",
                "optimizer",
                "learning_rate",
                "batch_size",
                "epoch",
                "train_time_sec",
                "energy_kwh",
                "emissions_kg",
                "cost_usd",
                "avg_ram_mb",
                "peak_ram_mb",
                "avg_gflops_per_batch",
                "total_gflops",
                "status",
                "timestamp",
                "eval_precision",
                "eval_recall",
                "eval_f1",
                "eval_accuracy",
                "eval_source",
            ])

        writer.writerow([
            experiment_id,
            json_filename,
            exp["seed"],
            device_type,
            train["optimizer"],
            train["learning_rate"],
            train["batch_size"],
            train["epoch"],
            f"{exec_time:.2f}",
            energy_kwh,
            emissions_kg,
            f"{cost_usd:.6f}" if cost_usd is not None else None,
            avg_ram,
            peak_ram,
            avg_gflops_per_batch,
            total_gflops,
            status,
            end_iso,
            f"{eval_metrics['precision']:.4f}" if eval_metrics else None,
            f"{eval_metrics['recall']:.4f}" if eval_metrics else None,
            f"{eval_metrics['f1_score']:.4f}" if eval_metrics else None,
            f"{eval_metrics['accuracy']:.4f}" if eval_metrics and 'accuracy' in eval_metrics else None,
            eval_metrics.get("source") if eval_metrics else None,
        ])

    print(f"[OK] Wrapper finalizou em {exec_time:.2f} segundos - {exp['name']} ({status})")


# =========================
# CLI
# =========================
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: uv run python run_experiment.py <config_path> [gpu_id ...]")
        print("  Ex. (single GPU): uv run python run_experiment.py config/experiments/BertPLI.config 0")
        print("  Ex. (multi-GPU):  uv run python run_experiment.py config/experiments/BertPLI.config 0 1")
        print("  Ex. (CPU):        uv run python run_experiment.py config/experiments/BertPLI.config")
        exit(1)

    _gpus = [int(g) for g in sys.argv[2:]] if len(sys.argv) > 2 else None
    execute_experiment(sys.argv[1], gpu_list=_gpus)
