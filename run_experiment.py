from pathlib import Path
import subprocess
import configparser
import time
import uuid
import os
import json
import csv
from datetime import datetime, timezone
from utils.util import get_torch_device

import psutil

try:
    from codecarbon import EmissionsTracker
except ImportError:
    EmissionsTracker = None

# Constantes de diretórios e arquivos
OUTPUT_DIR = "output/experiments"
METRICS_DIR = "output/experiments/metrics"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(METRICS_DIR).mkdir(parents=True, exist_ok=True)
CSV_PATH = Path(METRICS_DIR) / "experiment_summary.csv"


# =========================
# UTILS
# =========================
def now_iso():
    return datetime.now(timezone.utc).isoformat()


def load_config(path):
    cfg = configparser.ConfigParser()
    cfg.read(path)
    return cfg


def estimate_bert_flops(
    seq_len,
    hidden_size=768,
    num_layers=12,
    num_heads=12
):
    attention = (
        4 * seq_len * hidden_size * hidden_size +
        2 * num_heads * seq_len * seq_len * (hidden_size // num_heads)
    )
    ffn = 8 * seq_len * hidden_size * hidden_size
    return num_layers * (attention + ffn) / 1e9  # GFLOPs

# info de processamento (CPU/GPU)
device_type, device_name, device_info = get_torch_device()

# =========================
# MAIN WRAPPER
# =========================
def execute_experiment(config_path):
    cfg = load_config(config_path)

    exp =   cfg["experiment"]
    train = cfg["train"]
    env =   cfg["environment"]
    mon =   cfg["monitoring"]

    experiment_id = str(uuid.uuid4())
    start_time = time.time()
    start_iso = now_iso()

    # -------- ENERGY TRACKER --------
    tracker = None
    if EmissionsTracker and mon.getboolean("enable_monitoring"):
        tracker = EmissionsTracker(
            project_name=exp["name"],
            output_dir=OUTPUT_DIR,
            log_level="error"
        )
        tracker.start()

    # -------- EXEC TRAIN (EXTERNAL) --------
    process = subprocess.Popen(
        ["uv", "run", "python", "train.py", "-c", config_path, "-g", "0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )

    ps_proc = psutil.Process(process.pid)
    ram_samples = []
    output_lines = []

    # Stream output in real-time
    for line in process.stdout:
        print(line, end='')  # Print to console
        output_lines.append(line)  # Store for later
        try:
            ram_samples.append(ps_proc.memory_info().rss / (1024 ** 2))
        except psutil.NoSuchProcess:
            break

    process.wait()
    stdout = ''.join(output_lines)
    stderr = ""

    status = "success" if process.returncode == 0 else "failed"

    if tracker:
        energy_kwh = tracker.stop()
    else:
        energy_kwh = None

    end_time = time.time()
    end_iso = now_iso()

    # -------- METRICS (proxy / external) --------
    avg_ram = sum(ram_samples) / len(ram_samples) if ram_samples else None
    peak_ram = max(ram_samples) if ram_samples else None

    total_gflops = estimate_bert_flops(seq_len=256)

    # =========================
    # JSON OUTPUT
    # =========================
    result = {
        "experiment": {
            "id": experiment_id,
            "config_name": exp["name"],
            "seed": int(exp["seed"]),
            "status": status,
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
            "batch_size": int(train["batch_size"]),
            "epoch": int(train["epoch"])
        },
        "resources": {
            "train_time_sec": end_time - start_time,
            "energy_kwh": energy_kwh,
            "avg_ram_mb": avg_ram,
            "peak_ram_mb": peak_ram,
            "total_gflops": total_gflops
        },
        "logs": {
            "stdout_tail": stdout[-1000:],
            "stderr_tail": stderr[-1000:]
        }
    }

    json_path = os.path.join(
        OUTPUT_DIR,
        f"{exp['name']}_{exp['seed']}.json"
    )

    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    # =========================
    # CSV AGGREGATION
    # =========================
    write_header = not os.path.exists(CSV_PATH)

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
                "avg_ram_mb",
                "peak_ram_mb",
                "total_gflops",
                "status",
                "timestamp"
            ])

        writer.writerow([
            experiment_id,
            exp["name"],
            exp["seed"],
            device_type,
            train["optimizer"],
            train["learning_rate"],
            train["batch_size"],
            train["epoch"],
            end_time - start_time,
            energy_kwh,
            avg_ram,
            peak_ram,
            total_gflops,
            status,
            end_iso
        ])

    print(f"[OK] Wrapper finalizou: {exp['name']} ({status})")


# =========================
# CLI
# =========================
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Uso: python run_experiment_wrapper.py <config_path>")
        exit(1)

    execute_experiment(sys.argv[1])
