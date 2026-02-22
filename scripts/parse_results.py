"""
scripts/parse_results.py — CLI de avaliação e parsing de resultados
===================================================================
Modos de uso:

1. Avaliar predições contra ground truth:
   python -m scripts.parse_results evaluate <labels.json> <predicted.json> [output.json]
   bert-pli-parse-results evaluate labels.json predicted.json [output.json]

2. Parsear saída bruta do GRU/LSTM para formato task1:
   GRU=true python -m scripts.parse_results
   bert-pli-parse-results
"""
import sys
from pathlib import Path

# Deve vir antes de qualquer import do projeto
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import logging
import os

from tools.eval_tool import evaluate_predictions, parse_gru_results

# Configura logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    """Ponto de entrada do CLI ``bert-pli-parse-results``.

    Modos de operação:

    - ``evaluate``: compara predições contra ground truth e exibe métricas.
    - padrão (sem subcomando): converte resultados GRU/LSTM para o formato
      task1 da COLIEE, usando a variável de ambiente ``GRU`` para escolher
      entre prefixos ``gru`` e ``lstm``.
    """
    parser = argparse.ArgumentParser(
        description="Avalia predições ou converte resultados GRU/LSTM para o formato task1."
    )
    subparsers = parser.add_subparsers(dest="command")

    # Subcomando: evaluate
    eval_p = subparsers.add_parser("evaluate", help="Avalia predições contra ground truth")
    eval_p.add_argument("labels_file", help="Arquivo JSON com ground truth")
    eval_p.add_argument("predicted_file", help="Arquivo JSON com predições")
    eval_p.add_argument("output_file", nargs="?", default=None, help="Caminho de saída (opcional)")

    args = parser.parse_args()

    if args.command == "evaluate":
        evaluate_predictions(args.labels_file, args.predicted_file, args.output_file)
    else:
        use_gru = os.environ.get("GRU", "False").lower() in ("true", "1", "t")
        prefix = "gru" if use_gru else "lstm"
        input_file = f"output/results/{prefix}_results.json"
        output_file = f"output/results/{prefix}_parsed_result.json"
        logger.info("Parseando resultados de %s -> %s", input_file, output_file)
        parse_gru_results(input_file, output_file)


if __name__ == "__main__":
    main()
