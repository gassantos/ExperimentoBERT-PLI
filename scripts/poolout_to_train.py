"""
scripts/poolout_to_train.py — CLI de conversão pool-out → treino
================================================================
Combina o arquivo de embeddings extraídos (pool-out) com os labels
originais, gerando o formato de entrada para o AttenRNN.

Uso:
    python -m scripts.poolout_to_train \\
        --paras-file data/train_task2.json \\
        --poolout-file output/results/poolout.json \\
        --result output/results/poolout_train.json
    bert-pli-poolout-to-train ...
"""

import sys
from pathlib import Path

# Deve vir antes de qualquer import do projeto
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import logging

from tools.dataset_tool import process_poolout_files

# Configura logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    """Ponto de entrada do CLI ``bert-pli-poolout-to-train``.

    Combina o arquivo de parágrafos originais (``--paras-file``) com os
    embeddings extraídos (``--poolout-file``) e grava o resultado em
    ``--result``, pronto para treino do AttentionRNN.
    """
    parser = argparse.ArgumentParser(
        description="Converte saída de pool-out para formato de treino RNN."
    )
    parser.add_argument("--paras-file", "-in", help="Arquivo de parágrafos de entrada", required=True)
    parser.add_argument("--poolout-file", "-out", help="Arquivo de pool-out gerado", required=True)
    parser.add_argument("--result", help="Caminho do arquivo de resultado", required=True)
    args = parser.parse_args()

    process_poolout_files(args.paras_file, args.poolout_file, args.result)
    logger.info("Processamento concluído.")


if __name__ == "__main__":
    main()
