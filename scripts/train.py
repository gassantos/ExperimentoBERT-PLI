"""
scripts/train.py — CLI de treinamento
======================================
Ponto de entrada para o pipeline de fine-tuning BERT-PLI.

Uso:
    python -m scripts.train -c config/nlp/BertPoint.config
    bert-pli-train -c config/nlp/BertPoint.config
"""

import sys
from pathlib import Path

# Deve vir antes de qualquer import do projeto
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import os
import logging
import torch

from utils.paths import PathManager  # configura HF_HOME antes de qualquer import transformers
from utils.config import create_config
from tools.init_tool import init_all
from tools.train_tool import train

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    """Ponto de entrada do CLI ``bert-pli-train``.

    Lê ``--config``, seleciona GPUs via ``--gpu`` e inicia (ou retoma de
    ``--checkpoint``) o pipeline de fine-tuning do modelo configurado.
    """
    parser = argparse.ArgumentParser(description="Treina um modelo BERT-PLI.")
    parser.add_argument("--config", "-c", help="Arquivo de configuração", required=True)
    parser.add_argument("--gpu", "-g", help="IDs de GPU separados por vírgula (ex: 0,1)")
    parser.add_argument("--checkpoint", help="Caminho para checkpoint de retomada")
    args = parser.parse_args()

    gpu_list = []
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        gpu_list = list(range(len(args.gpu.split(","))))

    config = create_config(args.config)

    cuda = torch.cuda.is_available()
    logger.info("CUDA available: %s", cuda)
    if not cuda and gpu_list:
        logger.warning("CUDA indisponível mas GPU solicitada. Usando CPU.")
        gpu_list = []

    parameters = init_all(config, gpu_list, args.checkpoint, "train")
    train(parameters, config, gpu_list)


if __name__ == "__main__":
    main()
