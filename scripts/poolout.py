"""
scripts/poolout.py — CLI de extração de embeddings (pool out)
==============================================================
Extrai representações de parágrafos usando BertPoolOutMax e salva
em arquivo JSON para treino subsequente via AttenRNN.

Uso:
    python -m scripts.poolout -c config/nlp/BertPoolOutMax.config \\
        --checkpoint output/checkpoints/bert_poolout/10.pkl \\
        --result output/results/poolout.json
    bert-pli-poolout -c ...
"""

import sys
from pathlib import Path

# Deve vir antes de qualquer import do projeto
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import logging
import os
import torch

from utils.paths import PathManager  # configura HF_HOME antes de qualquer import transformers
from utils.config import create_config
from tools.init_tool import init_all
from tools.poolout_tool import pool_out

# Raiz do projeto no sys.path para executar script
sys.path.insert(0, str(PathManager.get_projectdir()))

# Configura logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    """Ponto de entrada do CLI ``bert-pli-poolout``.

    Carrega um checkpoint de ``BertPoolOutMax``, processa todos os pares de
    parágrafos configurados e salva os embeddings em arquivo JSON.
    Argumentos: ``--config``, ``--gpu``, ``--checkpoint``, ``--result``.
    """
    parser = argparse.ArgumentParser(description="Extrai embeddings pool-out de parágrafos.")
    parser.add_argument("--config", "-c", help="Arquivo de configuração", required=True)
    parser.add_argument("--gpu", "-g", help="IDs de GPU separados por vírgula")
    parser.add_argument("--checkpoint", help="Caminho para checkpoint do modelo")
    parser.add_argument("--result", help="Caminho do arquivo de saída JSON", required=True)
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

    parameters = init_all(config, gpu_list, args.checkpoint, "poolout")

    result_path = Path(args.result)
    PathManager.ensure_dir(result_path.parent)

    outputs = pool_out(parameters, config, gpu_list, args.result)
    logger.info("Total de outputs gerados: %d", outputs)

    with open(result_path, "w", encoding="utf-8") as out_file:
        for output in outputs:
            out_file.write(
                json.dumps({"id_": output[0], "res": output[1]}, ensure_ascii=False) + "\n"
            )


if __name__ == "__main__":
    main()
