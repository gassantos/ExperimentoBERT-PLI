"""
scripts/test.py — CLI de inferência
=====================================
Ponto de entrada para avaliação/inferência de modelos BERT-PLI.

Uso:
    python -m scripts.test -c config/nlp/BertPoint.config \\
        --checkpoint output/checkpoints/bertpoint/10.pkl \\
        --result output/results/test_results.json
    bert-pli-test -c ...
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
from utils.config import create_config, ConfigParser
from tools.init_tool import init_all
from tools.test_tool import test

# Configura logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    """Ponto de entrada do CLI ``bert-pli-test``.

    Inicializa o modelo a partir de ``--checkpoint`` e ``--config``,
    executa a inferência na partição configurada e grava as predições
    em ``--result`` no formato JSON.
    """
    parser = argparse.ArgumentParser(description="Executa inferência com modelo BERT-PLI.")
    parser.add_argument("--config", "-c", help="Arquivo de configuração", required=True)
    parser.add_argument("--gpu", "-g", help="IDs de GPU separados por vírgula")
    parser.add_argument("--checkpoint", help="Caminho para checkpoint do modelo", required=True)
    parser.add_argument("--result", help="Caminho do arquivo de saída JSON", required=True)
    args = parser.parse_args()

    gpu_list = []
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        gpu_list = list(range(len(args.gpu.split(","))))

    config: ConfigParser = create_config(args.config) # type: ignore

    cuda = torch.cuda.is_available()
    logger.info("CUDA available: %s", cuda)
    if not cuda and gpu_list:
        logger.warning("CUDA indisponível mas GPU solicitada. Usando CPU.")
        gpu_list = []

    parameters = init_all(config, gpu_list, args.checkpoint, "test")

    PathManager.ensure_dir(Path(args.result).parent)

    if config.getboolean("output", "save_as_dict"):
        with open(args.result, "w", encoding="utf-8") as out_file:
            for output in test(parameters, config, gpu_list):
                out_file.write(
                    json.dumps({"id_": output[0], "res": output[1]}, ensure_ascii=False) + "\n"
                )
    else:
        json.dump(
            test(parameters, config, gpu_list),
            open(args.result, "w", encoding="utf-8"),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )


if __name__ == "__main__":
    main()
