# -*- coding: utf-8 -*-
__author__ = 'yshao'

import argparse
import os
import torch
import json
import logging
from pathlib import Path
from utils.paths import PathManager  # configura HF_HOME antes de qualquer import transformers

from tools.init_tool import init_all
from tools.poolout_tool import pool_out
from config_parser import create_config

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help="specific config file", required=True)
    parser.add_argument('--gpu', '-g', help="gpu id list")
    parser.add_argument('--checkpoint', help="checkpoint file path")
    parser.add_argument('--result', help="result file path", required=True)
    args = parser.parse_args()


    configFilePath = args.config

    use_gpu = True
    gpu_list = []
    if args.gpu is None:
        use_gpu = False
    else:
        use_gpu = True
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        device_list = args.gpu.split(",")
        for a in range(0, len(device_list)):
            gpu_list.append(int(a))

    os.system("clear")

    config = create_config(configFilePath)

    cuda = torch.cuda.is_available()
    logger.info("CUDA available: %s" % str(cuda))
    if not cuda and len(gpu_list) > 0:
        logger.warning("CUDA is not available but GPU was requested. Falling back to CPU execution.")
        use_gpu = False
        gpu_list = []

    # Initialize all directories and components
    # Set directory used in config file based on NLP directory files for training
    # For this example, we are using BertPoolOutMax.config (README.md for more details)
    # - directory: output/results/
    parameters = init_all(config, gpu_list, args.checkpoint, "poolout")

    # Create output directory if it doesn't exist
    result_path = Path(args.result)
    PathManager.ensure_dir(result_path.parent)
    
    out_file = open(result_path, 'w', encoding='utf-8')
    outputs = pool_out(parameters, config, gpu_list, args.result)
    logger.info(f"Total number of outputs: {outputs}")
    for output in outputs:
        tmp_dict = {
            'id_': output[0],
            'res': output[1]
        }
        out_line = json.dumps(tmp_dict, ensure_ascii=False) + '\n'
        out_file.write(out_line)
    out_file.close()

    # train(parameters, config, gpu_list)
