import json
import logging
from pathlib import Path
from typing import Union, List

from tqdm import tqdm

logger = logging.getLogger(__name__)


def dfs_search(path: Union[str, Path], recursive: bool) -> List[str]:
    """
    Busca recursiva de arquivos em um diretório.
    
    Args:
        path: Caminho do diretório ou arquivo
        recursive: Se True, busca recursivamente
        
    Returns:
        Lista de caminhos absolutos de arquivos como strings
    """
    path = Path(path)
    
    if path.is_file():
        return [str(path)]
    
    file_list = []
    name_list = sorted(path.iterdir())
    
    for item in name_list:
        if item.is_dir():
            if recursive:
                file_list.extend(dfs_search(item, recursive))
        else:
            file_list.append(str(item))
    
    return file_list


# ---------------------------------------------------------------------------
# Pool-out → treino
# ---------------------------------------------------------------------------

def load_json_lines(file_path: Union[str, Path]) -> dict:
    """
    Carrega um arquivo JSON Lines (um objeto JSON por linha) em um dicionário
    indexado por ``id_`` (se presente) ou ``guid``.

    Args:
        file_path: Caminho do arquivo JSON Lines

    Returns:
        Dicionário ``{chave: item}``
    """
    data: dict = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            key = item.get("id_") or item.get("guid")
            if key is None:
                logger.warning("Item sem 'id_' ou 'guid' ignorado: %s", line[:80])
                continue
            data[key] = item
    return data


def process_poolout_files(
    in_file_path: Union[str, Path],
    out_file_path: Union[str, Path],
    result_file_path: Union[str, Path],
) -> None:
    """
    Combina o arquivo de parágrafos (``in_file_path``) com o arquivo de
    embeddings extraídos pelo pool-out (``out_file_path``) e grava o resultado
    final em ``result_file_path`` no formato JSON Lines::

        {"guid": "...", "res": [...], "label": 0|1}

    Args:
        in_file_path: Arquivo de parágrafos de entrada (JSON Lines)
        out_file_path: Arquivo gerado pelo pool-out (JSON Lines)
        result_file_path: Caminho de saída do resultado (JSON Lines)
    """
    logger.info("Carregando arquivo de entrada: %s", in_file_path)
    in_data = load_json_lines(in_file_path)

    logger.info("Carregando arquivo de pool-out: %s", out_file_path)
    out_data = load_json_lines(out_file_path)

    result_path = Path(result_file_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Processando e gravando resultado em: %s", result_file_path)
    with open(result_path, "w", encoding="utf-8") as f:
        for guid, in_item in tqdm(in_data.items(), desc="Processando itens"):
            if guid in out_data:
                record = {
                    "guid": guid,
                    "res": out_data[guid]["res"],
                    "label": in_item["label"],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                logger.warning("Nenhuma saída encontrada para guid: %s", guid)
