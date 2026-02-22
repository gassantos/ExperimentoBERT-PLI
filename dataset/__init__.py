"""Pacote de datasets do projeto BERT-PLI.

Registra as implementações de ``torch.utils.data.Dataset`` disponíveis:

- **JsonFromFiles**: carrega exemplos de arquivos JSON (formato ``single`` ou
  JSONL).  Suporta leitura em memória ou acesso sequencial lazy.
- **FilenameOnly**: expõe apenas os caminhos de arquivo, útil na etapa de
  pool-out onde o batch carrega o embedding pré-computado.

Uso::

    from dataset import dataset_list
    DatasetClass = dataset_list[config.get('data', 'dataset_type')]
"""
from .nlp.JsonFromFiles import JsonFromFilesDataset
from .others.FilenameOnly import FilenameOnlyDataset

dataset_list = {
    "JsonFromFiles": JsonFromFilesDataset,
    "FilenameOnly": FilenameOnlyDataset
}
