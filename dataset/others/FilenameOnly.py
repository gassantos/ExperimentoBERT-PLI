"""Dataset que expõe apenas os caminhos dos arquivos de entrada.

Utilizado na etapa de extraçào de pool-out (:mod:`tools.poolout_tool`),
onde cada item do lote é o caminho de um arquivo e o embedding é calculado
pelo modelo ``BertPoolOutMax`` sob demanda.
"""
from pathlib import Path
from torch.utils.data import Dataset

from tools.dataset_tool import dfs_search


class FilenameOnlyDataset(Dataset):
    """Dataset PyTorch que retorna apenas caminhos de arquivo como itens.

    A lista de arquivos é construída via busca DFS nos diretórios configurados,
    suportando padrões glob e recursividade.
    """

    def __init__(self, config, mode, *args, **params):
        """Constrói a lista de arquivos a partir das configurações de ``[data]``.

        Args:
            config: instância de :class:`utils.config.ConfigParser` com as
                chaves ``{mode}_data_path``, ``{mode}_file_list`` e
                ``recursive``.
            mode: partição — ``'train'``, ``'valid'`` ou ``'test'``.
        """
        self.config = config
        self.mode = mode
        self.file_list = []
        self.data_path = Path(config.get("data", "%s_data_path" % mode))

        filename_list = config.get("data", "%s_file_list" % mode).replace(" ", "").split(",")
        recursive = config.getboolean("data", "recursive")

        for name in filename_list:
            self.file_list = self.file_list + dfs_search(self.data_path / name, recursive)
        self.file_list.sort()

    def __getitem__(self, item):
        """Retorna o caminho do arquivo de índice *item*."""
        return self.file_list[item]

    def __len__(self):
        """Retorna o número de arquivos no dataset."""
        return len(self.file_list)
