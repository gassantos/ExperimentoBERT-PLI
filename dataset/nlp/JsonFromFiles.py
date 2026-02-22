"""Dataset que carrega exemplos a partir de arquivos JSON ou JSONL.

Suporta dois formatos:

- ``single``: cada arquivo contém uma lista JSON completa.
- ``jsonl`` (padrão): cada linha é um objeto JSON independente (JSONL).

Os exemplos podem ser carregados em memória (``load_into_mem = true``) ou
acessados de forma lazy via busca binária sobre prefixos de contagem de linhas.
"""
import json
from pathlib import Path
from torch.utils.data import Dataset

from tools.dataset_tool import dfs_search


class JsonFromFilesDataset(Dataset):
    """Dataset PyTorch que lê exemplos de um ou mais arquivos JSON/JSONL.

    Suporta:

    - Múltiplos arquivos concatenados virtualmente.
    - Modo em memória (``load_into_mem = true``) para datasets pequenos.
    - Acesso lazy com busca binária para datasets grandes.
    - Formato JSON completo por arquivo (``json_format = single``) ou JSONL.
    """

    def __init__(self, config, mode, encoding="utf8", *args, **params):
        """Inicializa o dataset a partir das configurações da secção ``[data]``.

        Args:
            config: instância de :class:`utils.config.ConfigParser` com as
                chaves ``{mode}_data_path``, ``{mode}_file_list``,
                ``recursive``, ``load_into_mem`` e ``json_format``.
            mode: partição do dataset — ``'train'``, ``'valid'`` ou ``'test'``.
            encoding: codificação dos arquivos (padrão: ``'utf8'``).
        """
        self.config = config
        self.mode = mode
        self.file_list = []
        self.data_path = Path(config.get("data", "%s_data_path" % mode))
        self.encoding = encoding

        filename_list = config.get("data", "%s_file_list" % mode).replace(" ", "").split(",")
        recursive = config.getboolean("data", "recursive")
        for name in filename_list:
            self.file_list = self.file_list + dfs_search(self.data_path / name, recursive)
        self.file_list.sort()

        self.load_mem = config.getboolean("data", "load_into_mem")
        self.json_format = config.get("data", "json_format")

        if self.load_mem:
            self.data = []
            for filename in self.file_list:
                if self.json_format == "single":
                    self.data = self.data + json.load(open(filename, "r", encoding=encoding))
                else:
                    f = open(filename, "r", encoding=encoding)
                    for line in f:
                        self.data.append(json.loads(line))

        else:
            self.total = 0
            self.prefix_file_cnt = []

            if self.json_format == "single":
                self.temp_data = {
                    "data": json.load(open(self.file_list[0], "r", encoding=encoding)),
                    "file_id": 0
                }
            else:
                self.temp_file_list = []

            for filename in self.file_list:
                if self.json_format == "single":
                    data = json.load(open(filename, "r", encoding=encoding))
                    self.prefix_file_cnt.append(len(data))
                else:
                    f = open(filename, "r", encoding=encoding)
                    cnt = 0
                    for line in f:
                        cnt += 1
                    f.close()
                    self.temp_file_list.append({
                        "file": open(filename, "r", encoding=encoding),
                        "cnt": 0
                    })
                    self.prefix_file_cnt.append(cnt)

            for a in range(1, len(self.prefix_file_cnt)):
                self.prefix_file_cnt[a] += self.prefix_file_cnt[a - 1]
            self.total = self.prefix_file_cnt[-1]

    def get_file_id(self, item):
        """Retorna o índice do arquivo que contém o exemplo de posição *item*.

        Utiliza busca binária sobre ``self.prefix_file_cnt`` (prefixo de somas
        de tamanhos), operando em O(log F) onde F é o número de arquivos.

        Args:
            item: índice global do exemplo no dataset concatenado.

        Returns:
            Índice do arquivo em ``self.file_list``.
        """
        l = 0
        r = len(self.prefix_file_cnt)
        while l + 1 != r:
            m = (l + r) // 2
            if self.prefix_file_cnt[m-1] <= item:
                l = m
            else:
                r = m

        return l

    def __getitem__(self, item):
        """Retorna o exemplo de índice *item*.

        Se ``load_into_mem`` estiver ativo, acessa ``self.data[item]``
        diretamente; caso contrário, determina o arquivo via
        :meth:`get_file_id` e lê o item correspondente do disco.

        Args:
            item: índice do exemplo (0-based).

        Returns:
            Dicionário com os campos do exemplo JSON.
        """
        if self.load_mem:
            return self.data[item]
        else:
            which = self.get_file_id(item)
            if which == 0:
                idx = item
            else:
                idx = item - self.prefix_file_cnt[which - 1]

            if self.json_format == "single":
                if self.temp_data["file_id"] != which:
                    self.temp_data = {
                        "data": json.load(open(self.file_list[which], "r", encoding=self.encoding)),
                        "file_id": 0
                    }

                return self.temp_data["data"][idx]

            else:
                if self.temp_file_list[which]["cnt"] > idx:
                    self.temp_file_list[which] = {
                        "file": open(self.file_list[which], "r", encoding=self.encoding),
                        "cnt": 0
                    }

                delta = idx - self.temp_file_list[which]["cnt"]
                self.temp_file_list[which]["file"].readlines(delta)

                data = json.loads(self.temp_file_list[which]["file"].readline())
                self.temp_file_list[which]["cnt"] = idx + 1

                return data

    def __len__(self):
        """Retorna o número total de exemplos no dataset."""
        if self.load_mem:
            return len(self.data)
        else:
            return self.total
